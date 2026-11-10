from __future__ import annotations

from pathlib import Path
import weakref

import numpy as np

from .RenderContext import (
    BufferUsage,
    ComputePipeline,
    GpuBuffer,
    RenderContext,
    Shader,
    Texture,
    TextureUsage,
)


_default_blue_noise = None


def get_default_blue_noise():
	"""Lazily upload the packaged tile, sharing it among active device users.

	The weak cache does not retain GPU devices after renderers are released.
	A closed texture or a changed RenderContext device/generation is a cache miss.
	Only NumPy is needed to decode the packaged sample data.
	"""
	global _default_blue_noise
	texture = _default_blue_noise() if _default_blue_noise is not None else None
	if texture is not None:
		try:
			texture._check_device()
			return texture
		except RuntimeError:
			pass
	from .Assets import load_blue_noise
	thresholds = load_blue_noise()
	height, width = thresholds.shape
	texture = Texture((width, height), format="r32float",
		usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST, label="default_blue_noise")
	RenderContext.device.queue.write_texture({"texture": texture.handle}, thresholds,
		{"bytes_per_row": width*4}, (width, height, 1))
	_default_blue_noise = weakref.ref(texture)
	return texture


def build_material_texture_array(images, materials, size):
	"""Upload source images, then resample/filter/correct/encode mips on the GPU.

	Uncorrected premultiplied linear RGBA32F scratch mips retain tiny alpha values.
	MASK coverage search uses GPU reductions; no CPU filtering or read-back occurs.
	Encoded bytes are copied from a storage texture into the sRGB sampling array.
	This also works with WGPU backends that do not support mutable texture views.
	"""
	width, height = size
	levels = max(width, height).bit_length()
	result = Texture((*size, len(images)), format="rgba8unorm-srgb", mip_level_count=levels,
		usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="material_mip_array")
	encoded = Texture((*size, 1), format="rgba8unorm", mip_level_count=levels,
		usage=TextureUsage.STORAGE_BINDING | TextureUsage.COPY_SRC, label="encoded_mip_scratch")
	shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "material_mipmaps.shader"))
	pipelines = {entry: ComputePipeline(shader, entry=entry) for entry in (
		"decode", "resample", "init_source", "measure_source", "begin_search", "measure_candidate", "update_search", "encode")}
	def dispatch(compute, entry, resources, x=1, y=1):
		compute.set_pipeline(pipelines[entry])
		compute.set_bind_group(shader.bind_group(0, **resources))
		compute.dispatch(x, y)

	for layer, (image, material) in enumerate(zip(images, materials)):
		source_height, source_width, _ = image.shape
		source_size = (source_width, source_height)
		upload = Texture(source_size, format="rgba8unorm-srgb",
			usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": upload.handle}, np.ascontiguousarray(image),
			{"bytes_per_row": source_width*4, "rows_per_image": source_height}, (*source_size, 1))
		scratch_usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
		original = Texture(source_size, format="rgba32float", usage=scratch_usage)
		working = Texture(size, format="rgba32float", usage=scratch_usage, mip_level_count=levels)
		mode, factor = int(material["surface"][2]), float(material["base_color"][3])
		cutoff = float(material["surface"][3])/factor if factor > 0 else 0
		preserve = mode == 1 and 0 < cutoff <= 1
		params = shader.uniform_buffer("params")
		params.content["info"] = (0, mode, preserve, 0)
		params.content["cutoff"] = (cutoff, 0, 0, 0)
		params.upload()
		state = GpuBuffer(np.zeros(12, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		cmd = RenderContext.commands(f"material_mips_{layer}")
		with cmd.compute_pass(label="decode_and_source_coverage") as compute:
			dispatch(compute, "decode", dict(params=params, source=upload, linear_destination=original),
				(source_width+7)//8, (source_height+7)//8)
			if preserve:
				dispatch(compute, "init_source", dict(source=original, coverage=state))
				dispatch(compute, "measure_source", dict(params=params, source=original, coverage=state),
					(source_width+7)//8, (source_height+7)//8)
		previous = original.view()
		for level in range(levels):
			mip_width, mip_height = max(1, width >> level), max(1, height >> level)
			current = working.view(base_mip_level=level, mip_level_count=1, dimension="2d")
			destination = encoded.view(base_mip_level=level, mip_level_count=1, dimension="2d-array")
			groups = ((mip_width+7)//8, (mip_height+7)//8)
			with cmd.compute_pass(label=f"generate_mip_{level}") as compute:
				dispatch(compute, "resample", dict(source=previous, linear_destination=current), *groups)
				if preserve:
					dispatch(compute, "begin_search", dict(source=current, coverage=state))
					# Bracket by doubling, then bisect. Converged searches skip sample
					# work; fixed bounds cover RGBA8 source alpha through tiny mips.
					for _ in range(56):
						dispatch(compute, "measure_candidate", dict(params=params, source=current, coverage=state), *groups)
						dispatch(compute, "update_search", dict(params=params, coverage=state))
				dispatch(compute, "encode", dict(params=params, source=current,
					encoded_destination=destination, coverage=state), *groups)
			cmd.handle.copy_texture_to_texture(
				{"texture": encoded.handle, "mip_level": level},
				{"texture": result.handle, "mip_level": level, "origin": (0, 0, layer)},
				(mip_width, mip_height, 1))
			previous = current
		RenderContext.submit(cmd.finish())
	return result


def build_correlated_noise_hierarchy(source):
	"""Build nested rank-preserving noise levels on the GPU, without readback.

	Parents average each 2x2 child block, then receive unique integer ranks again.
	Uniform per-tile rank jitter at sampling time retains continuous marginals,
	including the 1x1 root. Small noise tiles (up to 64x64) bound the rank pass cost.
	"""
	width, height, layers = source.size
	if width != height or layers != 1 or width > 64 or width & (width-1):
		raise ValueError("Noise hierarchy requires a square power-of-two tile up to 64x64")
	levels = width.bit_length()
	usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
	result = Texture((width, height), format="r32float", usage=usage,
		mip_level_count=levels, label="correlated_noise_ranks")
	shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "noise_hierarchy.shader"))
	pipelines = {name: ComputePipeline(shader, entry=name) for name in ("rank_values", "average_children")}
	commands = RenderContext.commands("correlated_noise_hierarchy")
	previous = source
	for level in range(levels):
		size = width >> level
		current = result.view(base_mip_level=level, mip_level_count=1, dimension="2d")
		if level:
			scratch = Texture((size, size), format="r32float", usage=usage)
			with commands.compute_pass(label=f"noise_parent_{level}") as compute:
				compute.set_pipeline(pipelines["average_children"])
				compute.set_bind_group(shader.bind_group(0, source=previous, destination=scratch))
				compute.dispatch((size+7)//8, (size+7)//8)
			previous = scratch
		with commands.compute_pass(label=f"noise_ranks_{level}") as compute:
			compute.set_pipeline(pipelines["rank_values"])
			compute.set_bind_group(shader.bind_group(0, source=previous, destination=current))
			compute.dispatch((size+7)//8, (size+7)//8)
		previous = current
	RenderContext.submit(commands.finish())
	return result

