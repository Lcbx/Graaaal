"""Read-back checks of sample coverage, opaque occlusion and live MSAA switches."""
from dataclasses import replace

import numpy as np
import pytest

from numpyecs import TransparencyPass as transparency
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.TransparencyData import Primitive, MATERIAL_DTYPE, SHADING_VERTEX_DTYPE, VERTEX_DTYPE
from scenes import transparency_example as scene


def _geometry(opaque_occluder):
	materials = np.zeros(2, dtype=MATERIAL_DTYPE)
	materials["base_color"] = (0, 0, 0, 1)
	materials["emission"][0] = (1, 0, 0, 0)
	materials["opacity_rgb"] = 1
	materials["surface"][:, :2] = (1, 1)
	materials["surface"][:, 2] = (2, 0)
	materials["flags"][:, 1] = 1
	primitives, stored = [], []
	positions = [(-1.31, -1.07, 0.2), (1.23, -0.83, 0.2), (-0.17, 1.26, 0.2)]
	geometries = [(positions, 0)]
	if opaque_occluder:
		geometries = [([(-100, -100, 0), (100, -100, 0), (0, 100, 0)], 0), (positions, 1)]
	for index, (positions, material) in enumerate(geometries):
		vertices = np.zeros(3, dtype=VERTEX_DTYPE)
		vertices["position"] = positions
		vertices["normal"] = (0, 0, 1)
		vertices["barycentric"] = [(1, 0), (0, 1), (0, 0)]
		vertices["uv"] = [(0, 0), (1, 0), (0, 1)]
		primitives.append(Primitive(vertices, material, index, index))
		shading = np.zeros(3, dtype=SHADING_VERTEX_DTYPE)
		shading["position"][:, :3] = vertices["position"]
		shading["normal"][:, :3] = vertices["normal"]
		shading["uv"][:, :2] = vertices["uv"]
		stored.append(shading)
	images = [np.full((1, 1, 4), 255, dtype=np.uint8) for _ in materials]
	return primitives, materials, images, np.concatenate(stored)


def _read(texture, dtype, channels, layers=1):
	width, height = texture.size[:2]
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": width*channels*np.dtype(dtype).itemsize, "rows_per_image": height}, (width, height, layers))
	return np.frombuffer(data, dtype=dtype).reshape(layers, height, width, channels).astype(np.float32)


def _linear(encoded):
	encoded = encoded/255
	return np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)


@pytest.mark.parametrize("opaque_occluder", [False, True])
@pytest.mark.parametrize("initial_count", [1, 4])
def test_msaa_resolves_coverage_in_linear_hdr_and_preserves_live_batches(monkeypatch, opaque_occluder, initial_count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: _geometry(opaque_occluder))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	monkeypatch.setattr(RenderContext, "window_dimensions", (48, 48))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	# Native MSAA attachments cannot have COPY_SRC; only read the shaded array.
	def readable_texture(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1:
			usage |= TextureUsage.COPY_SRC
		return Texture(*args, usage=usage, **kwargs)
	monkeypatch.setattr(transparency, "Texture", readable_texture)
	scene.init_scene(sample_count=initial_count, bilateral_filter=scene.BilateralFilter(kernel_size=1))
	oit = scene.transparency
	assert (oit.noise_mode, oit.winner_count, oit.winner_mix) == ("anchored_blue", 2, "average")
	hiz_calls = []
	original_hiz = scene.HiZCuller.record
	original_build = scene.hzb.build
	def record_hiz(culler, *args, **kwargs):
		assert oit.sample_count == 1, "Pixel-center Hi-Z must not cull MSAA sample visibility"
		hiz_calls.append(oit.sample_count)
		return original_hiz(culler, *args, **kwargs)
	def build_hzb(*args, **kwargs):
		assert oit.sample_count == 1, "Skip the unused single-sample depth prepass in MSAA mode"
		return original_build(*args, **kwargs)
	monkeypatch.setattr(scene.HiZCuller, "record", record_hiz)
	monkeypatch.setattr(scene.hzb, "build", build_hzb)
	registered = scene.transparent.data.shaders[0]
	scene.camera.position, scene.camera.target = scene.Vec3(0, 0, 3), scene.Vec3(0, 0, 0)
	oit.set_noise_mode("triplanar_correlated")
	oit.set_noise_scale(1.5)
	oit.set_noise_footprint_balance(0.5)
	hierarchy = oit._noise_hierarchy
	for count, width, kernel in ((initial_count, 48, 1), (5-initial_count, 40, 9), (initial_count, 48, 9)):
		output = Texture((width, 48), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		monkeypatch.setattr(RenderContext, "window_dimensions", (width, 48))
		monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
		oit.set_sample_count(count)
		assert not oit.set_sample_count(count)
		assert oit.noise_scale == 1.5
		assert oit.noise_footprint_balance == 0.5
		assert oit.uniforms.content["noise_footprint_balance"] == 0.5
		assert oit.effective_noise_scale == 1.5/(count**0.5)
		assert oit.uniforms.content["clip"][..., 2] == oit.effective_noise_scale
		assert scene.transparent.data.shaders[0] is registered
		assert scene.transparent.passes.render_shader is registered
		assert oit._noise_hierarchy is hierarchy
		assert oit.noise_mode == "triplanar_correlated"
		oit.set_bilateral_filter(replace(oit.bilateral_filter, kernel_size=kernel))
		oit.set_debug_view("final")
		previous_hiz_calls = len(hiz_calls)
		scene.render_frame()
		assert oit.uniforms.content["clip"][..., 2] == oit.effective_noise_scale
		assert len(hiz_calls)-previous_hiz_calls == (2 if count == 1 else 0)
		winners = _read(oit.winner_color, np.float16, 4, count)
		flags = winners[..., 3]
		assert set(np.unique(flags)) == {0, 1}
		assert np.all(winners[..., 0] == flags)  # exact red shade, including split payloads
		assert not winners[..., 1:3].any()
		coverage = flags.mean(axis=0)
		assert np.any((coverage > 0) & (coverage < 1)) == (count == 4)
		hdr = coverage[..., None]*np.array([1, 0, 0])
		if not opaque_occluder:
			hdr += (1-coverage[..., None])*np.array([0.035, 0.05, 0.08])
		np.testing.assert_allclose(_linear(_read(output, np.uint8, 4)[0, ..., :3]), hdr/(1+hdr), atol=0.004)
		oit.set_debug_view("transmittance")
		scene.render_frame()
		# Guide/transmittance must agree with per-sample winner occlusion.
		np.testing.assert_allclose(_linear(_read(output, np.uint8, 4)[0, ..., 0]), 1-coverage, atol=0.004)
	# Editing the base width while MSAA is active must survive repeated toggles.
	oit.set_sample_count(4)
	oit.set_noise_scale(1.25)
	assert oit.effective_noise_scale == 0.625
	for count in (1, 4, 1):
		oit.set_sample_count(count)
		assert oit.noise_scale == 1.25
		assert oit.uniforms.content["clip"][..., 2] == 1.25/(count**0.5)


@pytest.mark.parametrize("count", [0, 2, 8, True, 4.0, None])
def test_sample_count_validation_precedes_resource_changes(count):
	oit = object.__new__(transparency.TransparencyPass)
	with pytest.raises(ValueError, match="sample_count"):
		oit.set_sample_count(count)


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_footprint_feature_requires_boolean(value):
	oit = object.__new__(transparency.TransparencyPass)
	with pytest.raises(ValueError, match="boolean"): oit.set_noise_footprint_balancing(value)


@pytest.mark.parametrize("pixels,subdivisions", [(0, 4), (float("nan"), 4), (257, 4), (True, 4), (24, 0), (24, 33), (24, 1.5)])
def test_anchor_settings_validate_before_uniform_changes(pixels, subdivisions):
	oit = object.__new__(transparency.TransparencyPass)
	with pytest.raises(ValueError): oit.set_noise_anchor(pixels, subdivisions)
