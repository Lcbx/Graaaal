"""CPU loader checks and offscreen, read-back checks of the actual OIT passes."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.Assets import gltf_node_matrix, load_blue_noise
from numpyecs import TransparencyData as transparency_data
from numpyecs import TextureProcessing as texture_processing
from scenes import transparency_example as scene
from numpyecs import TransparencyPass as transparency


def test_bundled_blue_noise_loads_as_deterministic_linear_thresholds():
	noise = load_blue_noise(scene.BLUE_NOISE_PATH)
	np.testing.assert_array_equal(noise, load_blue_noise(scene.BLUE_NOISE_PATH))
	assert noise.shape == (64, 64) and noise.dtype == np.float32
	assert 0 < noise.min() < noise.max() < 1
	assert len(np.unique(noise)) == noise.size  # retain the full 16-bit ranks
	# No sRGB conversion: a uniform threshold distribution stays uniform.
	# The source stores 4096 discrete ranks in a 16-bit container; allow one
	# rank's quantization error rather than assuming it spans every 16-bit bin.
	assert abs(noise.mean()-0.5) < 1/noise.size
	np.testing.assert_array_equal(np.histogram(noise, bins=16, range=(0, 1))[0], 256)


@pytest.mark.parametrize("options", [
	{"kernel_size": 4}, {"kernel_size": 3.0}, {"spatial_sigma": 0},
	{"depth_sigma": -1}, {"transmittance_sigma": float("nan")},
	{"min_depth_sigma": float("inf")},
	{"kernel_size": 7}, {"kernel_size": True}, {"kernel_size": 0},
])
def test_bilateral_filter_rejects_invalid_footprints_and_tolerances(options):
	with pytest.raises(ValueError):
		scene.BilateralFilter(**options)


def test_node_matrix_uses_gltf_column_major_matrix_and_trs():
	trs = SimpleNamespace(matrix=None, rotation=[0, 0, 0, 1], scale=[2, 3, 4], translation=[5, 6, 7])
	matrix = gltf_node_matrix(trs)
	np.testing.assert_allclose(matrix @ [1, 1, 1, 1], [7, 9, 11, 1])
	np.testing.assert_array_equal(gltf_node_matrix(SimpleNamespace(matrix=matrix.T.reshape(-1).tolist())), matrix)


def test_loader_bakes_hierarchy_reflection_and_default_material(tmp_path):
	positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
	# A non-indexed primitive with no normals or UVs also exercises fallbacks.
	gltf = {
		"asset": {"version": "2.0"},
		"buffers": [{"byteLength": positions.nbytes, "uri": "data:application/octet-stream;base64," + base64.b64encode(positions.tobytes()).decode()}],
		"bufferViews": [{"buffer": 0, "byteLength": positions.nbytes}],
		"accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
		"nodes": [{"translation": [2, 3, 4], "children": [1]}, {"mesh": 0, "scale": [-2, 1, 1]}],
		"scenes": [{"nodes": [0]}], "scene": 0,
	}
	path = tmp_path / "triangle.gltf"
	path.write_text(json.dumps(gltf))
	primitives, materials, images, stored = transparency_data.load_transparency_scene(path)
	vertices = primitives[0].vertices
	np.testing.assert_allclose(vertices["position"], [[2, 3, 4], [2, 4, 4], [0, 3, 4]])
	np.testing.assert_allclose(vertices["normal"], [[0, 0, 1]]*3)
	np.testing.assert_array_equal(stored["position"][:, :3], vertices["position"])
	assert primitives[0].material_id == 0
	assert materials[0]["surface"][2] == 0
	assert len(images) == 1


def test_real_scene_has_opaque_mask_and_blended_geometry():
	primitives, materials, images, vertices = transparency_data.load_transparency_scene(scene.PATH)
	assert {int(materials[p.material_id]["surface"][2]) for p in primitives} == {0, 1, 2}
	assert len(images) == len(materials)
	for primitive in primitives:
		start = primitive.triangle_offset*3
		count = len(primitive.vertices)
		np.testing.assert_array_equal(vertices["position"][start:start+count, :3], primitive.vertices["position"])
	assert np.isfinite(vertices["normal"]).all()
	# The example overrides only the named canopy, keeping generic loading faithful.
	_, cutout_materials, _, _ = scene.load_transparency_scene(scene.PATH)
	changed = cutout_materials["surface"][:, 2] != materials["surface"][:, 2]
	assert changed.sum() == 1
	assert materials["surface"][changed, 2] == 2
	assert cutout_materials["surface"][changed, 2] == 1
	assert cutout_materials["flags"][changed, 2] == 0  # no geometry-specific anchoring flags


def mip_material(mode=2, alpha_factor=1, cutoff=0.5):
	material = np.zeros(1, dtype=transparency.MATERIAL_DTYPE)[0]
	material["base_color"] = (1, 1, 1, alpha_factor)
	material["surface"] = (1, 0, mode, cutoff)
	return material


def alpha_coverage(alpha, cutoff):
	"""Independent CPU measurement of GPU-produced repeat/bilinear coverage."""
	count = 0
	for y in (-1, 1):
		row = np.roll(alpha, y, axis=0)
		for x in (-1, 1):
			samples = alpha*0.5625 + row*0.1875 + np.roll(alpha, x, axis=1)*0.1875 + np.roll(row, x, axis=1)*0.0625
			count += np.count_nonzero(samples >= cutoff)
	return count/(4*alpha.size)


@pytest.fixture
def gpu_mips(monkeypatch):
	RenderContext.setup_graphics_backend(True)
	monkeypatch.setattr(texture_processing, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	def generate(image, material, size=None):
		size = size or (image.shape[1], image.shape[0])
		texture = texture_processing.build_material_texture_array([image], np.array([material], dtype=transparency.MATERIAL_DTYPE), size)
		mips = []
		for level in range(max(size).bit_length()):
			width, height = max(1, size[0] >> level), max(1, size[1] >> level)
			data = RenderContext.device.queue.read_texture({"texture": texture.handle, "mip_level": level},
				{"bytes_per_row": width*4, "rows_per_image": height}, (width, height, 1))
			mips.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4).copy())
		return mips
	return generate


def test_blend_mips_preserve_mean_opacity_and_ignore_transparent_rgb(gpu_mips):
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	visible = np.indices((16, 16)).sum(axis=0) % 2 == 0
	image[visible] = (255, 0, 0, 255)
	image[~visible] = (0, 255, 0, 0)
	mips = gpu_mips(image, mip_material())
	assert [mip.shape[:2] for mip in mips] == [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
	for mip in mips[1:]:
		assert abs(mip[:, :, 3].mean()/255-0.5) <= 1/255
		np.testing.assert_array_equal(mip[:, :, :3], np.broadcast_to((255, 0, 0), mip[:, :, :3].shape))


def test_opaque_mips_filter_in_linear_space_and_ignore_alpha_for_rgb(gpu_mips):
	image = np.array([[(255, 255, 255, 0), (0, 0, 0, 255)]], dtype=np.uint8)
	last = gpu_mips(image, mip_material(mode=0))[-1]
	np.testing.assert_allclose(last[0, 0, :3], (188, 188, 188), atol=1)


@pytest.mark.parametrize("mode", [1, 2])
def test_cutout_mips_pad_transparent_rgb_without_changing_alpha(gpu_mips, mode):
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	color = (32, 220, 72)
	image[4:12, :4] = (*color, 255)
	mips = gpu_mips(image, mip_material(mode=mode, cutoff=0.2))
	for mip in mips:
		alpha = mip[:, :, 3]
		neighbor = np.zeros_like(alpha, dtype=bool)
		for y in (-1, 0, 1):
			for x in (-1, 0, 1):
				neighbor |= np.roll(alpha > 0, (y, x), axis=(0, 1))
		fringe = (alpha == 0) & neighbor
		if fringe.any():
			np.testing.assert_allclose(mip[:, :, :3][fringe], np.broadcast_to(color, mip[:, :, :3][fringe].shape), atol=1)
	# This donor crosses the repeat boundary. RGB may bleed; alpha must not.
	np.testing.assert_allclose(mips[0][8, -1, :3], color, atol=1)
	assert mips[0][8, -1, 3] == 0
	if mode == 2:
		for mip in mips:
			assert abs(mip[:, :, 3].mean()-image[:, :, 3].mean()) <= 1


@pytest.mark.parametrize("alpha_factor,cutoff", [(1, 0.5), (0.5, 0.35)])
def test_mask_mips_preserve_cutoff_coverage_including_material_factor(gpu_mips, alpha_factor, cutoff):
	rng = np.random.default_rng(13)
	image = np.full((64, 64, 4), 255, dtype=np.uint8)
	image[:, :, 3] = (rng.random((64, 64)) < 0.4)*255
	mask = mip_material(mode=1, alpha_factor=alpha_factor, cutoff=cutoff)
	plain = mip_material(mode=2, alpha_factor=alpha_factor, cutoff=cutoff)
	# Include the initial source-to-array resize in the coverage guarantee.
	corrected = gpu_mips(image, mask, (32, 32))
	unscaled = gpu_mips(image, plain, (32, 32))
	effective_cutoff = cutoff/alpha_factor
	target = alpha_coverage(image[:, :, 3]/255, effective_cutoff)
	best_improvement = 0
	for mip, raw in zip(corrected, unscaled):
		error = abs(alpha_coverage(mip[:, :, 3]/255, effective_cutoff)-target)
		raw_error = abs(alpha_coverage(raw[:, :, 3]/255, effective_cutoff)-target)
		assert error <= raw_error + 1e-6
		best_improvement = max(best_improvement, raw_error-error)
	assert best_improvement > 0.1
	assert abs(alpha_coverage(corrected[0][:, :, 3]/255, effective_cutoff)-target) < 0.02


@pytest.mark.parametrize("alpha", [0, 255])
def test_mips_handle_constant_and_degenerate_alpha(gpu_mips, alpha):
	image = np.full((8, 8, 4), 255, dtype=np.uint8)
	image[:, :, 3] = alpha
	for mode in (1, 2):
		for mip in gpu_mips(image, mip_material(mode)):
			np.testing.assert_array_equal(mip[:, :, 3], alpha)
	# An unreachable cutoff and zero material alpha need no coverage adjustment.
	for mip in gpu_mips(image, mip_material(mode=1, alpha_factor=0)):
		np.testing.assert_array_equal(mip[:, :, 3], alpha)


def test_gpu_mips_preserve_npot_mean_alpha_and_repeat_edges_when_upsampling(gpu_mips):
	image = np.full((5, 7, 4), 255, dtype=np.uint8)
	image[:, :, 3] = np.arange(35).reshape(5, 7)*7
	mips = gpu_mips(image, mip_material())
	assert [mip.shape[:2] for mip in mips] == [(5, 7), (2, 3), (1, 1)]
	for mip in mips:
		assert abs(mip[:, :, 3].mean()-image[:, :, 3].mean()) <= 1
	image = np.full((1, 2, 4), 255, dtype=np.uint8)
	image[0, :, 3] = (0, 255)
	base = gpu_mips(image, mip_material(), (4, 1))[0]
	# Linear repeat sampling blends the final texel into the first on upscale.
	np.testing.assert_allclose(base[0, :, 3], [64, 64, 191, 191], atol=1)


def make_layers():
	"""Two unsorted blended sheets, an occluded sheet, and an opaque backdrop."""
	materials = np.zeros(4, dtype=transparency.MATERIAL_DTYPE)
	materials["base_color"] = [(1, 0.1, 0.1, 0.25), (0.1, 1, 0.1, 0.5), (0.1, 0.1, 1, 1), (0.4, 0.4, 0.4, 1)]
	materials["opacity_rgb"] = 1
	materials["surface"][:, 0] = 1
	materials["surface"][:, 2] = [2, 2, 2, 0]
	materials["flags"][:, 0] = np.arange(4)
	materials["flags"][:, 1] = 1
	primitives, storage = [], []
	for i, z in enumerate((0.6, 0.0, -1.2, -0.6)):
		vertices = np.zeros(6, dtype=transparency.VERTEX_DTYPE)
		vertices["position"] = [(-10, -10, z), (10, -10, z), (10, 10, z), (-10, -10, z), (10, 10, z), (-10, 10, z)]
		vertices["normal"] = (0, 0, 1)
		vertices["barycentric"] = [(1, 0), (0, 1), (0, 0)]*2
		vertices["triangle"] = [0, 0, 0, 1, 1, 1]
		primitives.append(transparency_data.Primitive(vertices, i, i*2, i))
		stored = np.zeros(6, dtype=transparency.SHADING_VERTEX_DTYPE)
		stored["position"][:, :3] = vertices["position"]
		stored["normal"][:, :3] = vertices["normal"]
		storage.append(stored)
	images = [np.full((1, 1, 4), 255, dtype=np.uint8) for _ in materials]
	return primitives, materials, images, np.concatenate(storage)


@pytest.fixture(params=[3, 5, 9])
def gpu_scene(monkeypatch, request):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: make_layers())
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	# Read-back usage is confined to tests; normal rendering needs none of it.
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=request.param))
	renderer = scene
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	return renderer, output


def render_frame(render_scene):
	"""Exercise the example's complete culling and transparency pipeline."""
	render_scene.render_frame()


def read_texture(texture, dtype, components):
	width, height, _ = texture.size
	row_bytes = width*np.dtype(dtype).itemsize*components
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": row_bytes, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype=dtype).reshape(height, width, components).copy()


def test_gpu_transmittance_selection_occlusion_order_and_empty_samples(gpu_scene):
	renderer, output = gpu_scene
	render_frame(renderer)
	t = read_texture(renderer.transparency.transmittance, np.float16, 4)
	np.testing.assert_allclose(t, 0.75*0.5, atol=0.001)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	color = read_texture(renderer.transparency.winner_color, np.float16, 4)
	stable_ids = renderer.transparent.metadata.content[:, 2]
	occupied = payload[:, :, 0] != 0
	winners = stable_ids[payload[:, :, 0][occupied]-1]
	assert set(winners) == {0, 1}  # sheet 2 is behind opaque depth
	assert 0.15 < np.mean(occupied & (payload[:, :, 0] == 1)) < 0.35
	assert 0.50 < occupied.mean() < 0.75
	assert np.count_nonzero(color[~occupied]) == 0
	image = read_texture(output, np.uint8, 4)
	assert np.isfinite(color).all() and image[:, :, :3].max() > 0
	renderer.transparent.data.draw_batches.reverse()
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), payload)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), image)
	# Moving all geometry behind the camera clears the previous frame's samples.
	renderer.camera.target = scene.Vec3(0, 0, 10)
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner_color, np.float16, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), 1)


def test_static_primitive_input_order_preserves_packed_metadata_and_shading(gpu_scene, monkeypatch):
	renderer, output = gpu_scene
	render_frame(renderer)
	before = read_texture(output, np.uint8, 4)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	options = renderer.transparency.bilateral_filter
	primitives, materials, images, vertices = make_layers()
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives[::-1], materials, images, vertices))
	scene.init_scene(bilateral_filter=options)
	renderer.camera.position, renderer.camera.target = scene.Vec3(0, 0, 3), scene.Vec3(0, 0, 0)
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), before)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), payload)


def test_transparent_draws_are_skipped_in_opaque_depth_prepass(gpu_scene):
	renderer, _ = gpu_scene
	render_frame(renderer)
	render = Mock()
	renderer.transparent.draw(render, "prepass")
	render.draw_indexed_indirect.assert_not_called()
	render.set_pipeline.assert_not_called()
	# The separate transmittance stage still renders the transparent geometry.
	renderer.transparent.draw(render, "transmittance")
	assert render.draw_indexed_indirect.call_count == len(renderer.transparent.data.draw_batches)
	assert not renderer.transparent.passes.render_shader["transmittance"].pipeline.depth_write


def test_gpu_zero_full_rgb_and_low_opacity(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][:3, 3] = 0
	buffer.upload()
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), 1)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner_color, np.float16, 4), 0)
	buffer.content["base_color"][0] = (0, 0, 0, 1)
	buffer.content["surface"][0, 1] = 1
	buffer.content["emission"][0, :3] = (1, 0.3, 0.1)
	buffer.upload()
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0], 1)
	# The compute stage returns straight color even at fractional RGB opacity.
	for alpha, opacity_rgb in ((0.5, (1, 0.5, 0.25)), (0.01, (1, 1, 1))):
		buffer.content["base_color"][0, 3] = alpha
		buffer.content["opacity_rgb"][0, :3] = opacity_rgb
		buffer.upload()
		render_frame(renderer)
		t = read_texture(renderer.transparency.transmittance, np.float16, 4)
		expected = [*(1-alpha*np.array(opacity_rgb)), 1-alpha*max(opacity_rgb)]
		np.testing.assert_allclose(t, np.broadcast_to(expected, t.shape), atol=0.001)
		color = read_texture(renderer.transparency.winner_color, np.float16, 4)
		occupied = color[:, :, 3] != 0
		assert abs(occupied.mean() - alpha) < (0.06 if alpha > 0.1 else 0.009)
		assert occupied.any()  # low-alpha contributions must survive selection
		np.testing.assert_allclose(color[occupied], np.broadcast_to((1, 0.3, 0.1, 1), color[occupied].shape), atol=0.001)


def test_gpu_resize_rebuilds_targets_and_bindings(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	render_frame(renderer)
	previous = renderer.transparency.winner
	previous_hzb = renderer.hzb.view
	output = Texture((48, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "window_dimensions", (48, 32))
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	render_frame(renderer)
	assert renderer.transparency.winner is not previous
	assert renderer.hzb.view is not previous_hzb
	assert renderer.hzb.size == (48, 32)
	assert renderer.transparency.winner.size == (48, 32, 1)
	assert np.any(read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0] != 0)
	assert np.any(read_texture(output, np.uint8, 4)[:, :, :3] != 0)
	# Minimized windows leave valid resources alone and submit no new work.
	previous = renderer.transparency.winner
	previous_hzb = renderer.hzb.view
	monkeypatch.setattr(RenderContext, "window_dimensions", (0, 0))
	render_frame(renderer)
	assert renderer.transparency.winner is previous
	assert renderer.hzb.view is previous_hzb


def test_gpu_filter_changes_only_reconstruction_without_recreating_targets(gpu_scene):
	renderer, output = gpu_scene
	render_frame(renderer)
	original = renderer.transparency.bilateral_filter
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	image = read_texture(output, np.uint8, 4)
	compose, target = renderer.transparency.compose_pipeline, renderer.transparency.winner
	other_size = 5 if original.kernel_size == 3 else 3
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=other_size))
	render_frame(renderer)
	assert renderer.transparency.compose_pipeline is compose and renderer.transparency.winner is target
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), payload)
	assert not np.array_equal(read_texture(output, np.uint8, 4), image)
	# Tunable spatial spread is independent of the selected footprint.
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=original.kernel_size, spatial_sigma=0.5))
	render_frame(renderer)
	assert not np.array_equal(read_texture(output, np.uint8, 4), image)
	renderer.transparency.set_bilateral_filter(original)
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), image)


def create_textured_gpu_scene(monkeypatch, uv_scale=2):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	primitives, materials, images, vertices = make_layers()
	materials["base_color"][0] = (1, 1, 1, 1)
	materials["base_color"][1, 3] = 0  # keep one visible transparent layer
	visible = np.indices((64, 64)).sum(axis=0) % 2 == 0
	image = np.empty((64, 64, 4), dtype=np.uint8)
	image[visible] = (255, 0, 0, 255)
	image[~visible] = (0, 255, 0, 0)
	images[0] = image
	for primitive in primitives:
		primitive.vertices["uv"] = primitive.vertices["position"][:, :2]*uv_scale
		start = primitive.triangle_offset*3
		vertices["uv"][start:start+len(primitive.vertices), :2] = primitive.vertices["uv"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 64)
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene()
	renderer = scene
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	return renderer, output


def test_gpu_cutout_sampling_has_no_black_bilinear_fringe(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	primitives, materials, images, vertices = make_layers()
	materials["base_color"][0] = (1, 1, 1, 1)
	materials["surface"][0, 2:] = (1, 0.2)
	materials["base_color"][1, 3] = 0
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :8] = (255, 255, 255, 255)
	images[0] = image
	for primitive in primitives:
		primitive.vertices["uv"] = primitive.vertices["position"][:, :2]*0.05
		start = primitive.triangle_offset*3
		vertices["uv"][start:start+len(primitive.vertices), :2] = primitive.vertices["uv"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 16)
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene()
	renderer = scene
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	render_frame(renderer)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	foreground = payload[:, :, 0] == 1
	assert 0.2 < foreground.mean() < 0.9
	color = read_texture(renderer.transparency.winner_color, np.float16, 4)[:, :, :3][foreground]
	# With black transparent neighbors, accepted alpha-edge samples darken to
	# roughly 20% of this brightness. Padding keeps straight white color intact.
	assert color.min() > 0.47
	assert color.max() < 0.51
	guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)[:, :, 0]
	t = read_texture(renderer.transparency.transmittance, np.float16, 4)
	# Sharpened MASK probability is identical in guide, accumulation and selection.
	np.testing.assert_allclose(1-t[:, :, 3], guide, atol=0.001)
	assert np.any((guide > 0) & (guide < 1))
	assert np.any(guide == 1) and np.any(guide == 0)
	assert np.all(foreground[guide == 1])
	assert not np.any(foreground[guide == 0])


def test_gpu_minified_alpha_and_deferred_color_use_matching_mips(monkeypatch):
	renderer, _ = create_textured_gpu_scene(monkeypatch)
	render_frame(renderer)
	t = read_texture(renderer.transparency.transmittance, np.float16, 4)
	np.testing.assert_allclose(t, 1-128/255, atol=0.001)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	occupied = payload[:, :, 0] != 0
	assert abs(occupied.mean()-128/255) < 0.04
	lod_face = payload[:, :, 3][occupied].copy().view(np.float16).reshape(-1, 2)
	assert np.all(lod_face[:, 0] > 1.5)
	assert np.all(lod_face[:, 1] == 1)
	color = read_texture(renderer.transparency.winner_color, np.float16, 4)[occupied]
	# At these footprints all mips contain straight red. Sampling level zero in
	# compute would darken accepted edge samples despite correctly filtered alpha.
	assert color[:, 0].min() > 0.47
	# The lighting model adds neutral dielectric specular, equally to G and B.
	np.testing.assert_allclose(color[:, 1], color[:, 2], atol=0.0001)
	assert color[:, 1:3].max() < 0.011
	# Subpixel camera translation leaves the minified opacity stable.
	renderer.camera.position.x += 0.005
	renderer.camera.target.x += 0.005
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), t)
	# Increasing distance selects a coarser mip without thinning the sheet.
	renderer.camera.position.z = 6
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), t)
	distant = read_texture(renderer.transparency.winner, np.uint32, 4)
	distant_lod = distant[:, :, 3][distant[:, :, 0] != 0].copy().view(np.float16).reshape(-1, 2)[:, 0]
	assert distant_lod.mean() > lod_face[:, 0].mean()+0.5


def test_gpu_debug_views_separate_transmittance_winners_and_reconstruction(gpu_scene):
	renderer, output = gpu_scene
	render_frame(renderer)
	final = read_texture(output, np.uint8, 4)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	occupied = payload[:, :, 0] != 0
	renderer.transparency.set_debug_view("transmittance")
	render_frame(renderer)
	gray = read_texture(output, np.uint8, 4)[:, :, :3]
	expected = round((1.055*0.375**(1/2.4)-0.055)*255)
	np.testing.assert_allclose(gray, expected, atol=1)
	# Display settings and filter changes do not change stochastic selection.
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=5))
	assert renderer.transparency.debug_view == "transmittance"
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4)[:, :, :3], gray)
	np.testing.assert_array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), payload)
	renderer.transparency.set_debug_view("winner")
	render_frame(renderer)
	winner = read_texture(output, np.uint8, 4)[:, :, :3]
	assert np.count_nonzero(winner[~occupied]) == 0
	assert np.all(np.any(winner[occupied] != 0, axis=1))
	renderer.transparency.set_debug_view("reconstruction")
	render_frame(renderer)
	reconstructed = read_texture(output, np.uint8, 4)
	material_buffer = renderer.transparency.shared["materials"]
	material_buffer.content["base_color"][3, :3] = (1, 0, 1)
	material_buffer.upload()
	render_frame(renderer)
	# The reconstruction view omits the opaque background.
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), reconstructed)
	renderer.transparency.set_debug_view("final")
	render_frame(renderer)
	assert not np.array_equal(read_texture(output, np.uint8, 4), final)
	with pytest.raises(ValueError):
		renderer.transparency.set_debug_view("unknown")


def decode_display(image):
	color = image.astype(np.float32)/255
	return np.where(color <= 0.04045, color/12.92, ((color+0.055)/1.055)**2.4)


def test_gpu_self_normalisation_preserves_constant_radiance_and_empty_kernels(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][:3] = 0
	buffer.content["base_color"][0, 3] = 0.15
	buffer.content["surface"][0, 1] = 1  # black metal has no diffuse/specular term
	buffer.content["emission"][0, :3] = (1, 0.5, 0.25)
	buffer.upload()
	renderer.transparency.set_debug_view("reconstruction")
	noise = load_blue_noise(scene.BLUE_NOISE_PATH)
	for offset in ((0, 0), (9, 23), (31, 11)):
		pixels = np.ascontiguousarray(np.roll(noise, offset, axis=(0, 1)))
		RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle}, pixels,
			{"bytes_per_row": 256}, (64, 64, 1))
		render_frame(renderer)
		mapped = decode_display(read_texture(output, np.uint8, 4)[:, :, :3])
		hdr = mapped/np.maximum(1-mapped, 1e-6)
		occupied = read_texture(renderer.transparency.winner_color, np.float16, 4)[:, :, 3] > 0
		# Any kernel containing a winner of this constant color must reconstruct
		# exactly A_rgb * color, independent of the random number of survivors.
		supported = np.zeros_like(occupied)
		radius = renderer.transparency.bilateral_filter.kernel_size//2
		for y in range(-radius, radius+1):
			for x in range(-radius, radius+1):
				supported |= np.roll(occupied, (y, x), axis=(0, 1))
		supported = supported[radius:-radius, radius:-radius]
		values = hdr[radius:-radius, radius:-radius][supported]
		np.testing.assert_allclose(values, np.broadcast_to((0.15, 0.075, 0.0375), values.shape), atol=0.003)
	# A wholly empty neighborhood has a finite zero contribution (no stale data).
	RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle},
		np.ones((64, 64), dtype=np.float32), {"bytes_per_row": 256}, (64, 64, 1))
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4)[:, :, :3], 0)


def test_gpu_certain_winners_bypass_filter_even_with_varying_shading(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][0, 3] = 1
	buffer.upload()
	renderer.transparency.set_debug_view("winner")
	render_frame(renderer)
	winner = read_texture(output, np.uint8, 4)
	assert np.ptp(winner[:, :, 0]) > 0  # spatially varying specular lighting
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(spatial_sigma=10))
	renderer.transparency.set_debug_view("reconstruction")
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), winner)
	guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)
	np.testing.assert_array_equal(guide[:, :, 0], 1)


def test_gpu_1x1_bypasses_reconstruction_without_extra_alpha_attenuation(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][:3] = 0
	buffer.content["base_color"][0, 3] = 0.2
	buffer.content["surface"][0, 1] = 1
	buffer.content["emission"][0, :3] = (1, 0.5, 0.25)
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=1))
	renderer.transparency.set_debug_view("reconstruction")
	for opacity_rgb in ((1, 1, 1), (1, 0.5, 0.25)):
		buffer.content["opacity_rgb"][0, :3] = opacity_rgb
		buffer.upload()
		render_frame(renderer)
		t = read_texture(renderer.transparency.transmittance, np.float16, 4).astype(np.float32)
		winner = read_texture(renderer.transparency.winner_color, np.float16, 4).astype(np.float32)
		assert np.any(winner[:, :, 3] == 0) and np.any(winner[:, :, 3] == 1)
		hdr = (1-t[:, :, :3])*winner[:, :, :3]/np.maximum(1-t[:, :, 3:], 1e-6)
		mapped = hdr/(1+hdr)
		expected = np.rint(np.where(mapped <= 0.0031308, 12.92*mapped, 1.055*mapped**(1/2.4)-0.055)*255)
		image = read_texture(output, np.uint8, 4)[:, :, :3]
		np.testing.assert_allclose(image, expected, atol=1)
		# Point mode is independent of all reconstruction tolerances.
		renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=1,
			spatial_sigma=0.1, depth_sigma=0.0001, transmittance_sigma=0.001, min_depth_sigma=0.000001))
		render_frame(renderer)
		np.testing.assert_array_equal(read_texture(output, np.uint8, 4)[:, :, :3], image)
	# Empty winners remain visibly empty in point mode, whereas a real footprint
	# reconstructs the constant color from neighboring accepted samples.
	empty = winner[:, :, 3] == 0
	np.testing.assert_array_equal(image[empty], 0)
	renderer.transparency.set_bilateral_filter(scene.BilateralFilter(kernel_size=9))
	render_frame(renderer)
	assert np.any(read_texture(output, np.uint8, 4)[:, :, 0][empty] > 0)


@pytest.mark.parametrize("alpha", [0.95, 0.9998])
def test_gpu_nearly_opaque_missing_winners_use_full_reconstruction(gpu_scene, alpha):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][:3] = 0
	buffer.content["base_color"][0, 3] = alpha
	buffer.content["surface"][0, 1] = 1
	buffer.content["emission"][0, :3] = 1
	buffer.upload()
	renderer.transparency.set_debug_view("reconstruction")
	if alpha > 0.99:
		assert np.float16(alpha) == 1  # reproduce the guide's half-float rounding
		noise = load_blue_noise(scene.BLUE_NOISE_PATH)
		# Deliberately exercise rare rejections without needing a huge image.
		noise = np.where(noise > 0.8, np.nextafter(np.float32(1), np.float32(0)), np.float32(0.25))
		RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle}, noise,
			{"bytes_per_row": 256}, (64, 64, 1))
	render_frame(renderer)
	guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)[:, :, 0]
	assert np.all(guide < 1)  # fractional opacity must never encode certainty
	missing = read_texture(renderer.transparency.winner_color, np.float16, 4)[:, :, 3] == 0
	assert missing.any()
	image = read_texture(output, np.uint8, 4)[:, :, :3]
	# Previously the own/filtered mix showed rare black holes as cards moved.
	# Every supported pixel now gets the same constant reconstructed radiance.
	np.testing.assert_allclose(image[missing], np.broadcast_to(image[32, 32], image[missing].shape), atol=1)
	np.testing.assert_allclose(decode_display(image), alpha/(1+alpha), atol=0.004)


def test_gpu_fixed_footprint_is_distance_and_noise_independent(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][:3, 3] = (0.1, 0, 0)
	buffer.upload()
	options = renderer.transparency.bilateral_filter
	renderer.transparency.set_debug_view("filter_support")
	supports = []
	for camera_z in (3, 6):
		renderer.camera.position.z = camera_z
		render_frame(renderer)
		supports.append(read_texture(output, np.uint8, 4)[5:-5, 5:-5, 0])
	np.testing.assert_array_equal(*supports)
	radius = options.kernel_size//2
	y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
	w = np.exp(-(x*x+y*y)/(2*options.spatial_sigma**2))
	np.testing.assert_allclose(decode_display(supports[0]), w.sum()**2/(w*w).sum()/options.kernel_size**2, atol=0.01)
	before = read_texture(output, np.uint8, 4)
	RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle},
		np.ones((64, 64), dtype=np.float32), {"bytes_per_row": 256}, (64, 64, 1))
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), before)


def test_gpu_noise_scale_changes_sampling_without_affecting_opacity_or_filter(gpu_scene):
	renderer, _ = gpu_scene
	render_frame(renderer)
	transmittance = read_texture(renderer.transparency.transmittance, np.float16, 4)
	guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)
	winners = read_texture(renderer.transparency.winner, np.uint32, 4)
	options = renderer.transparency.bilateral_filter
	for invalid in (0, -1, float("nan"), float("inf"), 1e300, 1e-300):
		with pytest.raises(ValueError):
			renderer.transparency.set_noise_scale(invalid)
		assert renderer.transparency.noise_scale == 1
	renderer.transparency.set_noise_scale(2)
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), transmittance)
	np.testing.assert_array_equal(read_texture(renderer.transparency.transparent_guide, np.float16, 4), guide)
	assert not np.array_equal(read_texture(renderer.transparency.winner, np.uint32, 4), winners)
	assert renderer.transparency.bilateral_filter is options
	assert renderer.transparency.uniforms.content["clip"][..., 2] == 2


@pytest.mark.parametrize("usable_uvs", [False, True])
@pytest.mark.parametrize("noise_mode", ["uv_blue_lerp", "uv_blue_correlated", "triplanar_blue", "triplanar_correlated"])
def test_gpu_curved_mesh_noise_tracks_pan_and_rigid_object_transforms(gpu_scene, monkeypatch, usable_uvs, noise_mode):
	renderer, _ = gpu_scene
	primitives, materials, images, _ = make_layers()
	materials["base_color"][:3, 3] = (0.5, 0, 0)
	# A genuinely curved triangle mesh with no UVs, component IDs or planar units.
	axis = np.linspace(-1, 1, 13)
	grid = np.array([[(x+0.137, y-0.231, 0.6+0.3*np.sin(2.7*x)*np.cos(2.3*y))
		for x in axis] for y in axis], dtype=np.float32)
	positions = []
	for y in range(12):
		for x in range(12):
			positions.extend((grid[y, x], grid[y, x+1], grid[y+1, x+1],
				grid[y, x], grid[y+1, x+1], grid[y+1, x]))
	vertices = np.zeros(len(positions), dtype=transparency.VERTEX_DTYPE)
	vertices["position"] = positions
	vertices["normal"] = (0, 0, 1)
	if usable_uvs:
		vertices["uv"] = vertices["position"][:, :2]*(0.37, -0.91)  # nonuniform, mirrored UVs
	vertices["triangle"] = np.repeat(np.arange(len(vertices)//3), 3)
	vertices["barycentric"] = np.tile(((1, 0), (0, 1), (0, 0)), (len(vertices)//3, 1))
	primitives[0].vertices = vertices
	stored, offset = [], 0
	for primitive in primitives:
		primitive.triangle_offset = offset
		v = primitive.vertices
		record = np.zeros(len(v), dtype=transparency.SHADING_VERTEX_DTYPE)
		for name, width in (("position", 3), ("normal", 3), ("uv", 2)):
			record[name][:, :width] = v[name]
		stored.append(record)
		offset += len(v)//3
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, np.concatenate(stored)))
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=1))
	renderer.transparency.set_noise_mode(noise_mode)
	renderer.camera = scene.Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 3,
		perspective=False, near=0.01, far=100)
	renderer.transparency.set_noise_scale(2)
	render_frame(renderer)
	first = read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0] != 0
	assert first.sum() > 500
	# Orthographic translation preserves the surface footprint. Large and curved
	# meshes must track the pan, with no projected-radius cutoff or card fallback.
	renderer.camera.position.x += 3/64
	renderer.camera.target.x += 3/64
	render_frame(renderer)
	second = read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0] != 0
	assert np.mean(second[:, :-1] == first[:, 1:]) > 0.995
	# Co-transform camera and every object: screen geometry is identical while
	# world coordinates change. A world-space anchor would fail this comparison.
	angle = 0.73
	rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
	translation = np.array([4, -2, 1])
	for geometry in (renderer.opaque, renderer.transparent):
		geometry.data.write_instance_fields(iPosition=translation,
			iRotation=scene.pack_quaternion([0, np.sin(angle/2), 0, np.cos(angle/2)]),
			iScale=scene.pack_scale([2, 2, 2]))
	renderer.camera.position = scene.Vec3(np.asarray(renderer.camera.position) @ rotation.T * 2 + translation)
	renderer.camera.target = scene.Vec3(np.asarray(renderer.camera.target) @ rotation.T * 2 + translation)
	renderer.camera.fovy_deg *= 2
	render_frame(renderer)
	transformed = read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0] != 0
	assert np.mean(transformed == second) > 0.995


def test_gpu_object_noise_has_no_triangle_diagonal_or_uv_seam(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	primitives, materials, images, stored = make_layers()
	materials["base_color"][:3, 3] = (0.5, 0, 0)
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, stored))
	masks = []
	for triangulation in range(2):
		if triangulation:
			# Flip the diagonal and change every UV. This is the same geometric
			# surface, but its local triangle IDs and texture parameterization differ.
			v = primitives[0].vertices
			v["position"] = [(-10, -10, 0.6), (10, -10, 0.6), (-10, 10, 0.6),
				(10, -10, 0.6), (10, 10, 0.6), (-10, 10, 0.6)]
			v["uv"] = np.random.default_rng(9).random((6, 2))*7
			stored["position"][:6, :3] = v["position"]
			stored["uv"][:6, :2] = v["uv"]
		scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=1))
		renderer.transparency.set_noise_mode("object_hash")
		renderer.camera.position = scene.Vec3(0, 0, 3)
		renderer.camera.target = scene.Vec3(0, 0, 0)
		render_frame(renderer)
		masks.append(read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0] != 0)
	assert 0.4 < masks[0].mean() < 0.6
	assert np.mean(masks[0] == masks[1]) > 0.995


def test_gpu_uv_noise_modes_change_only_selection_and_preserve_continuous_uvs(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	primitives, materials, images, stored = make_layers()
	for primitive in primitives:
		v = primitive.vertices
		v["uv"] = v["position"][:, :2]*(0.13, -0.27)+(0.037, -0.113)
		start = primitive.triangle_offset*3
		stored["uv"][start:start+len(v), :2] = v["uv"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, stored))
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=1))
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	frames = []
	hierarchy = None
	for mode in scene.NOISE_MODES:
		renderer.transparency.set_noise_mode(mode)
		if mode in ("uv_blue_correlated", "triplanar_correlated"):
			if hierarchy is None:
				hierarchy = renderer.transparency._noise_hierarchy
			assert renderer.transparency._noise_hierarchy is hierarchy
		render_frame(renderer)
		frames.append((read_texture(renderer.transparency.transmittance, np.float16, 4),
			read_texture(renderer.transparency.winner, np.uint32, 4)))
		assert renderer.transparency.uniforms.content["clip"][..., 3] == scene.NOISE_MODES.index(mode)
	for t, _ in frames[1:]:
		np.testing.assert_array_equal(t, frames[0][0])
	assert len({payload[:, :, 0].tobytes() for _, payload in frames}) == len(scene.NOISE_MODES)
	previous = renderer.transparency.noise_mode
	with pytest.raises(ValueError):
		renderer.transparency.set_noise_mode("unknown")
	assert renderer.transparency.noise_mode == previous
	# The UV mode must also ignore triangulation when UVs remain continuous.
	renderer.transparency.set_noise_mode("uv_blue_pick")
	render_frame(renderer)
	before = read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0]
	v = primitives[0].vertices
	v["position"] = [(-10, -10, 0.6), (10, -10, 0.6), (-10, 10, 0.6),
		(10, -10, 0.6), (10, 10, 0.6), (-10, 10, 0.6)]
	v["uv"] = v["position"][:, :2]*(0.13, -0.27)+(0.037, -0.113)
	stored["position"][:6, :3] = v["position"]
	stored["uv"][:6, :2] = v["uv"]
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=1))
	renderer.transparency.set_noise_mode("uv_blue_pick")
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	render_frame(renderer)
	after = read_texture(renderer.transparency.winner, np.uint32, 4)[:, :, 0]
	assert np.mean(before == after) > 0.995


def test_gpu_slope_aware_weights_recover_support_on_oblique_card(gpu_scene, monkeypatch):
	renderer, output = gpu_scene
	primitives, materials, images, vertices = make_layers()
	materials["base_color"][:3, 3] = (0.5, 0, 0)
	card = primitives[0].vertices
	card["position"][:, 2] += card["position"][:, 0]*0.8
	vertices["position"][:6, :3] = card["position"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
	options = scene.BilateralFilter(kernel_size=5, spatial_sigma=2, depth_sigma=0.002, min_depth_sigma=0.0001)
	scene.init_scene(bilateral_filter=options)
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	renderer.transparency.set_debug_view("filter_support")
	render_frame(renderer)
	aware = decode_display(read_texture(output, np.uint8, 4)[32, 32, 0])
	# Compare to independent raw-depth Gaussian weights on the same rasterized
	# plane. This assesses the always-on correction without a production toggle.
	d = read_texture(renderer.transparency.transparent_depth, np.float32, 1)[30:35, 30:35, 0].astype(np.float64)
	near, far = renderer.camera.near, renderer.camera.far
	z = near*far/(far-d*(far-near))
	y, x = np.mgrid[-2:3, -2:3]
	spatial = np.exp(-(x*x+y*y)/(2*options.spatial_sigma**2))
	raw_weights = spatial*np.exp(-0.5*((z-z[2, 2])/max(z[2, 2]*options.depth_sigma, options.min_depth_sigma))**2)
	raw = raw_weights.sum()**2/np.square(raw_weights).sum()/25
	assert aware > raw*2
	assert aware > 0.85


def test_gpu_transparent_depth_guide_is_deterministic_and_alpha_occlusion_tested(gpu_scene):
	renderer, _ = gpu_scene
	render_frame(renderer)
	guide = read_texture(renderer.transparency.transparent_depth, np.float32, 1)[:, :, 0]
	near, far = renderer.camera.near, renderer.camera.far
	view_depth = near*far/(far-guide.astype(np.float64)*(far-near))
	np.testing.assert_allclose(view_depth, 2.4, atol=0.001)
	payload = read_texture(renderer.transparency.winner, np.uint32, 4)
	assert np.any(payload[:, :, 0] == 0)
	assert np.all(guide[payload[:, :, 0] == 0] < 1)  # includes no-winner pixels
	buffer = renderer.transparency.shared["materials"]
	buffer.content["base_color"][0, 3] = 0
	buffer.upload()
	render_frame(renderer)
	guide = read_texture(renderer.transparency.transparent_depth, np.float32, 1)[:, :, 0]
	view_depth = near*far/(far-guide.astype(np.float64)*(far-near))
	np.testing.assert_allclose(view_depth, 3, atol=0.001)  # skips the zero-alpha front sheet
	buffer.content["base_color"][1, 3] = 0
	buffer.upload()
	render_frame(renderer)
	# The remaining alpha=1 sheet is behind opaque depth and cannot guide filtering.
	np.testing.assert_array_equal(read_texture(renderer.transparency.transparent_depth, np.float32, 1), 1)


def test_gpu_grazing_fade_angles_front_back_and_all_passes_agree(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	primitives, materials, images, stored = make_layers()
	primitive = primitives[0]
	primitive.vertices["position"][:, 2] = 0
	# Deliberately incorrect shading normals prove the fade uses geometry.
	primitive.vertices["normal"] = (1, 0, 0)
	stored = stored[:6].copy()
	stored["position"][:, :3] = primitive.vertices["position"]
	stored["normal"][:, :3] = primitive.vertices["normal"]
	materials = materials[:1].copy()
	materials["base_color"] = (0, 0, 0, 0.8)
	materials["surface"][:, 1] = 1
	materials["emission"][:, :3] = 1
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: ([primitive], materials, images[:1], stored))
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=5))
	renderer.camera = scene.Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 3,
		perspective=False, near=0.1, far=100)
	# A constant .6 threshold survives p=.8 but rejects the half-faded p=.4,
	# independently of the selected noise mode and its octave CDF.
	RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle},
		np.full((64, 64), 0.6, dtype=np.float32), {"bytes_per_row": 256}, (64, 64, 1))
	for angle, expected in ((0, 0.8), (65, 0.8), (77.5, 0.4), (85, 0), (88, 0), (102.5, 0.4), (180, 0.8)):
		radians = np.deg2rad(angle)
		renderer.transparent.data.write_instance_fields(iPosition=(0, 0, 0.6),
			iRotation=scene.pack_quaternion([0, np.sin(radians/2), 0, np.cos(radians/2)]))
		renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=False, start_angle=70, end_angle=85))
		render_frame(renderer)
		baseline = read_texture(renderer.transparency.transmittance, np.float16, 4)
		visible = baseline[:, :, 3] < 1
		assert visible.mean() > 0.1
		np.testing.assert_allclose(1-baseline[:, :, 3][visible], 0.8, atol=0.001)
		target = renderer.transparency.transmittance
		renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=True, start_angle=70, end_angle=85))
		render_frame(renderer)
		assert renderer.transparency.transmittance is target
		t = read_texture(target, np.float16, 4)
		np.testing.assert_allclose(1-t[visible], expected, atol=0.001)
		guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)
		np.testing.assert_allclose(guide[:, :, 0][visible], expected, atol=0.001)
		payload = read_texture(renderer.transparency.winner, np.uint32, 4)
		if expected == 0.8:
			assert np.all(payload[:, :, 0][visible] != 0)
		else:
			np.testing.assert_array_equal(payload, 0)
		if expected == 0:
			np.testing.assert_array_equal(t, 1)
		if angle > 85 and expected == 0:
			# The packed rotation can place nominally 85-degree geometry just
			# below the endpoint. Beyond it, even guide depth must disappear.
			np.testing.assert_array_equal(guide, 0)
			np.testing.assert_array_equal(read_texture(renderer.transparency.transparent_depth, np.float32, 1), 1)
	# Perspective rays vary across a plane: its centre stays face-on while the
	# edges fade. Compare the complete probability image to independent angles.
	renderer.transparent.data.write_instance_fields(iPosition=(0, 0, 0.6),
		iRotation=scene.pack_quaternion([0, 0, 0, 1]))
	renderer.camera.perspective = True
	renderer.camera.fovy_deg = 55
	renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=True, start_angle=10, end_angle=30))
	render_frame(renderer)
	y, x = np.indices((64, 64))
	slope_x = (2*(x+0.5)/64-1)*np.tan(np.deg2rad(55/2))
	slope_y = (2*(y+0.5)/64-1)*np.tan(np.deg2rad(55/2))
	angles = np.rad2deg(np.arctan(np.sqrt(slope_x*slope_x+slope_y*slope_y)))
	u = np.clip((angles-10)/20, 0, 1)
	expected = 0.8*(1-u*u*(3-2*u))
	t = read_texture(renderer.transparency.transmittance, np.float16, 4)
	guide = read_texture(renderer.transparency.transparent_guide, np.float16, 4)
	np.testing.assert_allclose(1-t[:, :, 3], expected, atol=0.001)
	np.testing.assert_allclose(guide[:, :, 0], expected, atol=0.001)


@pytest.mark.parametrize("edge", ["depth", "transmittance"])
def test_gpu_relative_depth_edges_and_transmittance_tolerance(monkeypatch, edge):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	supports = []
	for step in range(2):
		scale = 1+step if edge == "depth" else 1
		primitives, materials, images, vertices = make_layers()
		materials["base_color"][:2, 3] = (0.3, 0.3) if edge == "depth" else (0.1, 0.8)
		for primitive in primitives:
			if primitive.stable_id == 0:
				primitive.vertices["position"][:, 0] = np.minimum(primitive.vertices["position"][:, 0], 0)
			elif primitive.stable_id == 1:
				primitive.vertices["position"][:, 0] = np.maximum(primitive.vertices["position"][:, 0], 0)
				primitive.vertices["position"][:, 2] = 0.55 if edge == "depth" else 0.6
			primitive.vertices["position"] *= scale
			start = primitive.triangle_offset*3
			vertices["position"][start:start+len(primitive.vertices), :3] = primitive.vertices["position"]
		monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
		options = scene.BilateralFilter(kernel_size=5, spatial_sigma=2,
			transmittance_sigma=1 if edge == "transmittance" and step else (1/24)**0.5)
		scene.init_scene(bilateral_filter=options)
		renderer = scene
		renderer.camera.position = scene.Vec3(0, 0, 3*scale)
		renderer.camera.target = scene.Vec3(0, 0, 0)
		renderer.transparency.set_debug_view("filter_support")
		render_frame(renderer)
		actual = float(decode_display(read_texture(output, np.uint8, 4)[32, 31, 0]))
		y, x = np.mgrid[-2:3, -2:3]
		weights = np.exp(-(x*x+y*y)/8)
		if edge == "depth":
			# Across x=0: depths 2.4 and 2.45 in scene 1, doubled in scene 2.
			# Relative Gaussian weighting is invariant to this uniform rescaling.
			weights[:, x[0] > 0] *= np.exp(-0.5*(0.05/(2.4*options.depth_sigma))**2)
		else:
			# Opacity-edge rejection remains active and changes only with the
			# explicit tolerance, never with distance or acceptance randomness.
			weights[:, x[0] > 0] *= np.exp(-4*0.7**2/(2*options.transmittance_sigma**2))
		expected = weights.sum()**2/np.sum(weights*weights)/25
		assert abs(actual-expected) < 0.01
		supports.append(actual)
	if edge == "depth":
		assert abs(supports[1]-supports[0]) < 0.01
	else:
		assert supports[1] > supports[0]+0.2


def test_wide_filter_reduces_canopy_boundary_grain(monkeypatch):
	"""Compare actual canopy noise under wide and restrictive fixed footprints.

	Independent static-noise offsets are used only for this diagnostic; production
	rendering still uses a single static tile and has no temporal accumulation.
	"""
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (320, 240))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((320, 240), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene(bilateral_filter=scene.BilateralFilter(kernel_size=9, spatial_sigma=3,
		depth_sigma=0.02, transmittance_sigma=(1/24)**0.5))
	renderer = scene
	distance = renderer.radius*1.4
	renderer.camera.position = scene.Vec3(renderer.center+distance*np.array([np.cos(0.65), 0.5, np.sin(0.65)]))
	render_frame(renderer)
	opacity = 1-read_texture(renderer.transparency.transmittance, np.float16, 4)[:, :, 3]
	boundary = (opacity > 0.02) & (opacity < 0.98)
	assert boundary.sum() > 100
	noise = load_blue_noise(scene.BLUE_NOISE_PATH)
	shifts = np.random.default_rng(12).integers(0, 64, (12, 2))
	wide = renderer.transparency.bilateral_filter
	deviations, supports = [], []
	for options in (scene.replace(wide, kernel_size=3, spatial_sigma=0.8, transmittance_sigma=0.1), wide):
		renderer.transparency.set_bilateral_filter(options)
		estimates = []
		for shift in shifts:
			pixels = np.ascontiguousarray(np.roll(noise, shift, axis=(0, 1)))
			RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle}, pixels,
				{"bytes_per_row": noise.shape[1]*4}, (*noise.shape, 1))
			render_frame(renderer)
			mapped = decode_display(read_texture(output, np.uint8, 4)[:, :, :3])
			estimates.append((mapped/np.maximum(1-mapped, 1e-6))[boundary])
		deviations.append(np.std(estimates, axis=0).mean())
		renderer.transparency.set_debug_view("filter_support")
		render_frame(renderer)
		# Debug support is normalized by kernel area; compare absolute effective
		# sample counts when the two profiles have different kernel sizes.
		supports.append(decode_display(read_texture(output, np.uint8, 4)[:, :, :3])[boundary].mean()*options.kernel_size**2)
		renderer.transparency.set_debug_view("final")
	assert supports[1] > supports[0]*1.5
	assert deviations[1] < deviations[0]*0.9


def test_gpu_hiz_culls_opaque_occlusion_and_preserves_cutout_holes(gpu_scene):
	renderer, _ = gpu_scene
	# Keep the transparent bounds inside the opaque backdrop's screen coverage.
	renderer.transparent.data.write_instance_fields(iScale=scene.pack_scale([0.05, 0.05, 1]))
	view = renderer.transparent.visibility._view
	for name in ("prepass_draw_cmd", "main_draw_cmd"):
		buffer = view.buffers[name]
		view.buffers[name] = scene.GpuBuffer(buffer.content.copy(), buffer.usage | scene.BufferUsage.COPY_SRC)
	view.bindings.clear()

	def instance_counts(name):
		buffer = view.buffers[name]
		raw = RenderContext.device.queue.read_buffer(buffer.handle)
		return np.frombuffer(raw, dtype=buffer.dtype)["instance_count"][:len(renderer.transparent.data.draw_batches)]

	render_frame(renderer)
	np.testing.assert_array_equal(instance_counts("prepass_draw_cmd"), [1, 1, 1])
	np.testing.assert_array_equal(instance_counts("main_draw_cmd"), [1, 1, 0])
	# Transparent layers do not occlude one another, even at full opacity.
	materials = renderer.transparency.shared["materials"]
	materials.content["surface"][3, 2:] = (1, 0.5)
	materials.content["base_color"][3, 3] = 0
	materials.upload()
	render_frame(renderer)
	np.testing.assert_array_equal(instance_counts("main_draw_cmd"), [1, 1, 1])
