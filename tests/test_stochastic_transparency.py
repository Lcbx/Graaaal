"""CPU loader checks and offscreen, read-back checks of the actual OIT passes."""

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from RenderContext import RenderContext, Texture, TextureUsage
from scenes import transparency_example as scene


def test_bundled_blue_noise_loads_as_deterministic_linear_thresholds():
	noise = scene.load_blue_noise()
	np.testing.assert_array_equal(noise, scene.load_blue_noise())
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
	{"adaptive": 1}, {"filter_near": -1}, {"filter_near": 50, "filter_far": 50},
	{"filter_far": float("inf")}, {"far_spatial_sigma": 0},
	{"far_depth_sigma": -1}, {"far_transmittance_sigma": float("nan")},
	{"far_transmittance_weight": -0.1}, {"far_transmittance_weight": 1.1},
])
def test_bilateral_filter_rejects_invalid_footprints_and_tolerances(options):
	with pytest.raises(ValueError):
		scene.BilateralFilter(**options)


def test_node_matrix_uses_gltf_column_major_matrix_and_trs():
	trs = SimpleNamespace(matrix=None, rotation=[0, 0, 0, 1], scale=[2, 3, 4], translation=[5, 6, 7])
	matrix = scene.node_matrix(trs)
	np.testing.assert_allclose(matrix @ [1, 1, 1, 1], [7, 9, 11, 1])
	np.testing.assert_array_equal(scene.node_matrix(SimpleNamespace(matrix=matrix.T.reshape(-1).tolist())), matrix)


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
	primitives, materials, images, stored = scene.load_scene(path)
	vertices = primitives[0].vertices
	np.testing.assert_allclose(vertices["position"], [[2, 3, 4], [2, 4, 4], [0, 3, 4]])
	np.testing.assert_allclose(vertices["normal"], [[0, 0, 1]]*3)
	np.testing.assert_array_equal(stored["position"][:, :3], vertices["position"])
	assert primitives[0].material_id == 0
	assert materials[0]["surface"][2] == 0
	assert len(images) == 1


def test_real_scene_has_opaque_mask_and_blended_geometry():
	primitives, materials, images, vertices = scene.load_scene()
	assert {int(materials[p.material_id]["surface"][2]) for p in primitives} == {0, 1, 2}
	assert len(images) == len(materials)
	for primitive in primitives:
		start = primitive.triangle_offset*3
		count = len(primitive.vertices)
		np.testing.assert_array_equal(vertices["position"][start:start+count, :3], primitive.vertices["position"])
	assert np.isfinite(vertices["normal"]).all()


def mip_material(mode=2, alpha_factor=1, cutoff=0.5):
	material = np.zeros(1, dtype=scene.MATERIAL_DTYPE)[0]
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
	monkeypatch.setattr(scene, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	def generate(image, material, size=None):
		size = size or (image.shape[1], image.shape[0])
		texture = scene.build_material_texture_array([image], np.array([material], dtype=scene.MATERIAL_DTYPE), size)
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
	materials = np.zeros(4, dtype=scene.MATERIAL_DTYPE)
	materials["base_color"] = [(1, 0.1, 0.1, 0.25), (0.1, 1, 0.1, 0.5), (0.1, 0.1, 1, 1), (0.4, 0.4, 0.4, 1)]
	materials["opacity_rgb"] = 1
	materials["surface"][:, 0] = 1
	materials["surface"][:, 2] = [2, 2, 2, 0]
	materials["flags"][:, 0] = np.arange(4)
	materials["flags"][:, 1] = 1
	primitives, storage = [], []
	for i, z in enumerate((0.6, 0.0, -1.2, -0.6)):
		vertices = np.zeros(6, dtype=scene.VERTEX_DTYPE)
		vertices["position"] = [(-10, -10, z), (10, -10, z), (10, 10, z), (-10, -10, z), (10, 10, z), (-10, 10, z)]
		vertices["normal"] = (0, 0, 1)
		vertices["barycentric"] = [(1, 0), (0, 1), (0, 0)]*2
		vertices["triangle"] = [0, 0, 0, 1, 1, 1]
		primitives.append(scene.Primitive(vertices, i, i*2, i))
		stored = np.zeros(6, dtype=scene.SHADING_VERTEX_DTYPE)
		stored["position"][:, :3] = vertices["position"]
		stored["normal"][:, :3] = vertices["normal"]
		storage.append(stored)
	images = [np.full((1, 1, 4), 255, dtype=np.uint8) for _ in materials]
	return primitives, materials, images, np.concatenate(storage)


@pytest.fixture(params=[3, 5])
def gpu_scene(monkeypatch, request):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "load_scene", lambda path: make_layers())
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	# Read-back usage is confined to tests; normal rendering needs none of it.
	monkeypatch.setattr(scene, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "windowDimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	renderer = scene.TransparencyRenderer(bilateral_filter=scene.BilateralFilter(kernel_size=request.param))
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	return renderer, output


def read_texture(texture, dtype, components):
	width, height, _ = texture.size
	row_bytes = width*np.dtype(dtype).itemsize*components
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": row_bytes, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype=dtype).reshape(height, width, components).copy()


def test_gpu_transmittance_selection_occlusion_order_and_empty_samples(gpu_scene):
	renderer, output = gpu_scene
	renderer.render()
	t = read_texture(renderer.transmittance, np.float16, 4)
	np.testing.assert_allclose(t, 0.75*0.5, atol=0.001)
	payload = read_texture(renderer.winner, np.uint32, 4)
	color = read_texture(renderer.winner_color, np.float16, 4)
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
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4), payload)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), image)
	# Moving all geometry behind the camera clears the previous frame's samples.
	renderer.camera.target = scene.Vec3(0, 0, 10)
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.winner_color, np.float16, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), 1)


def test_gpu_zero_full_rgb_and_low_opacity(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.shared["materials"]
	buffer.content["base_color"][:3, 3] = 0
	buffer.upload()
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), 1)
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.winner_color, np.float16, 4), 0)
	buffer.content["base_color"][0] = (0, 0, 0, 1)
	buffer.content["surface"][0, 1] = 1
	buffer.content["emission"][0, :3] = (1, 0.3, 0.1)
	buffer.upload()
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), 0)
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4)[:, :, 0], 1)
	# The compute stage returns straight color even at fractional RGB opacity.
	for alpha, opacity_rgb in ((0.5, (1, 0.5, 0.25)), (0.01, (1, 1, 1))):
		buffer.content["base_color"][0, 3] = alpha
		buffer.content["opacity_rgb"][0, :3] = opacity_rgb
		buffer.upload()
		renderer.render()
		t = read_texture(renderer.transmittance, np.float16, 4)
		expected = [*(1-alpha*np.array(opacity_rgb)), 1-alpha*max(opacity_rgb)]
		np.testing.assert_allclose(t, np.broadcast_to(expected, t.shape), atol=0.001)
		color = read_texture(renderer.winner_color, np.float16, 4)
		occupied = color[:, :, 3] != 0
		assert abs(occupied.mean() - alpha) < (0.06 if alpha > 0.1 else 0.009)
		assert occupied.any()  # low-alpha contributions must survive selection
		np.testing.assert_allclose(color[occupied], np.broadcast_to((1, 0.3, 0.1, 1), color[occupied].shape), atol=0.001)


def test_gpu_resize_rebuilds_targets_and_bindings(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	renderer.render()
	previous = renderer.winner
	output = Texture((48, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "windowDimensions", (48, 32))
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	renderer.render()
	assert renderer.winner is not previous
	assert renderer.winner.size == (48, 32, 1)
	assert np.any(read_texture(renderer.winner, np.uint32, 4)[:, :, 0] != 0)
	assert np.any(read_texture(output, np.uint8, 4)[:, :, :3] != 0)
	# Minimized windows leave valid resources alone and submit no new work.
	previous = renderer.winner
	monkeypatch.setattr(RenderContext, "windowDimensions", (0, 0))
	renderer.render()
	assert renderer.winner is previous


def test_gpu_filter_changes_only_reconstruction_without_recreating_targets(gpu_scene):
	renderer, output = gpu_scene
	renderer.render()
	original = renderer.bilateral_filter
	payload = read_texture(renderer.winner, np.uint32, 4)
	image = read_texture(output, np.uint8, 4)
	compose, target = renderer.compose, renderer.winner
	other_size = 5 if original.kernel_size == 3 else 3
	renderer.set_bilateral_filter(scene.BilateralFilter(kernel_size=other_size))
	renderer.render()
	assert renderer.compose is compose and renderer.winner is target
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4), payload)
	assert not np.array_equal(read_texture(output, np.uint8, 4), image)
	# Tunable spatial spread is independent of the selected footprint.
	renderer.set_bilateral_filter(scene.BilateralFilter(kernel_size=original.kernel_size, spatial_sigma=0.5))
	renderer.render()
	assert not np.array_equal(read_texture(output, np.uint8, 4), image)
	renderer.set_bilateral_filter(original)
	renderer.render()
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
	monkeypatch.setattr(scene, "load_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 64)
	monkeypatch.setattr(scene, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "windowDimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	renderer = scene.TransparencyRenderer()
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
	monkeypatch.setattr(scene, "load_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 16)
	monkeypatch.setattr(scene, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "windowDimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	renderer = scene.TransparencyRenderer()
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	renderer.render()
	depth = read_texture(renderer.depth, np.float32, 1)[:, :, 0].astype(np.float64)
	near, far = renderer.camera.near, renderer.camera.far
	foreground = near*far/(far-depth*(far-near)) < 3
	assert 0.2 < foreground.mean() < 0.9
	color = read_texture(renderer.main_color, np.float16, 4)[:, :, :3][foreground]
	# With black transparent neighbors, accepted alpha-edge samples darken to
	# roughly 20% of this brightness. Padding keeps straight white color intact.
	assert color.min() > 0.47
	assert color.max() < 0.51


def test_gpu_minified_alpha_and_deferred_color_use_matching_mips(monkeypatch):
	renderer, _ = create_textured_gpu_scene(monkeypatch)
	renderer.render()
	t = read_texture(renderer.transmittance, np.float16, 4)
	np.testing.assert_allclose(t, 1-128/255, atol=0.001)
	payload = read_texture(renderer.winner, np.uint32, 4)
	occupied = payload[:, :, 0] != 0
	assert abs(occupied.mean()-128/255) < 0.04
	lod_face = payload[:, :, 3][occupied].copy().view(np.float16).reshape(-1, 2)
	assert np.all(lod_face[:, 0] > 1.5)
	assert np.all(lod_face[:, 1] == 1)
	color = read_texture(renderer.winner_color, np.float16, 4)[occupied]
	# At these footprints all mips contain straight red. Sampling level zero in
	# compute would darken accepted edge samples despite correctly filtered alpha.
	assert color[:, 0].min() > 0.47
	# The lighting model adds neutral dielectric specular, equally to G and B.
	np.testing.assert_allclose(color[:, 1], color[:, 2], atol=0.0001)
	assert color[:, 1:3].max() < 0.011
	# Subpixel camera translation leaves the minified opacity stable.
	renderer.camera.position.x += 0.005
	renderer.camera.target.x += 0.005
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), t)
	# Increasing distance selects a coarser mip without thinning the sheet.
	renderer.camera.position.z = 6
	renderer.render()
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), t)
	distant = read_texture(renderer.winner, np.uint32, 4)
	distant_lod = distant[:, :, 3][distant[:, :, 0] != 0].copy().view(np.float16).reshape(-1, 2)[:, 0]
	assert distant_lod.mean() > lod_face[:, 0].mean()+0.5


def test_gpu_debug_views_separate_transmittance_winners_and_reconstruction(gpu_scene):
	renderer, output = gpu_scene
	renderer.render()
	final = read_texture(output, np.uint8, 4)
	payload = read_texture(renderer.winner, np.uint32, 4)
	occupied = payload[:, :, 0] != 0
	renderer.set_debug_view("transmittance")
	renderer.render()
	gray = read_texture(output, np.uint8, 4)[:, :, :3]
	expected = round((1.055*0.375**(1/2.4)-0.055)*255)
	np.testing.assert_allclose(gray, expected, atol=1)
	# Display settings and filter changes do not change stochastic selection.
	renderer.set_bilateral_filter(scene.BilateralFilter(kernel_size=5))
	assert renderer.debug_view == "transmittance"
	renderer.render()
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4)[:, :, :3], gray)
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4), payload)
	renderer.set_debug_view("winner")
	renderer.render()
	winner = read_texture(output, np.uint8, 4)[:, :, :3]
	assert np.count_nonzero(winner[~occupied]) == 0
	assert np.all(np.any(winner[occupied] != 0, axis=1))
	renderer.set_debug_view("reconstruction")
	renderer.render()
	reconstructed = read_texture(output, np.uint8, 4)
	material_buffer = renderer.shared["materials"]
	material_buffer.content["base_color"][3, :3] = (1, 0, 1)
	material_buffer.upload()
	renderer.render()
	# The reconstruction view omits the opaque background.
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), reconstructed)
	renderer.set_debug_view("final")
	renderer.render()
	assert not np.array_equal(read_texture(output, np.uint8, 4), final)
	with pytest.raises(ValueError):
		renderer.set_debug_view("unknown")


def decode_display(image):
	color = image.astype(np.float32)/255
	return np.where(color <= 0.04045, color/12.92, ((color+0.055)/1.055)**2.4)


def test_gpu_transparent_depth_guide_is_deterministic_and_alpha_occlusion_tested(gpu_scene):
	renderer, _ = gpu_scene
	renderer.render()
	guide = read_texture(renderer.transparent_depth, np.float32, 1)[:, :, 0]
	near, far = renderer.camera.near, renderer.camera.far
	view_depth = near*far/(far-guide.astype(np.float64)*(far-near))
	np.testing.assert_allclose(view_depth, 2.4, atol=0.001)
	payload = read_texture(renderer.winner, np.uint32, 4)
	assert np.any(payload[:, :, 0] == 0)
	assert np.all(guide[payload[:, :, 0] == 0] < 1)  # includes no-winner pixels
	buffer = renderer.shared["materials"]
	buffer.content["base_color"][0, 3] = 0
	buffer.upload()
	renderer.render()
	guide = read_texture(renderer.transparent_depth, np.float32, 1)[:, :, 0]
	view_depth = near*far/(far-guide.astype(np.float64)*(far-near))
	np.testing.assert_allclose(view_depth, 3, atol=0.001)  # skips the zero-alpha front sheet
	buffer.content["base_color"][1, 3] = 0
	buffer.upload()
	renderer.render()
	# The remaining alpha=1 sheet is behind opaque depth and cannot guide filtering.
	np.testing.assert_array_equal(read_texture(renderer.transparent_depth, np.float32, 1), 1)


def test_gpu_adaptive_filter_uses_transparent_distance_and_reduces_variance(gpu_scene):
	renderer, output = gpu_scene
	buffer = renderer.shared["materials"]
	buffer.content["base_color"][:3, :3] = 0
	buffer.content["surface"][:3, 1] = 1
	buffer.content["emission"][0, :3] = 1  # constant straight color, no lighting variance
	buffer.content["base_color"][0, 3] = 0.1
	buffer.content["base_color"][1, 3] = 0
	buffer.upload()
	options = scene.BilateralFilter(kernel_size=5, filter_near=3, filter_far=5,
		spatial_sigma=0.8, transmittance_sigma=0.1)
	renderer.set_bilateral_filter(options)
	reconstruction_std = []
	for camera_z, sigma in ((3, 0.8), (4.6, 1.4), (6, 2.0)):
		renderer.camera.position.z = camera_z
		renderer.set_debug_view("filter_support")
		renderer.render()
		support = decode_display(read_texture(output, np.uint8, 4)[4:-4, 4:-4, 0])
		# The guide is at z=2.4/4.0/5.4; opaque depth would give incorrect blend
		# factors at the first two distances. Flat depth/transmittance isolate the
		# analytically known effective sample count of the spatial Gaussian.
		y, x = np.mgrid[-2:3, -2:3]
		weights = np.exp(-(x*x+y*y)/(2*sigma*sigma))
		expected = weights.sum()**2/np.sum(weights*weights)/25
		np.testing.assert_allclose(support, expected, atol=0.01)
		renderer.set_debug_view("reconstruction")
		renderer.render()
		reconstruction_std.append(decode_display(read_texture(output, np.uint8, 4)[4:-4, 4:-4, 0]).std())
	assert reconstruction_std[-1] < reconstruction_std[0]*0.9
	payload = read_texture(renderer.winner, np.uint32, 4)
	t = read_texture(renderer.transmittance, np.float16, 4)
	renderer.set_bilateral_filter(scene.replace(options, adaptive=False))
	renderer.set_debug_view("filter_support")
	renderer.render()
	weights = np.exp(-(x*x+y*y)/(2*options.spatial_sigma**2))
	expected = weights.sum()**2/np.sum(weights*weights)/25
	np.testing.assert_allclose(decode_display(read_texture(output, np.uint8, 4)[4:-4, 4:-4, 0]), expected, atol=0.01)
	np.testing.assert_array_equal(read_texture(renderer.winner, np.uint32, 4), payload)
	np.testing.assert_array_equal(read_texture(renderer.transmittance, np.float16, 4), t)


@pytest.mark.parametrize("edge", ["depth", "transmittance"])
def test_gpu_relative_depth_edges_and_distant_transmittance_relaxation(monkeypatch, edge):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	monkeypatch.setattr(RenderContext, "windowDimensions", (64, 64))
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
		monkeypatch.setattr(scene, "load_scene", lambda path: (primitives, materials, images, vertices))
		options = scene.BilateralFilter(spatial_sigma=2, far_spatial_sigma=2,
			adaptive=edge == "transmittance", filter_near=3, filter_far=5)
		renderer = scene.TransparencyRenderer(bilateral_filter=options)
		renderer.camera.position = scene.Vec3(0, 0, (3 if edge == "depth" else 3+3*step)*scale)
		renderer.camera.target = scene.Vec3(0, 0, 0)
		renderer.set_debug_view("filter_support")
		renderer.render()
		actual = float(decode_display(read_texture(output, np.uint8, 4)[32, 31, 0]))
		y, x = np.mgrid[-2:3, -2:3]
		weights = np.exp(-(x*x+y*y)/8)
		if edge == "depth":
			# Across x=0: depths 2.4 and 2.45 in scene 1, doubled in scene 2.
			# Relative Gaussian weighting is invariant to this uniform rescaling.
			weights[:, x[0] > 0] *= np.exp(-0.5*(0.05/(2.4*0.02))**2)
		elif step == 0:
			# Near transmittance differs by 0.7 in each of RGBA. At far distance
			# this constraint fades to zero while the depth guide stays active.
			weights[:, x[0] > 0] *= np.exp(-4*0.7**2/(2*options.transmittance_sigma**2))
		expected = weights.sum()**2/np.sum(weights*weights)/25
		assert abs(actual-expected) < 0.01
		supports.append(actual)
	if edge == "depth":
		assert abs(supports[1]-supports[0]) < 0.01
	else:
		assert supports[1] > supports[0]+0.2


def test_default_filter_restores_canopy_boundary_support_and_reduces_grain(monkeypatch):
	"""Compare actual canopy noise under restored and overly restrictive defaults.

	Independent static-noise offsets are used only for this diagnostic; production
	rendering still uses a single static tile and has no temporal accumulation.
	"""
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "windowDimensions", (320, 240))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((320, 240), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	renderer = scene.TransparencyRenderer()
	distance = renderer.radius*1.4
	renderer.camera.position = scene.Vec3(renderer.center+distance*np.array([np.cos(0.65), 0.5, np.sin(0.65)]))
	renderer.render()
	opacity = 1-read_texture(renderer.transmittance, np.float16, 4)[:, :, 3]
	boundary = (opacity > 0.02) & (opacity < 0.98)
	assert boundary.sum() > 100
	noise = scene.load_blue_noise()
	shifts = np.random.default_rng(12).integers(0, 64, (12, 2))
	default = renderer.bilateral_filter
	assert default.spatial_sigma == pytest.approx(2**0.5)
	assert default.transmittance_sigma == pytest.approx((1/24)**0.5)
	deviations, supports = [], []
	for options in (scene.replace(default, spatial_sigma=0.8, transmittance_sigma=0.1), default):
		renderer.set_bilateral_filter(options)
		estimates = []
		for shift in shifts:
			pixels = np.ascontiguousarray(np.roll(noise, shift, axis=(0, 1)))
			RenderContext.device.queue.write_texture({"texture": renderer.noise.handle}, pixels,
				{"bytes_per_row": noise.shape[1]*4}, (*noise.shape, 1))
			renderer.render()
			mapped = decode_display(read_texture(output, np.uint8, 4)[:, :, :3])
			estimates.append((mapped/np.maximum(1-mapped, 1e-6))[boundary])
		deviations.append(np.std(estimates, axis=0).mean())
		renderer.set_debug_view("filter_support")
		renderer.render()
		supports.append(decode_display(read_texture(output, np.uint8, 4)[:, :, :3])[boundary].mean())
		renderer.set_debug_view("final")
	assert supports[1] > supports[0]*1.5
	assert deviations[1] < deviations[0]*0.9
