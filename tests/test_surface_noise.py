"""GPU checks of the production 3D threshold field and its scale transitions."""

import numpy as np
import pytest

from numpyecs.Assets import load_blue_noise
from numpyecs.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, RenderPipeline, Shader, Texture, TextureUsage
from numpyecs.TextureProcessing import build_correlated_noise_hierarchy
from numpyecs import TextureProcessing as texture_processing


@pytest.fixture
def thresholds(request):
	RenderContext.setup_graphics_backend(True, required_features=getattr(request, "param", ()))
	shader = Shader(source='''
		#include "utils.shaderlib"
		#include "stochastic_noise.shaderlib"
		<Sample> position: vec4f;
		<Sample> dx: vec4f;
		<Sample> dy: vec4f;
		<Sample> parameters: vec4u;
		<Sample> uv: vec4f;
		<Sample> uv_gradients: vec4f;
		<Sample> normal: vec4f;
		<Sample> screen: vec4f;
		storage<0, 0, read> samples: array<Sample>;
		storage<0, 1, read_write> values: array<f32>;
		{{ binding('noise', 'texture_2d<f32>', 0, 2) }}
		{{ binding('hierarchy', 'texture_2d<f32>', 0, 3) }}
		@compute @workgroup_size(64)
		fn evaluate(@builtin(global_invocation_id) id: vec3u) {
			if id.x >= arrayLength(&samples) { return; }
			let s = samples[id.x];
			if s.parameters.y == 7u {
				values[id.x] = anchored_blue_threshold(noise, s.screen.xy, s.position.xyz, s.dx.xyz, s.dy.xyz,
					s.position.w, hash_u32(s.parameters.x), s.screen.z, s.screen.w);
				return;
			}
			values[id.x] = surface_noise_threshold_balanced(noise, hierarchy, s.position.xyz, s.dx.xyz, s.dy.xyz, s.normal.xyz,
				s.uv.xy, s.uv_gradients.xy, s.uv_gradients.zw, s.position.w,
				hash_u32(s.parameters.x), s.parameters.y, bitcast<f32>(s.parameters.z));
		}
	''', features=("NOISE_FOOTPRINT_BALANCE",))
	pipeline = ComputePipeline(shader, entry="evaluate")
	pixels = load_blue_noise()
	noise = Texture((64, 64), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, pixels, {"bytes_per_row": 256}, (64, 64, 1))
	hierarchy = None
	dtype = np.dtype([("position", "f4", 4), ("dx", "f4", 4), ("dy", "f4", 4), ("parameters", "u4", 4),
		("uv", "f4", 4), ("uv_gradients", "f4", 4), ("normal", "f4", 4), ("screen", "f4", 4)])
	def evaluate(positions, *, log_scale=0.5, seed=17, pixels=1.0, mode=0, uv=None, gradients=None, object_gradients=None, normal=(0, 0, 1), balance=0.0,
		screen=None, anchor_pixels=24.0, subdivisions=4):
		nonlocal hierarchy
		if mode in (4, 6) and hierarchy is None:
			hierarchy = build_correlated_noise_hierarchy(noise)
		data = np.zeros(len(positions), dtype=dtype)
		data["position"][:, :3] = positions
		data["position"][:, 3] = pixels
		data["dx"][:, 0] = 2.0**(-log_scale)/pixels
		data["dy"][:, 1] = 2.0**(-log_scale)/pixels
		if object_gradients is not None:
			data["dx"][:, :3] = object_gradients[:, :3]
			data["dy"][:, :3] = object_gradients[:, 3:]
		data["parameters"][:, 0] = seed
		data["parameters"][:, 1] = mode
		data["parameters"][:, 2] = np.float32(balance).view(np.uint32)
		data["uv"][:, :2] = positions[:, :2] if uv is None else uv
		footprint = 2.0**log_scale/(64*pixels)
		data["uv_gradients"] = (footprint, 0, 0, footprint) if gradients is None else gradients
		data["normal"][:, :3] = normal
		data["screen"][:, :2] = positions[:, :2] if screen is None else screen
		data["screen"][:, 2:] = (anchor_pixels, subdivisions)
		inputs = GpuBuffer(data, BufferUsage.STORAGE | BufferUsage.COPY_DST)
		output = GpuBuffer(np.zeros(len(data), dtype=np.float32), BufferUsage.STORAGE | BufferUsage.COPY_SRC)
		binding = shader.bind_group(0, samples=inputs, values=output, noise=noise,
			hierarchy=hierarchy if hierarchy is not None else noise)
		commands = RenderContext.commands("surface_noise_distribution")
		with commands.compute_pass() as compute:
			compute.set_pipeline(pipeline)
			compute.set_bind_group(binding)
			compute.dispatch((len(data)+63)//64)
		RenderContext.submit(commands.finish())
		return np.frombuffer(RenderContext.device.queue.read_buffer(output.handle), dtype=np.float32).copy()
	return evaluate


def test_gpu_thresholds_preserve_opacity_through_noise_scale_transitions(thresholds):
	positions = np.random.default_rng(41).uniform(-30, 30, (32768, 3))
	for scale in (-1.0, -0.75, -0.5, -0.25, 0, 0.5, 1):
		u = thresholds(positions, log_scale=scale)
		assert np.isfinite(u).all() and u.min() >= 0 and u.max() <= 1
		for alpha in (0.01, 0.1, 0.25, 0.5, 0.9, 0.99):
			assert abs(np.mean(u < alpha)-alpha) < 0.012


def test_gpu_thresholds_are_continuous_at_discrete_noise_scale_boundaries(thresholds):
	positions = np.random.default_rng(92).uniform(-8, 8, (8192, 3))
	for boundary in (-2, 0, 2):
		below = thresholds(positions, log_scale=boundary-1e-5)
		above = thresholds(positions, log_scale=boundary+1e-5)
		np.testing.assert_allclose(below, above, atol=0.0001)


def test_gpu_overlapping_instance_seeds_and_spatial_depth_are_decorrelated(thresholds):
	positions = np.random.default_rng(7).uniform(-30, 30, (32768, 3))
	a = thresholds(positions, seed=23) < 0.5
	b = thresholds(positions, seed=42) < 0.5
	# Even coincident instances need independent thresholds: correlated layers
	# would cover only 50%, instead of the required 1-(1-.5)^2 = 75%.
	assert abs(np.mean(a | b)-0.75) < 0.012
	positions[:, 2] += 8
	c = thresholds(positions, seed=23) < 0.5
	assert abs(np.mean(a == c)-0.5) < 0.012


def test_gpu_thresholds_remain_finite_at_extreme_coordinate_and_footprint_scales(thresholds):
	positions = np.array([[0, -0.0, 0], [1e8, -1e8, 1e8], [-1e-20, 1e-20, 0]], dtype=np.float32)
	for scale in (-60, 60):
		u = thresholds(positions, log_scale=scale)
		assert np.isfinite(u).all() and np.all((u >= 0) & (u <= 1))


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_gpu_uv_octaves_preserve_opacity_under_magnification_and_minification(thresholds, mode):
	positions = np.random.default_rng(61).uniform(-30, 30, (32768, 3))
	for octave in (-3, -2.5, -0.75, -0.25, 0, 0.25, 0.5, 0.75, 2.5):
		u = thresholds(positions, log_scale=octave, mode=mode)
		assert np.isfinite(u).all() and 0 < u.min() < u.max() < 1
		for alpha in (0.01, 0.1, 0.25, 0.5, 0.9, 0.99):
			assert abs(np.mean(u < alpha)-alpha) < 0.012


@pytest.mark.parametrize("mode", [1, 2])
def test_gpu_single_uv_octave_retains_blue_tile_ranks_and_spectrum(thresholds, mode):
	y, x = np.indices((64, 64))
	positions = np.zeros((4096, 3))
	noise = load_blue_noise()
	for octave in (-3, 0, 3):
		uv = np.stack((x-64+0.5, y-64+0.5), axis=-1).reshape(-1, 2)/64*2.0**octave
		u = thresholds(positions, uv=uv, mode=mode, log_scale=octave).reshape(64, 64)
		np.testing.assert_array_equal(np.sort(u.ravel()), np.sort(noise.ravel()))
		for alpha in (0.1, 0.5, 0.9):
			power = np.abs(np.fft.fft2(u < alpha))**2
			reference = np.abs(np.fft.fft2(noise < alpha))**2
			np.testing.assert_allclose(power, reference, rtol=1e-8, atol=1e-6)
		# Signed wrapping must repeat exactly across negative UV tile boundaries.
		u_repeat = thresholds(positions, uv=uv+2.0**octave, mode=mode, log_scale=octave)
		np.testing.assert_array_equal(u.ravel(), u_repeat)


def test_gpu_bayer_tile_contains_every_midpoint_rank(thresholds):
	y, x = np.indices((64, 64))
	uv = np.stack((x+0.5, y+0.5), axis=-1).reshape(-1, 2)/64
	u = thresholds(np.zeros((4096, 3)), uv=uv, log_scale=0, mode=3)
	np.testing.assert_array_equal(np.sort(u), (np.arange(4096)+0.5)/4096)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_gpu_uv_modes_fall_back_for_degenerate_uvs_and_accept_mirrored_uvs(thresholds, mode):
	positions = np.random.default_rng(23).uniform(-8, 8, (8192, 3))
	reference = thresholds(positions, mode=0)
	for gradients in ((0, 0, 0, 0), (0.25, 0, 0.5, 0)):
		u = thresholds(positions, mode=mode, gradients=gradients)
		np.testing.assert_array_equal(u, reference)
	footprint = 2.0**0.5/64
	u = thresholds(positions, mode=mode)
	mirrored = thresholds(positions, mode=mode, gradients=(footprint, 0, 0, -footprint))
	np.testing.assert_array_equal(mirrored, u)


@pytest.mark.parametrize("mode", [1, 2])
def test_gpu_uv_blue_modes_decorrelate_overlapping_instances(thresholds, mode):
	positions = np.random.default_rng(35).uniform(-30, 30, (32768, 3))
	a = thresholds(positions, seed=23, mode=mode) < 0.5
	b = thresholds(positions, seed=42, mode=mode) < 0.5
	assert abs(np.mean(a | b)-0.75) < 0.02


def test_gpu_hierarchy_levels_are_unique_ranks_ordered_by_their_child_averages(monkeypatch):
	RenderContext.setup_graphics_backend(True)
	pixels = load_blue_noise()
	noise = Texture((64, 64), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, pixels, {"bytes_per_row": 256}, (64, 64, 1))
	monkeypatch.setattr(texture_processing, "Texture", lambda *a, usage, **kw:
		Texture(*a, usage=usage | TextureUsage.COPY_SRC, **kw))
	hierarchy = build_correlated_noise_hierarchy(noise)
	previous = None
	for level in range(7):
		size = 64 >> level
		raw = RenderContext.device.queue.read_texture({"texture": hierarchy.handle, "mip_level": level},
			{"bytes_per_row": size*4}, (size, size, 1))
		ranks = np.frombuffer(raw, dtype=np.float32).reshape(size, size)
		np.testing.assert_array_equal(np.sort(ranks.ravel()), np.arange(size*size))
		if previous is not None:
			means = previous.reshape(size, 2, size, 2).mean(axis=(1, 3))
			assert np.all(np.diff(means.ravel()[np.argsort(ranks.ravel())]) >= 0)
		previous = ranks


@pytest.mark.parametrize("mode", [4, 5, 6])
def test_gpu_correlated_and_triplanar_modes_preserve_opacity(thresholds, mode):
	positions = np.random.default_rng(116).uniform(-30, 30, (32768, 3))
	for octave in (-5, -3.5, -1.5, 0.5, 2.5, 4.5):
		footprint = 2.0**octave/64
		derivatives = np.tile((footprint, 0, 0, 0, footprint, 0), (len(positions), 1))
		u = thresholds(positions, log_scale=octave, mode=mode, normal=(1, 2, 1), object_gradients=derivatives)
		assert np.isfinite(u).all() and 0 < u.min() < u.max() < 1
		for alpha in (0.01, 0.1, 0.25, 0.5, 0.9, 0.99):
			assert abs(np.mean(u < alpha)-alpha) < 0.015, (mode, octave, alpha, np.mean(u < alpha))


@pytest.mark.parametrize("mode", [5, 6])
def test_gpu_triplanar_projection_is_uv_independent_and_blends_axes_smoothly(thresholds, mode):
	positions = np.random.default_rng(81).uniform(-5, 5, (8192, 3))
	u = thresholds(positions, mode=mode, normal=(1, 1, 0.3))
	changed_uv = thresholds(positions, mode=mode, normal=(-1, -1, -0.3),
		uv=np.zeros((len(positions), 2)), gradients=(0, 0, 0, 0))
	np.testing.assert_array_equal(u, changed_uv)
	a = thresholds(positions, mode=mode, normal=(1, 1-1e-5, 0.3))
	b = thresholds(positions, mode=mode, normal=(1-1e-5, 1, 0.3))
	np.testing.assert_allclose(a, b, atol=0.0002)


def test_gpu_correlated_octaves_keep_continuous_mixture_limits_and_object_fallback(thresholds):
	positions = np.random.default_rng(73).uniform(-5, 5, (8192, 3))
	for boundary in (-4, -1, 2, 3):
		a = thresholds(positions, mode=4, log_scale=boundary-1e-6)
		b = thresholds(positions, mode=4, log_scale=boundary+1e-6)
		assert np.mean(np.abs(a-b) > 0.001) < 0.001
	missing_uv = thresholds(positions, mode=4, gradients=(0, 0, 0, 0))
	np.testing.assert_array_equal(missing_uv, thresholds(positions, mode=0))


@pytest.mark.parametrize("mode", range(7))
def test_gpu_distinct_surface_ids_seed_all_anchoring_modes_at_coincident_positions(thresholds, mode):
	positions = np.random.default_rng(701).uniform(-30, 30, (8192, 3))
	step = 1/64
	derivatives = np.tile((step, 0, 0, 0, step, 0), (len(positions), 1))
	options = dict(mode=mode, log_scale=0, object_gradients=derivatives)
	a = thresholds(positions, seed=4, **options)
	b = thresholds(positions, seed=17, **options)
	assert np.mean(a != b) > 0.9
	# Seed changes do not guarantee independent acceptance: Bayer and small
	# correlated hierarchy levels retain substantial cross-ID correlation.
	if mode in (0, 1, 2, 5):
		for alpha in (0.1, 0.5, 0.9):
			assert abs(np.mean((a < alpha) | (b < alpha))-(1-(1-alpha)**2)) < 0.025


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_gpu_scalar_uv_footprint_is_an_approximate_size_at_oblique_angles(thresholds, mode):
	# A 4:1 footprint selects cells spanning 4x1 screen pixels at an integer
	# octave. Record the existing limitation rather than claiming square cells.
	screen_x = np.arange(0.125, 20, 0.25)
	uv = np.stack((screen_x/64, np.full(len(screen_x), 0.5/64)), axis=-1)
	u = thresholds(np.zeros((len(uv), 3)), uv=uv, gradients=(1/64, 0, 0, 4/64), mode=mode)
	# Hierarchy origins shift the boundaries, so measure complete interior cells.
	boundaries = np.flatnonzero(np.diff(u) != 0)+1
	assert len(boundaries) >= 3
	np.testing.assert_array_equal(np.diff(screen_x[boundaries]), np.full(len(boundaries)-1, 4.0))


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_gpu_uv_footprint_balance_matches_geometric_scale_and_keeps_opacity(thresholds, mode):
	positions = np.random.default_rng(74).uniform(-30, 30, (32768, 3))
	for balance in (0.0, 0.5, 1.0):
		# fmax=4/64, fmin=1/64; selected scale is 4^(1-a)/64.
		step = 4**(1-balance)/64
		balanced = thresholds(positions, mode=mode, gradients=(1/64, 0, 0, 4/64), balance=balance)
		equivalent = thresholds(positions, mode=mode, gradients=(step, 0, 0, step))
		np.testing.assert_allclose(balanced, equivalent, atol=2e-6)
		mirrored = thresholds(positions, mode=mode, gradients=(-1/64, 0, 0, 4/64), balance=balance)
		np.testing.assert_array_equal(balanced, mirrored)
		for alpha in (0.1, 0.5, 0.9):
			assert abs(np.mean(balanced < alpha)-alpha) < 0.015
	# Rank-one UVs still take the original object-space fallback.
	fallback = thresholds(positions, mode=mode, gradients=(1, 0, 2, 0), balance=1)
	np.testing.assert_array_equal(fallback, thresholds(positions, mode=0))


@pytest.mark.parametrize("mode", [0, 5, 6])
def test_gpu_uv_balance_does_not_change_other_anchoring_modes(thresholds, mode):
	positions = np.random.default_rng(75).uniform(-8, 8, (8192, 3))
	before = thresholds(positions, mode=mode)
	np.testing.assert_array_equal(before, thresholds(positions, mode=mode, balance=1))


def test_gpu_anchored_blue_preserves_opacity_through_grid_octaves(thresholds):
	positions = np.random.default_rng(812).uniform(-30, 30, (32768, 3))
	screen = np.floor(positions[:, :2]*24)+0.5
	for size in (16, 24, 32, 48):
		for footprint in (1/48, 1/24, 1/12):
			derivatives = np.tile((footprint, 0, 0, 0, footprint, 0), (len(positions), 1))
			u = thresholds(positions, mode=7, screen=screen, object_gradients=derivatives, anchor_pixels=size)
			assert np.isfinite(u).all() and np.all((u >= 0) & (u < 1))
			for alpha in (0.1, 0.5, 0.9): assert abs(np.mean(u < alpha)-alpha) < 0.015


def test_gpu_anchored_blue_tracks_integer_screen_translation_and_ignores_uvs(thresholds):
	positions = np.random.default_rng(813).uniform(-8, 8, (8192, 3))
	screen = np.floor(positions[:, :2]*24)+0.5
	derivatives = np.tile((1/24, 0, 0, 0, 1/24, 0), (len(positions), 1))
	a = thresholds(positions, mode=7, screen=screen, object_gradients=derivatives)
	b = thresholds(positions, mode=7, screen=screen+(17, -29), object_gradients=derivatives,
		uv=np.zeros((len(positions), 2)), gradients=(0, 0, 0, 0))
	np.testing.assert_array_equal(a, b)
	other = thresholds(positions, mode=7, screen=screen, object_gradients=derivatives, seed=42)
	assert abs(np.mean((a < 0.5) | (other < 0.5))-0.75) < 0.02


def test_gpu_anchored_blue_falls_back_for_singular_rest_derivatives(thresholds):
	positions = np.random.default_rng(814).uniform(-8, 8, (8192, 3))
	for values in ((0, 0, 0, 0, 0, 0), (1, 0, 0, 2, 0, 0), (1e-30, 0, 0, 0, 1e-30, 0)):
		derivatives = np.tile(values, (len(positions), 1))
		a = thresholds(positions, mode=7, object_gradients=derivatives)
		b = thresholds(positions, mode=0, object_gradients=derivatives)
		np.testing.assert_array_equal(a, b)


def test_gpu_anchored_fine_shift_cells_restore_rear_layer_wins_at_separated_sites(thresholds):
	positions = np.random.default_rng(815).uniform(-30, 30, (32768, 3))
	screen = np.floor(positions[:, :2]*24)+0.5
	derivatives = np.tile((1/24, 0, 0, 0, 1/24, 0), (len(positions), 1))
	a = thresholds(positions, mode=7, screen=screen, object_gradients=derivatives) < 0.5
	coincident = thresholds(positions, mode=7, screen=screen, object_gradients=derivatives) < 0.5
	assert not np.any(~a & coincident)
	for separation in (0.25, 0.5, 1.0, 2.0):
		back = positions.copy()
		back[:, 2] += separation
		b = thresholds(back, mode=7, screen=screen, object_gradients=derivatives) < 0.5
		assert abs(np.mean(~a & b)-0.25) < 0.015


def _native_sample_positions(count):
	shader = Shader(source='''
        @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            return vec4f(vec2f(f32((i << 1u)&2u), f32(i&2u))*2.0-1.0, 0.0, 1.0);
        }
        @fragment fn fragment(@builtin(position) p: vec4f, @builtin(sample_index) sample: u32) -> @location(0) vec4f {
            return vec4f(p.xy, f32(sample), 1.0);
        }
    ''')
	raster = RenderPipeline(shader, sample_count=count, depth_test=None, depth_write=False, cull_mode="none")
	texture = Texture((1, 1), format="rgba16float", sample_count=count,
		usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING)
	reader = Shader(source='''
        #if PARAMS.samples > 1
        {{ binding('samples', 'texture_multisampled_2d<f32>', 0, 0) }}
        #else
        {{ binding('samples', 'texture_2d<f32>', 0, 0) }}
        #endif
        storage<0, 1, read_write> values: array<vec4f>;
        @compute @workgroup_size(1) fn read(@builtin(global_invocation_id) id: vec3u) {
            values[id.x] = textureLoad(samples, vec2i(0), i32(id.x));
        }
    ''', params={"samples": count})
	compute_pipeline = ComputePipeline(reader)
	output = GpuBuffer(np.zeros((count, 4), dtype=np.float32), BufferUsage.STORAGE | BufferUsage.COPY_SRC)
	bindings = reader.bind_group(0, samples=texture, values=output)
	commands = RenderContext.commands()
	with commands.render_pass(color=texture.color_attachment(clear=(0, 0, 0, 0))) as render:
		render.set_pipeline(raster)
		render.draw(3)
	with commands.compute_pass() as compute:
		compute.set_pipeline(compute_pipeline)
		compute.set_bind_group(bindings)
		compute.dispatch(count)
	RenderContext.submit(commands.finish())
	return np.frombuffer(RenderContext.device.queue.read_buffer(output.handle), dtype=np.float32).reshape(count, 4)[:, :2].copy()


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("thresholds", [("indirect-first-instance",)], indirect=True)
def test_gpu_anchored_selection_matches_one_field_at_native_msaa_positions(thresholds, monkeypatch, count):
	from numpyecs import TransparencyPass as transparency
	from numpyecs.DefaultMaterial import DefaultMaterialShader
	from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from numpyecs.Math import Camera
	from numpyecs.TransparencyData import VERTEX_DTYPE
	sample_positions = _native_sample_positions(count)
	assert len(np.unique(sample_positions, axis=0)) == count
	def readable_texture(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1: usage |= TextureUsage.COPY_SRC
		return Texture(*args, usage=usage, **kwargs)
	monkeypatch.setattr(transparency, "Texture", readable_texture)
	vertices = np.zeros(6, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
	vertices["normal"] = (0, 0, 1)
	material = DefaultMaterialShader(transparent=True, default_material={"base_color": (1, 1, 1, 0.5)})
	item = DrawInstance(MaterialMesh(vertices, material))
	renderer = MaterialRenderer([item], target_format="rgba8unorm", sample_count=count, noise_mode="anchored_blue", winner_count=1)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 2, perspective=False, near=0.05, far=100)
	width = height = 16
	output = Texture((width, height), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	commands = RenderContext.commands()
	renderer.record(commands, (width, height), camera, output.color_attachment(), time=0)
	RenderContext.submit(commands.finish())
	flags = RenderContext.device.queue.read_texture({"texture": renderer.transparency.winner_color.handle},
		{"bytes_per_row": width*8, "rows_per_image": height}, (width, height, count))
	flags = np.frombuffer(flags, dtype=np.float16).reshape(count, height, width, 4)[..., 3] != 0
	y, x = np.indices((height, width))
	pixel = np.stack((x, y), axis=-1)[None]+sample_positions[:, None, None, :]
	pixel = pixel.reshape(-1, 2)
	vp = camera.view() @ camera.projection(1)
	center = np.array([0, 0, 0, 1]) @ vp
	def unproject(pixel):
		clip = np.column_stack((pixel[:, 0]*2/width-1, 1-pixel[:, 1]*2/height,
			np.full(len(pixel), center[2]/center[3]), np.ones(len(pixel))))
		local = clip @ np.linalg.inv(vp)
		return local[:, :3]/local[:, 3:4]
	local = unproject(pixel)
	dx = unproject(pixel+(1, 0))-local
	dy = unproject(pixel+(0, 1))-local
	u = thresholds(local, screen=pixel, mode=7, object_gradients=np.column_stack((dx, dy)),
		seed=item.surface_id, pixels=1/np.sqrt(count))
	np.testing.assert_array_equal(flags, (u < 0.5).reshape(count, height, width))
