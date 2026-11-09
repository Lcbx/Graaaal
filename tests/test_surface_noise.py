"""GPU checks of the production 3D threshold field and its scale transitions."""
from pathlib import Path

import numpy as np
import pytest

from numpyecs.Assets import load_blue_noise
from numpyecs.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, Shader, Texture, TextureUsage
from numpyecs.TextureProcessing import build_correlated_noise_hierarchy
from numpyecs import TextureProcessing as texture_processing


@pytest.fixture
def thresholds():
	RenderContext.setup_graphics_backend(True)
	shader = Shader(source='''
		#include "utils.shaderlib"
		#include "stochastic_noise.shaderlib"
		struct Sample {
			position: vec4f,
			dx: vec4f,
			dy: vec4f,
			parameters: vec4u,
			uv: vec4f,
			uv_gradients: vec4f,
			normal: vec4f,
		};
		@group(0) @binding(0) var<storage, read> samples: array<Sample>;
		@group(0) @binding(1) var<storage, read_write> values: array<f32>;
		@group(0) @binding(2) var noise: texture_2d<f32>;
		@group(0) @binding(3) var hierarchy: texture_2d<f32>;
		@compute @workgroup_size(64)
		fn evaluate(@builtin(global_invocation_id) id: vec3u) {
			if id.x >= arrayLength(&samples) { return; }
			let s = samples[id.x];
			values[id.x] = surface_noise_threshold(noise, hierarchy, s.position.xyz, s.dx.xyz, s.dy.xyz, s.normal.xyz,
				s.uv.xy, s.uv_gradients.xy, s.uv_gradients.zw, s.position.w,
				hash_u32(s.parameters.x), s.parameters.y);
		}
	''')
	pipeline = ComputePipeline(shader, entry="evaluate")
	pixels = load_blue_noise(Path(__file__).resolve().parents[1] / "scenes/resources/blue_noise/HDR_L_0.png")
	noise = Texture((64, 64), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, pixels, {"bytes_per_row": 256}, (64, 64, 1))
	hierarchy = None
	dtype = np.dtype([("position", "f4", 4), ("dx", "f4", 4), ("dy", "f4", 4), ("parameters", "u4", 4),
		("uv", "f4", 4), ("uv_gradients", "f4", 4), ("normal", "f4", 4)])
	def evaluate(positions, *, log_scale=0.5, seed=17, pixels=1.0, mode=0, uv=None, gradients=None, object_gradients=None, normal=(0, 0, 1)):
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
		data["uv"][:, :2] = positions[:, :2] if uv is None else uv
		footprint = 2.0**log_scale/(64*pixels)
		data["uv_gradients"] = (footprint, 0, 0, footprint) if gradients is None else gradients
		data["normal"][:, :3] = normal
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
	noise = load_blue_noise(Path(__file__).resolve().parents[1] / "scenes/resources/blue_noise/HDR_L_0.png")
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
	pixels = load_blue_noise(Path(__file__).resolve().parents[1] / "scenes/resources/blue_noise/HDR_L_0.png")
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
