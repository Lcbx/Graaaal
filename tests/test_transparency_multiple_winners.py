"""Read-back checks for draws with replacement and pooled reconstruction."""
from dataclasses import replace

import numpy as np
import pytest

from numpyecs import TransparencyPass as transparency
from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.TransparencyData import VERTEX_DTYPE


def read(texture, dtype=np.float16):
	width, height, layers = texture.size
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": width*4*np.dtype(dtype).itemsize, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype=dtype).reshape(layers, height, width, 4).astype(np.float32)


@pytest.fixture
def factory(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	def readable(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1: usage |= TextureUsage.COPY_SRC
		return Texture(*args, usage=usage, **kwargs)
	monkeypatch.setattr(transparency, "Texture", readable)
	def make(*, sample_count=1, kernel=3, alpha=0.5, layers=1, **options):
		shader = DefaultMaterialShader(transparent=True, source='''
            fn shade(material: Material, p: Params, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
                return p.emission;
            }
        ''')
		vertices = np.zeros(6, dtype=VERTEX_DTYPE)
		vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
		vertices["normal"] = (0, 0, 1)
		vertices["uv"] = [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]
		mesh = MaterialMesh(vertices, shader)
		items = [DrawInstance(mesh, material={"base_color": (1, 1, 1, alpha),
			"emission": (1, 0, 0) if i == 0 else (0, 0, 1)}, position=(0, 0, -i*0.3), surface_id=17+i) for i in range(layers)]
		renderer = MaterialRenderer(items, target_format="rgba8unorm", sample_count=sample_count,
			winner_count=2, bilateral_filter=transparency.BilateralFilter(kernel_size=kernel), **options)
		return renderer
	return make


def render(renderer, size=(96, 96)):
	output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("two_winner_test")
	renderer.record(commands, size, Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=0.01, far=100),
		output.color_attachment(), time=0)
	RenderContext.submit(commands.finish())
	return read(output, np.uint8)


@pytest.mark.parametrize("sample_count", [1, 4])
@pytest.mark.parametrize("kernel", [1, 3])
def test_dual_stream_validity_and_unfiltered_estimator(factory, sample_count, kernel):
	renderer = factory(sample_count=sample_count, kernel=kernel)
	image = render(renderer)
	oit = renderer.transparency
	first, mixed = read(oit.winner_first), read(oit.winner_color)
	a, b = first[..., 3], 2*mixed[..., 3]-first[..., 3]
	assert set(np.unique(mixed[..., 3])) == {0, 0.5, 1}
	assert set(np.unique(b)) == {0, 1}
	for observed, expected in ((np.mean((a+b) == 0), 0.25), (np.mean((a+b) == 1), 0.5), (np.mean((a+b) == 2), 0.25)):
		assert abs(observed-expected) < 0.03
	np.testing.assert_array_equal(mixed[..., 0], mixed[..., 3])
	assert not mixed[..., 1:3].any()
	if kernel == 1:
		assert oit._winner_settings[1].content["info"][0, 1] == 0
		encoded = image[0, ..., :3]/255
		linear = np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)
		hdr = mixed[..., :3].mean(axis=0) + np.array([0.035, 0.05, 0.08])*0.5
		np.testing.assert_allclose(linear, hdr/(1+hdr), atol=0.004)
	else:
		assert oit._winner_settings[1].content["info"][0, 1] == 1
	assert np.isfinite(image).all()


@pytest.mark.parametrize("sample_count", [1, 4])
def test_certain_pixels_skip_second_selection_but_keep_first_color(factory, sample_count):
	renderer = factory(sample_count=sample_count, alpha=1)
	render(renderer)
	oit = renderer.transparency
	first, mixed = read(oit.winner_first), read(oit.winner_color)
	np.testing.assert_array_equal(first, mixed)
	np.testing.assert_array_equal(mixed[..., 0], 1)
	np.testing.assert_array_equal(mixed[..., 3], 1)
	if sample_count == 1:
		np.testing.assert_array_equal(read(oit.winner, np.uint32), 0)


def test_runtime_switch_resize_and_noise_hierarchies_preserve_registration(factory):
	renderer = factory(noise_mode="uv_blue_correlated")
	render(renderer)
	oit = renderer.transparency
	assert oit.noise_second is not oit.noise
	assert oit._noise_hierarchy_second is not oit._noise_hierarchy
	shader = oit.shader
	for count, samples in ((1, 4), (2, 4), (2, 1), (1, 1), (2, 1)):
		renderer.set_winner_count(count)
		renderer.set_sample_count(samples)
		renderer.transparency.set_bilateral_filter(replace(oit.bilateral_filter, kernel_size=1))
		render(renderer, (80, 64))
		assert oit.winner_count == count
		assert (oit.winner_first is not None) == (count == 2)
		assert ("second" in renderer.draws.shaders[1].passes) == (count == 2)
		assert np.isfinite(read(oit.winner_color)).all()
	assert oit.shader is shader
	assert not oit.set_winner_count(2)


def test_two_colored_layers_keep_marginals_and_submission_order(factory):
	renderer = factory(layers=2, kernel=1, noise_mode="object_hash")
	before = render(renderer)
	mixed = read(renderer.transparency.winner_color)
	assert abs(mixed[..., 0].mean()-0.5) < 0.025
	assert abs(mixed[..., 2].mean()-0.25) < 0.025
	renderer.instances.reverse()
	np.testing.assert_array_equal(before, render(renderer))


@pytest.mark.parametrize("mode", transparency.NOISE_MODES)
@pytest.mark.parametrize("sample_count", [1, 4])
def test_each_noise_mode_preserves_per_stream_marginals(factory, mode, sample_count):
	renderer = factory(noise_mode=mode, sample_count=sample_count)
	render(renderer)
	first = read(renderer.transparency.winner_first)
	mixed = read(renderer.transparency.winner_color)
	b = 2*mixed[..., 3]-first[..., 3]
	assert set(np.unique(b)) <= {0, 1}
	assert abs(first[..., 3].mean()-0.5) < 0.04
	assert abs(b.mean()-0.5) < 0.04
	np.testing.assert_array_equal(mixed[..., 0], mixed[..., 3])


def test_nearly_certain_guide_does_not_skip_second_stream(factory):
	def constant(value):
		texture = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": texture.handle}, np.array([value], np.float32),
			{"bytes_per_row": 4}, (1, 1, 1))
		return texture
	renderer = factory(alpha=0.9998, blue_noise=constant(0.99999), blue_noise_second=constant(0.25))
	renderer.transparency.set_debug_view("reconstruction")
	image = render(renderer)
	np.testing.assert_array_equal(read(renderer.transparency.winner_first), 0)
	np.testing.assert_array_equal(read(renderer.transparency.winner_color)[..., 3], 0.5)
	assert np.min(image[..., 0]) > 180  # Full reconstructed red, despite only B winning.


def test_caller_targets_validate_optional_scratch_and_single_mode_has_none(factory):
	renderer = factory()
	render(renderer)
	oit = renderer.transparency
	targets = oit._targets
	with pytest.raises(ValueError, match="winner_first"):
		oit.resize(oit.size, targets=replace(targets, winner_first=None))
	assert oit._targets is targets
	assert oit.resize(oit.size, targets=replace(targets))
	renderer.set_winner_count(1)
	render(renderer)
	assert oit._targets.winner_first is None and oit.mix_targets is None
	assert oit.mix_pipeline is None


def test_dual_debug_views_and_alpha_count(factory):
	renderer = factory(kernel=1)
	for view in ("winner", "winner_a", "winner_b", "winner_count", "filter_support"):
		renderer.transparency.set_debug_view(view)
		image = render(renderer)
		assert np.isfinite(image).all()
		if view == "winner_count":
			encoded = image[0, ..., 0]/255
			linear = np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)
			np.testing.assert_allclose(linear, read(renderer.transparency.winner_color)[0, ..., 3], atol=0.004)


@pytest.mark.parametrize("mode", ["uv_blue_lerp", "anchored_blue"])
@pytest.mark.parametrize("samples", [1, 4])
def test_stream_a_is_unchanged_by_dual_mode(factory, mode, samples):
	renderer = factory(noise_mode=mode, sample_count=samples)
	renderer.transparency.set_debug_view("transmittance")
	t = render(renderer)
	first = read(renderer.transparency.winner_first)
	renderer.set_winner_count(1)
	np.testing.assert_array_equal(t, render(renderer))
	np.testing.assert_array_equal(first, read(renderer.transparency.winner_color))


def test_zero_opacity_and_empty_draws_clear_both_streams(factory):
	renderer = factory()
	render(renderer)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	render(renderer)
	for texture in (renderer.transparency.winner_first, renderer.transparency.winner_color):
		np.testing.assert_array_equal(read(texture), 0)
	renderer.instances.clear()
	render(renderer)
	np.testing.assert_array_equal(read(renderer.transparency.winner_color), 0)


def test_same_id_close_layers_remain_correlated_in_each_stream():
	from scenes.benchmarks.diagnose_anchored_noise import NoiseProbe
	from scenes.benchmarks.diagnose_multiple_winners import same_id_audit
	RenderContext.setup_graphics_backend(True)
	rows = list(same_id_audit(NoiseProbe(), count=8192))
	for row in rows:
		if row["separation_cells"] == 0: assert row["rear_win"] == 0
		elif row["separation_cells"] == 0.1: assert row["rear_win"] < 0.16
		else: assert abs(row["rear_win"]-0.25) < 0.025


@pytest.mark.parametrize("kernel", [1, 3])
def test_rgb_opacity_uses_exact_transmittance_and_pooled_validity(factory, kernel):
	renderer = factory(kernel=kernel)
	color = np.array([0.4, 0.6, 0.8])
	opacity = np.array([1, 0.5, 0.25])
	renderer.instances[0].material.update(emission=tuple(color), opacity_rgb=tuple(opacity))
	image = render(renderer)[0, ..., :3]/255
	linear = np.where(image <= 0.04045, image/12.92, ((image+0.055)/1.055)**2.4)
	mixed = read(renderer.transparency.winner_color)[0]
	background = (1-0.5*opacity)*np.array([0.035, 0.05, 0.08])
	if kernel == 1:
		hdr = background + mixed[..., :3]*opacity
	else:
		valid = np.pad(mixed[..., 3] > 0, 1)
		support = sum(valid[y:y+96, x:x+96] for y in range(3) for x in range(3)) > 0
		hdr = background + support[..., None]*(0.5*opacity*color)
	np.testing.assert_allclose(linear, hdr/(1+hdr), atol=0.004)


@pytest.mark.parametrize("count", [0, 3, True, 2.0, None])
def test_invalid_winner_counts_do_not_mutate_resources(count):
	oit = object.__new__(transparency.TransparencyPass)
	with pytest.raises(ValueError, match="winner_count"): oit.set_winner_count(count)
