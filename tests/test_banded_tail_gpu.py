"""Native band allocation and dual Kawase checks against deterministic stacks."""
import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render
from test_transparency_exact_nearest import encoded
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, RenderPipeline, Shader, Texture, TextureUsage


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("passes", [0, 1, 3])
def test_single_layer_stays_exact_without_any_tail(factory, samples, passes):
	renderer = factory(sample_count=samples, alpha=0.1, tail_allocation="banded", winner_coupling="complement", kawase_passes=passes)
	image = render(renderer, (31, 23))
	oit = renderer.transparency
	assert oit.exact_nearest and oit.reconstruction == "kawase"
	np.testing.assert_array_equal(read(oit.winner_color), 0)
	np.testing.assert_array_equal(read(oit.reconstruction_color), 0)
	expected = encoded(np.array([0.1, 0, 0])+0.9*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.8])
def test_band_totals_payload_marginals_and_denominator(factory, samples, alpha):
	renderer = factory(sample_count=samples, alpha=alpha, layers=4, tail_allocation="banded", winner_coupling="complement", kawase_passes=0)
	render(renderer, (128, 128))
	oit = renderer.transparency
	payloads = read(oit.band_payloads, np.uint32)
	counts = np.frombuffer(RenderContext.device.queue.read_buffer(oit.band_counts.handle), dtype=np.uint32).reshape(128, 128, samples, 2)
	pq = int(np.rint(np.float32(alpha)*65536))
	np.testing.assert_array_equal(counts, 3*pq)
	den = min(1.0, 3*pq/65536)
	np.testing.assert_allclose(read(oit.winner_color)[..., 3], den, atol=0.0005)
	for stream in range(2):
		# MSAA sample positions together cover the four parities of the half-pixel
		# tile lattice. One sample alone sees a subsampled tile histogram.
		ids = payloads[stream::2, ..., 0]
		assert not np.any(ids == 1)  # Exact nearest is never allocated a tail band.
		for instance in (2, 3, 4):
			assert abs(np.mean(ids == instance)-pq/max(65536, 3*pq)) < 0.004
		assert abs(np.mean(ids != 0)-den) < 0.004
		if den == 1: assert np.all(ids != 0)
	assert oit.winner is None and oit.selection_depth is None


@pytest.mark.parametrize("samples", [1, 4])
def test_half_alpha_tail_is_constant_and_nearest_is_not_blurred(factory, samples):
	renderer = factory(sample_count=samples, alpha=0.5, layers=2, tail_allocation="banded", winner_coupling="complement", kawase_passes=0)
	image = render(renderer, (43, 29))
	for passes in (1, 3, 8, 0):
		renderer.set_kawase_passes(passes)
		np.testing.assert_array_equal(render(renderer, (43, 29)), image)
		np.testing.assert_allclose(read(renderer.transparency.reconstruction_color)[..., :3], np.broadcast_to((0, 0, 0.5), (samples, 29, 43, 3)), atol=0.001)
		np.testing.assert_array_equal(read(renderer.transparency.nearest_color)[..., 0], 1)
	expected = encoded(np.array([0.5, 0, 0.25])+0.25*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)


def test_blur_reduces_sparse_tail_variance_and_resize_switching_clears(factory):
	renderer = factory(alpha=0.1, layers=2, tail_allocation="banded", winner_coupling="complement", kawase_passes=0)
	render(renderer, (128, 128))
	oit = renderer.transparency
	sharp = read(oit.reconstruction_color)[..., 2]
	renderer.set_kawase_passes(3)
	render(renderer, (128, 128))
	blurred = read(oit.reconstruction_color)[..., 2]
	assert blurred.std() < sharp.std()*0.2
	assert abs(blurred.mean()-0.1) < 0.005
	for mode, samples, size in (("stochastic", 4, (31, 23)), ("banded", 4, (63, 47)), ("banded", 1, (19, 17))):
		renderer.set_tail_allocation(mode)
		renderer.set_sample_count(samples)
		assert np.isfinite(render(renderer, size)).all()
		assert (oit.band_counts is not None) == (mode == "banded")
	for item in renderer.instances: item.material["base_color"] = (1, 1, 1, 0)
	render(renderer, (19, 17))
	np.testing.assert_array_equal(read(oit.band_payloads, np.uint32), 0)
	np.testing.assert_array_equal(read(oit.winner_color), 0)
	np.testing.assert_array_equal(read(oit.reconstruction_color), 0)


@pytest.mark.parametrize("value", [-1, 9, True, 1.5])
def test_invalid_kawase_counts_are_rejected(factory, value):
	with pytest.raises(ValueError, match="kawase_passes"):
		factory(tail_allocation="banded", winner_coupling="complement", kawase_passes=value)


def test_incompatible_banded_settings_are_rejected_without_mutation(factory):
	renderer = factory(tail_allocation="banded", winner_coupling="complement")
	oit = renderer.transparency
	for setter, value in ((oit.set_exact_nearest, False), (oit.set_winner_count, 1), (oit.set_winner_mix, "fill"),
		(oit.set_winner_coupling, "independent"), (oit.set_force_single_contributor, True), (oit.set_early_b_rejection, True),
		(oit.set_reconstruction, "atrous"), (oit.set_reconstruction_passes, 2), (oit.set_debug_view, "winner_count")):
		with pytest.raises(ValueError): setter(value)
	assert oit.exact_nearest and oit.winner_count == 2 and oit.reconstruction == "kawase"


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("variant", ["baseline", "proxy", "single"])
def test_external_opaque_depth_blocks_atomic_writes_and_preserves_color(factory, samples, variant):
	renderer = factory(sample_count=samples, layers=3, tail_allocation="banded", winner_coupling="complement",
		banded_proxy=variant != "baseline", banded_subset=variant == "single", banded_tail_msaa=variant != "single")
	usage = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
	color = Texture((64, 32), format="rgba16float", usage=usage, sample_count=samples)
	depth = Texture((64, 32), format="depth32float", usage=usage, sample_count=samples)
	renderer.transparency.set_opaque_inputs(color, depth)
	shader = Shader(source='''
<Output> @location(0) color: vec4f;
<Output> @builtin(frag_depth) depth: f32;
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
	return vec4f(vec2f(f32((i << 1u) & 2u), f32(i & 2u))*2.0-1.0, 0.0, 1.0);
}
@fragment fn fragment(@builtin(position) p: vec4f) -> Output {
	return Output(vec4f(0.0, 0.2, 0.3, 1.0), select(1.0, 0.2, p.x < 32.0));
}
''')
	pipeline = RenderPipeline(shader, cull_mode="none", depth_test="always", sample_count=samples)
	output = Texture((64, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands()
	with commands.render_pass(color=color.color_attachment(clear=(0, 0, 0, 1)), depth=depth.depth_attachment(clear=1)) as draw:
		draw.set_pipeline(pipeline)
		draw.draw(3)
	renderer.record(commands, (64, 32), Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55), output.color_attachment())
	RenderContext.submit(commands.finish())
	oit = renderer.transparency
	counts = np.frombuffer(RenderContext.device.queue.read_buffer(oit.band_counts.handle), dtype=np.uint32).reshape(32, 64, oit.tail_sample_count, 2)
	np.testing.assert_array_equal(counts[:, :32], 0)
	if oit.banded_subset:
		np.testing.assert_array_equal(counts[:, 32:, ..., 0], 65536)
		assert np.count_nonzero(counts[:, 32:, ..., 1]) == 32*32*oit.tail_sample_count//4
	else: np.testing.assert_array_equal(counts[:, 32:], 65536)
	np.testing.assert_array_equal(read(oit.band_payloads, np.uint32)[:, :, :16 if oit.banded_subset else 32], 0)
	np.testing.assert_array_equal(read(oit.nearest_color)[:, :, :32], 0)
	image = read(output, np.uint8)
	expected = encoded(np.array([0, 0.2, 0.3]))
	np.testing.assert_allclose(image[0, :, :32, :3], np.broadcast_to(expected, (32, 32, 3)), atol=1)


@pytest.mark.parametrize("samples", [1, 4])
def test_coplanar_distinct_fragments_are_not_dropped_by_nearest_exclusion(factory, samples):
	renderer = factory(sample_count=samples, layers=2, tail_allocation="banded", winner_coupling="complement")
	renderer.instances[1].position = renderer.instances[0].position
	render(renderer, (32, 32))
	oit = renderer.transparency
	counts = np.frombuffer(RenderContext.device.queue.read_buffer(oit.band_counts.handle), dtype=np.uint32)
	np.testing.assert_array_equal(counts, 32768)
	np.testing.assert_allclose(read(oit.winner_color)[..., 3], 0.5)


def test_rgb_nearest_mix_uses_per_channel_opacity(factory):
	renderer = factory(layers=2, alpha=0.5, tail_allocation="banded", winner_coupling="complement")
	renderer.instances[0].material["opacity_rgb"] = (1, 0.5, 0.25)
	image = render(renderer, (32, 32))
	a = np.array([0.5, 0.25, 0.125])
	hdr = a*np.array([1, 0, 0])+(1-a)*(0.5*np.array([0, 0, 1])+0.5*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(encoded(hdr), (32, 32, 3)), atol=1)


@pytest.mark.parametrize("samples", [1, 4])
def test_contrasting_tail_matches_p_weighting_after_submission_reversal(factory, samples):
	renderer = factory(sample_count=samples, layers=4, alpha=0.1, tail_allocation="banded", winner_coupling="complement", kawase_passes=0)
	p = np.array([0.05, 0.1, 0.2])
	for item, alpha, color in zip(renderer.instances[1:], p, np.eye(3)):
		item.material["base_color"] = (1, 1, 1, alpha)
		item.material["emission"] = tuple(color)
	expected = (1-np.prod(1-p))*p/p.sum()
	for reverse in (False, True):
		if reverse: renderer.instances.reverse()
		render(renderer, (128, 128))
		np.testing.assert_allclose(read(renderer.transparency.winner_color)[..., :3].mean(axis=(0, 1, 2)), p, atol=0.004)
		np.testing.assert_allclose(read(renderer.transparency.reconstruction_color)[..., :3].mean(axis=(0, 1, 2)), expected, atol=0.004)
