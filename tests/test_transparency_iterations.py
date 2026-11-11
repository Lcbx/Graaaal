"""Arbitrary fill attempts and repeated, unnormalized bilateral reconstruction."""
from dataclasses import replace

import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render
from numpyecs.RenderContext import RenderContext
from numpyecs.TransparencyPass import TransparencyPass


def source_ids(oit):
	texture = oit.winner_source
	width, height, layers = texture.size
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": width*4, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, np.uint32).reshape(layers, height, width)


def test_low_level_custom_shaders_reject_fill_even_with_one_attempt():
	with pytest.raises(ValueError, match="default shader set"):
		TransparencyPass(materials=None, base_textures=None, vertices=None, winner_count=1,
			winner_mix="fill", noise_mode="object_hash", geometry_shader=object(), shading_shader=object())


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("attempts", [1, 3, 4, 7])
def test_fill_budget_weights_sources_and_retention(factory, samples, attempts):
	renderer = factory(sample_count=samples, layers=2, kernel=1, noise_mode="object_hash",
		winner_mix="fill", winner_count=attempts)
	before = render(renderer, (127, 113))
	oit = renderer.transparency
	color, sources = read(oit.winner_color), source_ids(oit)
	valid = color[..., 3] > 0
	sum_t = sum(0.25**i for i in range(attempts))
	np.testing.assert_allclose(color[..., 3][valid], 1/sum_t, atol=0.0005)
	assert sources.max() <= attempts
	np.testing.assert_array_equal(sources > 0, valid)
	assert (sources == attempts).any() or attempts == 7  # Later attempts actually execute.
	for actual, expected in zip((np.mean(color[..., 0] > 0), np.mean(color[..., 2] > 0), np.mean(~valid)),
		(0.5*sum_t, 0.25*sum_t, 0.25**attempts)):
		assert abs(actual-expected) < 0.025
	renderer.instances.reverse()
	np.testing.assert_array_equal(render(renderer, (127, 113)), before)
	if attempts > 2:
		with pytest.raises(ValueError, match="at most two"): renderer.set_winner_mix("average")
		assert oit.winner_mix == "fill"
		renderer.set_winner_count(2)
		render(renderer, (127, 113))
		new_sources = source_ids(oit)
		np.testing.assert_array_equal(new_sources[sources == 1], 1)
		np.testing.assert_array_equal(new_sources[sources == 2], 2)
		renderer.set_winner_count(1)
		render(renderer, (31, 27))
		assert oit.winner_mix == "average"


@pytest.mark.parametrize("samples,size", [(1, 1024), (4, 512)])
@pytest.mark.parametrize("mode", ["object_hash", "anchored_blue"])
def test_four_attempts_match_million_sample_layer_frequencies(factory, samples, size, mode):
	renderer = factory(sample_count=samples, kernel=1, layers=2, noise_mode=mode, winner_mix="fill", winner_count=4)
	render(renderer, (size, size))
	color = read(renderer.transparency.winner_color)
	total = sum(0.25**i for i in range(4))
	for actual, expected in zip((np.mean(color[..., 0] > 0), np.mean(color[..., 2] > 0), np.mean(color[..., 3] == 0)),
		(0.5*total, 0.25*total, 0.25**4)):
		assert abs(actual-expected) <= 5*np.sqrt(expected*(1-expected)/(size*size*samples))


def gaussian_pair_pass(pair, sigma):
	"""Reference for constant depth/transmittance, with truncated image boundaries."""
	_, height, width, _ = pair.shape
	result = np.zeros_like(pair)
	weights = np.zeros((1, height, width, 1), np.float32)
	for dy in (-1, 0, 1):
		for dx in (-1, 0, 1):
			weight = np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))
			p = (slice(None), slice(max(0, -dy), min(height, height-dy)), slice(max(0, -dx), min(width, width-dx)))
			q = (slice(None), slice(max(0, dy), min(height, height+dy)), slice(max(0, dx), min(width, width+dx)))
			result[p] += weight*pair[q]
			weights[p] += weight
	return result/weights


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("backend", ["fragment", "compute"])
@pytest.mark.parametrize("passes", [2, 3, 6])
def test_stacked_filters_match_weighted_cpu_reference(factory, samples, backend, passes):
	renderer = factory(sample_count=samples, layers=2, noise_mode="object_hash", winner_mix="fill", winner_count=3,
		reconstruction=backend, reconstruction_passes=passes)
	renderer.transparency.set_debug_view("reconstruction")
	actual = render(renderer, (37, 29))
	oit = renderer.transparency
	pair = read(oit.winner_color)
	for _ in range(passes-1):
		pair = gaussian_pair_pass(pair, oit.bilateral_filter.spatial_sigma).astype(np.float16).astype(np.float32)
	last = oit.reconstruction_pair if (passes-2)%2 == 0 else oit.reconstruction_pair_scratch
	# Storage conversion may round half floats toward zero rather than nearest.
	# Bound accumulated error by one half-float ULP per intermediate store.
	np.testing.assert_allclose(read(last), pair, atol=(passes-1)*0.0005)
	filtered = gaussian_pair_pass(pair, oit.bilateral_filter.spatial_sigma)
	hdr = 0.75*filtered[..., :3]/np.maximum(filtered[..., 3:], 1e-6)
	hdr = hdr.mean(axis=0)
	linear = hdr/(1+hdr)
	srgb = np.where(linear <= 0.0031308, linear*12.92, 1.055*linear**(1/2.4)-0.055)
	np.testing.assert_allclose(actual[0, ..., :3], np.rint(srgb*255), atol=1)
	# Raw winner diagnostics must ignore filtered intermediate data.
	oit.set_debug_view("winner")
	before = render(renderer, (37, 29))
	renderer.set_reconstruction_passes(1)
	np.testing.assert_array_equal(render(renderer, (37, 29)), before)
	assert oit.reconstruction_pair is None and oit.reconstruction_pair_scratch is None


@pytest.mark.parametrize("backend", ["fragment", "compute"])
@pytest.mark.parametrize("samples", [1, 4])
def test_stacking_preserves_forced_color_clears_empty_and_bypasses_1x1(factory, backend, samples, monkeypatch):
	renderer = factory(sample_count=samples, alpha=0.3, force_single_contributor=True, noise_mode="anchored_blue",
		winner_mix="fill", winner_count=5, reconstruction=backend)
	before = render(renderer, (19, 13))
	renderer.set_reconstruction_passes(5)
	np.testing.assert_allclose(render(renderer, (19, 13)), before, atol=1)
	oit = renderer.transparency
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	render(renderer, (19, 13))
	np.testing.assert_array_equal(read(oit.reconstruction_pair), 0)
	np.testing.assert_array_equal(read(oit.reconstruction_pair_scratch), 0)
	oit.set_bilateral_filter(replace(oit.bilateral_filter, kernel_size=1))
	def fail(): raise AssertionError("1x1 must bypass every reconstruction pass")
	monkeypatch.setattr(oit.pair_pipeline, "_check_device", fail)
	render(renderer, (19, 13))


def test_repeated_filter_backend_switches_resize_and_partial_msaa_edges(factory):
	renderer = factory(layers=2, noise_mode="anchored_blue", winner_mix="fill", winner_count=4)
	renderer.instances[0].scale = (0.07, 0.04, 1)
	renderer.instances[0].rotation = (0, 0.3, 0, (1-0.3**2)**0.5)
	renderer.instances[1].scale = (0.1, 0.08, 1)
	for samples in (1, 4):
		renderer.set_sample_count(samples)
		for passes, size in ((2, (31, 27)), (5, (43, 33))):
			renderer.set_reconstruction_passes(passes)
			renderer.set_reconstruction("fragment")
			before = render(renderer, size)
			renderer.set_reconstruction("compute")
			np.testing.assert_allclose(render(renderer, size), before, atol=1)
			assert renderer.transparency.reconstruction_pair.size == (*size, samples)
			renderer.set_reconstruction_passes(1)
			assert renderer.transparency.reconstruction_pair is None


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, None, 2**32])
def test_invalid_iteration_counts_are_rejected_before_mutation(factory, invalid):
	renderer = factory()
	oit = renderer.transparency
	with pytest.raises(ValueError, match="reconstruction_passes"): oit.set_reconstruction_passes(invalid)
	assert oit.reconstruction_passes == 1
	with pytest.raises(ValueError, match="winner_count"): oit.set_winner_count(invalid)
	assert oit.winner_count == 2
