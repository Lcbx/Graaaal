"""Exact first contribution and an ID-excluded stochastic tail."""
import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render


def encoded(hdr):
	linear = np.asarray(hdr)/(1+np.asarray(hdr))
	return np.rint(255*np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055))


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("alpha", [0.001, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("mode,kernel", [("fragment", 1), ("fragment", 3), ("compute", 3), ("push_pull", 3), ("atrous", 3)])
def test_single_layer_is_exact_and_tail_is_empty(factory, samples, alpha, mode, kernel):
	renderer = factory(sample_count=samples, alpha=alpha, reconstruction=mode, kernel=kernel, exact_nearest=True)
	image = render(renderer, (31, 23))
	oit = renderer.transparency
	np.testing.assert_allclose(read(oit.nearest_color), np.broadcast_to((1, 0, 0, 1), (samples, 23, 31, 4)), atol=0.001)
	np.testing.assert_array_equal(read(oit.winner_color), 0)
	expected = encoded(alpha*np.array([1, 0, 0])+(1-alpha)*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)
	if samples == 1:
		np.testing.assert_array_equal(read(oit.transmittance), 1)
		assert np.all(read(oit.nearest_payload, np.uint32)[..., 0] == 1)


@pytest.mark.parametrize("samples", [1, 4])
def test_two_half_layers_preserve_tail_and_compose_exact_mean(factory, samples):
	renderer = factory(sample_count=samples, alpha=0.5, layers=2, exact_nearest=True, winner_coupling="complement")
	image = render(renderer, (63, 47))
	oit = renderer.transparency
	# Complementary selection of the one tail layer gives exactly half blue.
	np.testing.assert_allclose(read(oit.winner_color), np.broadcast_to((0, 0, 0.5, 0.5), (samples, 47, 63, 4)), atol=0.001)
	expected = encoded(np.array([0.5, 0, 0.25])+0.25*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("force", [False, True])
def test_faint_foreground_stays_in_front_and_switches_clear(factory, samples, force):
	renderer = factory(sample_count=samples, layers=2, exact_nearest=True, force_single_contributor=force, winner_coupling="complement")
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0.001)
	image = render(renderer, (43, 29))
	np.testing.assert_array_equal(read(renderer.transparency.nearest_color)[..., 0], 1)
	np.testing.assert_array_equal(read(renderer.transparency.nearest_color)[..., 2], 0)
	expected = encoded(np.array([0.001, 0, 0.999*0.5])+0.999*0.5*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)
	renderer.draws.draw_batches.reverse()
	np.testing.assert_array_equal(render(renderer, (43, 29)), image)
	renderer.transparency.set_debug_view("nearest_mask")
	renderer.set_exact_nearest(False)
	assert renderer.transparency.debug_view == "final"
	render(renderer, (19, 17))
	assert renderer.transparency.nearest_payload is None
	assert renderer.transparency.nearest_color is None
	renderer.set_exact_nearest(True)
	renderer.set_sample_count(5-samples)
	renderer.set_winner_mix("fill")
	renderer.set_winner_count(4)
	assert np.isfinite(render(renderer, (31, 23))).all()
	for item in renderer.instances: item.material["base_color"] = (1, 1, 1, 0)
	render(renderer, (31, 23))
	np.testing.assert_array_equal(read(renderer.transparency.nearest_color), 0)
	np.testing.assert_array_equal(read(renderer.transparency.winner_color), 0)


@pytest.mark.parametrize("layers", [2, 4, 8])
@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.7])
@pytest.mark.parametrize("winners", [1, 2])
def test_tail_mean_matches_sorted_scalar_stack(factory, layers, alpha, winners):
	renderer = factory(alpha=alpha, layers=layers, winner_count=winners, kernel=1, exact_nearest=True, winner_coupling="complement")
	colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0.2, 1, 0.8), (0.7, 0.1, 0.3), (0.1, 0.4, 0.9)])[:layers]
	for item, color in zip(renderer.instances, colors): item.material["emission"] = tuple(color)
	means = []
	for seed in range(8):
		for index, item in enumerate(renderer.instances): item.surface_id = 713+seed*101+index
		render(renderer, (96, 96))
		means.append(read(renderer.transparency.winner_color)[..., :3].mean(axis=(0, 1, 2)))
	expected_tail = np.sum((alpha*(1-alpha)**np.arange(layers-1))[:, None]*colors[1:], axis=0)
	np.testing.assert_allclose(np.mean(means, axis=0), expected_tail, atol=0.008)
	expected = np.sum((alpha*(1-alpha)**np.arange(layers))[:, None]*colors, axis=0)
	actual = alpha*colors[0]+(1-alpha)*np.mean(means, axis=0)
	np.testing.assert_allclose(actual, expected, atol=0.008)


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_exact_nearest_requires_boolean(factory, value):
	with pytest.raises(ValueError, match="boolean"): factory(exact_nearest=value)


def test_low_level_custom_shaders_reject_exact_nearest_before_allocating():
	from graaal.TransparencyPass import TransparencyPass
	with pytest.raises(ValueError, match="default shader set"):
		TransparencyPass(materials=None, base_textures=None, vertices=None, geometry_shader=object(), shading_shader=object(), exact_nearest=True)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("front_alpha", [0.5, 1.0])
def test_rgb_nearest_and_coplanar_tail_are_not_double_counted(factory, samples, front_alpha):
	renderer = factory(sample_count=samples, alpha=0.5, layers=2, exact_nearest=True, winner_coupling="complement")
	renderer.instances[0].material["base_color"] = (1, 1, 1, front_alpha)
	renderer.instances[0].material["opacity_rgb"] = (1, 0.5, 0.25)
	image = render(renderer, (41, 27))
	a1 = front_alpha*np.array([1, 0.5, 0.25])
	expected = encoded(a1*np.array([1, 0, 0])+(1-a1)*(0.5*np.array([0, 0, 1])+0.5*np.array([0.035, 0.05, 0.08])))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)
	# Equal-depth fragments with different IDs must leave one layer in the tail.
	renderer.instances[0].material["opacity_rgb"] = (1, 1, 1)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0.5)
	renderer.instances[1].position = renderer.instances[0].position
	render(renderer, (41, 27))
	assert np.all(read(renderer.transparency.winner_color)[..., 3] == 0.5)


def test_owned_nearest_targets_validate_and_diagnostics_follow_option(factory):
	renderer = factory(exact_nearest=True)
	render(renderer, (33, 25))
	oit = renderer.transparency
	targets = oit._targets
	from dataclasses import replace
	with pytest.raises(ValueError, match="nearest_payload"):
		oit.resize((33, 25), targets=replace(targets, nearest_payload=None))
	assert not oit.resize((33, 25), targets=targets)
	from scenes.rendering_settings import FilterSettings
	assert {"nearest", "nearest_mask", "tail_transmittance"} <= set(FilterSettings.available_debug_views(oit))
	for view in ("nearest", "nearest_mask", "tail_transmittance", "transmittance", "reconstruction", "winner"):
		oit.set_debug_view(view)
		assert np.isfinite(render(renderer, (33, 25))).all()
	renderer.set_exact_nearest(False)
	assert not {"nearest", "nearest_mask", "tail_transmittance"} & set(FilterSettings.available_debug_views(oit))
	with pytest.raises(ValueError, match="require"): oit.set_debug_view("nearest")


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("mode", ["fragment", "compute"])
@pytest.mark.parametrize("passes", [2, 3])
def test_repeated_bilateral_initializes_tail_expected_denominator(factory, samples, mode, passes):
	renderer = factory(sample_count=samples, layers=2, alpha=0.1, winner_count=1, exact_nearest=True, reconstruction=mode, reconstruction_passes=passes)
	render(renderer, (37, 29))
	oit = renderer.transparency
	# A single fractional tail layer has stochastic validity, but every
	# intermediate denominator must remain the deterministic tail acceptance.
	pair = read(oit.reconstruction_pair)
	np.testing.assert_allclose(pair[..., 3], 1-float(np.float16(0.9)), atol=0.0001)
	assert np.unique(read(oit.winner_color)[..., 3]).size == 2
