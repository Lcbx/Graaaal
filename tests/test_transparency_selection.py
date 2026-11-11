"""Per-sample fill stencil gating must preserve the ungated estimator exactly."""
import numpy as np
import pytest
from test_transparency_multiple_winners import factory, render, read


@pytest.mark.parametrize("factory", [("depth32float-stencil8",)], indirect=True)
@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("winners", [2, 4])
def test_fill_stencil_preserves_each_sample_and_submission_order(factory, samples, force, kernel, winners):
	renderer = factory(sample_count=samples, layers=2, noise_mode="anchored_blue", winner_mix="fill",
		force_single_contributor=force, kernel=kernel, winner_count=winners)
	# Include silhouette coverage, a sole contributor outside the front surface,
	# and overlap. At 4x these classifications differ between hardware samples.
	renderer.instances[0].scale = (0.07, 0.06, 1)
	renderer.instances[0].rotation = (0, 0.3, 0, (1-0.3**2)**0.5)
	before = render(renderer, (53, 41))
	colors = read(renderer.transparency.winner_color)
	renderer.set_early_b_rejection(True)
	after = render(renderer, (53, 41))
	np.testing.assert_array_equal(after, before)
	np.testing.assert_array_equal(read(renderer.transparency.winner_color), colors)
	assert renderer.transparency.selection_depth.format == "depth32float-stencil8"
	renderer.instances.reverse()
	np.testing.assert_array_equal(render(renderer, (53, 41)), before)
	renderer.set_early_b_rejection(False)
	np.testing.assert_array_equal(render(renderer, (53, 41)), before)
	assert renderer.transparency.selection_depth.format == "depth32float"


@pytest.mark.parametrize("factory", [("depth32float-stencil8",)], indirect=True)
def test_stencil_clears_on_new_frame_and_survives_resize_and_switches(factory):
	renderer = factory(layers=2, noise_mode="object_hash", winner_mix="fill", early_b_rejection=True)
	render(renderer)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	before = render(renderer)
	renderer.set_early_b_rejection(False)
	np.testing.assert_array_equal(render(renderer), before)
	renderer.set_early_b_rejection(True)
	renderer.set_sample_count(4)
	before = render(renderer, (37, 29))
	renderer.set_early_b_rejection(False)
	np.testing.assert_array_equal(render(renderer, (37, 29)), before)
	renderer.set_early_b_rejection(True)
	renderer.set_winner_mix("average")
	assert not renderer.transparency.early_b_rejection
	render(renderer)
	with pytest.raises(ValueError, match="fill"): renderer.set_early_b_rejection(True)
	renderer.set_winner_mix("fill")
	renderer.set_early_b_rejection(True)
	renderer.set_winner_count(1)
	assert not renderer.transparency.early_b_rejection
	assert renderer.transparency.winner_mix == "average"
	render(renderer)


def test_target_replacement_releases_cached_texture_bindings(factory):
	renderer = factory()
	render(renderer)
	oit = renderer.transparency
	assert any(layout._bind_groups for layout in oit._layout_cache.values())
	assert not oit.resize(oit.size)
	assert any(layout._bind_groups for layout in oit._layout_cache.values())
	assert oit.resize((37, 29))
	assert all(not layout._bind_groups for layout in oit._layout_cache.values())
	render(renderer, (37, 29))
	renderer.set_sample_count(4)
	assert all(not layout._bind_groups for layout in oit._layout_cache.values())
	render(renderer, (37, 29))
