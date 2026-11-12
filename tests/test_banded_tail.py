import numpy as np
import pytest

from scenes.benchmarks.diagnose_banded_tail import estimate, select, uniform_tile


@pytest.mark.parametrize("probabilities", [(0.03, 0.1, 0.2), (0.5, 0.5), (0.7, 0.4, 0.9), (0, 0.2, 0)])
def test_bands_tile_and_have_expected_marginals(probabilities):
	# Dense midpoint sweep tests interval boundaries without a random tolerance.
	noise = ((np.arange(65536)+0.5)/65536).reshape(256, 256)
	p = np.array(probabilities)
	for threshold in (noise, 1-noise):
		winner = select(threshold, p, banded=True)
		for i, expected in enumerate(p/max(1, p.sum())):
			assert abs(np.mean(winner == i)-expected) <= 1/noise.size
		assert abs(np.mean(winner >= 0)-min(1, p.sum())) <= 1/noise.size


@pytest.mark.parametrize("p", [(0.05, 0.1, 0.15), (0.7, 0.5, 0.2)])
def test_banded_mean_matches_p_weighting_under_submission_reversal(p):
	noise = ((np.arange(65536)+0.5)/65536).reshape(256, 256)
	p = np.array(p)
	colors = np.eye(3)
	expected = (1-np.prod(1-p))*(p@colors)/p.sum()
	for order in (np.arange(3), np.arange(2, -1, -1)):
		image = estimate(noise, p[order], colors[order], banded=True)
		np.testing.assert_allclose(image.mean(axis=(0, 1)), expected, atol=3/noise.size, rtol=0)


def test_complement_pair_fills_half_opacity_single_layer_exactly():
	noise = uniform_tile(np.arange(4096).reshape(64, 64))
	color = np.array([[0.7, 0.2, 0.1]])
	for banded in (False, True):
		image = estimate(noise, [0.5], color, banded=banded, shifts=[(3, 7)])
		np.testing.assert_allclose(image, np.broadcast_to(0.5*color, image.shape), atol=1e-12)


def test_layer_entry_can_replace_later_layers_entire_acceptance_mask():
	noise = uniform_tile(np.arange(4096).reshape(64, 64))
	before = select(noise, [0.1], banded=True) == 0
	after = select(noise, [0.1, 0.1], banded=True) == 1
	assert not np.any(before & after)
	assert abs(np.mean(before != after)-0.2) < 2/noise.size
