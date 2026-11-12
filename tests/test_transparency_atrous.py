"""Unnormalised dilated sums, output-level selection and retained support."""
import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render
from numpyecs.TransparencyPass import ConfidenceFilter
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage


def cascade_reference(pair, passes, budget, active=None):
	result = np.zeros_like(pair)
	result[..., 3] = 1
	active = pair[..., 3] > 0 if active is None else active
	states = []
	for level in range(passes):
		step = 1 << level
		acc = np.zeros_like(pair)
		height, width = pair.shape[-3:-1]
		for j in (-1, 0, 1):
			for i in (-1, 0, 1):
				dy, dx = j*step, i*step
				if abs(dy) >= height or abs(dx) >= width: continue
				p = (slice(None), slice(max(0, -dy), min(height, height-dy)), slice(max(0, -dx), min(width, width-dx)))
				q = (slice(None), slice(max(0, dy), min(height, height+dy)), slice(max(0, dx), min(width, width+dx)))
				weight = 0.25*(1 if i == 0 else 0.5)*(1 if j == 0 else 0.5)
				acc[p] += weight*(active[p] & active[q])[..., None]*pair[q]
		count = acc[..., 3:]*4**(level+1)
		t = np.clip((count-0.5*budget)/(0.5*budget), 0, 1)
		beta = np.ones_like(t) if level == passes-1 else t*t*(3-2*t)
		result[..., :3] += result[..., 3:]*beta*acc[..., :3]/np.maximum(acc[..., 3:], 1e-12)
		result[..., 3:] *= 1-beta
		pair = acc
		states.append(acc.copy())
	return result, states


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("passes", [1, 2, 3, 4])
def test_cascade_matches_cpu_sums_and_selection(factory, samples, passes):
	renderer = factory(sample_count=samples, alpha=0.1, winner_count=1, reconstruction="atrous", confidence_filter=ConfidenceFilter(cascade_passes=passes))
	render(renderer, (37, 29))
	oit = renderer.transparency
	pair = read(oit.winner_color)
	acceptance = 1-float(np.float16(0.9))
	pair[..., 3] = acceptance
	expected, states = cascade_reference(pair, passes, 4)
	np.testing.assert_allclose(read(oit.pyramid[(passes-1)%2]["pair"], np.float32), states[-1], atol=1e-6)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], acceptance*expected[..., :3], atol=0.0005)
	textures = [p["pair"] for p in oit.pyramid]
	renderer.set_confidence_filter(ConfidenceFilter(cascade_passes=passes, target_winners=8))
	assert textures == [p["pair"] for p in oit.pyramid]
	render(renderer, (37, 29))
	expected, _ = cascade_reference(pair, passes, 8)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], acceptance*expected[..., :3], atol=0.0005)
	assert not read(oit.pyramid[(passes-1)%2]["result"], np.float32)[..., 3].any()
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	render(renderer, (17, 13))
	np.testing.assert_array_equal(read(oit.reconstruction_color), 0)


def test_isolated_winner_is_a_translated_unit_peak_tent(factory, monkeypatch):
	from numpyecs import TransparencyPass as transparency
	original = transparency.Texture
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: original(*args, usage=usage | TextureUsage.COPY_DST, **kwargs))
	renderer = factory(alpha=0.1, winner_count=1, reconstruction="atrous", confidence_filter=ConfidenceFilter(target_winners=1e6))
	render(renderer, (65, 65))
	oit = renderer.transparency
	responses = []
	for x in (31, 32):
		winner = np.zeros((65, 65, 4), np.float16)
		winner[32, x] = (1, 0, 0, 1)
		RenderContext.device.queue.write_texture({"texture": oit.winner_color.handle}, winner, {"bytes_per_row": 65*8}, (65, 65, 1))
		output = Texture((65, 65), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
		commands = RenderContext.commands()
		oit.compose(commands, output.color_attachment())
		RenderContext.submit(commands.finish())
		pair = read(oit.pyramid[1]["pair"], np.float32)[0]
		axis = np.maximum(1-np.abs(np.arange(65)-32)/16, 0)
		tent = axis[:, None]*np.maximum(1-np.abs(np.arange(65)-x)/16, 0)[None, :]
		np.testing.assert_allclose(pair[..., 0]*256, tent, atol=1e-6)
		responses.append(read(oit.reconstruction_color)[0, :, :, 0])
	np.testing.assert_allclose(responses[1][:, 1:], responses[0][:, :-1], atol=0.0001)


def test_rejected_mass_and_overlap_match_composite_weights(factory, monkeypatch):
	from numpyecs import TransparencyPass as transparency
	original = transparency.Texture
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: original(*args, usage=usage | TextureUsage.COPY_DST, **kwargs))
	renderer = factory(alpha=0.1, winner_count=1, reconstruction="atrous", confidence_filter=ConfidenceFilter(target_winners=1e6))
	render(renderer, (39, 35))
	oit = renderer.transparency
	y, x = np.indices((35, 39))
	active = ((x+y < 47) & ((x < 18) | (y > 17)))[None]
	t = np.ones((1, 35, 39, 4), np.float16)
	t[active] = 0.9
	acceptance = 1-t[..., 3].astype(np.float32)
	pair = np.zeros((1, 35, 39, 4), np.float32)
	pair[0, 12, 12] = (1, 0, 0, 1)
	RenderContext.device.queue.write_texture({"texture": oit.transmittance.handle}, t, {"bytes_per_row": 39*8}, (39, 35, 1))
	RenderContext.device.queue.write_texture({"texture": oit.winner_color.handle}, pair.astype(np.float16), {"bytes_per_row": 39*8}, (39, 35, 1))
	pair[..., 3] = acceptance
	expected, states = cascade_reference(pair, 4, 1e6, active)
	output = Texture((39, 35), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	commands = RenderContext.commands()
	oit.compose(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	actual = read(oit.pyramid[1]["pair"], np.float32)
	np.testing.assert_allclose(actual, states[-1], atol=1e-6)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], acceptance[..., None]*expected[..., :3], atol=0.0005)
	# An impulse measures real composite W, including all rejected paths.
	weights = actual[..., 0]*256
	assert weights.max() <= 1+1e-6
	assert np.count_nonzero(weights[~active]) == 0


@pytest.mark.parametrize("samples", [1, 4])
def test_cascade_separated_depth_boundaries(factory, samples):
	renderer = factory(sample_count=samples, layers=2, alpha=0.5, winner_coupling="complement", reconstruction="atrous", confidence_filter=ConfidenceFilter(target_winners=32))
	renderer.instances[0].position = (-0.8, 0, 0)
	renderer.instances[1].position = (0.8, 0, -0.8)
	for item in renderer.instances: item.scale = (0.075, 0.2, 1)
	render(renderer, (83, 61))
	color = read(renderer.transparency.reconstruction_color)
	assert color[:, :, :27, 2].max() < 0.0001
	assert color[:, :, 56:, 0].max() < 0.0001


@pytest.mark.parametrize("passes", [0, 5, True, 1.5])
def test_invalid_cascade_count(passes):
	with pytest.raises(ValueError): ConfidenceFilter(cascade_passes=passes)
