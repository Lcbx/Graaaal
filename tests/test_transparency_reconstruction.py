"""Fragment/compute parity, boundary workgroups, and profile pass timestamps."""
from math import sin, cos
from unittest.mock import Mock
import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.GpuProfiler import GpuProfiler


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("kernel", [3, 5, 9])
@pytest.mark.parametrize("mix,force", [("average", False), ("average", True), ("fill", True)])
def test_compute_matches_fragment_for_tilted_overlap_and_partial_tiles(factory, samples, kernel, mix, force):
	renderer = factory(sample_count=samples, kernel=kernel, noise_mode="anchored_blue", layers=2,
		winner_mix=mix, force_single_contributor=force)
	renderer.instances[0].scale = (0.07, 0.04, 1)
	renderer.instances[0].rotation = (0, sin(0.5), 0, cos(0.5))
	before = render(renderer, (37, 29))
	shader = renderer.transparency.shader
	renderer.set_reconstruction("compute")
	after = render(renderer, (37, 29))
	assert renderer.transparency.shader is shader  # Geometry did not acquire new shader features.
	np.testing.assert_allclose(after, before, atol=1)
	assert renderer.transparency.reconstruction_color.size == (37, 29, samples)
	renderer.set_reconstruction("fragment")
	np.testing.assert_array_equal(before, render(renderer, (37, 29)))
	assert renderer.transparency.reconstruction_color is None


@pytest.mark.parametrize("alpha", [0, 1])
def test_compute_skips_shared_halo_for_empty_and_certain_tiles(factory, alpha):
	renderer = factory(alpha=alpha, kernel=9)
	before = render(renderer, (17, 11))
	renderer.set_reconstruction("compute")
	np.testing.assert_allclose(render(renderer, (17, 11)), before, atol=1)


def test_compute_does_not_present_stale_empty_pixels_after_resize_or_msaa_switch(factory):
	renderer = factory(alpha=1, reconstruction="compute", force_single_contributor=True)
	render(renderer, (17, 11))
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	after = render(renderer, (17, 11))
	renderer.set_reconstruction("fragment")
	np.testing.assert_array_equal(render(renderer, (17, 11)), after)
	renderer.set_reconstruction("compute")
	renderer.set_sample_count(4)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0.3)
	after = render(renderer, (23, 9))
	renderer.set_reconstruction("fragment")
	np.testing.assert_allclose(render(renderer, (23, 9)), after, atol=1)


def test_compute_debug_views_and_1x1_do_not_read_stale_intermediate(factory, monkeypatch):
	renderer = factory(sample_count=4, layers=2, noise_mode="object_hash", winner_mix="fill", force_single_contributor=True)
	for view in ("final", "transmittance", "winner", "reconstruction", "filter_support", "winner_a", "winner_b", "winner_count", "forced"):
		renderer.transparency.set_debug_view(view)
		renderer.set_reconstruction("fragment")
		before = render(renderer, (13, 19))
		renderer.set_reconstruction("compute")
		np.testing.assert_allclose(render(renderer, (13, 19)), before, atol=1)
	from dataclasses import replace
	renderer.transparency.set_bilateral_filter(replace(renderer.transparency.bilateral_filter, kernel_size=1))
	renderer.transparency.set_debug_view("final")
	monkeypatch.setattr(renderer.transparency.reconstruct_pipeline, "_check_device", Mock(side_effect=AssertionError("1x1 must skip compute")))
	after = render(renderer, (13, 19))
	renderer.set_reconstruction("fragment")
	np.testing.assert_array_equal(render(renderer, (13, 19)), after)


def test_gpu_profiler_measures_native_passes_and_rejects_unread_frames():
	RenderContext.setup_graphics_backend(True)
	if "timestamp-query" not in RenderContext.adapter.features: pytest.skip("No GPU timestamps")
	RenderContext.setup_graphics_backend(True, required_features=["timestamp-query"])
	profiler = GpuProfiler()
	texture = Texture((8, 8), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	try:
		cmd = profiler.wrap(RenderContext.commands())
		clear = cmd.render_pass(color=texture.color_attachment(clear=(0, 0, 0, 1)), label="clear")
		cmd.compute_pass(label="unused")  # An unentered scope must allocate no timestamp.
		with cmd.compute_pass(label="empty"): pass
		with clear: pass
		assert profiler.labels == ["empty", "clear"]
		RenderContext.submit(cmd.finish())
		with pytest.raises(RuntimeError, match="previous"): profiler.wrap(RenderContext.commands())
		measurements = profiler.read()
		assert set(measurements) == {"clear", "empty"}
		assert all(np.isfinite(t) and t >= 0 for t in measurements.values())
		assert profiler.last_frame_ms >= 0
		with pytest.raises(RuntimeError, match="Finish"): profiler.read()
	finally: profiler.close()
