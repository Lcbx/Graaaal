"""Keyboard settings and range validation without creating a GPU device."""

from types import SimpleNamespace

import pytest
import glfw

from graaal.TransparencyPass import BilateralFilter
from scenes.rendering_settings import FilterSettings, SETTINGS
from scenes import transparency_example as scene


def make_renderer(**overrides):
	values = dict(banded_tail=False, bilateral_filter=BilateralFilter(), winner_count=1, winner_mix="average",
		force_single_contributor=False, debug_view="final", reconstruction="fragment")
	values.update(overrides)
	renderer = SimpleNamespace(**values)
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	renderer.set_debug_view = lambda value: setattr(renderer, "debug_view", value)
	return renderer


def test_antialiasing_modes_disable_msaa_for_postprocess():
	renderer = make_renderer(sample_count=4)
	renderer.set_sample_count = lambda count: setattr(renderer, "sample_count", count)
	aa = SimpleNamespace(mode="off")
	panel = FilterSettings(renderer, aa_pass=aa)
	panel.set_aa_mode("lxaa")
	assert renderer.sample_count == 1 and aa.mode == "lxaa"
	panel.set_aa_mode("faaa")
	assert renderer.sample_count == 1 and aa.mode == "faaa"
	panel.set_aa_mode("msaa")
	assert renderer.sample_count == 4 and aa.mode == "msaa"
	panel.set_aa_mode("off")
	assert renderer.sample_count == 1 and aa.mode == "off"


def test_filter_reset_restores_all_startup_values():
	options = BilateralFilter(kernel_size=9, spatial_sigma=4, depth_sigma=0.12,
		transmittance_sigma=0.7, min_depth_sigma=0.03)
	renderer = make_renderer(bilateral_filter=options)
	panel = FilterSettings(renderer)
	renderer.set_bilateral_filter(BilateralFilter())
	panel.reset()
	assert renderer.bilateral_filter == options


def test_arrow_navigation_editing_fine_steps_and_reset():
	renderer = make_renderer()
	settings = FilterSettings(renderer)
	assert {row[0] for row in SETTINGS} == set(vars(renderer.bilateral_filter))
	def key(code, mods=0, action=glfw.PRESS):
		return settings.handle_key(code, 0, action, mods)
	assert key(glfw.KEY_UP)
	assert settings.selected == len(SETTINGS)-1
	assert key(glfw.KEY_DOWN)
	assert settings.selected == 0
	for expected in (1, 1):
		key(glfw.KEY_LEFT)
		assert renderer.bilateral_filter.kernel_size == expected
	for expected in (3, 5, 9, 9):
		key(glfw.KEY_RIGHT)
		assert renderer.bilateral_filter.kernel_size == expected
	settings.selected = next(i for i, row in enumerate(SETTINGS) if row[0] == "spatial_sigma")
	key(glfw.KEY_RIGHT, glfw.MOD_SHIFT)
	assert renderer.bilateral_filter.spatial_sigma == pytest.approx(settings.defaults.spatial_sigma+0.01)
	key(glfw.KEY_LEFT, action=glfw.RELEASE)
	assert renderer.bilateral_filter.spatial_sigma == pytest.approx(settings.defaults.spatial_sigma+0.01)
	key(glfw.KEY_HOME)
	assert renderer.bilateral_filter == settings.defaults
	key(glfw.KEY_F)
	assert settings.visible is False
	assert not key(glfw.KEY_RIGHT)
	key(glfw.KEY_F, action=glfw.REPEAT)
	assert settings.visible is False
	key(glfw.KEY_F)
	assert settings.visible is True
	assert not key(glfw.KEY_F5)
	assert not key(glfw.KEY_F9)
	assert renderer.bilateral_filter == settings.defaults


def test_arrow_edits_keep_sigmas_positive():
	renderer = make_renderer()
	settings = FilterSettings(renderer)
	for name in ("depth_sigma", "transmittance_sigma", "min_depth_sigma"):
		settings.selected = next(i for i, row in enumerate(SETTINGS) if row[0] == name)
		for _ in range(30):
			settings.adjust(-1)
		assert getattr(renderer.bilateral_filter, name) > 0


@pytest.mark.parametrize("name,value,expected", [
	("depth_sigma", -2, 0.0001), ("spatial_sigma", -2, 0.01),
	("transmittance_sigma", -2, 0.001), ("min_depth_sigma", -2, 0.000001),
])
def test_widget_edits_use_validated_ranges(name, value, expected):
	renderer = make_renderer()
	settings = FilterSettings(renderer)
	settings.set_value(name, value)
	assert getattr(renderer.bilateral_filter, name) == pytest.approx(expected)
	before = renderer.bilateral_filter
	settings.set_value(name, float("nan"))
	assert renderer.bilateral_filter is before


def test_noise_shortcuts_cycle_all_modes_and_adjust_grain(monkeypatch):
	renderer = SimpleNamespace(noise_mode="uv_blue_pick", noise_scale=1.0, winner_mix="average")
	renderer.set_noise_mode = lambda value: setattr(renderer, "noise_mode", value)
	renderer.set_noise_scale = lambda value: setattr(renderer, "noise_scale", value)
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	visited = set()
	for _ in scene.NOISE_MODES:
		scene.on_char(ord("n"))
		visited.add(renderer.noise_mode)
	assert visited == set(scene.NOISE_MODES) and renderer.noise_mode == "uv_blue_pick"
	scene.on_char(ord("]"))
	assert renderer.noise_scale > 1
	scene.on_char(ord("["))
	assert renderer.noise_scale == pytest.approx(1)
	for _ in range(30):
		scene.on_char(ord("["))
	assert renderer.noise_scale == 0.25


@pytest.mark.parametrize("count,mix,kernel,force,extras", [
	(1, "average", 1, False, ()),
	(1, "average", 3, True, ("filter_support", "forced")),
	(2, "average", 3, False, ("filter_support", "winner_a", "winner_b", "winner_count")),
	(4, "fill", 3, False, ("filter_support", "winner_count")),
	(8, "fill", 1, True, ("winner_count", "forced")),
])
def test_diagnostic_choices_and_shortcut_match_active_technique(monkeypatch, count, mix, kernel, force, extras):
	renderer = make_renderer(winner_count=count, winner_mix=mix, bilateral_filter=BilateralFilter(kernel_size=kernel),
		force_single_contributor=force)
	expected = {"final", "transmittance", "winner", "reconstruction", *extras}
	views = FilterSettings.available_debug_views(renderer)
	assert set(views) == expected
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	visited = set()
	for _ in views:
		scene.on_char(ord("d"))
		visited.add(renderer.debug_view)
	assert visited == expected and renderer.debug_view == "final"
	if mix == "fill":
		assert FilterSettings.debug_view_label(renderer, "winner_count") == "winner_validity"


def test_unavailable_selection_resets_on_feature_and_kernel_changes(monkeypatch):
	renderer = make_renderer(winner_count=2, debug_view="winner_b")
	renderer.winner_mix = "fill"
	FilterSettings.sync_debug_view(renderer)
	assert renderer.debug_view == "final"
	renderer.debug_view = "winner_count"
	renderer.winner_count = 1
	FilterSettings.sync_debug_view(renderer)
	assert renderer.debug_view == "final"
	renderer.debug_view = "forced"
	FilterSettings.sync_debug_view(renderer)
	assert renderer.debug_view == "final"
	settings = FilterSettings(renderer)
	renderer.debug_view = "filter_support"
	settings.set_value("kernel_size", 1)
	assert renderer.debug_view == "final"
	renderer.set_bilateral_filter(BilateralFilter(kernel_size=3))
	renderer.debug_view = "filter_support"
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	scene.on_char(ord("1"))
	assert renderer.debug_view == "final"
