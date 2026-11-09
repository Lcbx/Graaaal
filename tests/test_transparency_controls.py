"""Keyboard settings and profile validation without creating a GPU device."""

from types import SimpleNamespace
import json

import pytest
import glfw

from scenes.transparency_example import BilateralFilter, FilterSettings, SETTINGS
from scenes import transparency_example as scene


def test_filter_list_round_trips_all_settings_and_resets(tmp_path):
	options = BilateralFilter(kernel_size=9, spatial_sigma=4, depth_sigma=0.12,
		transmittance_sigma=0.7, min_depth_sigma=0.03)
	renderer = SimpleNamespace(bilateral_filter=options)
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	panel = FilterSettings(renderer)
	path = tmp_path / "filter.json"
	panel.save(path)
	renderer.set_bilateral_filter(BilateralFilter(kernel_size=1))
	panel.load(path)
	assert renderer.bilateral_filter == options
	renderer.set_bilateral_filter(BilateralFilter())
	panel.reset()
	assert renderer.bilateral_filter == options


def test_older_profile_keeps_supported_values_and_drops_retired_controls(tmp_path):
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter())
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	settings = FilterSettings(renderer)
	path = tmp_path / "old.json"
	path.write_text(json.dumps({"kernel_size": 5, "spatial_sigma": 2, "adaptive": True,
		"adaptation_mode": "distance", "slope_aware": False, "target_winners": 4,
		"filter_near": 3, "filter_far": 12, "far_spatial_sigma": 3,
		"far_depth_sigma": 0.1, "far_transmittance_sigma": 0.5, "far_transmittance_weight": 0}))
	settings.load(path)
	assert renderer.bilateral_filter == BilateralFilter(kernel_size=5, spatial_sigma=2)
	settings.save(path)
	assert set(json.loads(path.read_text())) == {row[0] for row in SETTINGS}


@pytest.mark.parametrize("content", ['{"depth_sigma": -1}', '{"unknown": 4}', '{broken', '[]'])
def test_invalid_filter_profiles_leave_running_settings_unchanged(tmp_path, content):
	options = BilateralFilter()
	renderer = SimpleNamespace(bilateral_filter=options)
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	panel = FilterSettings(renderer)
	path = tmp_path / "invalid.json"
	path.write_text(content)
	with pytest.raises((ValueError, TypeError)):
		panel.load(path)
	assert renderer.bilateral_filter is options


def test_arrow_navigation_editing_fine_steps_and_reset(tmp_path):
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter())
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	settings = FilterSettings(renderer, tmp_path / "profile.json")
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
	key(glfw.KEY_F5)
	settings.adjust(1)
	key(glfw.KEY_F9)
	assert renderer.bilateral_filter == settings.defaults


def test_arrow_edits_keep_sigmas_positive():
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter())
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
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
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter())
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	settings = FilterSettings(renderer)
	settings.set_value(name, value)
	assert getattr(renderer.bilateral_filter, name) == pytest.approx(expected)
	before = renderer.bilateral_filter
	settings.set_value(name, float("nan"))
	assert renderer.bilateral_filter is before


def test_noise_shortcuts_cycle_all_modes_and_adjust_grain(monkeypatch):
	renderer = SimpleNamespace(noise_mode="uv_blue_pick", noise_scale=1.0)
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
