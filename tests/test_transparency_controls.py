"""Keyboard settings and profile validation without creating a GPU device."""

from types import SimpleNamespace

import pytest
import glfw

from scenes.transparency_example import BilateralFilter, FilterSettings, SETTINGS


def test_filter_list_round_trips_all_settings_and_resets(tmp_path):
	options = BilateralFilter(filter_near=3, filter_far=12, far_spatial_sigma=3,
		far_depth_sigma=0.12, far_transmittance_sigma=0.7, far_transmittance_weight=0.25)
	renderer = SimpleNamespace(bilateral_filter=options)
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	panel = FilterSettings(renderer)
	path = tmp_path / "filter.json"
	panel.save(path)
	renderer.set_bilateral_filter(BilateralFilter(adaptive=False, kernel_size=3))
	panel.load(path)
	assert renderer.bilateral_filter == options
	renderer.set_bilateral_filter(BilateralFilter())
	panel.reset()
	assert renderer.bilateral_filter == options


@pytest.mark.parametrize("content", ['{"filter_far": -1}', '{"unknown": 4}', '{broken'])
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
	key(glfw.KEY_LEFT)
	assert renderer.bilateral_filter.adaptive is False
	key(glfw.KEY_RIGHT)
	assert renderer.bilateral_filter.adaptive is True
	key(glfw.KEY_DOWN)
	key(glfw.KEY_LEFT)
	assert renderer.bilateral_filter.kernel_size == 3
	key(glfw.KEY_RIGHT)
	assert renderer.bilateral_filter.kernel_size == 5
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


def test_arrow_edits_keep_distances_and_sigmas_valid():
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter(filter_near=1, filter_far=2))
	renderer.set_bilateral_filter = lambda value: setattr(renderer, "bilateral_filter", value)
	settings = FilterSettings(renderer)
	for name, direction in (("filter_near", 1), ("filter_far", -1), ("depth_sigma", -1), ("far_transmittance_weight", 1)):
		settings.selected = next(i for i, row in enumerate(SETTINGS) if row[0] == name)
		for _ in range(30):
			settings.adjust(direction)
	assert 0 <= renderer.bilateral_filter.filter_near < renderer.bilateral_filter.filter_far
	assert renderer.bilateral_filter.depth_sigma > 0
	assert renderer.bilateral_filter.far_transmittance_weight == 1


@pytest.mark.parametrize("name,value,expected", [
	("filter_near", 100, 49.9999), ("filter_far", 0, 10.0001),
	("depth_sigma", -2, 0.0001), ("far_transmittance_weight", 2, 1),
	("far_transmittance_weight", -2, 0),
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
