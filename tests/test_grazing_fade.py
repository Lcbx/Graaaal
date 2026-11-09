"""Grazing-fade configuration and settings edits without initializing graphics."""
from types import SimpleNamespace

import pytest

from numpyecs.TransparencyPass import BilateralFilter, GrazingFade
from scenes import transparency_example as scene


@pytest.mark.parametrize("options", [
	{"enabled": 1}, {"start_angle": -1}, {"end_angle": 91},
	{"start_angle": 85, "end_angle": 85}, {"start_angle": 86, "end_angle": 85},
	{"start_angle": float("nan")}, {"end_angle": float("inf")},
	{"start_angle": 85, "end_angle": 85+1e-9},
])
def test_grazing_fade_rejects_invalid_or_unrepresentable_intervals(options):
	with pytest.raises(ValueError):
		GrazingFade(**options)


def test_grazing_fade_accepts_full_angular_range_and_defaults_disabled():
	assert GrazingFade() == GrazingFade(enabled=False, start_angle=70, end_angle=85)
	assert GrazingFade(enabled=True, start_angle=0, end_angle=90).enabled


def test_grazing_settings_and_toggle_keep_angles_ordered(monkeypatch):
	renderer = SimpleNamespace(bilateral_filter=BilateralFilter(), grazing_fade=GrazingFade())
	renderer.set_grazing_fade = lambda value: setattr(renderer, "grazing_fade", value)
	settings = scene.FilterSettings(renderer)
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	scene.on_char(ord("g"))
	assert renderer.grazing_fade.enabled
	settings.set_grazing_angle("start_angle", 100)
	assert renderer.grazing_fade.start_angle == pytest.approx(84.99)
	settings.set_grazing_angle("end_angle", -10)
	assert renderer.grazing_fade.end_angle == pytest.approx(85)
	settings.set_grazing_angle("start_angle", -10)
	settings.set_grazing_angle("end_angle", 100)
	assert renderer.grazing_fade == GrazingFade(enabled=True, start_angle=0, end_angle=90)
	before = renderer.grazing_fade
	settings.set_grazing_angle("start_angle", float("nan"))
	assert renderer.grazing_fade is before
	scene.on_char(ord("G"))
	assert not renderer.grazing_fade.enabled
