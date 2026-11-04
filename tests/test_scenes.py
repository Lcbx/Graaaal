import importlib
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scenes import prototype as scene


def test_prototype_creates_original_actors_and_converts_predicate_rows_to_owner_ids():
	world = scene.create_world()
	actors = world.query(scene.Damageable, scene.Attributes, scene.ActorMetadata)
	ids = actors.entities()
	assert world.get(scene.Attributes)[ids].name.tolist() == ["player", "mosquito", "goblin"]
	assert world.get(scene.Damageable)[ids].health.tolist() == [150, 30, 50]
	assert world.get(scene.Essences)[ids].essence1.tolist() == [None] * 3
	idle = world.query(scene.ActorMetadata, exclude=scene.Action)
	assert scene.inactive_enemy_ids(world, idle).tolist() == ids[1:].tolist()
	world.add(ids[1], scene.Action("attack", 0, 1, ids[0]))
	assert scene.inactive_enemy_ids(world, idle).tolist() == ids[2:].tolist()


def test_prototype_panels_show_health_and_refresh_after_field_and_membership_edits():
	world = scene.create_world()
	query = world.query(scene.Damageable, scene.Attributes, scene.ActorMetadata)
	ids = query.entities()
	panels, title = scene.actor_panels(world, query)
	assert panels.dtype == scene.PANEL_DTYPE
	assert len(panels) == 9
	assert "player: 150/100" in title
	np.testing.assert_allclose(panels[2::3]["iRect"][:, 2], panels[1::3]["iRect"][:, 2])
	world.set(ids[1], scene.Damageable(5))
	updated, title = scene.actor_panels(world, query)
	assert "mosquito: 5/30" in title
	assert updated[5]["iRect"][2] == pytest.approx(updated[4]["iRect"][2] / 6)
	world.get(scene.Attributes)[ids[1]].max_health = 0
	updated, _ = scene.actor_panels(world, query)
	assert updated[5]["iRect"][2] == 0
	assert np.isfinite(updated["iRect"]).all()
	world.delete(ids[1])
	updated, title = scene.actor_panels(world, query)
	assert len(updated) == 6
	assert "mosquito" not in title
	world.delete(world.entities())
	assert len(scene.actor_panels(world, query)[0]) == 0


@pytest.mark.parametrize("entrypoint", [True, False])
def test_scene_launcher_caps_frames_for_entrypoint_and_legacy_scenes(monkeypatch, capsys, entrypoint):
	frames, presents, imports = [], [], []
	def window_loop():
		presents.append(1)
		return True
	context = SimpleNamespace(window_loop=window_loop)
	render_module = ModuleType("RenderContext")
	render_module.RenderContext = context
	monkeypatch.setitem(sys.modules, "RenderContext", render_module)
	def run_scene():
		while context.window_loop():
			frames.append(1)
	def load(name):
		imports.append(name)
		if entrypoint:
			return SimpleNamespace(run_scene=run_scene)
		run_scene()
		return SimpleNamespace()
	monkeypatch.setattr(importlib, "import_module", load)
	monkeypatch.setattr(sys, "argv", ["main.py", "scenes/example.py", "--frames", "3"])
	runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
	assert imports == ["scenes.example"]
	assert len(frames) == 3
	assert len(presents) == 4  # Last submitted frame is also presented.
	assert context.window_loop is window_loop
	assert "Scene completed: 3 frames" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
	["scenes/example.py", "--frames", "0"],
	["scenes/example.py", "--frames", "-1"],
	["--frames", "3"],
	["scenes/example.py", "--frames", "3", "--compile"],
])
def test_scene_launcher_rejects_invalid_frame_modes_before_importing_scene(monkeypatch, argv):
	monkeypatch.setattr(sys, "argv", ["main.py", *argv])
	with pytest.raises(SystemExit) as caught:
		runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
	assert caught.value.code == 2
