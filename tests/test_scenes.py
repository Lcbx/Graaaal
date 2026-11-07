import importlib
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

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
	actors = scene.actor_statuses(world, query)
	assert len(actors) == 3
	assert actors[0] == scene.ActorStatus("player", 150, 100, True)
	assert all(actor.health_fraction == 1 for actor in actors)
	world.set(ids[1], scene.Damageable(5))
	updated = scene.actor_statuses(world, query)
	assert updated[1].name == "mosquito" and updated[1].health == 5
	assert updated[1].health_fraction == pytest.approx(1/6)
	world.get(scene.Attributes)[ids[1]].max_health = 0
	updated = scene.actor_statuses(world, query)
	assert updated[1].health_fraction == 0
	world.delete(ids[1])
	updated = scene.actor_statuses(world, query)
	assert len(updated) == 2
	assert "mosquito" not in [actor.name for actor in updated]
	world.delete(world.entities())
	assert scene.actor_statuses(world, query) == ()


@pytest.mark.parametrize("entrypoint", [True, False])
def test_scene_launcher_caps_frames_for_entrypoint_and_legacy_scenes(monkeypatch, capsys, entrypoint):
	frames, presents, imports = [], [], []
	def window_loop():
		presents.append(1)
		return True
	context = SimpleNamespace(window_loop=window_loop)
	render_module = ModuleType("numpyecs.RenderContext")
	render_module.RenderContext = context
	monkeypatch.setitem(sys.modules, "numpyecs.RenderContext", render_module)
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
