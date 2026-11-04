import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from ECS import ECS, component
from scenes.direct_draw_example import CubeInstances
from scenes import prototype


@component
class Position:
	x: float
	y: float
	z: float


@component
class Rotation:
	x: float
	y: float
	z: float
	w: float


@component
class Scale:
	x: float
	y: float
	z: float


@component
class Tint:
	value: np.uint32


def make_cache():
	world = ECS()
	world.register(Position, Rotation, Scale, Tint)
	ids = world.create(3)
	world.add(ids, Position(1, 2, 3), Rotation(0, 0, 0, 1), Scale(1, 2, 3), Tint(42))
	counts = {"rotation": 0, "scale": 0}
	def pack(field):
		def convert(values):
			counts[field] += 1
			return values[:, :2].astype(np.uint32)
		return convert
	dtype = np.dtype([("iPosition", "f4", 3), ("iTint", "u4", 1), ("iRotation", "u4", 2), ("iScale", "u4", 2)])
	cache = CubeInstances(world, world.query(Position, Rotation, Scale, Tint),
		(Position, Rotation, Scale, Tint), dtype, pack("rotation"), pack("scale"))
	return world, ids, cache, counts


def test_direct_instances_skip_static_packing_and_unchanged_uploads():
	world, ids, cache, counts = make_cache()
	assert cache.update()
	content = cache.content
	rotation, scale, tint = (content[name].copy() for name in ("iRotation", "iScale", "iTint"))
	assert not cache.update()
	world.get(Position)[ids].x += 10
	assert cache.update()
	assert cache.content is content
	np.testing.assert_array_equal(content["iPosition"][:, 0], [11] * 3)
	for name, expected in (("iRotation", rotation), ("iScale", scale), ("iTint", tint)):
		np.testing.assert_array_equal(content[name], expected)
	assert counts == {"rotation": 1, "scale": 1}
	world.set(ids[:1], Rotation(4, 5, 6, 7))
	assert cache.update()
	assert counts == {"rotation": 2, "scale": 1}
	world.get(Tint)[ids[1]].value = 77
	assert cache.update()
	assert content["iTint"][:, 0].tolist() == [42, 77, 42]
	assert counts == {"rotation": 2, "scale": 1}
	assert not cache.update()


def test_direct_instances_refresh_membership_generation_order_layout_and_empty_world():
	world, ids, cache, counts = make_cache()
	cache.update()
	world.delete(ids[0])
	new = world.create()
	world.add(new, Position(99, 0, 0), Rotation(1, 2, 3, 4), Scale(4, 5, 6), Tint(77))
	assert cache.update()
	assert cache.content["iPosition"][:, 0].tolist() == [99, 1, 1]
	assert cache.content["iTint"][:, 0].tolist() == [77, 42, 42]
	assert counts == {"rotation": 2, "scale": 2}
	store = world.get(Position)._store
	store._grow_dense(store._capacity + 1)
	assert cache.update()
	assert counts == {"rotation": 3, "scale": 3}
	world.remove(ids[1], Scale)
	assert cache.update()
	assert len(cache.content) == 2
	world.delete(world.entities())
	assert cache.update()
	assert len(cache.content) == 0
	assert not cache.update()


def test_panel_cache_tracks_only_displayed_fields_and_actor_membership(monkeypatch):
	world = prototype.create_world()
	query = world.query(prototype.Damageable, prototype.Attributes, prototype.ActorMetadata)
	cache = prototype.PanelCache(world, query)
	builds = []
	original = prototype.actor_panels
	def build(*args):
		builds.append(1)
		return original(*args)
	monkeypatch.setattr(prototype, "actor_panels", build)
	assert cache.update()
	panels = cache.panels
	assert not cache.update()
	world.get(prototype.Attributes)[query.entities()[:1]].damage = 9
	assert not cache.update()
	assert cache.panels is panels
	assert len(builds) == 1
	world.get(prototype.Damageable)[query.entities()[1]].health = 5
	assert cache.update()
	assert "mosquito: 5/30" in cache.title
	assert len(builds) == 2
	world.get(prototype.Attributes)[query.entities()[0]].name = "hero"
	assert cache.update()
	assert "hero" in cache.title
	world.delete(query.entities()[-1])
	assert cache.update()
	assert len(cache.panels) == 6
	assert not cache.update()


def test_legacy_ecs_benchmark_queries_refresh_after_spawns_and_deletion(monkeypatch):
	# Test the real benchmark callbacks with a tiny Raylib stub; no optional GUI backend.
	import scenes.benchmarks
	rays = ModuleType("pyray")
	rays.Color = type("Color", (), {})
	rays.LIGHTGRAY = rays.Color()
	rays.get_random_value = lambda low, high: low
	drawn = []
	rays.draw_cube = lambda *args: drawn.append(args)
	loop = ModuleType("scenes.benchmarks.benchmark_loop")
	loop.SPACE_SIZE, loop.CUBE_MAX = 30, 7
	loop.rnd_color = rays.Color
	loop.run = lambda: None
	monkeypatch.setitem(sys.modules, "pyray", rays)
	monkeypatch.setitem(sys.modules, loop.__name__, loop)
	monkeypatch.setattr(scenes.benchmarks, "benchmark_loop", loop, raising=False)
	path = Path(__file__).resolve().parents[1] / "scenes/benchmarks/benchmark_optimized.py"
	spec = importlib.util.spec_from_file_location("_legacy_benchmark_callbacks", path)
	module = importlib.util.module_from_spec(spec)
	monkeypatch.setitem(sys.modules, spec.name, module)
	spec.loader.exec_module(module)
	module.add_cubes()
	assert len(module.movement_query.fetch()) == 250
	module.draw()
	assert len(drawn) == 251  # Includes the stationary ground.
	before = module.movement_query.fetch()
	module.update(0.5)
	assert module.movement_query.fetch() is before
	assert before[module.Position].selection.x[0] == -32
	module.add_cubes()
	assert len(module.movement_query.fetch()) == 500
	module.world.delete(module.movement_query.entities()[0])
	drawn.clear()
	module.draw()
	assert len(drawn) == 500
