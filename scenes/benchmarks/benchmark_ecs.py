"""CPU-only ECS benchmarks; setup is excluded from timing and traced peaks.

Run from any directory: python scenes/benchmarks/benchmark_ecs.py --output result.json
Timing and memory passes are separate. Peaks are tracemalloc allocations during
the action (including NumPy buffers where supported), not process RSS. Persistent
world storage is measured separately with array.nbytes. Results are observations,
not performance assertions for CI.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ECS import ECS, Entity, Index, NO_ENTITY, component, entity_index


@component
class Position:
	x: float
	y: float
	z: float


@component
class Velocity:
	x: float
	y: float
	z: float


@component
class Rare:
	pass


@component(multiples=True)
class Item:
	value: float


def measure(setup, action, repeat, loops=1):
	action(setup())  # Untimed warmup; use fresh state for mutating operations.
	samples = []
	for _ in range(repeat):
		state = setup()
		start = time.perf_counter_ns()
		for _ in range(loops):
			action(state)
		samples.append((time.perf_counter_ns() - start) / loops / 1e6)
	state = setup()
	gc.collect()
	tracemalloc.start()
	action(state)
	_, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	return {"median_ms": statistics.median(samples), "min_ms": min(samples), "peak_traced_bytes": peak, "loops_per_sample": loops}


def make_world(n, rare_count=None):
	w = ECS()
	p, v, r = w.register(Position, Velocity, Rare)
	ids = w.create(n)
	w.add(ids, Position(1, 2, 3), Velocity(0.1, 0.2, 0.3))
	if rare_count:
		# Include the highest index and scatter owners through the index space.
		w.add(ids[np.linspace(0, n - 1, rare_count, dtype=int)], Rare)
	return w, p, v, r


def scan(query):
	w = query._ecs
	current = w._current_entities[:w._next_entity_index]
	masks = w.entity_masks[:current.size]
	selected = (current != NO_ENTITY) & ((masks & query._required) == query._required)
	if query._excluded:
		selected &= (masks & query._excluded) == 0
	if query._any:
		selected &= (masks & query._any) != 0
	return current[selected]


def candidates(query):
	w = query._ecs
	store = w._get_store(Rare)
	indices = np.sort(entity_index(store._dense_entities[:store._size]))
	masks = w.entity_masks[indices]
	return w._current_entities[indices[(masks & query._required) == query._required]]


def main():
	global ECS
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--entities", type=int, default=100_000)
	parser.add_argument("--repeat", type=int, default=7)
	parser.add_argument("--loops", type=int, default=30)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--implementation", type=Path, help="Optional compatible ECS.py snapshot for baseline comparisons")
	args = parser.parse_args()
	if args.entities < 100 or args.repeat < 1 or args.loops < 1:
		parser.error("entities must be >= 100; repeat and loops must be positive")
	n, repeat, loops = args.entities, args.repeat, args.loops
	source = args.implementation or Path(__file__).resolve().parents[2] / "src" / "ECS.py"
	if args.implementation:
		spec = importlib.util.spec_from_file_location("_benchmark_ecs_snapshot", source)
		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)
		ECS = module.ECS
	records = {}
	def bench(name, setup, action, count=loops):
		records[name] = measure(setup, action, repeat, count)
		print(f"{name}: {records[name]['median_ms']:.4f} ms", flush=True)

	for divisor in (1, 10, 100, 1000):
		w, *_ = make_world(n, max(1, n // divisor))
		query = w.query(Position, Rare)
		np.testing.assert_array_equal(scan(query), candidates(query))
		for name, action in (("scan", scan), ("candidates", candidates), ("production", lambda q: q._scan_entities())):
			bench(f"query_{n // divisor}_of_{n}_{name}", lambda: query, action)
		query.fetch()
		bench(f"query_{n // divisor}_of_{n}_cached", lambda: query, lambda q: q.fetch())

	w, p, v, _ = make_world(n)
	query = w.query(Position, Velocity)
	query.fetch()
	def move(state, cached):
		if cached:
			result = query.fetch()
			pos, vel = result[Position].selection, result[Velocity].selection
		else:
			ids = w.where(Position, Velocity)
			pos, vel = p[ids], v[ids]
		for field in ("x", "y", "z"):
			pos[field] += vel[field] * 0.01
	bench("movement_gather_scatter_uncached", lambda: None, lambda s: move(s, False))
	bench("movement_gather_scatter_cached", lambda: None, lambda s: move(s, True))

	def deleted_world():
		world, *_ = make_world(n)
		world.delete(world.entities()[:-100])
		return world.query(Position)
	bench("query_heavily_deleted_world", deleted_world, lambda q: q._scan_entities())

	def removal_state(k):
		world, positions, *_ = make_world(n)
		rows = np.unique(np.linspace(0, n - 1, k, dtype=Index))
		return positions._store, rows
	for k in (1, 4, 8, 16, 64):
		bench(f"remove_{k}_from_{n}", lambda k=k: removal_state(k), lambda s: s[0]._remove_rows(s[1]), 1)
		rows = removal_state(k)[1]
		tail = np.arange(n - len(rows), n, dtype=Index)
		def mask_sources(_):
			mask = np.zeros(n, dtype=bool)
			mask[rows] = True
			return tail[~mask[tail]]
		def tail_sources(_):
			return tail[~np.any(tail[:, None] == rows[None, :], axis=1)]
		np.testing.assert_array_equal(mask_sources(None), tail_sources(None))
		bench(f"remove_{k}_sources_full_mask", lambda: None, mask_sources)
		bench(f"remove_{k}_sources_tail_mask", lambda: None, tail_sources)

	def churn_state():
		world, *_ = make_world(n)
		return world, world.entities()[::max(1, n // 100)], world.query(Position, exclude=Rare)
	def churn(state):
		world, owners, query = state
		world.add(owners, Rare)
		query.fetch()
		world.remove(owners, Rare)
		query.fetch()
	bench("membership_churn_100_owners_two_queries", churn_state, churn)

	def item_world():
		world = ECS()
		world.register(Item, capacity=1)
		return world, world.create(2)
	for count in (256, 1024):
		def append(state, alternating=False):
			world, ids = state
			for i in range(count):
				world.add(ids[i % 2 if alternating else 0], Item(float(i)))
		def batch(state):
			world, ids = state
			world.add(np.repeat(ids[:1], count), Item, np.arange(count, dtype=float))
		bench(f"append_{count}_tail_owner", item_world, append, 1)
		bench(f"append_{count}_alternating_owners", item_world, lambda s: append(s, True), 1)
		bench(f"append_{count}_one_batch", item_world, batch, 1)

	world = ECS()
	rare, = world.register(Rare)
	ids = world.create(n)
	world.add(ids[-1], Rare)
	store = rare._store
	memory = {"high_index": int(ids[-1]), "owners": 1, "sparse_bytes": store._sparse.nbytes,
		"dense_bytes": sum(a.nbytes for a in store._dense.values()),
		"world_state_bytes": world.entity_masks.nbytes + world._current_entities.nbytes,
		"hypothetical_one_page_bytes": 256 * np.dtype(Index).itemsize}
	dirty = {"entities": n, "changed_entities": 100, "position_field_bytes": n * 3 * 8,
		"changed_position_bytes": 100 * 3 * 8,
		"note": "Byte estimates only; no dirty-tracking implementation or upload benchmark."}
	result = {"environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
		"processor": platform.processor(), "entities": n, "repeat": repeat, "loops": loops,
		"implementation": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
		"memory_metric": "Peak action-only tracemalloc allocation, separate from timing; excludes preexisting storage and is not RSS."},
		"benchmarks": records, "rare_high_index_memory": memory, "dirty_tracking_estimate": dirty}
	args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
