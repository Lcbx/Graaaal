"""Structural query joins, lifetimes, and cache behavior against a reference model."""

import numpy as np
import pytest

from graaal.ECS import ECS, Entity, INDEX_MAX, NO_ENTITY, component, entity_index


@component
class Position:
	x: float


@component
class Velocity:
	x: float


@component
class Health:
	x: float


@component
class Blocked:
	x: float


@component(multiples=True)
class MultiA:
	x: float


@component(multiples=True)
class MultiB:
	x: float


TYPES = (Position, Velocity, Health, Blocked, MultiA, MultiB)


def world():
	ecs = ECS()
	ecs.register(*TYPES, capacity=2)
	return ecs


def test_inspection_preserves_input_order_duplicates_and_generations():
	ecs = world()
	ids = ecs.create(4)
	ecs.add(ids[[0, 2]], Position(7))
	ecs.delete(ids[0])
	replacement = ecs.create()[0]
	ecs.add(replacement, Position(9))
	mixed = [ids[0], replacement, ids[2], ids[1], replacement, NO_ENTITY, Entity(10_000)]
	assert ecs.is_alive(mixed).tolist() == [False, True, True, True, True, False, False]
	assert ecs.has(mixed, Position).tolist() == [False, True, True, False, True, False, False]
	assert ecs.is_alive(replacement).tolist() == [True]
	assert ecs.has(ids[1], Position).tolist() == [False]
	assert ecs.is_alive([]).shape == ecs.has([], Position).shape == (0,)
	assert ecs.is_alive([]).dtype == np.bool_
	assert ecs.entities().tolist() == [int(replacement), *map(int, ids[1:])]
	exported = ecs.entities()
	exported[:] = NO_ENTITY
	assert len(ecs.entities()) == 4
	with pytest.raises(KeyError):
		ecs.has([], type("Unregistered", (), {}))


def test_lifecycle_versions_advance_only_for_nonempty_successful_operations():
	ecs = world()
	assert ecs.entity_version == 0
	ids = ecs.create(3)
	assert ecs.entity_version == 1
	ecs.add(ids, Position(1))
	ecs.set(ids, Position(2))
	ecs.remove(ids, Position)
	ecs.create(0)
	ecs.delete([])
	assert ecs.entity_version == 1
	ecs.delete([ids[0], ids[0]])
	assert ecs.entity_version == 2
	ecs.delete(ids[0])
	assert ecs.entity_version == 2
	with pytest.raises(ValueError):
		ecs.create(-1)
	ecs.create()
	assert ecs.entity_version == 3


def test_exhausted_creation_does_not_consume_recycled_handles_or_change_version():
	ecs = world()
	id_ = ecs.create()
	ecs.delete(id_)
	ecs._next_entity_index = INDEX_MAX  # Exercise the boundary without a huge allocation.
	version, free, count = ecs.entity_version, ecs._free_entities.copy(), len(ecs)
	with pytest.raises(OverflowError):
		ecs.create(2)
	assert ecs._free_entities == free
	assert ecs.entity_version == version
	assert len(ecs) == count
	assert not ecs.is_alive(id_).any()


def test_required_excluded_any_and_optional_terms_compose():
	ecs = world()
	ids = ecs.create(7)
	ecs.add(ids[:5], Position, np.arange(5.))
	ecs.add(ids[[0, 2, 4, 5]], Velocity, np.array([10., 20., 40., 50.]))
	ecs.add(ids[[1, 2]], Health, np.array([11., 22.]))
	ecs.add(ids[2], Blocked(1))
	query = ecs.query(Position, exclude=Blocked, any_of=(Velocity, Health), optional=MultiA)
	result = query.fetch()
	assert result.entities.tolist() == ids[[0, 1, 4]].tolist()
	assert result[Position].selection.x.tolist() == [0, 1, 4]
	assert result[Velocity].present.tolist() == [True, False, True]
	assert result[Velocity].selection.x.tolist() == [10, 40]
	assert result[Velocity].offsets.tolist() == [0, 1, 1, 2]
	assert result[Health].present.tolist() == [False, True, False]
	assert result[Health].selection.x.tolist() == [11]
	assert result[MultiA].present.tolist() == [False] * 3
	assert result[MultiA].offsets.tolist() == [0] * 4
	assert len(result[MultiA].selection) == 0
	with pytest.raises(KeyError):
		result[Blocked]  # Excluded-only components are filters, not returned terms.


def test_empty_optional_only_any_only_and_exclusion_only_queries():
	ecs = world()
	all_query = ecs.query()
	assert len(all_query.fetch()) == 0
	assert ecs.query(Position).fetch()[Position].offsets.tolist() == [0]
	ids = ecs.create(3)
	ecs.add(ids[1], Velocity(2))
	ecs.add(ids[2], Blocked(1))
	assert all_query.entities().tolist() == ids.tolist()
	assert ecs.query(exclude=Blocked).entities().tolist() == ids[:2].tolist()
	assert ecs.query(any_of=Velocity).entities().tolist() == ids[1:2].tolist()
	optional = ecs.query(optional=Velocity).fetch()
	assert optional.entities.tolist() == ids.tolist()
	assert optional[Velocity].present.tolist() == [False, True, False]
	assert optional[Velocity].offsets.tolist() == [0, 0, 1, 1]
	assert ecs.query(any_of=()).entities().tolist() == ids.tolist()
	assert ecs.query(Position, exclude=Position).entities().size == 0
	assert ecs.where().size == 0  # Preserve the legacy no-required-terms behavior.
	assert ecs.where(exclude=Blocked).size == 0


def test_multicomponent_rows_are_grouped_per_owner_without_cartesian_expansion():
	ecs = world()
	ids = ecs.create(4)
	ecs.add(ids, Position, np.arange(4.))
	ecs.add(ids[[2, 0, 2, 0, 0]], MultiA, np.array([20., 1., 21., 2., 3.]))
	ecs.add(ids[[1, 0, 1, 2]], MultiB, np.array([10., 4., 11., 22.]))
	result = ecs.query(Position, optional=(MultiA, MultiB)).fetch()
	a, b = result[MultiA], result[MultiB]
	assert a.present.tolist() == [True, False, True, False]
	assert a.offsets.tolist() == [0, 3, 3, 5, 5]
	assert a.selection.x.tolist() == [1, 2, 3, 20, 21]
	assert b.offsets.tolist() == [0, 1, 3, 4, 4]
	assert b.selection.x.tolist() == [4, 10, 11, 22]
	assert ecs.get(MultiA).owner_ids(a.selection).tolist() == ids[[0, 0, 0, 2, 2]].tolist()
	assert ecs.query(MultiA, MultiB).entities().tolist() == ids[[0, 2]].tolist()
	assert len(result) == 4
	# Scatter an owner-wise update to each owner's variable number of rows.
	counts = np.diff(a.offsets)
	a.selection.x += np.repeat(result[Position].selection.x, counts)
	assert ecs.get(MultiA)[ids[2]].x.tolist() == [22, 23]


def count_scans(monkeypatch, query):
	scans = []
	original = query._scan_entities
	def scan():
		scans.append(1)
		return original()
	monkeypatch.setattr(query, "_scan_entities", scan)
	return scans


def test_field_writes_and_unrelated_changes_reuse_the_query_cache(monkeypatch):
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position(1), Velocity(2))
	query = ecs.query(Position, Velocity)
	scans = count_scans(monkeypatch, query)
	result = query.fetch()
	p, v = result[Position].selection, result[Velocity].selection
	p.x += v.x * 0.5
	ecs.set(ids, Velocity(7))
	ecs.get(Position).touch("x")
	ecs.add(ids, Health(8))
	ecs.create()  # Bare entities cannot join a required-component query.
	assert query.fetch() is result
	assert result[Position].selection is p
	assert p.x.tolist() == [2, 2, 2]
	assert v.x.tolist() == [7, 7, 7]
	assert query.entities().tolist() == ids.tolist()
	assert len(scans) == 1


def test_optional_layout_refreshes_rows_without_rescanning_membership(monkeypatch):
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position(1))
	query = ecs.query(Position, optional=Velocity)
	scans = count_scans(monkeypatch, query)
	before = query.fetch()
	saved_term = before[Velocity]
	detached = before[Position].selection
	ecs.add(ids[1], Velocity(2))
	with pytest.raises(RuntimeError, match="stale query result"):
		saved_term.present
	assert len(detached) == 3  # Detached selections retain their own store lifetime.
	result = query.fetch()
	assert result is not before
	assert result.entities.tolist() == ids.tolist()
	assert result[Velocity].present.tolist() == [False, True, False]
	assert len(scans) == 1
	ecs.get(Velocity)._store._grow_dense(100)
	assert query.fetch() is not result
	assert len(scans) == 1


def test_compaction_refreshes_multicomponent_rows_without_rescanning(monkeypatch):
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, MultiA, np.arange(3.))
	ecs.remove(ids[1], MultiA)
	query = ecs.query(MultiA)
	scans = count_scans(monkeypatch, query)
	before = query.fetch()
	ecs.get(MultiA)._store._defragment(4)
	after = query.fetch()
	assert after is not before
	assert after.entities.tolist() == ids[[0, 2]].tolist()
	assert after[MultiA].selection.x.tolist() == [0, 2]
	assert after[MultiA].offsets.tolist() == [0, 1, 2]
	assert len(scans) == 1


def test_excluded_only_layout_changes_do_not_invalidate_cached_result(monkeypatch):
	ecs = world()
	ids = ecs.create(2)
	ecs.add(ids, Position(1))
	query = ecs.query(Position, exclude=Blocked)
	scans = count_scans(monkeypatch, query)
	result = query.fetch()
	ecs.get(Blocked)._store._grow_dense(100)
	ecs.register(type("Unused", (), {"__annotations__": {"x": float}}))
	assert query.fetch() is result
	assert len(result) == 2
	assert len(scans) == 1


def test_filter_membership_changes_expire_all_result_metadata(monkeypatch):
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position(1))
	query = ecs.query(Position, exclude=Blocked)
	scans = count_scans(monkeypatch, query)
	before = query.fetch()
	term = before[Position]
	ecs.add(ids[1], Blocked(1))
	for read in (lambda: before.entities, lambda: len(before), lambda: before[Position],
		lambda: term.present, lambda: term.offsets, lambda: term.selection):
		with pytest.raises(RuntimeError, match="stale query result"):
			read()
	assert query.entities().tolist() == ids[[0, 2]].tolist()
	assert len(scans) == 2
	result = query.fetch()
	assert query.fetch() is result
	ecs.remove(ids[1], Blocked)
	assert query.entities().tolist() == ids.tolist()
	assert len(scans) == 3


@pytest.mark.parametrize("kind", ["all", "excluded", "optional", "any"])
def test_queries_without_required_terms_follow_entity_lifecycle_and_reuse(kind):
	ecs = world()
	query = {"all": lambda: ecs.query(), "excluded": lambda: ecs.query(exclude=Blocked),
		"optional": lambda: ecs.query(optional=Velocity), "any": lambda: ecs.query(any_of=Velocity)}[kind]()
	before = query.fetch()
	ids = ecs.create(2)
	with pytest.raises(RuntimeError, match="stale query result"):
		before.entities
	ecs.add(ids, Velocity(1))
	ecs.delete(ids[0])
	reused = ecs.create()[0]
	ecs.add(reused, Velocity(2))
	assert query.entities().tolist() == [int(reused), int(ids[1])]
	assert query.fetch().entities.tolist() == [int(reused), int(ids[1])]
	assert not ecs.is_alive(ids[0]).any()


def test_required_query_refreshes_after_delete_and_same_index_reinsertion():
	ecs = world()
	query = ecs.query(Position)
	ids = ecs.create(2)
	ecs.add(ids, Position(1))
	before = query.fetch()
	ecs.delete(ids[0])
	replacement = ecs.create()[0]
	ecs.add(replacement, Position(2))
	assert query.entities().tolist() == [int(replacement), int(ids[1])]
	assert query.fetch()[Position].selection.x.tolist() == [2, 1]
	with pytest.raises(RuntimeError, match="stale"):
		before[Position]


def test_exported_query_arrays_cannot_corrupt_cached_membership_or_alignment():
	ecs = world()
	ids = ecs.create(2)
	ecs.add(ids[0], Position(3))
	query = ecs.query(optional=Position)
	result = query.fetch()
	query.entities()[:] = NO_ENTITY
	result.entities[:] = NO_ENTITY
	result[Position].present[:] = False
	result[Position].offsets[:] = 99
	result[Position].selection.get_rows()[:] = 1
	assert query.fetch() is result
	assert result.entities.tolist() == ids.tolist()
	assert result[Position].present.tolist() == [True, False]
	assert result[Position].offsets.tolist() == [0, 1, 1]
	assert result[Position].selection.x.tolist() == [3]


def test_value_predicates_are_reevaluated_after_cached_query_field_writes():
	ecs = world()
	ids = ecs.create(2)
	ecs.add(ids, Position, np.array([1., 2.]))
	query = ecs.query(Position)
	result = query.fetch()
	positions = ecs.get(Position)
	assert positions.owner_ids(positions.query_rows(lambda x: x > 1, query.entities())).tolist() == ids[1:].tolist()
	result[Position].selection.x = [3, 0]
	assert query.fetch() is result
	assert positions.owner_ids(positions.query_rows(lambda x: x > 1, query.entities())).tolist() == ids[:1].tolist()


@pytest.mark.parametrize("kw", ["required", "exclude", "any_of", "optional"])
def test_query_validates_registered_component_types_at_construction(kw):
	ecs = world()
	unregistered = type("Unregistered", (), {})
	for value, error in ((unregistered, KeyError), (42, TypeError)):
		with pytest.raises(error):
			if kw == "required":
				ecs.query(value)
			else:
				ecs.query(**{kw: [value]})
	assert len(ecs) == 0


def test_duplicate_and_overlapping_terms_are_normalized():
	ecs = world()
	ids = ecs.create(2)
	ecs.add(ids, Position(2))
	query = ecs.query(Position, Position, any_of=(Position, Position), optional=(Position, Velocity, Velocity))
	assert query.entities().tolist() == ids.tolist()
	assert query.fetch()[Position].selection.x.tolist() == [2, 2]
	assert not query.fetch()[Velocity].present.any()


def test_query_masks_support_the_highest_registered_component_bit():
	ecs = ECS()
	types = [component(type(f"C{i}", (), {"__annotations__": {"x": float}})) for i in range(64)]
	ecs.register(*types)
	ids = ecs.create(3)
	ecs.add(ids[0], types[63](1))
	ecs.add(ids[1], types[0](2))
	assert ecs.query(any_of=(types[0], types[63])).entities().tolist() == ids[:2].tolist()
	assert ecs.query(types[63], exclude=types[0]).fetch()[types[63]].selection.x.tolist() == [1]
	assert ecs.query(exclude=types[63]).entities().tolist() == ids[1:].tolist()


@pytest.mark.parametrize("seed", [12, 43])
def test_reusable_queries_match_uncached_reference_after_mixed_mutations(seed):
	rng = np.random.default_rng(seed)
	ecs = world()
	# required, excluded, any_of, optional
	specs = [
		((), (), (), (Position, MultiA)),
		((Position,), (Blocked,), (), (Velocity, MultiA, MultiB)),
		((), (Blocked,), (Velocity, Health), (Position, MultiA)),
		((MultiA,), (), (Position, MultiB), (Velocity,)),
		((Position, Velocity), (), (), (Health,)),
		((), (Blocked, Health), (), (MultiB,)),
		((Position,), (Position,), (), ()),
	]
	queries = [ecs.query(*req, exclude=exc, any_of=any_, optional=opt) for req, exc, any_, opt in specs]
	model = {}
	for step in range(140):
		op = int(rng.integers(6))
		if not model or op == 0:
			for owner in ecs.create(int(rng.integers(1, 4))):
				model[int(owner)] = {}
		else:
			owner = sorted(model)[int(rng.integers(len(model)))]
			cls = TYPES[int(rng.integers(len(TYPES)))]
			if op == 1:
				ecs.delete(owner)
				del model[owner]
			elif op in (2, 3):
				if cls in (MultiA, MultiB):
					values = [float(step)] * int(rng.integers(1, 4))
					ecs.add([owner] * len(values), cls, np.array(values))
					model[owner].setdefault(cls, []).extend(values)
				else:
					(ecs.set if cls in model[owner] else ecs.add)(owner, cls(float(step)))
					model[owner][cls] = [float(step)]
			elif op == 4:
				ecs.remove(owner, cls)
				model[owner].pop(cls, None)
			elif cls in (MultiA, MultiB):
				accessor = ecs.get(cls)
				accessor.remove(accessor.query_rows(lambda x: x % 2 == 0, owner))
				kept = [v for v in model[owner].get(cls, []) if v % 2 != 0]
				if kept:
					model[owner][cls] = kept
				else:
					model[owner].pop(cls, None)
			elif step % 17 == 0:
				store = ecs.get(cls)._store
				store._grow_dense(store._capacity + 1)

		assert len(ecs) == len(model)
		assert ecs.entities().tolist() == sorted(model, key=lambda e: int(entity_index(e)))
		for query, (req, exc, any_, opt) in zip(queries, specs):
			expected = sorted((owner for owner, data in model.items()
				if all(t in data for t in req) and not any(t in data for t in exc)
				and (not any_ or any(t in data for t in any_))), key=lambda e: int(entity_index(e)))
			result = query.fetch()
			assert query.entities().tolist() == result.entities.tolist() == expected
			for cls in dict.fromkeys((*req, *any_, *opt)):
				values = [model[owner].get(cls, []) for owner in expected]
				assert result[cls].present.tolist() == [bool(v) for v in values]
				assert result[cls].offsets.tolist() == [0, *np.cumsum([len(v) for v in values]).tolist()]
				assert result[cls].selection.x.tolist() == [x for group in values for x in group]
