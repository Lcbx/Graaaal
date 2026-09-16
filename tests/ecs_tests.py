from __future__ import annotations

from enum import IntFlag, auto
from typing import Any

import numpy as np
import pytest

from ECS import *


def test_higher_pow2() -> None:
	for i in range(0, 512):
		res = higher_pow2(i)
		assert res >= max(1, i)
		assert res & (res - 1) == 0


def test_create_entities() -> None:
	ecs = ECS()
	ids = ecs.create(3)

	assert isinstance(ids, np.ndarray)
	assert ids.dtype == Entity
	assert ids.tolist() == [0, 1, 2]
	assert ecs._next_entity_index == 3
	assert len(ecs) == 3


@component
class Foo:
	a: float
	b: str

@component
class Bar:
	a: float
	b: float

def test_single_component_storage_basic() -> None:
	store = ComponentStorage(Foo, capacity=2)

	store._add_range(as_entities([1]), Foo(1.23, "first"))
	selection = ComponentSelection(store, store._get_rows(as_entities([1])))
	assert selection.a.tolist() == pytest.approx([1.23])
	assert selection.b.tolist() == ["first"]

	emptied = store._remove_entities(as_entities([1]))
	assert emptied.tolist() == [1]
	assert store._get_rows(as_entities([1])).size == 0

	store._add_range(as_entities([1]), Foo(4.56, "second"))
	store._add_range(
		as_entities([2, 3]),
		(np.array([7.89, 0.12]), np.array(["third", "fourth"])),
	)

	# forces dense growth from initial capacity 2
	assert store._capacity >= 3

	selection = ComponentSelection(store, store._get_rows(as_entities([3])))
	assert selection.a.tolist() == pytest.approx([0.12])
	assert selection.b.tolist() == ["fourth"]

	selection.a = 0.69
	assert store._dense["a"][store._get_rows(as_entities([3]))][0] == pytest.approx(0.69)

	store._remove_entities(as_entities([2]))
	assert store._get_rows(as_entities([2])).size == 0

	# forces flat sparse growth
	store._add_range(as_entities([10_000]), Foo(0.5, "fifth"))
	row = store._get_rows(as_entities([10_000]))
	assert store._dense["a"][row][0] == pytest.approx(0.5)
	assert store._dense["b"][row][0] == "fifth"


def test_single_component_storage_sparse_dense_integrity() -> None:
	store = ComponentStorage(Foo, capacity=1)
	ids = as_entities([5, 6, 7])
	store._add_range(
		ids,
		(
			np.array([10.0, 20.0, 30.0]),
			np.array(["val5", "val6", "val7"]),
		),
	)

	assert store._capacity >= 3
	dense_rows = store._get_dense_indices(ids)
	assert np.all(dense_rows != np.iinfo(Index).max)
	assert np.all(dense_rows < store._size)

	store._remove_entities(as_entities([6]))

	assert store._get_rows(as_entities([6])).size == 0
	for eid, expected in ((5, 10.0), (7, 30.0)):
		row = store._get_rows(as_entities([eid]))
		assert store._dense["a"][row][0] == pytest.approx(expected)


@component
class Position2D:
	x: float
	y: float


@component
class Velocity2D:
	x: float
	y: float


class TagEnum(IntFlag):
	Enemy = auto()
	Flying = auto()
	Dazed = auto()


@component
class Tag:
	value: TagEnum


@component
class Position:
	x: float
	y: float
	z: float


@component
class Orientation:
	qx: float
	qy: float
	qz: float
	qw: float


@component
class LocalAABB:
	x_min: float
	y_min: float
	z_min: float
	x_max: float
	y_max: float
	z_max: float


@component
class AxisAlignedBoundingBox:
	x_min: float
	y_min: float
	z_min: float
	x_max: float
	y_max: float
	z_max: float


@component(multiples=True)
class MultiComp:
	val: float


def make_pos2d_ecs(vals: list[tuple[float, float]]) -> tuple[ECS, EntityArray, ComponentAccessor]:
	ecs = ECS()
	ecs.register(Position2D, capacity=len(vals))
	ids = ecs.create(len(vals))
	ecs.add(ids, Position2D, np.asarray(vals, dtype=float))
	return ecs, ids, ecs.get(Position2D)


def make_multitag_ecs(
	blocks_per_entity: dict[int, list[TagEnum | int]],
) -> tuple[ECS, EntityArray, ComponentAccessor]:
	ecs = ECS()
	ecs.register(Tag, allow_same_type_components_per_entity=True, capacity=32)

	max_id = max(blocks_per_entity, default=-1)
	ids = ecs.create(max_id + 1)

	owners: list[int] = []
	values: list[TagEnum | int] = []
	for eid, tags in blocks_per_entity.items():
		owners.extend([eid] * len(tags))
		values.extend(tags)

	if owners:
		ecs.add(as_entities(owners), Tag, np.asarray(values, dtype=object))

	return ecs, ids, ecs.get(Tag)


def test_add_payload_forms() -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D)

	a = ecs.create(3)
	positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
	vx = np.array([10.0, 20.0, 30.0])
	vy = np.array([-1.0, -2.0, -3.0])

	ecs.add(a, Position2D, positions, Velocity2D, (vx, vy))

	pos = ecs.get(Position2D)[a]
	vel = ecs.get(Velocity2D)[a]
	assert np.allclose(pos.x, positions[:, 0])
	assert np.allclose(pos.y, positions[:, 1])
	assert np.allclose(vel.x, vx)
	assert np.allclose(vel.y, vy)

	b = ecs.create(2)
	ecs.add(b, Position2D(7.0, 8.0))
	assert np.allclose(ecs.get(Position2D)[b].x, [7.0, 7.0])
	assert np.allclose(ecs.get(Position2D)[b].y, [8.0, 8.0])


def test_get_and_modify_component_range() -> None:
	ecs, ids, pos = make_pos2d_ecs([(0, 0), (1, 1), (2, 2)])

	pos[ids].x += 10
	pos[ids[1:]].y = np.array([20.0, 30.0])

	assert pos[ids].x.tolist() == pytest.approx([10.0, 11.0, 12.0])
	assert pos[ids].y.tolist() == pytest.approx([0.0, 20.0, 30.0])


def test_component_rows_follow_dense_compaction() -> None:
	ecs, ids, pos = make_pos2d_ecs([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

	ecs.remove(ids[[3, 0]], Position2D)
	new_id = ecs.create()
	ecs.add(new_id, Position2D(-1, -1))

	remaining = ecs.where(Position2D)
	assert sorted(remaining.tolist()) == [1, 2, 4, 5]

	active_rows = pos.get_rows()
	assert sorted(active_rows.tolist()) == [0, 1, 2, 3]
	assert sorted(pos.owner_ids(active_rows).tolist()) == [1, 2, 4, 5]


def test_query_simple_vectorized() -> None:
	ecs, _, pos = make_pos2d_ecs([(0, 0), (1, 2), (5, -3), (-1, 1), (4, 10)])
	out = pos.query(lambda x, y: (x > 0) & (np.abs(y) < 3))

	assert out.dtype == Index
	assert pos.owner_ids(out).tolist() == [1]


def test_query_annotated_types() -> None:
	_, _, pos = make_pos2d_ecs([(0, 0), (1, 2), (5, -3), (-1, 1), (4, 10)])

	def condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return (x > 0) & (np.abs(y) < 3)

	out = pos.query(condition)
	assert out.dtype == Index
	assert pos.owner_ids(out).tolist() == [1]


def test_query_subset_filtering() -> None:
	_, ids, pos = make_pos2d_ecs([(10, 0), (9, 0), (11, 0), (8, 0)])
	subset = ids[[0, 1, 3]]

	out = pos.query(lambda x, y: x >= 10, subset)
	assert pos.owner_ids(out).tolist() == [0]


def test_query_returns_empty_when_no_match() -> None:
	_, _, pos = make_pos2d_ecs([(0, 0), (1, 1)])
	out = pos.query(lambda x, y: x < 0)
	assert isinstance(out, np.ndarray)
	assert out.dtype == Index
	assert out.size == 0


def test_query_can_use_entity_id_in_predicate() -> None:
	_, _, pos = make_pos2d_ecs([(5, 0), (5, 0), (5, 0)])
	out = pos.query(lambda x, y, _entity: (x == 5) & (_entity != 1))
	assert pos.owner_ids(out).tolist() == [0, 2]


def test_query_raises_on_length_mismatch() -> None:
	_, _, pos = make_pos2d_ecs([(0, 0), (1, 1), (2, 2)])
	with pytest.raises(ValueError):
		pos.query(lambda x, y: np.array([True, False]))


def test_query_on_empty_storage() -> None:
	ecs = ECS()
	ecs.register(Position2D)
	out = ecs.get(Position2D).query(lambda x: x > 0)
	assert out.dtype == Index
	assert out.size == 0


def test_query_mult_comp_returns_matching_rows() -> None:
	_, _, tags = make_multitag_ecs(
		{
			0: [1, 0],
			1: [0, 0],
			2: [2, 1],
		}
	)

	out = tags.query(lambda value: value == 1)
	assert out.dtype == Index
	assert tags.owner_ids(out).tolist() == [0, 2]
	assert tags._store._dense["value"][out].tolist() == [1, 1]


def test_query_mult_comp_subset_filtering() -> None:
	_, ids, tags = make_multitag_ecs(
		{
			2: [4, 5],
			5: [3, 3],
			7: [1, 2],
			9: [1],
		}
	)

	out = tags.query(lambda value: value == 1, ids[[5, 7, 9]])
	assert tags.owner_ids(out).tolist() == [7, 9]


def test_query_mult_comp_no_match_returns_empty() -> None:
	_, _, tags = make_multitag_ecs({0: [0, 0], 1: [2], 3: [4, 5]})
	out = tags.query(lambda value: value == 9)
	assert out.size == 0


def test_mult_comp_get_returns_all_components_for_entities() -> None:
	_, ids, tags = make_multitag_ecs({0: [0, 0], 1: [2], 3: [4, 5], 4: [6, 6]})
	selection = tags[ids[[0, 3]]]
	assert selection.value.tolist() == [0, 0, 4, 5]


def test_query_intflag_predicate() -> None:
	ecs = ECS()
	ecs.register(Tag)
	ids = ecs.create(3)
	ecs.add(
		ids,
		Tag,
		np.asarray(
			[
				TagEnum.Enemy | TagEnum.Dazed,
				TagEnum.Flying,
				TagEnum.Enemy,
			],
			dtype=object,
		),
	)

	tags = ecs.get(Tag)
	out = tags.query(lambda value: (value & TagEnum.Enemy) != 0)
	assert tags.owner_ids(out).tolist() == [0, 2]


def test_movement_system() -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D, Tag)

	entities_ = ecs.create(3)
	ecs.add(entities_[1:], Position2D, np.array([[0.0, 0.0], [5.0, -3.0]]))
	ecs.add(entities_[1:], Velocity2D, (np.array([1.0, -1.0]), np.array([2.0, 0.5])))
	ecs.add(entities_[2], Tag(TagEnum.Enemy | TagEnum.Dazed))

	positions = ecs.get(Position2D)
	velocities = ecs.get(Velocity2D)

	assert positions[entities_[1]].x[0] == pytest.approx(0.0)
	assert positions[entities_[1]].y[0] == pytest.approx(0.0)
	assert positions[entities_[2]].x[0] == pytest.approx(5.0)
	assert positions[entities_[2]].y[0] == pytest.approx(-3.0)

	targets = ecs.where(Tag, Position2D, Velocity2D)
	positions[targets].x -= velocities[targets].x * 0.5
	positions[targets].y -= velocities[targets].y * 0.5

	assert positions[entities_[2]].x[0] == pytest.approx(5.5)
	assert positions[entities_[2]].y[0] == pytest.approx(-3.25)
	assert positions[entities_[1]].x[0] == pytest.approx(0.0)
	assert positions[entities_[1]].y[0] == pytest.approx(0.0)

	untagged = ecs.where(Position2D, Velocity2D, exclude=[Tag])
	assert entities_[2] not in untagged
	assert entities_[1] in untagged

	tag_value = ecs.get(Tag)[entities_[2]].value[0]
	assert tag_value & TagEnum.Enemy
	assert not (tag_value & TagEnum.Flying)

	tag_rows = ecs.get(Tag).query(lambda value: (value & TagEnum.Enemy) != 0)
	assert ecs.get(Tag).owner_ids(tag_rows).tolist() == [2]


def rotate_vectors_by_quaternions(q: np.ndarray, v: np.ndarray) -> np.ndarray:
	q_vec = q[:, None, :3]
	qw = q[:, None, 3:4]
	t = 2.0 * np.cross(q_vec, v, axis=2)
	return v + qw * t + np.cross(q_vec, t, axis=2)


def update_world_aabb(ecs: ECS) -> None:
	nr_ents = ecs.where(LocalAABB, Position)
	la = ecs.get(LocalAABB)[nr_ents]
	pos = ecs.get(Position)[nr_ents]

	positions = np.column_stack((pos.x, pos.y, pos.z))
	mins = np.column_stack((la.x_min, la.y_min, la.z_min))
	maxs = np.column_stack((la.x_max, la.y_max, la.z_max))

	wa_min = mins + positions
	wa_max = maxs + positions

	aabbs = ecs.get(AxisAlignedBoundingBox)[nr_ents]
	aabbs.x_min = wa_min[:, 0]
	aabbs.y_min = wa_min[:, 1]
	aabbs.z_min = wa_min[:, 2]
	aabbs.x_max = wa_max[:, 0]
	aabbs.y_max = wa_max[:, 1]
	aabbs.z_max = wa_max[:, 2]

	rot_ents = ecs.where(LocalAABB, Position, Orientation)
	if rot_ents.size == 0:
		return

	la = ecs.get(LocalAABB)[rot_ents]
	pos = ecs.get(Position)[rot_ents]
	ori = ecs.get(Orientation)[rot_ents]

	positions = np.column_stack((pos.x, pos.y, pos.z))
	orientations = np.column_stack((ori.qx, ori.qy, ori.qz, ori.qw))
	mins = np.column_stack((la.x_min, la.y_min, la.z_min))
	maxs = np.column_stack((la.x_max, la.y_max, la.z_max))

	corners = np.array(
		[[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
		dtype=int,
	)
	offsets = (
		mins[:, None, :] * (1 - corners)[None, :, :]
		+ maxs[:, None, :] * corners[None, :, :]
	)
	points = rotate_vectors_by_quaternions(orientations, offsets) + positions[:, None, :]
	wa_min = points.min(axis=1)
	wa_max = points.max(axis=1)

	aabbs = ecs.get(AxisAlignedBoundingBox)[rot_ents]
	aabbs.x_min = wa_min[:, 0]
	aabbs.y_min = wa_min[:, 1]
	aabbs.z_min = wa_min[:, 2]
	aabbs.x_max = wa_max[:, 0]
	aabbs.y_max = wa_max[:, 1]
	aabbs.z_max = wa_max[:, 2]


def detect_aabb_overlaps(ecs: ECS) -> list[tuple[int, int]]:
	ents = ecs.where(AxisAlignedBoundingBox)
	boxes = ecs.get(AxisAlignedBoundingBox)[ents]
	mins = np.column_stack((boxes.x_min, boxes.y_min, boxes.z_min))
	maxs = np.column_stack((boxes.x_max, boxes.y_max, boxes.z_max))

	ok1 = mins[:, None, :] <= maxs[None, :, :]
	ok2 = maxs[:, None, :] >= mins[None, :, :]
	overlap = np.logical_and(ok1, ok2).all(axis=2)

	i, j = np.triu_indices(ents.size, k=1)
	hits = overlap[i, j]
	return [
		(int(ents[ii]), int(ents[jj]))
		for ii, jj, hit in zip(i, j, hits)
		if hit
	]


def test_aabb_system_and_overlap() -> None:
	ecs = ECS()
	ecs.register(LocalAABB, Position, AxisAlignedBoundingBox, Orientation)

	all_entities = ecs.create(6)
	e1 = all_entities[0:1]
	e2 = all_entities[2:3]

	ecs.add(
		e1,
		LocalAABB(-1, -1, -1, 1, 1, 1),
		Position(5, 0, 0),
		AxisAlignedBoundingBox(0, 0, 0, 0, 0, 0),
		Orientation(0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)),
	)
	ecs.add(
		e2,
		LocalAABB(-2, -2, -2, 2, 2, 2),
		Position(5, 0, 0),
		AxisAlignedBoundingBox(0, 0, 0, 0, 0, 0),
	)

	update_world_aabb(ecs)
	wa1 = ecs.get(AxisAlignedBoundingBox)[e1]

	half_diag = np.sqrt(2)
	assert wa1.x_min[0] == pytest.approx(5 - half_diag)
	assert wa1.x_max[0] == pytest.approx(5 + half_diag)
	assert wa1.y_min[0] == pytest.approx(-half_diag)
	assert wa1.y_max[0] == pytest.approx(half_diag)
	assert wa1.z_min[0] == pytest.approx(-1)
	assert wa1.z_max[0] == pytest.approx(1)

	hits = detect_aabb_overlaps(ecs)
	assert hits == [(int(e1[0]), int(e2[0]))]


def test_storage_multicomp_add_get_remove() -> None:
	store = MultiComponentStorage(MultiComp, capacity=4)
	entity_id = Entity(42)

	store._add_range(
		as_entities([entity_id, entity_id, entity_id]),
		np.array([1.0, 2.0, 3.0]),
	)

	component_rows = store._get_rows(as_entities([entity_id]))
	assert store._dense["val"][component_rows].tolist() == pytest.approx([1.0, 2.0, 3.0])

	store._remove_rows(component_rows[1:2])
	component_rows = store._get_rows(as_entities([entity_id]))
	assert store._dense["val"][component_rows].tolist() == pytest.approx([1.0, 3.0])


def test_ecs_multicomp_reassignment() -> None:
	ecs = ECS()
	ecs.register(MultiComp, allow_same_type_components_per_entity=True, capacity=5)
	ecs.register(Position2D, capacity=5)

	ecs.create(12)
	e = ecs.create()
	e2 = ecs.create()

	ecs.add(np.repeat(e, 3), MultiComp, np.array([1.1, 2.2, 3.3]))
	ecs.add(e, Position2D(1, 1))
	ecs.add(np.repeat(e2, 3), MultiComp, np.array([5.5, 6.6, 7.7]))
	ecs.add(e2, Position2D(2, 2))
	ecs.add(e, MultiComp, 4.4)

	comps = ecs.get(MultiComp)
	assert sorted(comps[e].val.tolist()) == pytest.approx([1.1, 2.2, 3.3, 4.4])

	to_remove = comps.query(lambda val: (val == 2.2) | (val == 3.3), e)
	comps.remove(to_remove)
	assert sorted(comps[e].val.tolist()) == pytest.approx([1.1, 4.4])
	assert sorted(comps[e2].val.tolist()) == pytest.approx([5.5, 6.6, 7.7])

	positions = ecs.get(Position2D)
	assert positions[e2].x[0] == 2
	assert positions[e2].y[0] == 2

	ecs.delete(e2)
	assert e2[0] not in ecs.where(MultiComp)
	assert e2[0] not in ecs.where(Position2D)

	ecs.add(e, MultiComp, 8.8)
	assert sorted(comps[e].val.tolist()) == pytest.approx([1.1, 4.4, 8.8])

	ecs.add(e, MultiComp, 9.9)
	assert sorted(comps[e].val.tolist()) == pytest.approx([1.1, 4.4, 8.8, 9.9])


def test_multicomp_accessor_remove_clears_entity_mask_only_when_empty() -> None:
	ecs = ECS()
	ecs.register(MultiComp, allow_same_type_components_per_entity=True)
	ids = ecs.create(2)

	owners = np.array([ids[0], ids[0], ids[1]], dtype=Entity)
	ecs.add(owners, MultiComp, np.array([1.0, 2.0, 3.0]))

	comps = ecs.get(MultiComp)
	first = comps.query(lambda val: val == 1.0)
	comps.remove(first)
	assert ids.tolist() == ecs.where(MultiComp).tolist()

	second = comps.query(lambda val: val == 2.0)
	comps.remove(second)
	assert ecs.where(MultiComp).tolist() == [int(ids[1])]


def test_ecs_multicomp_defragment() -> None:
	ecs = ECS()
	ecs.register(MultiComp, allow_same_type_components_per_entity=True, capacity=12)
	ids = ecs.create(16)
	comps = ecs.get(MultiComp)

	owners = ids[[0, 4, 8, 12]]
	for owner in owners:
		owner_array = as_entities([owner] * 3)
		ecs.add(owner_array, MultiComp, np.array([owner, owner + 1, owner + 2], dtype=float))
		remove_rows = comps.query(lambda val: val != float(owner), owner_array)
		comps.remove(remove_rows)

	store = ecs._get_store(MultiComp)
	owner_indices = store._get_dense_indices( entity_index(owners) )
	assert store._dense_counts[owner_indices].tolist() == [1, 1, 1, 1]
	assert sorted(store._dense_entities[: store._size][store._dense_entities[: store._size] != NO_ENTITY].tolist()) == [0, 4, 8, 12]

	# Adding 8 rows with a capacity of 12 forces defragmentation of the holes.
	for owner in owners:
		owner_array = as_entities([owner] * 2)
		ecs.add(
			owner_array,
			MultiComp,
			np.array([owner + 1, owner + 2], dtype=float),
		)

	owner_indices = store._get_dense_indices( entity_index(owners) )
	assert store._dense_counts[owner_indices].tolist() == [3, 3, 3, 3]
	assert store._live_count == 12
	assert store._size == 12
	assert np.all(store._dense_entities[: store._size] != NO_ENTITY)

	for owner in owners:
		vals = np.sort(comps[as_entities([owner])].val)
		assert vals.tolist() == pytest.approx(
			[float(owner), float(owner + 1), float(owner + 2)]
		)


def test_batched_remove_and_delete() -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D)
	ids = ecs.create(8)

	ecs.add(ids, Position2D, np.column_stack((ids, ids)).astype(float))
	ecs.add(ids, Velocity2D, (np.ones(8), np.ones(8)))

	ecs.remove(ids[[1, 3, 5]], Velocity2D)
	assert ecs.where(Velocity2D).tolist() == [0, 2, 4, 6, 7]
	assert ecs.where(Position2D, Velocity2D).tolist() == [0, 2, 4, 6, 7]

	ecs.delete(ids[[0, 2, 7]])
	assert len(ecs) == 5
	assert ecs.where(Position2D).tolist() == [1, 3, 4, 5, 6]
	assert ecs.where(Position2D, Velocity2D).tolist() == [4, 6]

	assert len(ecs.get(Position2D)[ids[0]]) == 0


def test_generational_id_reuse() -> None:
	ecs = ECS()
	ids = ecs.create(3)
	stale = ids[1:2]

	ecs.delete(stale)
	replacement = ecs.create()

	stale_index = entity_index(stale)
	replacement_index = entity_index(replacement)
	stale_generation = entity_generation(stale)
	replacement_generation = entity_generation(replacement)

	assert isinstance(stale_index, np.ndarray)
	assert isinstance(replacement_index, np.ndarray)
	assert isinstance(stale_generation, np.ndarray)
	assert isinstance(replacement_generation, np.ndarray)

	assert replacement_index.tolist() == stale_index.tolist()
	assert replacement_generation.tolist() == [int(stale_generation[0]) + 1]
	assert replacement[0] == stale[0] + GENERATION_INCREMENT
	assert replacement[0] != stale[0]
	assert len(ecs) == 3


def test_stale_entity_id_is_ignored_after_reuse() -> None:
	ecs = ECS()
	ecs.register(Position2D)

	stale = ecs.create()
	ecs.add(stale, Position2D(1.0, 2.0))
	ecs.delete(stale)

	current = ecs.create()
	assert entity_index(current[0]) == entity_index(stale[0])
	assert current[0] != stale[0]

	ecs.add(current, Position2D(3.0, 4.0))
	assert ecs.get(Position2D)[current].x.tolist() == pytest.approx([3.0])

	assert len(ecs.get(Position2D)[stale]) == 0
	ecs.add(stale, Position2D(5.0, 6.0))
	ecs.delete(stale)

	assert ecs.get(Position2D)[current].x.tolist() == pytest.approx([3.0])
	assert ecs.where(Position2D).tolist() == current.tolist()
	assert len(ecs) == 1


def test_reused_entity_starts_without_old_component_mask() -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D)

	old = ecs.create()
	ecs.add(old, Position2D(1.0, 2.0), Velocity2D(3.0, 4.0))
	ecs.delete(old)

	new = ecs.create()
	assert entity_index(new[0]) == entity_index(old[0])
	assert ecs.where(Position2D).size == 0
	assert ecs.where(Velocity2D).size == 0

	ecs.add(new, Position2D(10.0, 20.0))
	assert ecs.where(Position2D).tolist() == new.tolist()
	assert ecs.where(Velocity2D).size == 0


def test_component_storage_checks_full_generation() -> None:
	store = ComponentStorage(Foo)

	old = as_entities([5])
	newer = old + GENERATION_INCREMENT
	store._add_range(old, Foo(1.0, "old"))

	assert store._get_rows(newer).size == 0
	assert store._get_rows(old).size == 1


def test_batch_reuse_increments_each_generation() -> None:
	ecs = ECS()
	ids = ecs.create(6)
	deleted = ids[[1, 3, 5]]
	ecs.delete(deleted)

	reused = ecs.create(3)

	reused_indices = entity_index(reused)
	reused_generations = entity_generation(reused)
	assert isinstance(reused_indices, np.ndarray)
	assert isinstance(reused_generations, np.ndarray)

	assert sorted(reused_indices.tolist()) == [1, 3, 5]
	assert reused_generations.tolist() == [1, 1, 1]
	assert len(ecs) == 6


def test_mixed_live_and_stale_add_filters_values_in_lockstep() -> None:
	ecs = ECS()
	ecs.register(Position2D)

	stale = ecs.create()
	ecs.delete(stale)
	current = ecs.create()
	other = ecs.create()

	mixed = np.asarray([stale[0], current[0], other[0]], dtype=Entity)
	values = np.asarray([[100.0, 101.0], [2.0, 3.0], [4.0, 5.0]])
	ecs.add(mixed, Position2D, values)

	pos = ecs.get(Position2D)
	assert len(pos[stale]) == 0
	assert pos[current].x.tolist() == pytest.approx([2.0])
	assert pos[current].y.tolist() == pytest.approx([3.0])
	assert pos[other].x.tolist() == pytest.approx([4.0])
	assert pos[other].y.tolist() == pytest.approx([5.0])


def test_add_parse_error_does_not_partially_mutate() -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D)
	ids = ecs.create(2)

	positions = np.asarray([[1.0, 2.0], [3.0, 4.0]])
	with pytest.raises(TypeError):
		ecs.add(ids, Position2D, positions, Velocity2D)

	assert ecs.where(Position2D).size == 0
	assert ecs.where(Velocity2D).size == 0


def recycled_entities(ecs: ECS, quantity: int) -> tuple[EntityArray, EntityArray]:
	old = ecs.create(quantity)
	ecs.delete(old)
	new = ecs.create(quantity)

	assert np.all(new > INDEX_MASK)
	assert np.array_equal(entity_index(new), entity_index(old))

	return old, new


def test_high_entity_ids_create_delete() -> None:
	ecs = ECS()

	old, ids = recycled_entities(ecs, 4)

	assert len(ecs) == 4
	assert np.all(ids > INDEX_MASK)

	ecs.delete(ids)

	assert len(ecs) == 0

	reused = ecs.create(4)
	assert np.all(reused > ids)
	assert np.array_equal(entity_index(reused), entity_index(ids))


def test_high_entity_ids_add_get_remove() -> None:
	ecs = ECS()
	ecs.register(Foo)

	_, ids = recycled_entities(ecs, 3)

	ecs.add(
		ids,
		Foo,
		(
			np.array([1.0, 2.0, 3.0]),
			np.array(["a", "b", "c"], dtype=object),
		),
	)

	foo = ecs.get(Foo)

	assert foo[ids].a.tolist() == pytest.approx([1.0, 2.0, 3.0])
	assert foo[ids].b.tolist() == ["a", "b", "c"]

	ecs.remove(ids[[0, 2]], Foo)

	assert ecs.where(Foo).tolist() == [ids[1]]


def test_high_entity_ids_where() -> None:
	ecs = ECS()
	ecs.register(Foo, Bar)

	_, ids = recycled_entities(ecs, 4)

	ecs.add(ids, Foo, (np.arange(4.0), ["a", "b", "c", "d"]))
	ecs.add(ids[[1, 3]], Bar, np.array([10.0, 20.0]))

	assert np.array_equal(
		ecs.where(Foo),
		ids,
	)

	assert np.array_equal(
		ecs.where(Foo, Bar),
		ids[[1, 3]],
	)

	assert np.array_equal(
		ecs.where(Foo, exclude=Bar),
		ids[[0, 2]],
	)


def test_high_entity_ids_accessor_query() -> None:
	ecs = ECS()
	ecs.register(Foo)

	_, ids = recycled_entities(ecs, 4)

	ecs.add(
		ids,
		Foo,
		(
			np.array([1.0, 2.0, 3.0, 4.0]),
			["a", "b", "c", "d"],
		),
	)

	foo = ecs.get(Foo)

	rows = foo.query(lambda a: a >= 3, ids)

	assert foo[ids[[2, 3]]].a.tolist() == pytest.approx([3.0, 4.0])
	assert foo[ids[[0, 1]]].a.tolist() == pytest.approx([1.0, 2.0])

	foo.remove(rows)

	assert np.array_equal(
		ecs.where(Foo),
		ids[:2],
	)


def test_high_entity_ids_multicomponent() -> None:
	ecs = ECS()
	ecs.register(
		MultiComp,
		allow_same_type_components_per_entity=True,
	)

	_, ids = recycled_entities(ecs, 3)

	owners = np.repeat(ids, 2)
	values = np.array([
		10.0, 11.0,
		20.0, 21.0,
		30.0, 31.0,
	])

	ecs.add(owners, MultiComp, values)

	comps = ecs.get(MultiComp)

	assert np.sort(comps[ids[0]].val).tolist() == pytest.approx([10.0, 11.0])
	assert np.sort(comps[ids[1]].val).tolist() == pytest.approx([20.0, 21.0])
	assert np.sort(comps[ids[2]].val).tolist() == pytest.approx([30.0, 31.0])

	rows = comps.query(lambda val: val % 10 == 1, ids)
	comps.remove(rows)

	assert comps[ids[0]].val.tolist() == pytest.approx([10.0])
	assert comps[ids[1]].val.tolist() == pytest.approx([20.0])
	assert comps[ids[2]].val.tolist() == pytest.approx([30.0])

	assert np.array_equal(ecs.where(MultiComp), ids)


def test_stale_low_generation_ids_are_ignored_by_public_api() -> None:
	ecs = ECS()
	ecs.register(Foo)

	stale, ids = recycled_entities(ecs, 3)

	ecs.add(ids, Foo, (np.array([1.0, 2.0, 3.0]), ["a", "b", "c"]))

	# Same sparse indices, different generations.
	assert np.array_equal(entity_index(stale), entity_index(ids))
	assert np.all(stale != ids)

	# Stale IDs must not resolve to the live generation.
	assert len(ecs.get(Foo)[stale]) == 0
	assert ecs.get(Foo).query(lambda a: a > 0, stale).size == 0

	ecs.remove(stale, Foo)
	assert np.array_equal(ecs.where(Foo), ids)

	ecs.delete(stale)
	assert len(ecs) == 3
	assert np.array_equal(ecs.where(Foo), ids)


def test_mixed_stale_and_high_entity_ids() -> None:
	ecs = ECS()
	ecs.register(Foo)

	stale, ids = recycled_entities(ecs, 3)

	mixed = np.array(
		[stale[0], ids[0], stale[1], ids[1], ids[2]],
		dtype=Entity,
	)

	ecs.add(
		mixed,
		Foo,
		(
			np.array([100.0, 1.0, 200.0, 2.0, 3.0]),
			["stale0", "a", "stale1", "b", "c"],
		),
	)

	foo = ecs.get(Foo)

	assert np.array_equal(ecs.where(Foo), ids)
	assert foo[ids].a.tolist() == pytest.approx([1.0, 2.0, 3.0])
	assert foo[ids].b.tolist() == ["a", "b", "c"]