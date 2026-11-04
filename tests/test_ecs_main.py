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


def world_snapshot(ecs: ECS) -> tuple:
	"""Capture storage state, including counters, to check rejected operations."""
	return (
		ecs.entity_masks.copy(),
		{cls: (
			store._size, store._capacity, getattr(store, "_live_count", None),
			store.membership_version, store.layout_version, store.field_versions.copy(),
			store._sparse.copy(), {name: a.copy() for name, a in store._dense.items()},
		) for cls, store in ecs._stores.items()},
	)


def assert_world_unchanged(ecs: ECS, snapshot: tuple) -> None:
	masks, stores = snapshot
	np.testing.assert_array_equal(ecs.entity_masks, masks)
	assert ecs._stores.keys() == stores.keys()
	for cls, (size, capacity, live, membership, layout, versions, sparse, dense) in stores.items():
		store = ecs._get_store(cls)
		assert (store._size, store._capacity, getattr(store, "_live_count", None)) == (size, capacity, live)
		assert store.membership_version == membership
		assert store.layout_version == layout
		assert store.field_versions == versions
		np.testing.assert_array_equal(store._sparse, sparse)
		for field, array in dense.items():
			np.testing.assert_array_equal(store._dense[field], array)


@pytest.mark.parametrize("case", ["existing", "duplicate_owners", "repeated_type", "bad_shape", "bad_dtype"])
def test_add_preflight_leaves_entire_world_unchanged(case: str) -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D)
	ids = ecs.create(2)
	if case == "existing":
		ecs.add(ids, Velocity2D(3, 4))
	owners = np.repeat(ids[:1], 2) if case == "duplicate_owners" else ids
	params: tuple = (Position2D(1, 2), Velocity2D(5, 6))
	if case == "repeated_type":
		params = (Position2D(1, 2), Position2D, (3, 4))
	elif case == "bad_shape":
		params = (Position2D(1, 2), Velocity2D, np.ones((3, 3)))
	elif case == "bad_dtype":
		params = (Position2D(1, 2), Velocity2D, ("invalid", 2))
	snapshot = world_snapshot(ecs)
	with pytest.raises(ValueError):
		ecs.add(owners, *params)
	assert_world_unchanged(ecs, snapshot)


def test_single_component_rejection_does_not_leave_orphan_rows() -> None:
	ecs = ECS()
	positions, = ecs.register(Position2D)
	ids = ecs.create(2)
	ecs.add(ids, Position2D(1, 2))
	with pytest.raises(ValueError, match="already exists"):
		ecs.add(ids[:1], Position2D(3, 4))
	ecs.remove(ids[:1], Position2D)
	assert positions.owner_ids(positions.get_rows()).tolist() == ids[1:].tolist()
	ecs.delete(ids[1:])
	assert positions.get_rows().size == 0
	assert ecs.where(Position2D).size == 0


def test_direct_storage_rejects_duplicate_single_components() -> None:
	store = ComponentStorage(Position2D)
	store._add_range(as_entities([4]), Position2D(1, 2))
	with pytest.raises(ValueError):
		store._add_range(as_entities([4]), Position2D(3, 4))
	with pytest.raises(ValueError):
		store._add_range(as_entities([5, 5]), Position2D(3, 4))
	assert store._size == 1


def test_set_replaces_values_and_only_advances_field_versions() -> None:
	ecs = ECS()
	foo, bar = ecs.register(Foo, Bar)
	ids = ecs.create(3)
	ecs.add(ids, Foo(1, "old"), Bar(2, 3))
	rows = foo[ids].get_rows().copy()
	foo_before, bar_before = foo.version(), bar.version()
	ecs.set(ids[[2, 0]], Foo, (np.array([20., 10.]), ["twenty", "ten"]), Bar(4, 5))
	assert foo[ids].a.tolist() == [10, 1, 20]
	assert foo[ids].b.tolist() == ["ten", "old", "twenty"]
	assert bar[ids].a.tolist() == [4, 2, 4]
	np.testing.assert_array_equal(foo[ids].get_rows(), rows)
	assert foo.version() == (foo_before[0], foo_before[1] + 1, foo_before[2] + 1)
	assert bar.version() == (bar_before[0], bar_before[1] + 1, bar_before[2] + 1)


@pytest.mark.parametrize("case", ["missing", "duplicate_owners", "repeated_type", "bad_shape", "multi"])
def test_set_preflight_leaves_entire_world_unchanged(case: str) -> None:
	ecs = ECS()
	ecs.register(Position2D, Velocity2D, MultiComp)
	ids = ecs.create(2)
	ecs.add(ids, Position2D(1, 2))
	if case != "missing":
		ecs.add(ids, Velocity2D(3, 4))
	owners = np.repeat(ids[:1], 2) if case == "duplicate_owners" else ids
	params: tuple = (Position2D(5, 6), Velocity2D(7, 8))
	if case == "repeated_type":
		params = (Position2D(5, 6), Position2D(7, 8))
	elif case == "bad_shape":
		params = (Position2D(5, 6), Velocity2D, np.ones((3, 3)))
	elif case == "multi":
		params = (Position2D(5, 6), MultiComp(7))
	snapshot = world_snapshot(ecs)
	with pytest.raises((ValueError, TypeError)):
		ecs.set(owners, *params)
	assert_world_unchanged(ecs, snapshot)


def test_set_filters_stale_values_and_noops_for_empty_inputs() -> None:
	ecs = ECS()
	foo, = ecs.register(Foo)
	stale, ids = recycled_entities(ecs, 2)
	ecs.add(ids, Foo(1, "old"))
	owners = np.array([stale[0], ids[1], stale[1], ids[0]], dtype=Entity)
	ecs.set(owners, Foo, (np.array([90., 20., 90., 10.]), ["stale", "b", "stale", "a"]))
	assert foo[ids].a.tolist() == [10, 20]
	assert foo[ids].b.tolist() == ["a", "b"]
	snapshot = world_snapshot(ecs)
	ecs.set(stale, Foo(9, "ignored"))
	ecs.set([], Foo(9, "ignored"))
	ecs.set(ids)
	assert_world_unchanged(ecs, snapshot)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("repeated_payload", [False, True])
def test_multicomponent_count_overflow_is_rejected_before_any_mutation(existing: bool, repeated_payload: bool) -> None:
	ecs = ECS()
	ecs.register(Position2D, MultiComp)
	ids = ecs.create(2)
	limit = int(np.iinfo(ComponentPerEntityCount).max)
	if existing:
		ecs.add(np.repeat(ids[:1], limit), MultiComp, np.arange(limit, dtype=float))
		ecs.add(ids[1:], MultiComp(7))
	owners = ids if existing else np.repeat(ids[:1], limit + 1)
	params = (MultiComp(2), MultiComp(3)) if repeated_payload else (MultiComp(2),)
	if existing:
		params = (Position2D(1, 2), *params)
	snapshot = world_snapshot(ecs)
	with pytest.raises(OverflowError):
		ecs.add(owners, *params)
	assert_world_unchanged(ecs, snapshot)
	assert len(ecs.get(MultiComp)[ids[:1]]) == (limit if existing else 0)


def test_multicomponent_repeated_payloads_are_combined_and_limit_is_inclusive() -> None:
	ecs = ECS()
	comps, = ecs.register(MultiComp)
	id_ = ecs.create()
	limit = int(np.iinfo(ComponentPerEntityCount).max)
	ecs.add(np.repeat(id_, limit - 2), MultiComp(1))
	ecs.add(id_, MultiComp(2), MultiComp(3))
	assert len(comps[id_]) == limit
	assert comps[id_].val[-2:].tolist() == [2, 3]
	ecs.delete(id_)
	assert comps.get_rows().size == 0


def test_remove_validates_all_types_before_removing_any() -> None:
	ecs = ECS()
	ecs.register(Position2D)
	ids = ecs.create(2)
	ecs.add(ids, Position2D(1, 2))
	snapshot = world_snapshot(ecs)
	with pytest.raises(KeyError):
		ecs.remove(ids, Position2D, Velocity2D)
	assert_world_unchanged(ecs, snapshot)
	ecs.remove(ids, Position2D, Position2D)
	assert ecs.where(Position2D).size == 0


@pytest.mark.parametrize("name", ["_entity", "_count"])
def test_registration_rejects_metadata_fields_without_partial_registration(name: str) -> None:
	reserved = component(type("Reserved", (), {"__annotations__": {name: int}}))
	ecs = ECS()
	with pytest.raises(ValueError, match="reserved"):
		ecs.register(Position2D, reserved)
	assert not ecs._stores
	assert not ecs._comp_bits


def test_registration_preflights_duplicates_and_component_limit() -> None:
	ecs = ECS()
	with pytest.raises(ValueError, match="already registered"):
		ecs.register(Position2D, Position2D)
	assert not ecs._stores
	classes = [component(type(f"C{i}", (), {"__annotations__": {"value": int}})) for i in range(65)]
	with pytest.raises(ValueError, match="64"):
		ecs.register(*classes)
	assert not ecs._stores
	ecs.register(*classes[:64])
	assert len(ecs._stores) == 64
	with pytest.raises(ValueError, match="64"):
		ecs.register(classes[-1])
	assert len(ecs._stores) == 64


def test_metadata_sentinels_after_initialization_growth_and_compaction() -> None:
	store = MultiComponentStorage(MultiComp, capacity=4)
	assert np.all(store._dense_entities == NO_ENTITY)
	assert np.all(store._dense_counts == 0)
	store._add_range(as_entities([1, 1, 2]), np.array([1., 2., 3.]))
	store._remove_entities(as_rows([1]))
	store._defragment(3)
	assert store._dense_entities[:store._size].tolist() == [2]
	assert np.all(store._dense_entities[store._size:] == NO_ENTITY)
	assert np.all(store._dense_counts[store._size:] == 0)
	store._defragment(16)
	assert np.all(store._dense_entities[store._size:] == NO_ENTITY)
	assert np.all(store._dense_counts[store._size:] == 0)


def test_set_shaped_fields_and_preflight_of_malformed_object_arrays() -> None:
	shaped = component(type("Shaped", (), {"__annotations__": {"position": np.dtype((np.float32, (3,)))}}))
	ecs = ECS()
	positions, foo = ecs.register(shaped, Foo)
	ids = ecs.create(2)
	ecs.add(ids, shaped, (np.ones((2, 3)),), Foo(1, "old"))
	ecs.set(ids, shaped, (np.arange(6).reshape(2, 3),))
	np.testing.assert_array_equal(positions[ids].position, [[0, 1, 2], [3, 4, 5]])
	snapshot = world_snapshot(ecs)
	with pytest.raises(ValueError, match="object array"):
		ecs.set(ids, shaped, (np.zeros((2, 3)),), Foo, (np.ones(2), np.array([["a", "b"]], dtype=object)))
	assert_world_unchanged(ecs, snapshot)
	assert foo[ids].b.tolist() == ["old", "old"]


@pytest.mark.parametrize("seed", [11, 97])
def test_mutations_match_reference_model(seed: int) -> None:
	rng = np.random.default_rng(seed)
	ecs = ECS()
	foo, multi = ecs.register(Foo, MultiComp, capacity=2)
	live: set[int] = set()
	foos: dict[int, tuple[float, str]] = {}
	multis: dict[int, list[float]] = {}
	stale: list[int] = []
	for step in range(250):
		operation = int(rng.integers(6))
		if not live or operation == 0:
			live.update(map(int, ecs.create(int(rng.integers(1, 4)))))
		else:
			owner = sorted(live)[int(rng.integers(len(live)))]
			if operation == 1:
				ecs.delete(owner)
				live.remove(owner)
				foos.pop(owner, None)
				multis.pop(owner, None)
				stale.append(owner)
			elif operation == 2:
				value = (float(step), str(step))
				if owner in foos:
					ecs.set(owner, Foo(*value))
				else:
					ecs.add(owner, Foo(*value))
				foos[owner] = value
			elif operation == 3:
				ecs.remove(owner, Foo)
				foos.pop(owner, None)
			elif operation == 4:
				values = [float(step)] * int(rng.integers(1, 4))
				ecs.add([owner] * len(values), MultiComp, np.array(values))
				multis.setdefault(owner, []).extend(values)
			else:
				rows = multi.query_rows(lambda val: val % 2 == 0, owner)
				multi.remove(rows)
				kept = [value for value in multis.get(owner, []) if value % 2 != 0]
				if kept:
					multis[owner] = kept
				else:
					multis.pop(owner, None)

		assert len(ecs) == len(live)
		assert set(map(int, ecs.where(Foo))) == set(foos)
		assert set(map(int, ecs.where(MultiComp))) == set(multis)
		assert set(map(int, ecs.where(Foo, MultiComp))) == set(foos) & set(multis)
		for owner, (value, text) in foos.items():
			assert foo[owner].a.tolist() == [value]
			assert foo[owner].b.tolist() == [text]
		for owner, values in multis.items():
			assert multi[owner].val.tolist() == values
		assert foo.get_rows().size == len(foos)
		assert multi.get_rows().size == sum(map(len, multis.values()))
		for accessor in (foo, multi):
			store = accessor._store
			active = accessor.get_rows()
			owners = accessor.owner_ids(active)
			assert set(map(int, owners)).issubset(live)
			for owner in np.unique(owners):
				mapped = store._get_rows(as_rows([int(entity_index(owner))]))
				np.testing.assert_array_equal(store._dense_entities[mapped], owner)
			assert all(value is None for value in store._dense.get("b", [])[store._size:])
		if stale:
			assert len(foo[stale]) == 0
			assert len(multi[stale]) == 0


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


@pytest.mark.parametrize("operation", [
	lambda s: s.x,
	lambda s: s["x"],
	lambda s: setattr(s, "x", 999),
	lambda s: s.__setitem__("x", 999),
	lambda s: s.vector(),
	lambda s: s.set_vector(np.array([[999., 999.]])),
	lambda s: s.get_rows(),
	lambda s: s.checked_rows(),
	lambda s: len(s),
], ids=["attribute-read", "field-read", "attribute-write", "field-write", "vector-read", "vector-write", "rows", "checked-rows", "length"])
def test_expired_selection_rejects_every_access_before_touching_moved_entity(operation) -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0), (30, 0)])
	saved = pos[ids[:1]]
	ecs.delete(ids[:1])
	current = ecs.create()
	ecs.add(current, Position2D(40, 0))
	snapshot = world_snapshot(ecs)
	with pytest.raises(RuntimeError, match="stale component selection"):
		operation(saved)
	assert_world_unchanged(ecs, snapshot)
	assert pos[ids[2]].x.tolist() == [30]
	assert pos[current].x.tolist() == [40]


@pytest.mark.parametrize("edit", ["add", "remove", "checked-remove", "grow"])
def test_structural_edits_expire_rows_for_all_owners_in_the_store(edit: str) -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0), (30, 0)])
	saved = pos[ids[2]]
	checked = pos.query_rows(lambda x: x >= 20)
	before = pos.layout_version
	if edit == "add":
		ecs.add(ecs.create(), Position2D(40, 0))
	elif edit == "remove":
		ecs.remove(ids[:1], Position2D)
	elif edit == "checked-remove":
		pos.remove(pos[ids[:1]])
	else:
		pos._store._grow_dense(pos._store._capacity + 1)
	assert pos.layout_version > before
	snapshot = world_snapshot(ecs)
	for rows in (saved, checked):
		with pytest.raises(RuntimeError, match="stale"):
			pos.owner_ids(rows)
		with pytest.raises(RuntimeError, match="stale"):
			pos.remove(rows)
	assert_world_unchanged(ecs, snapshot)


def test_growth_during_add_expires_existing_selections() -> None:
	ecs = ECS()
	pos, = ecs.register(Position2D, capacity=1)
	ids = ecs.create(2)
	ecs.add(ids[:1], Position2D(10, 20))
	saved = pos[ids[:1]]
	old_capacity = pos._store._capacity
	ecs.add(ids[1:], Position2D(30, 40))
	assert pos._store._capacity > old_capacity
	with pytest.raises(RuntimeError, match="stale"):
		saved.x = 100
	assert pos[ids[:1]].x.tolist() == [10]


@pytest.mark.parametrize("edit", ["rebuild", "partial-remove", "compact", "compact-grow"])
def test_multicomponent_layout_changes_expire_selections(edit: str) -> None:
	ecs = ECS()
	comps, = ecs.register(MultiComp, capacity=8)
	ids = ecs.create(2)
	ecs.add(np.repeat(ids, 2), MultiComp, np.array([1., 2., 3., 4.]))
	if edit.startswith("compact"):
		ecs.remove(ids[:1], MultiComp)  # Leave a hole before the surviving owner.
	saved = comps[ids[1]]
	rows = saved.checked_rows()
	before = comps.layout_version
	if edit == "rebuild":
		ecs.add(ids[:1], MultiComp(5))
	elif edit == "partial-remove":
		comps.remove(comps.query_rows(lambda val: val == 3))
	else:
		store = comps._store
		store._defragment(store._capacity + (1 if edit == "compact-grow" else 0))
	assert comps.layout_version > before
	with pytest.raises(RuntimeError, match="stale"):
		saved.val
	with pytest.raises(RuntimeError, match="stale"):
		rows[:1]
	assert comps[ids[1]].val.tolist() == ([4] if edit == "partial-remove" else [3, 4])


def test_checked_row_slicing_preserves_store_and_lifetime() -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0), (30, 0)])
	checked = pos.query_rows(lambda x: x >= 20)
	assert isinstance(checked, ComponentRows)
	subset = checked[[1, 0, 1]]
	assert pos.owner_ids(subset).tolist() == ids[[2, 1, 2]].tolist()
	assert pos.owner_ids(checked[0]).tolist() == ids[1:2].tolist()
	assert pos.owner_ids(checked[np.array([False, True])]).tolist() == ids[2:].tolist()
	assert pos.owner_ids(pos[ids[[2, 0]]]).tolist() == ids[[2, 0]].tolist()
	pos.remove(subset)
	assert ecs.where(Position2D).tolist() == ids[:1].tolist()
	with pytest.raises(RuntimeError, match="stale"):
		checked.get_rows()


@pytest.mark.parametrize("other_world", [False, True])
def test_foreign_checked_rows_are_rejected_without_mutation(other_world: bool) -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0)])
	if other_world:
		other, _, foreign = make_pos2d_ecs([(30, 0), (40, 0)])
	else:
		other = ecs
		foreign, = ecs.register(Velocity2D)
		ecs.add(ids, Velocity2D(30, 40))
	snapshot = world_snapshot(ecs)
	other_snapshot = world_snapshot(other)
	for rows in (foreign.rows(), foreign[ids], foreign.query_rows(lambda x: x > 0)):
		with pytest.raises(ValueError, match="different component storage"):
			pos.owner_ids(rows)
		with pytest.raises(ValueError, match="different component storage"):
			pos.remove(rows)
	assert_world_unchanged(ecs, snapshot)
	assert_world_unchanged(other, other_snapshot)


def test_row_exports_and_constructor_inputs_cannot_change_selection_mapping() -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0)])
	input_rows = pos.get_rows()
	selection = ComponentSelection(pos._store, input_rows)
	input_rows[:] = 1
	for rows in (selection.get_rows(), selection.checked_rows().get_rows(), pos.rows().get_rows()):
		rows[:] = 1
	assert selection.x.tolist() == [10, 20]
	assert selection.get_rows().tolist() == [0, 1]


def test_field_writes_and_unrelated_edits_keep_selections_valid() -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0)])
	saved = pos[ids]
	checked = pos.query_rows(lambda x: x > 15)
	before = pos.layout_version
	ecs.set(ids, Position2D(0, 1))
	pos.touch("x")
	saved.x += 2
	saved.x[:] = 99  # A numeric read is a copy, including a sliced assignment.
	assert saved.x.tolist() == [2, 2]
	assert pos.owner_ids(checked).tolist() == ids[1:].tolist()  # Predicate snapshot.
	new = ecs.create()
	ecs.register(Velocity2D)
	ecs.add(ids, Velocity2D(5, 6))
	ecs.remove(new, Position2D)
	ecs.add([], Position2D(0, 0))
	pos.remove([])
	with pytest.raises(ValueError):
		ecs.add(ids, Position2D(0, 0))
	ecs.delete(new)
	assert pos.layout_version == before
	assert len(saved) == 2


def test_object_reads_alias_values_and_require_touch_without_expiring_selection() -> None:
	@component
	class Payload:
		data: object
	ecs = ECS()
	payloads, = ecs.register(Payload)
	id_ = ecs.create()
	value = {"count": 1}
	ecs.add(id_, Payload(value))
	saved = payloads[id_]
	version, layout = payloads.version(), payloads.layout_version
	saved.data[0]["count"] = 2
	assert value["count"] == 2
	assert payloads.version() == version
	payloads.touch("data")
	assert payloads.version() == (version[0], version[1] + 1)
	assert payloads.layout_version == layout
	assert saved.data[0] is value


def test_empty_checked_rows_expire_and_noop_compaction_keeps_layout() -> None:
	ecs = ECS()
	comps, = ecs.register(MultiComp)
	empty = comps.rows()
	ecs.add(ecs.create(), MultiComp(3))
	with pytest.raises(RuntimeError, match="stale"):
		comps.remove(empty)
	saved = comps.rows()
	before = comps.layout_version
	comps._store._defragment(comps._store._capacity)
	assert comps.layout_version == before
	assert len(saved) == 1
	assert len(comps.query_rows(lambda val: val < 0)) == 0


def test_checked_query_rejects_layout_changes_inside_predicate() -> None:
	ecs, ids, pos = make_pos2d_ecs([(10, 0), (20, 0)])
	def invalid_predicate(x):
		ecs.remove(ids[:1], Position2D)
		return x > 0
	with pytest.raises(RuntimeError, match="layout changed"):
		pos.query_rows(invalid_predicate)
