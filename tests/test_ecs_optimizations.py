import numpy as np
import pytest

from ECS import ECS, NO_ENTITY, component, entity_index


@component
class Position:
	value: float
	data: object


@component
class Marked:
	pass


@component(multiples=True)
class Item:
	value: float
	data: object


def test_sparse_queries_keep_index_order_after_swap_removal_and_id_reuse():
	w = ECS()
	pos, _ = w.register(Position, Marked)
	ids = w.create(2000)
	owners = ids[[1800, 20, 400, 50]]
	w.add(owners, Position, (np.array([18., 2., 4., 5.]), ["a", "b", "c", "d"]))
	w.delete(ids[20])
	reused = w.create()[0]
	w.add(reused, Position(99, "new"))
	w.add(ids[[50, 1800]], Marked)
	expected = [int(reused), 50, 400, 1800]
	assert w.where(Position).tolist() == expected
	assert w.query(Position).entities().tolist() == expected
	assert w.query(Position).fetch()[Position].selection.value.tolist() == [99, 5, 4, 18]
	assert w.where(Position, exclude=Marked).tolist() == [int(reused), 400]
	assert w.query(Position, any_of=Marked).entities().tolist() == [50, 1800]
	assert w.query(Position, exclude=Marked).entities().tolist() == [int(reused), 400]
	assert pos[reused].data.tolist() == ["new"]


def test_sparse_multicomponent_candidates_ignore_holes_and_deduplicate_owners():
	w = ECS()
	w.register(Item, Marked)
	ids = w.create(2000)
	w.add(ids[[1800, 50, 400, 1800, 50]], Item, (np.arange(5.), [None] * 5))
	w.remove(ids[400], Item)  # Interior hole before the last owner block.
	w.add(ids[1800], Marked)
	assert w.where(Item).tolist() == [50, 1800]
	assert w.query(Item, any_of=Marked).entities().tolist() == [1800]
	result = w.query(Item, exclude=Marked).fetch()
	assert result.entities.tolist() == [50]
	assert result[Item].offsets.tolist() == [0, 2]
	assert result[Item].selection.value.tolist() == [1, 4]
	w.remove(ids[[50, 1800]], Item)
	assert not w.where(Item).size
	assert len(w.query(Item).fetch()) == 0


@pytest.mark.parametrize("quantity", [1, 4, 8, 16, 64])
def test_small_and_batch_removals_preserve_fields_sparse_mappings_and_object_cleanup(quantity):
	rng = np.random.default_rng(45)
	w = ECS()
	pos, tags = w.register(Position, Marked)
	ids = w.create(5000)
	order = rng.permutation(ids)
	objects = [{"owner": int(owner)} for owner in order]
	w.add(order, Position, (order.astype(float), objects))
	w.add(ids, Marked)
	rows = np.linspace(0, len(ids) - 1, quantity, dtype=np.uint32)[::-1]
	removed = pos.owner_ids(rows)
	w.remove(removed, Position, Marked)
	surviving = ids[~np.isin(ids, removed)]
	assert w.where(Position).tolist() == surviving.tolist()
	assert w.where(Marked).tolist() == surviving.tolist()
	assert pos[surviving].value.tolist() == surviving.astype(float).tolist()
	assert [obj["owner"] for obj in pos[surviving].data] == surviving.tolist()
	assert not len(pos[removed])
	assert not len(tags[removed])
	store = pos._store
	assert np.all(store._dense_entities[store._size:] == NO_ENTITY)
	assert all(obj is None for obj in store._dense["data"][store._size:])
	for i, owner in enumerate(store._dense_entities[:store._size]):
		assert store._sparse[entity_index(owner)] == i


@pytest.mark.parametrize("mode", ["spare", "grow", "compact"])
def test_tail_owner_append_keeps_old_objects_and_reacquires_head_after_compaction(mode):
	w = ECS()
	items, = w.register(Item, capacity=8 if mode != "grow" else 2)
	ids = w.create(2)
	old_objects = [{"old": i} for i in range(3)]
	if mode == "compact":
		w.add(np.repeat(ids[:1], 3), Item, (np.array([0., 1., 2.]), [None] * 3))
	w.add(np.repeat(ids[1:], 3), Item, (np.array([3., 4., 5.]), old_objects))
	if mode == "compact":
		w.remove(ids[:1], Item)
	saved = items[ids[1]]
	new_objects = [{"new": i} for i in range(3)]
	w.add(np.repeat(ids[1:], 3), Item, (np.array([6., 7., 8.]), new_objects))
	assert items[ids[1]].value.tolist() == [3, 4, 5, 6, 7, 8]
	assert all(actual is expected for actual, expected in zip(items[ids[1]].data, old_objects + new_objects))
	assert len(items.rows()) == items._store._live_count == 6
	with pytest.raises(RuntimeError, match="stale"):
		saved.value
	w.delete(ids[1])
	assert not len(items.rows())
	assert all(obj is None for obj in items._store._dense["data"])
