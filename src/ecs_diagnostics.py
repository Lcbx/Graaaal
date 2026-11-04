"""Explicit inspection and invariant checks; never run automatically in hot paths."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np

from ecs_types import NONE, NO_ENTITY, Mask, _entity_indices

if TYPE_CHECKING:
	from ECS import ECS, ComponentStorage


def storage_info(store: ComponentStorage) -> dict[str, Any]:
	rows = store._active_rows()
	return {
		"component": store.component_cls,
		"fields": {name: {"dtype": str(dtype), "shape": store.schema.shapes[name],
			"hasobject": dtype.hasobject} for name, dtype in store.schema._field_dtypes.items()},
		"owners": int(np.count_nonzero(store._sparse != NONE)),
		"rows": int(rows.size), "span": store._size, "capacity": store._capacity,
		"holes": store._size - int(rows.size),
		"dense_bytes": sum(array.nbytes for array in store._dense.values()),
		"sparse_bytes": store._sparse.nbytes,
	}


def world_info(world: ECS) -> dict[str, Any]:
	components = tuple(storage_info(store) for store in world._stores.values())
	state_bytes = world._entity_masks.nbytes + world._current_entities.nbytes
	return {
		"entities": len(world), "entity_capacity": world._current_entities.size,
		"entity_state_bytes": state_bytes, "components": components,
		"array_bytes": state_bytes + sum(item["dense_bytes"] + item["sparse_bytes"] for item in components),
	}


def _references_clear(array: np.ndarray, rows: np.ndarray) -> bool:
	if not array.dtype.hasobject: return True
	if array.dtype.names:
		return all(_references_clear(array[name], rows) for name in array.dtype.names)
	return all(value is None for value in array[rows].flat)


def check_world(world: ECS) -> None:
	"""Raise AssertionError on inconsistent generations, pools, masks or references."""
	# Import here to avoid a runtime cycle with the ECS facade.
	from ECS import MultiComponentStorage
	current = world._current_entities
	assert current.size == world._entity_masks.size, "world array capacities differ"
	assert 0 <= world._next_entity_index <= current.size, "invalid next entity index"
	alive_indices = np.flatnonzero(current != NO_ENTITY)
	assert alive_indices.size == len(world), "live entity count mismatch"
	assert np.all(alive_indices < world._next_entity_index), "live entity outside allocated index range"
	assert np.array_equal(_entity_indices(current[alive_indices]), alive_indices), "handle index mismatch"
	free = np.asarray(world._free_entities, dtype=np.uint64)
	free_indices = _entity_indices(free)
	assert np.unique(free_indices).size == free_indices.size, "duplicate recycled index"
	assert np.all(free_indices < world._next_entity_index), "recycled index out of range"
	assert np.all(current[free_indices] == NO_ENTITY), "recycled index is live"
	assert free.size + alive_indices.size == world._next_entity_index, "lost allocated entity index"
	expected_masks = np.zeros(current.size, dtype=Mask)
	for cls, store in world._stores.items():
		assert 0 <= store._size <= store._capacity, "invalid dense size"
		assert all(array.shape[0] == store._capacity for array in store._dense.values()), "dense capacity mismatch"
		active = store._active_rows()
		owners = store._dense_entities[active]
		assert np.all(world._alive_mask(owners)), "component has a dead or stale owner"
		unique, counts = np.unique(owners, return_counts=True)
		indices = _entity_indices(unique)
		heads = store._get_dense_indices(indices)
		assert np.all(heads != NONE), "owner missing from sparse map"
		assert np.count_nonzero(store._sparse != NONE) == unique.size, "orphan sparse mapping"
		assert np.array_equal(store._dense_entities[heads], unique), "sparse head owner mismatch"
		if isinstance(store, MultiComponentStorage):
			assert store._live_count == active.size, "multicomponent live count mismatch"
			assert np.array_equal(store._dense_counts[heads], counts), "owner block count mismatch"
			mapped = store._block_rows(heads)
			assert np.array_equal(np.sort(mapped), active), "owner blocks overlap or omit rows"
			assert np.array_equal(store._dense_entities[mapped], np.repeat(unique, counts)), "owner block is not contiguous"
			nonheads = np.ones(store._capacity, dtype=bool)
			nonheads[heads] = False
			assert np.all(store._dense_counts[nonheads] == 0), "count metadata outside block heads"
		else:
			assert np.all(counts == 1), "duplicate single-component owner"
		assert np.all(store._dense_entities[store._size:] == NO_ENTITY), "owner metadata beyond size"
		inactive = np.flatnonzero(store._dense_entities == NO_ENTITY)
		for field in store.schema.object_fields:
			assert _references_clear(store._dense[field], inactive), "inactive object reference retained"
		expected_masks[indices] |= world._comp_bits[cls]
	assert np.array_equal(world._entity_masks, expected_masks), "component membership mask mismatch"
