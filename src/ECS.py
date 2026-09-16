from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import IntFlag
from typing import cast, Any, Callable, ClassVar, Iterable, Iterator, Mapping, Sequence, TypeAlias, TypeVar, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
def component(cls: T | None = None, *, multiples: bool = False) -> T | Callable[[T], T]:
	def decorator(target: T) -> T:
		setattr(target, "_multicomponent", multiples)
		return dataclass(target)
	if cls is None:
		return decorator
	return decorator(cls)


Entity                  = np.uint64
Index                   = np.uint32
Generation              = np.uint32
Mask                    = np.uint64
ComponentPerEntityCount = np.uint16

EntityArray     : TypeAlias = NDArray[Entity]
IndexArray      : TypeAlias = NDArray[Index]
GenerationArray : TypeAlias = NDArray[Generation]
MaskArray       : TypeAlias = NDArray[Mask]
CountArray      : TypeAlias = NDArray[ComponentPerEntityCount]
BoolArray       : TypeAlias = NDArray[np.bool_]
FieldArray      : TypeAlias = NDArray[Any]

EntityLike: TypeAlias = Entity | int | EntityArray | Sequence[int]
IndexLike:  TypeAlias = Index  | int | IndexArray  | Sequence[int]


INDEX_MAX            = int(np.iinfo(Index).max)
INDEX_BITS           = np.dtype(Index).itemsize * 8
INDEX_MASK           = Entity(INDEX_MAX)
GENERATION_INCREMENT = Entity(1 << INDEX_BITS)
NONE                 = Index(INDEX_MAX)
NO_ENTITY            = Entity(np.iinfo(Entity).max)
INITIAL_CAPACITY     = 128
OBJECT_DTYPE         = np.dtype(object)
_ENTITY_STR          = "_entity"
_COUNT_STR           = "_count"


# useful for buffer resizing
# 0->1, 1->2, 2->4, 3->4, 4->8, 5->8
def higher_pow2(n: int|np.uint32|np.uint64) -> int:
	return 1 << int(n).bit_length()

def entity_index(entity: Entity | EntityArray) -> Index | IndexArray:
	result = np.asarray(entity, dtype=Entity)
	if result.ndim == 0: return Index(result & INDEX_MASK)
	return _entity_indices(result)

def entity_generation(entity: Entity | EntityArray) -> Generation | GenerationArray:
	result = np.asarray(entity, dtype=Entity) >> Entity(INDEX_BITS)
	if result.ndim == 0: return Generation(result)
	return result.astype(Generation, copy=False)

def _entity_indices(entities: EntityArray) -> IndexArray:
	return (entities & INDEX_MASK).astype(Index, copy=False)

def as_entities(entities: EntityLike) -> EntityArray:
	return cast(EntityArray, _as_1D_dtype(entities, Entity))

def as_rows(rows: IndexLike) -> IndexArray:
	return cast(IndexArray, _as_1D_dtype(rows, Index))

def _as_1D_dtype(array:Any, ndtype:type) -> FieldArray:
	array = np.asarray(array, dtype=ndtype)
	if array.ndim == 0: return array.reshape(1)
	if array.ndim != 1: raise ValueError("parameter must be a scalar or a 1-D array")
	return array

def arg_names(func:Callable)->Sequence[str]:
    code = func.__code__
    n = code.co_argcount + code.co_kwonlyargcount
    return code.co_varnames[:n]

def _component_fields(component_cls: type[Any]) -> tuple[tuple[str, Any], ...]:
	hints = get_type_hints(component_cls)
	if is_dataclass(component_cls):
		return tuple(
			(field.name, hints.get(field.name, field.type))
			for field in dataclass_fields(component_cls)
			if get_origin(hints.get(field.name, field.type)) is not ClassVar
		)
	return tuple(
		(name, annotation)
		for name, annotation in hints.items()
		if get_origin(annotation) is not ClassVar
	)


def _dtype_for(annotation: Any) -> np.dtype[Any]:
	if isinstance(annotation, type):
		if issubclass(annotation, IntFlag):
			return np.dtype(np.int64)
		if hasattr(annotation, "storage_dtype"):
			return annotation.storage_dtype()
		if issubclass(annotation, (str, bytes)):
			return OBJECT_DTYPE
	try:
		return np.dtype(annotation)
	except TypeError:
		...
	return OBJECT_DTYPE

class ComponentSelection:
	__slots__ = ("_store", "_rows")

	def __init__(self, store: ComponentStorage, rows: IndexArray) -> None:
		self._store = store
		self._rows = rows

	def get_rows(self) -> IndexArray:
		return self._rows

	def vector(self, *field_names: str) -> FieldArray:
		field_names = field_names or self._store.fields
		return np.column_stack( tuple(self[field] for field in field_names) )

	def set_vector(self, value : FieldArray, *field_names: str) -> None:
		field_names = field_names or self._store.fields
		for i, field in enumerate(field_names):
			self[field] = value[:, i]

	def __len__(self) -> int:
		return int(self._rows.size)

	def __getitem__(self, field: str) -> FieldArray:
		return self._store._dense[field][self._rows]

	def __setitem__(self, field: str, value: Any) -> None:
		array = self._store._dense[field]
		if self._rows.size: self._store.touch(field)
		array[self._rows] = value

	def __getattr__(self, name: str) -> FieldArray:
		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None
		return array[self._rows]

	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith("_"):
			object.__setattr__(self, name, value)
			return

		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None

		if self._rows.size: self._store.touch(name)
		array[self._rows] = value


class ComponentAccessor:
	"""Selections copy numeric fields; assign edited arrays back to write them.
	
	You can monitor changes to array using membership / field version tokens.
	Version tokens count writes, so assigning the same value still invalidates a token.
	Token are affected by changes to number of owners.
	Numeric array reads do not advance counters.
	Modified objects fieds are not monitored, notify their in-place edits with touch().
	Direct dense array writes likewise need touch().
	Compare tokens only for the same accessor and ordered field names. 
	Tokens do not provide thread synchronization.
	"""
	__slots__ = ("_ecs", "_component_cls", "_store")

	def __init__(
		self,
		ecs: ECS,
		component_cls: type[Any],
		store: ComponentStorage,
	) -> None:
		self._ecs = ecs
		self._component_cls = component_cls
		self._store = store

	@property
	def membership_version(self) -> int:
		"""Conservative counter for nonempty component additions/removals."""
		return self._store.membership_version

	def version(self, *field_names: str) -> tuple[int, ...]:
		"""Membership plus per-field writes, in argument order; default: all fields."""
		fields = field_names or self._store.fields
		return (self.membership_version, *(self._store.field_versions[name] for name in fields))

	def touch(self, *field_names: str) -> None:
		"""Invalidate named fields (default: all) after otherwise untracked writes.
		Does not write arrays back, change membership, or compare field values.
		"""
		self._store.touch(*field_names)

	def get_rows(self) -> IndexArray:
		return self._store._active_rows()

	def owner_ids(self, component_rows: IndexArray) -> EntityArray:
		return self._store._dense_entities[component_rows].astype(Entity, copy=False)

	def __getitem__(self, entities: EntityLike) -> ComponentSelection:
		entities_ = self._ecs._filter_alive(as_entities(entities))
		indices = _entity_indices(entities_)
		return ComponentSelection(self._store, self._store._get_rows(indices))

	def query(
		self,
		condition: Callable[..., Any],
		entities: EntityLike | None = None,
	) -> IndexArray:
		if entities is None:
			rows = self._store._active_rows()
		else:
			entities_ = self._ecs._filter_alive(np.unique(as_entities(entities)))
			rows = self._store._get_rows(_entity_indices(entities_))
		return self._store._query_rows(condition, rows)

	def remove(self, rows: IndexLike) -> None:
		rows_ = np.unique(as_rows(rows))
		rows_ = rows_[rows_ < self._store._size]
		rows_ = rows_[self._store._dense_entities[rows_] != NO_ENTITY]
		if rows_.size == 0:
			return

		emptied = self._store._remove_rows(rows_)
		if emptied.size:
			self._ecs._clear_component_mask(_entity_indices(emptied), self._component_cls)


class ComponentStorage:

	def __init__(
		self,
		component_cls: type[Any],
		capacity: int = INITIAL_CAPACITY,
	) -> None:
		self.component_cls = component_cls

		field_defs = _component_fields(component_cls)
		if not field_defs:
			raise TypeError(f"{component_cls.__name__} has no annotated component fields")

		self.fields = tuple(name for name, _ in field_defs)
		self.membership_version = 0
		self.field_versions = {name: 0 for name in self.fields}
		self._field_dtypes = {
			name: _dtype_for(annotation)
			for name, annotation in field_defs
		}

		self._capacity = max(1, int(capacity))
		self._size = 0
		self._sparse = np.full(self._capacity, NONE, dtype=Index)

		self._dense: dict[str, FieldArray] = {}
		self._dense[_ENTITY_STR] = np.full(self._capacity, NO_ENTITY, dtype=Entity)
		self._field_dtypes[_ENTITY_STR] = _dtype_for(Entity)

		for name, dtype in self._field_dtypes.items():
			array = np.empty(self._capacity, dtype=dtype)
			if dtype == OBJECT_DTYPE:
				array.fill(None)
			self._dense[name] = array

	def touch(self, *field_names: str) -> None:
		fields = field_names or self.fields
		for name in fields:
			self.field_versions[name] += 1

	@property
	def _dense_entities(self) -> EntityArray:
		return self._dense[_ENTITY_STR]
	

	# ---------- sparse ----------

	def _grow_sparse(self, minimum_size: int) -> None:
		if minimum_size <= self._sparse.size:
			return
		new_size = max(higher_pow2(minimum_size), self._sparse.size * 2)
		sparse = np.full(new_size, NONE, dtype=Index)
		sparse[: self._sparse.size] = self._sparse
		self._sparse = sparse

	def _ensure_sparse(self, indices: IndexArray) -> None:
		if indices.size:
			self._grow_sparse(int(indices.max()) + 1)

	def _get_dense_indices(self, indices: IndexArray) -> IndexArray:
		rows = np.full(indices.size, NONE, dtype=Index)
		in_range = indices < self._sparse.size
		rows[in_range] = self._sparse[indices[in_range]]
		return rows

	def _set_dense_indices(self, indices: IndexArray, rows: IndexArray) -> None:
		self._ensure_sparse(indices)
		self._sparse[indices] = rows

	def _clear_dense_indices(self, indices: IndexArray) -> None:
		self._sparse[indices] = NONE

	# ---------- dense ----------

	def _grow_dense(self, minimum_capacity: int) -> None:
		if minimum_capacity <= self._capacity: return

		new_capacity = higher_pow2(minimum_capacity)
		for name, old in tuple(self._dense.items()):
			new = np.empty(new_capacity , dtype=self._field_dtypes[name])
			new[: self._capacity] = old
			if name == _ENTITY_STR:
				new[self._capacity :] = NO_ENTITY
			if name == _COUNT_STR:
				new[self._capacity :] = 0
			elif old.dtype == OBJECT_DTYPE:
				new[self._capacity :] = None
			self._dense[name] = new
		self._capacity = new_capacity

	def _clear_rows(self, rows: IndexArray) -> None:
		self._dense_entities[rows] = NO_ENTITY
		for field in self.fields:
			array = self._dense[field]
			if array.dtype == OBJECT_DTYPE:
				array[rows] = None

	def _active_rows(self) -> IndexArray:
		return np.arange(self._size, dtype=Index)

	def _get_rows(self, indices: IndexArray) -> IndexArray:
		rows = self._get_dense_indices(indices)
		return rows[rows != NONE]

	# ---------- values ----------

	def _broadcast_field(
		self,
		field: str,
		value: Any,
		quantity: int,
		*,
		single: bool = False,
	) -> FieldArray:
		dtype = self._field_dtypes[field]

		if dtype == OBJECT_DTYPE:

			if not single and isinstance(value, (np.ndarray, list, tuple)):
				n = len(value) if isinstance(value, (list, tuple)) else value.size
				if n == quantity:
					return np.asarray(value, dtype=object)
				if n == 1:
					value = value[0]
				else:
					raise ValueError(f"field {field!r} has {n} values for {quantity} entities")
			obj_array = np.empty(quantity, dtype=object)
			obj_array.fill(value)
			return obj_array

		base_dtype, subshape = dtype.subdtype or (dtype, ())

		target_shape = (quantity,) + subshape
		array = np.asarray(value, dtype=base_dtype)
		arr_shape = array.shape

		if single:
			if arr_shape != subshape and arr_shape != (1,) + subshape:
				raise ValueError(f"field {field!r} expects shape {subshape}, got {arr_shape}")
			return np.broadcast_to(array, target_shape)

		if arr_shape == target_shape:
			return array

		if arr_shape == subshape or arr_shape == (1,) + subshape:
			return np.broadcast_to(array, target_shape)

		raise ValueError(f"field {field!r} expects shape {subshape} or {target_shape}, got {arr_shape}")

	def _normalise_values(self, entities: EntityArray, values: object) -> dict[str, FieldArray]:
		quantity = entities.size
		n_fields = len(self.fields)
		field_values: tuple[Any, ...]

		if isinstance(values, self.component_cls):
			return {field: self._broadcast_field(field, getattr(values, field), quantity, single=True) for field in self.fields}

		if n_fields == 1 and not (isinstance(values, tuple) and len(values) == 1):
			field_values = (values,)
		elif isinstance(values, tuple):
			field_values = values
		else:
			if any(dtype.subdtype is not None for dtype in self._field_dtypes.values()):
				raise ValueError(f"{self.component_cls.__name__} has shaped fields; values must be supplied as a tuple of fields")
			packed = np.asarray(values)
			if packed.shape == (n_fields,):
				field_values = tuple(packed)
			elif packed.shape == (quantity, n_fields):
				field_values = tuple(packed.T)
			else:
				raise ValueError(f"cannot map values with shape {packed.shape} to {self.component_cls.__name__}{self.fields}")

		if len(field_values) != n_fields:
			raise ValueError(f"{self.component_cls.__name__} expects {n_fields} fields, got {len(field_values)}")

		return {field: self._broadcast_field(field, value, quantity) for field, value in zip(self.fields, field_values)}

	# ---------- mutation ----------

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		if not entities.size: return
		self.membership_version += 1
		indices = _entity_indices(entities) 
		self._ensure_sparse(indices)

		count = int(entities.size)
		start = self._size
		stop = start + count
		self._grow_dense(stop)

		rows = np.arange(start, stop, dtype=Index)
		self._dense_entities[start:stop] = entities
		for field, field_values_array in field_values.items():
			self._dense[field][start:stop] = field_values_array
		self._set_dense_indices(indices, rows)
		self._size = stop
	
	def _add_range(self, entities: EntityArray, values: object) -> None:
		self._add_normalised(entities, self._normalise_values(entities, values))


	def _remove_entities(self, indices: IndexArray) -> EntityArray:
		return self._remove_rows(self._get_rows(indices))

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		if rows.size == 0:
			return np.empty(0, dtype=Entity)

		self.membership_version += 1
		removed_entities = self._dense_entities[rows].astype(Entity, copy=True)
		old_size = self._size
		new_size = old_size - int(rows.size)

		removed_mask = np.zeros(old_size, dtype=np.bool_)
		removed_mask[rows] = True
		holes = rows[rows < new_size]
		tail = np.arange(new_size, old_size, dtype=Index)
		sources = tail[~removed_mask[tail]]

		self._clear_dense_indices(_entity_indices(removed_entities))

		if holes.size:
			moved_entities = self._dense_entities[sources].astype(Entity, copy=True)
			for array in self._dense.values():
				array[holes] = array[sources]
			self._set_dense_indices(_entity_indices(moved_entities), holes)

		self._clear_rows(np.arange(new_size, old_size, dtype=Index))
		self._size = new_size
		return removed_entities

	# ---------- query ----------

	def _query_rows(
		self,
		condition: Callable[..., Any],
		rows: IndexArray,
	) -> IndexArray:
		if rows.size == 0:
			return np.empty(0, dtype=Index)

		params = arg_names(condition)
		missing = [name for name in params if name not in self._dense]
		if missing:
			raise TypeError(
				f"query parameters are not component fields: {', '.join(missing)}"
			)

		result = np.asarray(
			condition(**{name: self._dense[name][rows] for name in params}),
			dtype=np.bool_,
		)
		if result.ndim == 0:
			return rows.copy() if bool(result) else np.empty(0, dtype=Index)
		if result.ndim != 1 or result.size != rows.size:
			raise ValueError("query condition must return one boolean per component row")
		return rows[result]


class MultiComponentStorage(ComponentStorage):

	def __init__(
		self,
		component_cls: type[Any],
		capacity: int = INITIAL_CAPACITY,
	) -> None:
		super().__init__(component_cls, capacity)
		self._dense[_COUNT_STR] = np.zeros( self._capacity, dtype=ComponentPerEntityCount )
		self._field_dtypes[_COUNT_STR] = _dtype_for(ComponentPerEntityCount)
		self._live_count = 0

	@property
	def _dense_counts(self) -> CountArray:
		return self._dense[_COUNT_STR]

	def _clear_rows(self, rows: IndexArray) -> None:
		super()._clear_rows(rows)
		self._dense_counts[rows] = 0

	def _active_rows(self) -> IndexArray:
		return np.flatnonzero(
			self._dense_entities[:self._size] != NO_ENTITY
		).astype(Index, copy=False)

	def _block_rows(
		self,
		heads: IndexArray,
		counts: CountArray | None = None,
	) -> IndexArray:
		"""Concatenate arange(head, head + count) for each block."""
		if heads.size == 0:
			return np.empty(0, dtype=Index)

		if counts is None:
			counts = self._dense_counts[heads]

		counts64 = counts.astype(np.int64, copy=False)
		total = int(counts64.sum())

		if total == 0:
			return np.empty(0, dtype=Index)

		offsets = np.cumsum(counts64) - counts64

		rows = (
			np.repeat(
				heads.astype(np.int64, copy=False) - offsets,
				counts64,
			)
			+ np.arange(total, dtype=np.int64)
		)

		return rows.astype(Index, copy=False)

	def _get_rows(self, indices: IndexArray) -> IndexArray:
		heads = self._get_dense_indices(indices)
		heads = heads[heads != NONE]
		return self._block_rows(heads)

	def _trim_tail(self) -> None:
		if self._live_count == 0:
			self._size = 0
			return

		if self._dense_entities[self._size - 1] != NO_ENTITY:
			return

		reverse_live = (
			self._dense_entities[:self._size][::-1] != NO_ENTITY
		)
		self._size -= int(np.argmax(reverse_live))

	def _defragment(self, required_capacity: int) -> None:
		rows = self._active_rows()
		live = int(rows.size)

		grow = required_capacity > self._capacity
		new_capacity = (
			higher_pow2(required_capacity)
			if grow
			else self._capacity
		)

		if live == self._size and not grow:
			return

		for name in self._dense:
			old = self._dense[name]

			if grow:
				new = np.empty(new_capacity , dtype=self._field_dtypes[name])
			else:
				new = old

			new[:live] = old[rows]

			if name == _ENTITY_STR:
				new[live:] = NO_ENTITY
			elif name == _ENTITY_STR:
				new[live:] = 0
			elif old.dtype == OBJECT_DTYPE:
				new[live:] = None

			self._dense[name] = new

		self._capacity = new_capacity
		self._size = live

		self._sparse.fill(NONE)

		owners = self._dense_entities[:live]
		starts = np.flatnonzero(
			np.diff(owners, prepend=NO_ENTITY)
		).astype(Index, copy=False)

		self._set_dense_indices(
			_entity_indices(owners[starts]),
			starts,
		)

	def _ensure_append_capacity(self, quantity: int) -> None:
		if self._size + quantity > self._capacity:
			self._defragment(self._live_count + quantity)

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		if entities.size == 0:
			return

		self.membership_version += 1

		# Group incoming components by owner.
		order = np.argsort(entities, kind="stable")
		entities = entities[order]

		starts = np.flatnonzero(
			np.r_[True, entities[1:] != entities[:-1]]
		)

		add_counts = np.diff(
			np.r_[starts, entities.size]
		).astype(ComponentPerEntityCount, copy=False)

		owners = entities[starts]
		owner_indices = _entity_indices(owners)

		self._ensure_sparse(owner_indices)

		old_heads = self._sparse[owner_indices]
		existing = old_heads != NONE

		old_counts = np.zeros_like(add_counts)
		old_counts[existing] = self._dense_counts[ old_heads[existing] ]

		final_counts = old_counts + add_counts

		# Save existing blocks before removing them.
		old_rows = self._block_rows( old_heads[existing], old_counts[existing] )

		old_values = {
			field: self._dense[field][old_rows].copy()
			for field in self.fields
		}

		if old_rows.size:
			self._clear_rows(old_rows)
			self._sparse[owner_indices[existing]] = NONE

			self._live_count -= int(old_rows.size)
			self._trim_tail()

		# Rebuild all affected blocks contiguously at the end.
		append_count = int(final_counts.sum())
		self._ensure_append_capacity(append_count)

		start = self._size

		offsets = (
			np.cumsum(final_counts, dtype=np.uint64)
			- final_counts
		)

		new_heads = (
			start + offsets
		).astype(Index, copy=False)

		stop = start + append_count

		self._dense_entities[start:stop] = np.repeat( owners, final_counts )

		# Count metadata exists only at block heads.
		self._dense_counts[start:stop] = 0
		self._dense_counts[new_heads] = final_counts

		old_destinations = self._block_rows( new_heads, old_counts )

		add_heads = new_heads.astype(Index) + old_counts.astype(Index)
		add_destinations = self._block_rows( add_heads, add_counts )

		if old_rows.size:
			for field in self.fields:
				self._dense[field][old_destinations] = old_values[field]

		for field in self.fields:
			self._dense[field][add_destinations] = field_values[field][order]

		self._sparse[owner_indices] = new_heads

		self._size = stop
		self._live_count += append_count

	def _remove_entities(self, indices: IndexArray) -> EntityArray:
		return self._remove_rows(
			np.sort(self._get_rows(indices))
		)

	def _remove_rows(self, rows: IndexArray) -> EntityArray:
		if not rows.size: return np.empty(0, dtype=Entity)
		self.membership_version += 1
		removed_owners = self._dense_entities[rows]

		owners, removed_counts_ = np.unique(
			removed_owners,
			return_counts=True,
		)

		owner_indices = _entity_indices(owners)
		heads = self._sparse[owner_indices]

		old_counts = self._dense_counts[heads]

		removed_counts = removed_counts_.astype(ComponentPerEntityCount, copy=False )

		new_counts = ( old_counts - removed_counts
		).astype(ComponentPerEntityCount, copy=False)

		block_rows = self._block_rows(heads, old_counts)

		# Determine which rows in the affected blocks survive.
		positions = np.searchsorted(rows, block_rows)

		removed = positions < rows.size
		removed[removed] &= (rows[positions[removed]] == block_rows[removed])

		kept_rows = block_rows[~removed]

		destinations = self._block_rows(heads, new_counts)

		# Advanced indexing materializes RHS, so overlap is safe.
		for field in self.fields:
			self._dense[field][destinations] = self._dense[field][kept_rows]

		self._dense_entities[destinations] = np.repeat(
			owners,
			new_counts,
		)

		# Clear the removed tails of each affected block.
		tail_heads = heads.astype(Index) + new_counts.astype(Index)
		tails = self._block_rows( tail_heads, removed_counts )

		self._clear_rows(tails)

		nonempty = new_counts != 0
		self._dense_counts[heads[nonempty]] =  new_counts[nonempty]

		emptied = ~nonempty
		self._sparse[owner_indices[emptied]] = NONE

		self._live_count -= int(rows.size)
		self._trim_tail()

		return owners[emptied]


class ECS:
	def __init__(self) -> None:
		self._stores: dict[type[Any], ComponentStorage] = {}
		self._comp_bits: dict[type[Any], Mask] = {}

		self.entity_masks = np.zeros(0, dtype=Mask)
		self._current_entities = np.full(0, NO_ENTITY, dtype=Entity)
		self._free_entities: list[Entity] = []
		self._next_entity_index = 0
		self._entity_count = 0

	def __len__(self) -> int:
		return self._entity_count

	# ---------- entity state ----------

	def _grow_entity_state(self, minimum_size: int) -> None:
		if minimum_size <= self.entity_masks.size:
			return
		old_size = self.entity_masks.size
		new_size = max(higher_pow2(minimum_size), max(1, old_size * 2))

		masks = np.zeros(new_size, dtype=Mask)
		masks[:old_size] = self.entity_masks
		self.entity_masks = masks

		current = np.full(new_size, NO_ENTITY, dtype=Entity)
		current[:old_size] = self._current_entities
		self._current_entities = current

	def _alive_mask(self, entities: EntityArray) -> BoolArray:
		indices = _entity_indices(entities)
		alive = indices < self._current_entities.size

		alive[alive] &= (
			self._current_entities[indices[alive]]
			== entities[alive]
		)
		return alive

	def _filter_alive(self, entities: EntityArray) -> EntityArray:
		return entities[self._alive_mask(entities)]

	# ---------- component lookup ----------

	def _component_info(
		self,
		component_cls: type[Any],
	) -> tuple[ComponentStorage, Mask]:
		try:
			return self._stores[component_cls], self._comp_bits[component_cls]
		except KeyError:
			raise KeyError(
				f"component {component_cls.__name__} is not registered"
			) from None

	def _component_bits(self, component_types: Iterable[type[Any]]) -> Mask:
		bits = Mask(0)
		for component_cls in component_types:
			bits |= self._component_info(component_cls)[1]
		return bits

	def _clear_component_mask(
		self,
		indices: IndexArray,
		component_cls: type[Any],
	) -> None:
		self.entity_masks[indices] &= ~self._component_info(component_cls)[1]

	def _parse_add_components(
		self,
		params: tuple[object, ...],
	) -> Iterator[tuple[ComponentStorage, Mask, object]]:
		i = 0

		while i < len(params):
			item = params[i]

			if not isinstance(item, type):
				store, bit = self._component_info(type(item))
				yield store, bit, item
				i += 1
				continue

			store, bit = self._component_info(item)

			if i + 1 >= len(params) or (
				isinstance(params[i + 1], type)
				and params[i + 1] in self._stores
			):
				raise TypeError(f"missing components for {item.__name__}")

			yield store, bit, params[i + 1]
			i += 2

	# ---------- registration ----------

	def register(
		self,
		*component_types: type[Any],
		allow_same_type_components_per_entity: bool = False,
		capacity: int = INITIAL_CAPACITY,
	) -> tuple[ComponentAccessor, ...]:
		res : list[ComponentAccessor] = []

		for cls in component_types:
			if cls in self._stores:
				raise ValueError(f"{cls.__name__} is already registered")
			
			bit_index = len(self._comp_bits)
			if bit_index >= 64: raise ValueError("ECS supports at most 64 component types")

			storage_cls: type[ComponentStorage] = (
				MultiComponentStorage if cls._multicomponent or allow_same_type_components_per_entity
				else ComponentStorage
			)
			
			store = storage_cls(cls, capacity)
			self._stores[cls] = store
			self._comp_bits[cls] = Mask(1 << bit_index)

			res.append( ComponentAccessor(self, cls, store) )

		return tuple(res)


	# ---------- entities ----------

	def create(self, quantity: int = 1) -> EntityArray:
		if quantity < 0:
			raise ValueError("quantity must be >= 0")
		if quantity == 0:
			return np.empty(0, dtype=Entity)

		reuse_count = min(quantity, len(self._free_entities))
		reused = np.empty(0, dtype=Entity)

		if reuse_count:
			old = np.asarray(self._free_entities[-reuse_count:], dtype=Entity)
			del self._free_entities[-reuse_count:]
			reused = old + GENERATION_INCREMENT
			indices = _entity_indices(reused)
			self._current_entities[indices] = reused
			self.entity_masks[indices] = Mask(0)

		new_count = quantity - reuse_count
		if new_count == 0:
			self._entity_count += quantity
			return reused

		stop = self._next_entity_index + new_count
		if stop > INDEX_MAX:
			raise OverflowError("entity index space exhausted")
		self._grow_entity_state(stop)
		indices = np.arange(self._next_entity_index, stop, dtype=Index)
		fresh = indices.astype(Entity)
		self._current_entities[indices] = fresh
		self.entity_masks[indices] = Mask(0)
		self._next_entity_index = stop
		self._entity_count += quantity

		if reuse_count == 0:
			return fresh
		return np.concatenate((reused, fresh))

	def delete(self, entities: EntityLike) -> None:
		entity_array = self._filter_alive(as_entities(entities))
		if entity_array.size == 0:
			return

		# Duplicate live IDs must not be recycled twice.
		entity_array = np.unique(entity_array)
		indices = _entity_indices(entity_array)
		masks = self.entity_masks[indices].copy()

		for component_cls, bit in self._comp_bits.items():
			selected = (masks & bit) != 0
			if np.any(selected):
				self._stores[component_cls]._remove_entities(indices[selected])

		self.entity_masks[indices] = Mask(0)
		self._current_entities[indices] = NO_ENTITY
		self._free_entities.extend(entity_array.tolist())
		self._entity_count -= int(entity_array.size)

	# ---------- components ----------

	def add(self, entities: EntityLike, *components: object) -> None:
		entity_array = as_entities(entities)
		if entity_array.size == 0 or not components:
			return

		alive = self._alive_mask(entity_array)
		if not np.any(alive):
			return

		live_entities = entity_array[alive]
		live_indices = _entity_indices(live_entities)

		for store, bit, values in list(self._parse_add_components(components)):
			values = store._normalise_values(entity_array, values)

			if not np.all(alive):
				values = {field: value[alive] for field, value in values.items()}

			store._add_normalised(live_entities, values)
			self.entity_masks[live_indices] |= bit

	def remove(self, entities: EntityLike, *component_types: type[Any]) -> None:
		entity_array = np.unique(self._filter_alive(as_entities(entities)))
		if entity_array.size == 0 or not component_types:
			return

		indices = _entity_indices(entity_array)
		for component_cls in component_types:
			store, bit = self._component_info(component_cls)
			store._remove_entities(indices)
			self.entity_masks[indices] &= ~bit

	def _get_store(self, component_cls: type) -> ComponentStorage:
		return self._component_info(component_cls)[0]

	def get(self, component_cls: type) -> ComponentAccessor:
		store, _ = self._component_info(component_cls)
		return ComponentAccessor(self, component_cls, store)

	def where(
		self,
		*component_types: type[Any],
		exclude: type[Any] | Sequence[type[Any]] | None = None,
	) -> EntityArray:
		if not component_types:
			return np.empty(0, dtype=Entity)

		required = self._component_bits(component_types)
		if exclude is None:
			excluded = Mask(0)
		elif isinstance(exclude, type):
			excluded = self._component_bits((exclude,))
		else:
			excluded = self._component_bits(exclude)

		masks = self.entity_masks[: self._next_entity_index]
		selected = (masks & required) == required
		if excluded: selected &= (masks & excluded) == 0

		indices = np.flatnonzero(selected).astype(Index, copy=False)
		return self._current_entities[indices].astype(Entity, copy=False)
