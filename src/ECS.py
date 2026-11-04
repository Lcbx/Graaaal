from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import IntFlag
from collections import deque
import inspect
from operator import index as integer_index
from typing import cast, Any, Callable, ClassVar, Iterable, Iterator, Mapping, Sequence, TypeAlias, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

def component(cls: type | None = None, *, multiples: bool = False) -> Any | Callable:
	def decorator(target: Any) -> Any:
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
	return tuple(param.name for param in _predicate_parameters(func))

def _predicate_parameters(func: Callable[..., Any]) -> tuple[inspect.Parameter, ...]:
	"""Validate a synchronous predicate that can receive component fields by name."""
	if not callable(func):
		raise TypeError("query predicate must be callable")
	for target in (func, getattr(func, "__call__", None)):
		if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target) or inspect.isgeneratorfunction(target):
			raise TypeError("query predicate must return booleans synchronously")
	try:
		params = tuple(inspect.signature(func).parameters.values())
	except (TypeError, ValueError) as error:
		raise TypeError("query predicate requires an inspectable signature; wrap it in a named-field function") from error
	if any(param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) for param in params):
		raise TypeError("query predicate requires named parameters; positional-only and variadic parameters are unsupported")
	return params

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

class _RowSelection:
	"""Common storage identity, row ownership, and lifetime checks."""
	__slots__ = ("_store", "_rows", "_layout_version")

	def __init__(self, store: ComponentStorage, rows: IndexArray) -> None:
		self._store = store
		self._rows = as_rows(rows).copy()
		self._rows.flags.writeable = False
		self._layout_version = store.layout_version

	def _validate(self, store: ComponentStorage | None = None) -> None:
		if store is not None and self._store is not store:
			raise ValueError("rows belong to a different component storage")
		if self._layout_version != self._store.layout_version:
			raise RuntimeError("stale component selection: acquire a new selection after structural changes")

	def get_rows(self) -> IndexArray:
		"""Export a copy of raw indices; use the wrapper to retain lifetime checks."""
		self._validate()
		return self._rows.copy()

	def __len__(self) -> int:
		self._validate()
		return int(self._rows.size)


class ComponentRows(_RowSelection):
	"""Rows tied to one component store and its current layout.

	Use accessor.rows() or accessor.query_rows() to acquire checked rows. Slicing
	keeps the checks; get_rows() exports a copy of the raw indices, whose lifetime
	the caller must manage. Field writes do not refresh a predicate result.
	"""
	__slots__ = ()

	def __getitem__(self, key: Any) -> ComponentRows:
		self._validate()
		return ComponentRows(self._store, as_rows(self._rows[key]))


class ComponentSelection(_RowSelection):
	"""Field arrays copy on read; assign them back to write changes.

	Use selection["field"] for fields whose names collide with methods such as
	vector or get_rows. Internal owner/count metadata is reserved by storage.
	A structural edit to this store expires the selection. Numeric `x += value`
	writes back through the setter; `x[:] = value` only edits a temporary array.
	Object fields copy references: in-place object edits require accessor.touch().
	"""
	__slots__ = ()

	def checked_rows(self) -> ComponentRows:
		"""Capture these rows for checked slicing, owner lookup, or removal."""
		self._validate()
		return ComponentRows(self._store, self._rows)

	def vector(self, *field_names: str) -> FieldArray:
		self._validate()
		field_names = field_names or self._store.fields
		if not field_names:
			return np.empty((len(self), 0))
		return np.column_stack( tuple(self[field] for field in field_names) )

	def set_vector(self, value : FieldArray, *field_names: str) -> None:
		self._validate()
		field_names = field_names or self._store.fields
		if not field_names and np.shape(value) != (len(self), 0):
			raise ValueError("tag vectors must have shape (number of rows, 0)")
		for i, field in enumerate(field_names):
			self[field] = value[:, i]

	def __getitem__(self, field: str) -> FieldArray:
		self._validate()
		return self._store._dense[field][self._rows]

	def __setitem__(self, field: str, value: Any) -> None:
		self._validate()
		array = self._store._dense[field]
		if self._rows.size: self._store.touch(field)
		array[self._rows] = value

	def __getattr__(self, name: str) -> FieldArray:
		self._validate()
		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None
		return array[self._rows]

	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith("_"):
			object.__setattr__(self, name, value)
			return

		self._validate()
		try:
			array = self._store._dense[name]
		except KeyError:
			raise AttributeError(name) from None

		if self._rows.size: self._store.touch(name)
		array[self._rows] = value


RowLike: TypeAlias = IndexLike | ComponentRows | ComponentSelection


class ComponentAccessor:
	"""Selections copy numeric fields; assign edited arrays back to write them.
	Selections and checked rows expire on any structural edit to their store.
	get_rows() and query() return legacy raw indices with no lifetime checking;
	use rows(), query_rows(), or selection.checked_rows() for retained results.
	
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

	@property
	def layout_version(self) -> int:
		"""Token for row lifetimes; may advance more than once per structural edit."""
		return self._store.layout_version

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
		"""Return ephemeral raw row indices; rows() retains lifetime checks."""
		return self._store._active_rows()

	def rows(self) -> ComponentRows:
		"""Select all active rows with storage identity and layout checks."""
		return ComponentRows(self._store, self._store._active_rows())

	def _resolve_rows(self, rows: RowLike) -> IndexArray:
		if isinstance(rows, _RowSelection):
			rows._validate(self._store)
			return rows._rows
		return as_rows(rows)

	def owner_ids(self, component_rows: RowLike) -> EntityArray:
		"""Resolve owners, checking lifetime and storage for checked row inputs."""
		rows = self._resolve_rows(component_rows)
		return self._store._dense_entities[rows].astype(Entity, copy=False)

	def __getitem__(self, entities: EntityLike) -> ComponentSelection:
		entities_ = self._ecs._filter_alive(as_entities(entities))
		indices = _entity_indices(entities_)
		return ComponentSelection(self._store, self._store._get_rows(indices))

	def query(
		self,
		condition: Callable[..., Any],
		entities: EntityLike | None = None,
	) -> IndexArray:
		"""Return ephemeral raw predicate rows; query_rows() retains lifetime checks."""
		if entities is None:
			rows = self._store._active_rows()
		else:
			entities_ = self._ecs._filter_alive(np.unique(as_entities(entities)))
			rows = self._store._get_rows(_entity_indices(entities_))
		return self._store._query_rows(condition, rows)

	def query_rows(
		self,
		condition: Callable[..., Any],
		entities: EntityLike | None = None,
	) -> ComponentRows:
		"""Evaluate a predicate and capture matching rows with lifetime checks.

		This is a snapshot of which rows matched, not a live value query. Predicate
		callbacks must not structurally edit this component store.
		"""
		version = self.layout_version
		rows = self.query(condition, entities)
		if version != self.layout_version:
			raise RuntimeError("component layout changed while evaluating the predicate")
		return ComponentRows(self._store, rows)

	def remove(self, rows: RowLike) -> None:
		"""Remove rows; checked selections reject a stale or foreign row mapping."""
		rows_ = np.unique(self._resolve_rows(rows))
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

		self.fields = tuple(name for name, _ in field_defs)
		reserved = {_ENTITY_STR, _COUNT_STR}.intersection(self.fields)
		if reserved:
			raise ValueError(f"reserved component fields: {', '.join(sorted(reserved))}")
		self.membership_version = 0
		self.layout_version = 0
		self.field_versions = {name: 0 for name in self.fields}
		self._field_dtypes = {
			name: _dtype_for(annotation)
			for name, annotation in field_defs
		}

		self._capacity = max(1, int(capacity))
		self._size = 0
		self._sparse = np.full(self._capacity, NONE, dtype=Index)

		self._dense: dict[str, FieldArray] = {}
		self._field_dtypes[_ENTITY_STR] = _dtype_for(Entity)

		for name, dtype in self._field_dtypes.items():
			array = np.empty(self._capacity, dtype=dtype)
			if name == _ENTITY_STR:
				array.fill(NO_ENTITY)
			elif dtype == OBJECT_DTYPE:
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
		self.layout_version += 1

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
					array = np.asarray(value, dtype=object)
					if array.shape != (quantity,):
						raise ValueError(f"field {field!r} expects a 1-D object array")
					return array
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
		if not self.fields:
			if values is self.component_cls or isinstance(values, self.component_cls):
				return {}
			raise TypeError("tags have no field payload; supply the component type or an instance")
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

	def _validate_add(self, entities: EntityArray) -> None:
		indices = _entity_indices(entities)
		if np.unique(indices).size != indices.size:
			raise ValueError(f"duplicate owners for {self.component_cls.__name__}")
		if np.any(self._get_dense_indices(indices) != NONE):
			raise ValueError(f"{self.component_cls.__name__} already exists; use set() to replace it")

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		if not entities.size: return
		self._validate_add(entities)
		self.membership_version += 1
		self.layout_version += 1
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
		self.layout_version += 1
		removed_entities = self._dense_entities[rows].astype(Entity, copy=True)
		old_size = self._size
		new_size = old_size - int(rows.size)

		holes = rows[rows < new_size]
		tail = np.arange(new_size, old_size, dtype=Index)
		if rows.size == 1:
			sources = tail[tail != rows[0]]
		elif rows.size <= 8 and old_size >= 4096:
			# Bound temporary storage by the small removal batch, not the pool size.
			sources = tail[~np.any(tail[:, None] == rows[None, :], axis=1)]
		else:
			removed_mask = np.zeros(old_size, dtype=np.bool_)
			removed_mask[rows] = True
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
		params = _predicate_parameters(condition)
		missing = [param.name for param in params if param.name not in self._dense and param.default is inspect.Parameter.empty]
		if missing:
			raise TypeError(
				f"query parameters are not component fields: {', '.join(missing)}"
			)
		if rows.size == 0:
			return np.empty(0, dtype=Index)

		result = np.asarray(
			condition(**{param.name: self._dense[param.name][rows] for param in params if param.name in self._dense}),
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
		if not self.fields:
			raise TypeError("tags do not support multiple instances per entity")
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
		counts: CountArray | NDArray[np.int64] | None = None,
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

		self.layout_version += 1

		for name in self._dense:
			old = self._dense[name]

			if grow:
				new = np.empty(new_capacity , dtype=self._field_dtypes[name])
			else:
				new = old

			new[:live] = old[rows]

			if name == _ENTITY_STR:
				new[live:] = NO_ENTITY
			elif name == _COUNT_STR:
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

	def _validate_add(self, entities: EntityArray) -> None:
		owners, counts = np.unique(_entity_indices(entities), return_counts=True)
		heads = self._get_dense_indices(owners)
		existing = heads != NONE
		counts[existing] += self._dense_counts[heads[existing]].astype(np.int64)
		if np.any(counts > np.iinfo(ComponentPerEntityCount).max):
			raise OverflowError(f"too many {self.component_cls.__name__} components for one owner")

	def _add_normalised(
		self,
		entities: EntityArray,
		field_values: Mapping[str, FieldArray],
	) -> None:
		if entities.size == 0:
			return

		self._validate_add(entities)
		self.membership_version += 1
		self.layout_version += 1

		# Group incoming components by owner.
		order = np.argsort(entities, kind="stable")
		entities = entities[order]

		starts = np.flatnonzero(
			np.r_[True, entities[1:] != entities[:-1]]
		)

		add_counts = np.diff(
			np.r_[starts, entities.size]
		)

		owners = entities[starts]
		owner_indices = _entity_indices(owners)

		self._ensure_sparse(owner_indices)

		old_heads = self._sparse[owner_indices]
		existing = old_heads != NONE

		old_counts = np.zeros_like(add_counts)
		old_counts[existing] = self._dense_counts[ old_heads[existing] ]

		final_counts = old_counts + add_counts

		if owners.size == 1 and existing[0] and int(old_heads[0]) + int(old_counts[0]) == self._size:
			# Extend the last block without copying its existing rows on every append.
			quantity = int(add_counts[0])
			self._ensure_append_capacity(quantity)
			# Capacity management may compact holes and relocate this block.
			head = self._sparse[owner_indices[0]]
			start, stop = self._size, self._size + quantity
			self._dense_entities[start:stop] = owners[0]
			self._dense_counts[start:stop] = 0
			self._dense_counts[head] = final_counts[0]
			for field in self.fields:
				self._dense[field][start:stop] = field_values[field][order]
			self._size = stop
			self._live_count += quantity
			return

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
			np.cumsum(final_counts, dtype=np.int64)
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
		self.layout_version += 1
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


def _query_types(types: type[Any] | Sequence[type[Any]] | None) -> tuple[type[Any], ...]:
	if types is None:
		return ()
	items = (types,) if isinstance(types, type) else tuple(types)
	if any(not isinstance(item, type) for item in items):
		raise TypeError("query terms must be component types")
	return tuple(dict.fromkeys(items))


class QueryComponent:
	"""Packed rows in query-entity order, with one presence bit and offset per owner.

	For single components, selection rows align with result.entities[present].
	For multicomponents, offsets[i]:offsets[i + 1] identifies owner i's rows.
	Missing owners have zero rows. Different multicomponent terms are never zipped.
	"""
	__slots__ = ("_result", "_selection", "_present", "_offsets")

	def __init__(self, result: QueryResult, store: ComponentStorage, indices: IndexArray) -> None:
		self._result = result
		heads = store._get_dense_indices(indices)
		self._present = heads != NONE
		counts = self._present.astype(np.int64)
		if isinstance(store, MultiComponentStorage):
			counts[self._present] = store._dense_counts[heads[self._present]]
			rows = store._block_rows(heads[self._present], counts[self._present])
		else:
			rows = heads[self._present]
		self._offsets = np.empty(indices.size + 1, dtype=np.int64)
		self._offsets[0] = 0
		self._offsets[1:] = np.cumsum(counts)
		self._selection = ComponentSelection(store, rows)

	@property
	def selection(self) -> ComponentSelection:
		"""Current field values at cached rows, using normal selection write-back."""
		self._result._validate()
		return self._selection

	@property
	def present(self) -> BoolArray:
		self._result._validate()
		return self._present.copy()

	@property
	def offsets(self) -> NDArray[np.int64]:
		self._result._validate()
		return self._offsets.copy()


class QueryResult:
	"""Cached entity membership and component row mappings, not cached field values.

	Refetch from the query after relevant membership or layout changes. An extracted
	ComponentSelection retains its own store's lifetime rules.
	"""
	__slots__ = ("_query", "_entities", "_membership", "_layout", "_components")

	def __init__(
		self, query: Query, entities: EntityArray, membership: tuple[int, ...], layout: tuple[int, ...],
	) -> None:
		self._query = query
		self._entities = entities
		self._membership = membership
		self._layout = layout
		indices = _entity_indices(entities)
		self._components = {
			cls: QueryComponent(self, store, indices) for cls, store in query._returned_stores.items()
		}

	def _validate(self) -> None:
		if self._membership != self._query._membership_token() or self._layout != self._query._layout_token():
			raise RuntimeError("stale query result: call fetch() again after membership or layout changes")

	@property
	def entities(self) -> EntityArray:
		self._validate()
		return self._entities.copy()

	def __len__(self) -> int:
		self._validate()
		return int(self._entities.size)

	def __getitem__(self, component_cls: type[Any]) -> QueryComponent:
		self._validate()
		return self._components[component_cls]


class Query:
	"""Reusable structural query. Results follow ascending entity-index order.

	All required terms, at least one any_of term (when nonempty), and no excluded
	terms must match. Optional terms do not filter membership. Returned terms are
	the union of required, any_of, and optional components in declaration order.
	Field writes keep cached membership/rows; field predicates remain separate,
	uncached ComponentAccessor.query_rows() evaluations.
	"""
	def __init__(
		self, ecs: ECS, required: Sequence[type[Any]], *,
		exclude: type[Any] | Sequence[type[Any]] | None = None,
		any_of: type[Any] | Sequence[type[Any]] | None = None,
		optional: type[Any] | Sequence[type[Any]] | None = None,
	) -> None:
		self._ecs = ecs
		required = _query_types(required)
		excluded = _query_types(exclude)
		any_types = _query_types(any_of)
		optional_types = _query_types(optional)
		self._required = ecs._component_bits(required)
		self._excluded = ecs._component_bits(excluded)
		self._any = ecs._component_bits(any_types)
		filters = tuple(dict.fromkeys((*required, *excluded, *any_types)))
		self._filter_stores = tuple(ecs._get_store(cls) for cls in filters)
		self._required_stores = tuple(ecs._get_store(cls) for cls in required)
		self._returned_stores = {
			cls: ecs._get_store(cls) for cls in dict.fromkeys((*required, *any_types, *optional_types))
		}
		self._track_entities = not required
		self._cached_membership: tuple[int, ...] | None = None
		self._entities = np.empty(0, dtype=Entity)
		self._result: QueryResult | None = None

	def _membership_token(self) -> tuple[int, ...]:
		versions = tuple(store.membership_version for store in self._filter_stores)
		return versions + ((self._ecs.entity_version,) if self._track_entities else ())

	def _layout_token(self) -> tuple[int, ...]:
		return tuple(store.layout_version for store in self._returned_stores.values())

	def _scan_entities(self) -> EntityArray:
		return self._ecs._matching_entities(self._required, self._excluded, self._any, self._required_stores)

	def _refresh_entities(self, membership: tuple[int, ...]) -> None:
		if membership != self._cached_membership:
			self._entities = self._scan_entities()
			self._cached_membership = membership
			self._result = None

	def entities(self) -> EntityArray:
		"""Return matching handles without constructing component row selections."""
		self._refresh_entities(self._membership_token())
		return self._entities.copy()

	def fetch(self) -> QueryResult:
		"""Reuse membership and rows until their structural version tokens change."""
		membership = self._membership_token()
		self._refresh_entities(membership)
		layout = self._layout_token()
		if self._result is None or self._result._layout != layout:
			self._result = QueryResult(self, self._entities, membership, layout)
		return self._result


class DeferredEntities:
	"""A creation result owned by one command buffer; resolve after its create runs.

	Pass the token directly as a command target before flushing. It cannot be mixed
	with other handles or sliced. Payload fields are not resolved automatically;
	resolve explicitly before supplying entity handles as component field values.
	"""
	__slots__ = ("_buffer", "_entities", "_failed")

	def __init__(self, buffer: CommandBuffer) -> None:
		self._buffer = buffer
		self._entities: EntityArray | None = None
		self._failed = False

	def resolve(self) -> EntityArray:
		if self._failed:
			raise RuntimeError("deferred creation failed or was cancelled")
		if self._entities is None:
			raise RuntimeError("deferred entities are unresolved; flush their creation first")
		return self._entities.copy()


@dataclass(frozen=True)
class _Command:
	index: int
	operation: str
	target: EntityArray | DeferredEntities
	args: tuple[object, ...]


class CommandError(RuntimeError):
	"""A consumed command failed; earlier commands applied and the tail is pending."""
	def __init__(self, command: _Command) -> None:
		self.command_index = command.index
		self.operation = command.operation
		super().__init__(f"command {command.index} ({command.operation}) failed")


class CommandBuffer:
	"""FIFO edits applied at flush(), each preflighted by the ordinary ECS APIs.

	Handle arrays are copied on enqueue. Component payloads are retained by reference;
	copy mutable payloads yourself if they must not change before execution. A flush
	processes the commands present at entry. Failure consumes the failed command,
	keeps the tail, and raises CommandError with the original exception as its cause.
	"""
	def __init__(self, ecs: ECS) -> None:
		self._ecs = ecs
		self._commands: deque[_Command] = deque()
		self._next_index = 0
		self._flushing = False

	def __len__(self) -> int:
		return len(self._commands)

	def _enqueue(self, operation: str, entities: EntityLike | DeferredEntities, args: tuple[object, ...]) -> None:
		if isinstance(entities, DeferredEntities):
			if entities._buffer is not self:
				raise ValueError("deferred entities belong to another command buffer; resolve them explicitly")
			target = entities
		else:
			target = as_entities(entities).copy()
		self._commands.append(_Command(self._next_index, operation, target, args))
		self._next_index += 1

	def create(self, quantity: int = 1) -> DeferredEntities:
		quantity = integer_index(quantity)
		if quantity < 0:
			raise ValueError("quantity must be >= 0")
		result = DeferredEntities(self)
		self._enqueue("create", result, (quantity,))
		return result

	def add(self, entities: EntityLike | DeferredEntities, *components: object) -> None:
		self._enqueue("add", entities, components)

	def set(self, entities: EntityLike | DeferredEntities, *components: object) -> None:
		self._enqueue("set", entities, components)

	def remove(self, entities: EntityLike | DeferredEntities, *types: type[Any]) -> None:
		self._enqueue("remove", entities, types)

	def delete(self, entities: EntityLike | DeferredEntities) -> None:
		self._enqueue("delete", entities, ())

	def clear(self) -> None:
		"""Discard pending commands; pending creation tokens become cancelled."""
		if self._flushing:
			raise RuntimeError("cannot clear a flushing command buffer")
		for command in self._commands:
			if command.operation == "create":
				cast(DeferredEntities, command.target)._failed = True
		self._commands.clear()

	def flush(self) -> int:
		if self._flushing:
			raise RuntimeError("command buffer flush is not reentrant")
		self._flushing = True
		completed = 0
		try:
			for _ in range(len(self._commands)):
				command = self._commands.popleft()
				try:
					if command.operation == "create":
						cast(DeferredEntities, command.target)._entities = self._ecs.create(cast(int, command.args[0]))
					else:
						target = command.target.resolve() if isinstance(command.target, DeferredEntities) else command.target
						getattr(self._ecs, command.operation)(target, *command.args)
				except Exception as error:
					if command.operation == "create":
						cast(DeferredEntities, command.target)._failed = True
					raise CommandError(command) from error
				completed += 1
		finally:
			self._flushing = False
		return completed


@dataclass(frozen=True)
class _System:
	name: str
	callback: Callable[[ECS, CommandBuffer, float], None]
	phase: str
	before: tuple[str, ...]
	after: tuple[str, ...]
	flush_after: bool


class Scheduler:
	"""Sequential fixed_update, update, render_prep phases with explicit visibility.

	Callbacks receive (world, commands, dt). Flush before a run and after each phase
	(or fixed tick); flush_after=True also flushes after that system. Dependencies
	order execution but do not themselves flush. Fixed tick counts are caller-owned.
	"""
	PHASES = ("fixed_update", "update", "render_prep")

	def __init__(self, ecs: ECS) -> None:
		self._ecs = ecs
		self.commands = CommandBuffer(ecs)
		self._systems: dict[str, _System] = {}
		self._order: tuple[_System, ...] | None = None
		self._running = False

	def add(
		self, name: str, callback: Callable[[ECS, CommandBuffer, float], None], *,
		phase: str = "update", before: str | Sequence[str] = (),
		after: str | Sequence[str] = (), flush_after: bool = False,
	) -> None:
		if self._running:
			raise RuntimeError("cannot change a running schedule")
		if not isinstance(name, str) or not name or name in self._systems:
			raise ValueError("system names must be nonempty and unique")
		if phase not in self.PHASES:
			raise ValueError(f"unknown phase: {phase}")
		if not callable(callback):
			raise TypeError("system callback must be callable")
		def names(value: str | Sequence[str]) -> tuple[str, ...]:
			result = (value,) if isinstance(value, str) else tuple(value)
			if any(not isinstance(item, str) or not item for item in result):
				raise ValueError("dependencies must be nonempty system names")
			return result
		self._systems[name] = _System(name, callback, phase, names(before), names(after), flush_after)
		self._order = None

	def compile(self) -> tuple[str, ...]:
		"""Validate all dependencies before execution; ties keep registration order."""
		if self._order is None:
			edges: dict[str, set[str]] = {name: set() for name in self._systems}
			for system in self._systems.values():
				dependencies = [(system.name, name) for name in system.before]
				dependencies.extend((name, system.name) for name in system.after)
				for source, dest in dependencies:
					if source not in edges or dest not in edges:
						raise ValueError(f"unknown system dependency: {source} -> {dest}")
					if self.PHASES.index(self._systems[source].phase) > self.PHASES.index(self._systems[dest].phase):
						raise ValueError(f"dependency contradicts phase order: {source} -> {dest}")
					edges[source].add(dest)
			incoming = {name: 0 for name in edges}
			for destinations in edges.values():
				for dest in destinations:
					incoming[dest] += 1
			order = []
			while incoming:
				ready = [name for name, count in incoming.items() if count == 0]
				if not ready:
					raise ValueError("system dependency cycle")
				name = min(ready, key=lambda n: self.PHASES.index(self._systems[n].phase))
				order.append(self._systems[name])
				del incoming[name]
				for dest in edges[name]:
					incoming[dest] -= 1
			self._order = tuple(order)
		return tuple(system.name for system in self._order)

	def _phase(self, phase: str, dt: float) -> None:
		for system in self._order or ():
			if system.phase == phase:
				system.callback(self._ecs, self.commands, dt)
				if system.flush_after:
					self.commands.flush()
		self.commands.flush()

	def run(self, dt: float, *, fixed_steps: int = 0, fixed_dt: float = 1 / 60) -> None:
		fixed_steps = integer_index(fixed_steps)
		if fixed_steps < 0 or not np.isfinite(dt) or dt < 0 or not np.isfinite(fixed_dt) or fixed_dt <= 0:
			raise ValueError("dt must be finite and nonnegative, fixed_dt positive, and fixed_steps nonnegative")
		if self._running:
			raise RuntimeError("schedule execution is not reentrant")
		self.compile()
		self._running = True
		try:
			self.commands.flush()
			for _ in range(fixed_steps):
				self._phase("fixed_update", fixed_dt)
			self._phase("update", dt)
			self._phase("render_prep", dt)
		finally:
			self._running = False


class ECS:
	def __init__(self) -> None:
		self._stores: dict[type[Any], ComponentStorage] = {}
		self._comp_bits: dict[type[Any], Mask] = {}

		self.entity_masks = np.zeros(0, dtype=Mask)
		self._current_entities = np.full(0, NO_ENTITY, dtype=Entity)
		self._free_entities: list[Entity] = []
		self._next_entity_index = 0
		self._entity_count = 0
		self._entity_version = 0

	def __len__(self) -> int:
		return self._entity_count

	def commands(self) -> CommandBuffer:
		"""Create an independent command buffer bound to this world."""
		return CommandBuffer(self)

	@property
	def entity_version(self) -> int:
		"""Lifecycle token advanced by each successful nonempty create/delete call."""
		return self._entity_version

	def is_alive(self, entities: EntityLike) -> BoolArray:
		"""Return one boolean per input handle, including duplicates and stale IDs."""
		return self._alive_mask(as_entities(entities))

	def has(self, entities: EntityLike, component_cls: type[Any]) -> BoolArray:
		"""Return one presence bit per handle; unregistered component types raise."""
		_, bit = self._component_info(component_cls)
		entities_ = as_entities(entities)
		present = self._alive_mask(entities_)
		indices = _entity_indices(entities_)
		present[present] &= (self.entity_masks[indices[present]] & bit) != 0
		return present

	def entities(self) -> EntityArray:
		"""Return all live handles, including empty entities, in entity-index order."""
		current = self._current_entities[:self._next_entity_index]
		return current[current != NO_ENTITY]

	def query(
		self, *required: type[Any],
		exclude: type[Any] | Sequence[type[Any]] | None = None,
		any_of: type[Any] | Sequence[type[Any]] | None = None,
		optional: type[Any] | Sequence[type[Any]] | None = None,
	) -> Query:
		"""Create a reusable structural query; no terms matches every live entity."""
		return Query(self, required, exclude=exclude, any_of=any_of, optional=optional)

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
			if not store.fields:
				yield store, bit, item
				i += 1
				continue

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
		"""Register a batch after validating every type and constructing its storage."""
		pending: dict[type[Any], ComponentStorage] = {}

		for cls in component_types:
			if cls in self._stores or cls in pending:
				raise ValueError(f"{cls.__name__} is already registered")
			
			bit_index = len(self._comp_bits) + len(pending)
			if bit_index >= 64: raise ValueError("ECS supports at most 64 component types")

			storage_cls: type[ComponentStorage] = (
				MultiComponentStorage if getattr(cls, "_multicomponent", False) or allow_same_type_components_per_entity
				else ComponentStorage
			)
			
			pending[cls] = storage_cls(cls, capacity)

		res: list[ComponentAccessor] = []
		for cls, store in pending.items():
			bit_index = len(self._comp_bits)
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
		new_count = quantity - reuse_count
		stop = self._next_entity_index + new_count
		if stop > INDEX_MAX:
			raise OverflowError("entity index space exhausted")
		reused = np.empty(0, dtype=Entity)

		if reuse_count:
			old = np.asarray(self._free_entities[-reuse_count:], dtype=Entity)
			del self._free_entities[-reuse_count:]
			reused = old + GENERATION_INCREMENT
			indices = _entity_indices(reused)
			self._current_entities[indices] = reused
			self.entity_masks[indices] = Mask(0)

		if new_count == 0:
			self._entity_count += quantity
			self._entity_version += 1
			return reused

		self._grow_entity_state(stop)
		indices = np.arange(self._next_entity_index, stop, dtype=Index)
		fresh = indices.astype(Entity)
		self._current_entities[indices] = fresh
		self.entity_masks[indices] = Mask(0)
		self._next_entity_index = stop
		self._entity_count += quantity
		self._entity_version += 1

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
		self._entity_version += 1

	# ---------- components ----------

	def _prepare_components(
		self, entities: EntityLike, components: tuple[object, ...], *, replace: bool = False,
	) -> list[tuple[ComponentStorage, Mask, EntityArray, dict[str, FieldArray]]]:
		"""Normalize and validate the complete request without changing world state."""
		entity_array = as_entities(entities)
		if entity_array.size == 0 or not components:
			return []

		alive = self._alive_mask(entity_array)
		if not np.any(alive):
			return []

		live_entities = entity_array[alive]
		filter_values = not np.all(alive)
		prepared: dict[ComponentStorage, tuple[Mask, list[dict[str, FieldArray]]]] = {}
		for store, bit, payload in self._parse_add_components(components):
			multiple = isinstance(store, MultiComponentStorage)
			if replace and multiple:
				raise TypeError("set() does not support multicomponent storage")
			if store in prepared and not multiple:
				raise ValueError(f"repeated component type: {store.component_cls.__name__}")
			values = store._normalise_values(entity_array, payload)
			if filter_values:
				values = {field: value[alive] for field, value in values.items()}
			if store not in prepared:
				prepared[store] = (bit, [])
			prepared[store][1].append(values)

		result = []
		for store, (bit, batches) in prepared.items():
			owners = live_entities
			values = batches[0]
			if len(batches) > 1:
				owners = np.tile(live_entities, len(batches))
				values = {field: np.concatenate([batch[field] for batch in batches]) for field in store.fields}
			if replace:
				indices = _entity_indices(owners)
				if np.unique(indices).size != indices.size:
					raise ValueError(f"duplicate owners for {store.component_cls.__name__}")
				if np.any(store._get_dense_indices(indices) == NONE):
					raise ValueError(f"{store.component_cls.__name__} is missing; use add() to insert it")
			else:
				store._validate_add(owners)
			result.append((store, bit, owners, values))
		return result

	def add(self, entities: EntityLike, *components: object) -> None:
		"""Insert components after preflight. Single components must not already exist.

		Duplicate live owners and repeated single-component types are errors.
		Multicomponents append; stale handles and matching payload entries are ignored.
		Expected validation errors leave the whole request unapplied.
		"""
		for store, bit, owners, values in self._prepare_components(entities, components):
			store._add_normalised(owners, values)
			self.entity_masks[_entity_indices(owners)] |= bit

	def set(self, entities: EntityLike, *components: object) -> None:
		"""Replace existing single-component values using the same forms as add().

		All live owners must have each component. Stale handles are ignored.
		Rows and membership stay unchanged; writes advance the field versions.
		"""
		for store, _, owners, values in self._prepare_components(entities, components, replace=True):
			rows = store._get_rows(_entity_indices(owners))
			for field, value in values.items():
				store._dense[field][rows] = value
			store.touch()

	def remove(self, entities: EntityLike, *component_types: type[Any]) -> None:
		entity_array = np.unique(self._filter_alive(as_entities(entities)))
		if entity_array.size == 0 or not component_types:
			return

		indices = _entity_indices(entity_array)
		stores = [self._component_info(cls) for cls in dict.fromkeys(component_types)]
		for store, bit in stores:
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

		return self._matching_entities(required, excluded, Mask(0), (self._stores[cls] for cls in dict.fromkeys(component_types)))

	def _matching_entities(
		self, required: Mask, excluded: Mask, any_mask: Mask, required_stores: Iterable[ComponentStorage],
	) -> EntityArray:
		store = min(required_stores, key=lambda candidate: candidate._size, default=None)
		if store is not None and store._size * 16 <= self._next_entity_index:
			if isinstance(store, MultiComponentStorage):
				indices = np.unique(_entity_indices(store._dense_entities[store._active_rows()]))
			else:
				indices = np.sort(_entity_indices(store._dense_entities[:store._size]))
			# Sorting by index preserves ordering after swap removal and generation reuse.
			current, masks = self._current_entities[indices], self.entity_masks[indices]
		else:
			current = self._current_entities[:self._next_entity_index]
			masks = self.entity_masks[:current.size]
		selected = (current != NO_ENTITY) & ((masks & required) == required)
		if excluded:
			selected &= (masks & excluded) == 0
		if any_mask:
			selected &= (masks & any_mask) != 0
		return current[selected]
