from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeAlias
import numpy as np
import inspect
try:
	from mypy_extensions import mypyc_attr
except ImportError:
	# Compilation support is optional for the NumPy-only runtime.
	def mypyc_attr(*attrs: str, **options: object) -> Any:  # type: ignore[misc]
		return lambda cls: cls
from numpy.typing import NDArray

from ecs_types import (
	Entity, Index, Generation, Mask, ComponentPerEntityCount,
	EntityArray, IndexArray, GenerationArray, MaskArray, CountArray, BoolArray, FieldArray,
	EntityLike, IndexLike, INDEX_MAX, INDEX_BITS, INDEX_MASK, GENERATION_INCREMENT,
	NONE, NO_ENTITY, INITIAL_CAPACITY, OBJECT_DTYPE,
	component, higher_pow2, entity_index, entity_generation, as_entities, as_rows, arg_names,
)
from ecs_types import _quantity, _entity_indices, _clear_object_references, _predicate_parameters, _dtype_for, _ENTITY_STR, _COUNT_STR
from ecs_schema import ComponentColumns, ComponentSchema, columns
from ecs_commands import CommandBuffer, CommandError, DeferredEntities
from ecs_schedule import Scheduler
from ecs_diagnostics import storage_info, world_info, check_world


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

	def take(self, key: Any) -> ComponentSelection:
		"""Slice/filter positions, preserving duplicates and lifetime checks."""
		self._validate()
		return ComponentSelection(self._store, as_rows(self._rows[key]))

	def assign(self, **fields: Any) -> None:
		"""Assign fields atomically; repeated rows use NumPy last-write semantics."""
		self._validate()
		self._store._write_fields(self._rows, fields)

	def vector(self, *field_names: str) -> FieldArray:
		"""Pack numeric fields in C order, using NumPy dtype promotion."""
		self._validate()
		field_names = field_names or self._store.fields
		if not field_names:
			return np.empty((len(self), 0))
		widths = self._store.schema.vector_widths(field_names)
		return np.concatenate([self[field].reshape(len(self), width) for field, width in zip(field_names, widths)], axis=1)

	def set_vector(self, value : FieldArray, *field_names: str) -> None:
		self._validate()
		field_names = field_names or self._store.fields
		widths = self._store.schema.vector_widths(field_names)
		value = np.asarray(value)
		if not field_names and value.shape != (len(self), 0):
			raise ValueError("tag vectors must have shape (number of rows, 0)")
		if value.shape != (len(self), sum(widths)):
			raise ValueError(f"vector expects shape {(len(self), sum(widths))}, got {value.shape}")
		values = {}
		start = 0
		for field, width in zip(field_names, widths):
			shape = self._store._dense[field].shape[1:]
			values[field] = value[:, start:start + width].reshape((len(self),) + shape)
			start += width
		self._store._write_fields(self._rows, values)

	def __getitem__(self, field: str) -> FieldArray:
		self._validate()
		return self._store._dense[field][self._rows]

	def __setitem__(self, field: str, value: Any) -> None:
		self._validate()
		self._store._write_fields(self._rows, {field: value})

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
		if name not in self._store.fields: raise AttributeError(name)
		self._store._write_fields(self._rows, {name: value})


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

	def inspect(self) -> dict[str, Any]:
		"""Array storage bytes exclude referenced object graphs and Python overhead."""
		return storage_info(self._store)

	def reserve(self, capacity: object, *, sparse_capacity: object = None) -> None:
		"""Reserve dense rows/owner indices. Dense reallocation expires selections."""
		capacity = _quantity(capacity)
		sparse = None if sparse_capacity is None else _quantity(sparse_capacity)
		if capacity > INDEX_MAX or (sparse is not None and sparse > INDEX_MAX):
			raise OverflowError("component capacity exceeds index space")
		self._store._grow_dense(capacity)
		if sparse is not None: self._store._grow_sparse(sparse)

	def compact(self) -> None:
		"""Pack multicomponent holes, retaining capacity; relocation expires selections."""
		if isinstance(self._store, MultiComponentStorage):
			self._store._defragment(self._store._live_count)

	def get_rows(self) -> IndexArray:
		"""Return ephemeral raw row indices; rows() retains lifetime checks."""
		return self._store._active_rows()

	def rows(self) -> ComponentRows:
		"""Select all active rows with storage identity and layout checks."""
		return ComponentRows(self._store, self._store._active_rows())

	def select_rows(self, rows: ComponentRows | ComponentSelection) -> ComponentSelection:
		"""Select exact checked instances without expanding owner membership."""
		if not isinstance(rows, _RowSelection): raise TypeError("select_rows requires checked component rows")
		rows._validate(self._store)
		return ComponentSelection(self._store, rows._rows)

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

		self.schema = ComponentSchema(component_cls)
		self.fields = self.schema.fields
		self.membership_version = 0
		self.layout_version = 0
		self.field_versions = {name: 0 for name in self.fields}
		self._field_dtypes = self.schema._field_dtypes.copy()

		self._capacity = max(1, int(capacity))
		self._size = 0
		self._sparse = np.full(self._capacity, NONE, dtype=Index)

		self._dense: dict[str, FieldArray] = {}
		self._field_dtypes[_ENTITY_STR] = _dtype_for(Entity)

		for name, dtype in self._field_dtypes.items():
			array = np.empty(self._capacity, dtype=dtype)
			if name == _ENTITY_STR:
				array.fill(NO_ENTITY)
			elif dtype.hasobject:
				_clear_object_references(array, slice(None))
			self._dense[name] = array

	def touch(self, *field_names: str) -> None:
		fields = field_names or self.fields
		for name in fields:
			if name not in self.field_versions: raise KeyError(name)
		for name in fields:
			self.field_versions[name] += 1

	def _write_fields(self, rows: IndexArray, values: Mapping[str, Any]) -> None:
		prepared: dict[str, FieldArray] = {}
		for field, value in values.items():
			if field not in self.fields: raise KeyError(field)
			array = self._dense[field]
			shape = (rows.size,) + array.shape[1:]
			if array.dtype.hasobject or array.dtype.names:
				ready = np.empty(shape, dtype=array.dtype)
				ready[...] = value
			else:
				ready = np.asarray(value, dtype=array.dtype)
				if ready.shape != shape: ready = np.broadcast_to(ready, shape)
			prepared[field] = ready
		self._commit_fields(rows, prepared)

	def _commit_fields(self, rows: IndexArray, values: Mapping[str, FieldArray]) -> None:
		for field, value in values.items():
			self._dense[field][rows] = value
		if rows.size:
			for field in values: self.field_versions[field] += 1

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
			elif old.dtype.hasobject:
				_clear_object_references(new, slice(self._capacity, None))
			self._dense[name] = new
		self._capacity = new_capacity

	def _clear_rows(self, rows: IndexArray) -> None:
		self._dense_entities[rows] = NO_ENTITY
		for field in self.schema.object_fields:
			_clear_object_references(self._dense[field], rows)

	def _active_rows(self) -> IndexArray:
		return np.arange(self._size, dtype=Index)

	def _get_rows(self, indices: IndexArray) -> IndexArray:
		rows = self._get_dense_indices(indices)
		return rows[rows != NONE]

	# ---------- values ----------

	def _normalise_values(self, entities: EntityArray, values: object) -> dict[str, FieldArray]:
		return self.schema._normalise_values(entities, values)

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
			elif old.dtype.hasobject:
				_clear_object_references(new, slice(live, None))

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


@dataclass(frozen=True)
class _PreparedMutation:
	store: ComponentStorage
	bit: Mask
	owners: EntityArray
	values: dict[str, FieldArray]


@mypyc_attr(allow_interpreted_subclasses=True)
class ECS:
	"""Explicit world state for procedural systems operating on NumPy arrays.

	Pass a world, accessors, or queries to ordinary functions. Scheduler and
	CommandBuffer are optional; no system base class or global world is required.
	"""
	def __init__(self) -> None:
		self._stores: dict[type[Any], ComponentStorage] = {}
		self._comp_bits: dict[type[Any], Mask] = {}

		self._entity_masks = np.zeros(0, dtype=Mask)
		self._current_entities = np.full(0, NO_ENTITY, dtype=Entity)
		self._free_entities: list[Entity] = []
		self._next_entity_index = 0
		self._entity_count = 0
		self._entity_version = 0

	def __len__(self) -> int:
		return self._entity_count

	@property
	def entity_masks(self) -> MaskArray:
		"""Defensive inspection snapshot; membership edits must use world APIs."""
		return self._entity_masks.copy()

	def inspect(self) -> dict[str, Any]:
		"""Snapshot counts, schemas, fragmentation, and array bytes (excluding objects)."""
		return world_info(self)

	def check_invariants(self) -> None:
		"""Explicit debug audit; raises AssertionError on inconsistent world storage."""
		check_world(self)

	def reserve(self, capacity: object) -> None:
		"""Reserve entity slots without creating entities or changing lifecycle tokens."""
		capacity = _quantity(capacity)
		if capacity > INDEX_MAX: raise OverflowError("entity capacity exceeds index space")
		self._grow_entity_state(capacity)

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
		present[present] &= (self._entity_masks[indices[present]] & bit) != 0
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
		if minimum_size <= self._entity_masks.size:
			return
		old_size = self._entity_masks.size
		new_size = max(higher_pow2(minimum_size), max(1, old_size * 2))

		masks = np.zeros(new_size, dtype=Mask)
		masks[:old_size] = self._entity_masks
		self._entity_masks = masks

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
		self._entity_masks[indices] &= ~self._component_info(component_cls)[1]

	def _parse_add_components(
		self,
		params: tuple[object, ...],
	) -> Iterator[tuple[ComponentStorage, Mask, object]]:
		i = 0

		while i < len(params):
			item = params[i]
			if isinstance(item, ComponentColumns):
				store, bit = self._component_info(item.component_cls)
				yield store, bit, item
				i += 1
				continue

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

	def create(self, quantity: object = 1) -> EntityArray:
		quantity = _quantity(quantity)
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
			self._entity_masks[indices] = Mask(0)

		if new_count == 0:
			self._entity_count += quantity
			self._entity_version += 1
			return reused

		self._grow_entity_state(stop)
		indices = np.arange(self._next_entity_index, stop, dtype=Index)
		fresh = indices.astype(Entity)
		self._current_entities[indices] = fresh
		self._entity_masks[indices] = Mask(0)
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
		masks = self._entity_masks[indices].copy()

		for component_cls, bit in self._comp_bits.items():
			selected = (masks & bit) != 0
			if np.any(selected):
				self._stores[component_cls]._remove_entities(indices[selected])

		self._entity_masks[indices] = Mask(0)
		self._current_entities[indices] = NO_ENTITY
		self._free_entities.extend(entity_array.tolist())
		self._entity_count -= int(entity_array.size)
		self._entity_version += 1

	# ---------- components ----------

	def _prepare_components(
		self, entities: EntityLike, components: tuple[object, ...], *, replace: bool = False,
	) -> list[_PreparedMutation]:
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
			result.append(_PreparedMutation(store, bit, owners, values))
		return result

	def add(self, entities: EntityLike, *components: object) -> None:
		"""Insert components after preflight. Single components must not already exist.

		Duplicate live owners and repeated single-component types are errors.
		Multicomponents append; stale handles and matching payload entries are ignored.
		Expected validation errors leave the whole request unapplied.
		"""
		for mutation in self._prepare_components(entities, components):
			mutation.store._add_normalised(mutation.owners, mutation.values)
			self._entity_masks[_entity_indices(mutation.owners)] |= mutation.bit

	def set(self, entities: EntityLike, *components: object) -> None:
		"""Replace existing single-component values using the same forms as add().

		All live owners must have each component. Stale handles are ignored.
		Rows and membership stay unchanged; writes advance the field versions.
		"""
		for mutation in self._prepare_components(entities, components, replace=True):
			rows = mutation.store._get_rows(_entity_indices(mutation.owners))
			mutation.store._commit_fields(rows, mutation.values)

	def remove(self, entities: EntityLike, *component_types: type[Any]) -> None:
		entity_array = np.unique(self._filter_alive(as_entities(entities)))
		if entity_array.size == 0 or not component_types:
			return

		indices = _entity_indices(entity_array)
		stores = [self._component_info(cls) for cls in dict.fromkeys(component_types)]
		for store, bit in stores:
			store._remove_entities(indices)
			self._entity_masks[indices] &= ~bit

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
			current, masks = self._current_entities[indices], self._entity_masks[indices]
		else:
			current = self._current_entities[:self._next_entity_index]
			masks = self._entity_masks[:current.size]
		selected = (current != NO_ENTITY) & ((masks & required) == required)
		if excluded:
			selected &= (masks & excluded) == 0
		if any_mask:
			selected &= (masks & any_mask) != 0
		return current[selected]
