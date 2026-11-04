from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import numpy as np
from ecs_types import EntityArray, FieldArray, OBJECT_DTYPE, _component_fields, _dtype_for, _ENTITY_STR, _COUNT_STR


@dataclass(frozen=True)
class ComponentColumns:
	"""Named insertion/replacement payload; field values are retained by reference."""
	component_cls: type[Any]
	values: Mapping[str, Any]


def columns(component_cls: type[Any], /, **fields: Any) -> ComponentColumns:
	"""Build a complete named batch, e.g. columns(Position, x=xs, y=ys)."""
	if not isinstance(component_cls, type): raise TypeError("columns requires a component type")
	return ComponentColumns(component_cls, MappingProxyType(fields))


class ComponentSchema:
	"""Resolved field metadata and payload conversion, independent of storage layout."""
	def __init__(self, component_cls: type[Any]) -> None:
		self.component_cls = component_cls
		definitions = _component_fields(component_cls)
		self.fields = tuple(name for name, _ in definitions)
		reserved = {_ENTITY_STR, _COUNT_STR}.intersection(self.fields)
		if reserved: raise ValueError(f"reserved component fields: {', '.join(sorted(reserved))}")
		self._field_dtypes = {name: _dtype_for(annotation) for name, annotation in definitions}
		self.shapes = {name: dtype.subdtype[1] if dtype.subdtype else () for name, dtype in self._field_dtypes.items()}
		self.object_fields = tuple(name for name, dtype in self._field_dtypes.items() if dtype.hasobject)

	def vector_widths(self, fields: Sequence[str]) -> tuple[int, ...]:
		if len(set(fields)) != len(fields): raise ValueError("duplicate vector fields")
		widths = []
		for field in fields:
			if field not in self.fields: raise KeyError(field)
			dtype = self._field_dtypes[field]
			base = dtype.subdtype[0] if dtype.subdtype else dtype
			if base.kind not in "biufc": raise TypeError("vectors require numeric or boolean fields")
			widths.append(int(np.prod(self.shapes[field], dtype=np.int64)))
		return tuple(widths)

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
		if isinstance(values, ComponentColumns):
			if values.component_cls is not self.component_cls: raise TypeError("column payload belongs to a different component")
			if set(values.values) != set(self.fields):
				raise ValueError(f"{self.component_cls.__name__} requires exactly fields {self.fields}")
			return {field: self._broadcast_field(field, values.values[field], entities.size) for field in self.fields}
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

