from __future__ import annotations

import inspect
from collections import deque
from dataclasses import (
    dataclass,
    field as dataclass_field,
    fields as dataclass_fields,
    is_dataclass,
)
from enum import IntFlag
from operator import index as integer_index
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
    cast,
    get_origin,
    get_type_hints,
)

import numpy as np
from numpy.typing import NDArray

from .utils import higher_pow2


def component(cls: type | None = None, *, multiples: bool = False) -> Any | Callable:
	def decorator(target: Any) -> Any:
		setattr(target, "_multicomponent", multiples)
		return dataclass(target)
	if cls is None:
		return decorator
	return decorator(cls)


def array_field(dtype: Any, *, shape: Sequence[int] = ()) -> Any:
	"""Declare a dataclass column, e.g. position: np.ndarray = array_field('f4', shape=(3,))."""
	shape = tuple(integer_index(size) for size in shape)
	if any(size <= 0 for size in shape): raise ValueError("Field dimensions must be positive")
	base = np.dtype(dtype)
	if base.hasobject and shape: raise TypeError("Shaped object fields are not supported")
	return dataclass_field(metadata={"numpyecs.dtype": np.dtype((base, shape)) if shape else base})


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


def entity_index(entity: Entity | EntityArray) -> Index | IndexArray:
	result = as_entities(entity)
	if np.ndim(entity) == 0: return Index(result[0] & INDEX_MASK)
	return _entity_indices(result)

def entity_generation(entity: Entity | EntityArray) -> Generation | GenerationArray:
	result = as_entities(entity) >> Entity(INDEX_BITS)
	if np.ndim(entity) == 0: return Generation(result[0])
	return result.astype(Generation, copy=False)

def _entity_indices(entities: EntityArray) -> IndexArray:
	return (entities & INDEX_MASK).astype(Index, copy=False)

def as_entities(entities: EntityLike) -> EntityArray:
	return cast(EntityArray, _as_1D_dtype(entities, Entity))

def as_rows(rows: IndexLike) -> IndexArray:
	return cast(IndexArray, _as_1D_dtype(rows, Index))

def _as_1D_dtype(array:Any, ndtype:type) -> FieldArray:
	# Object conversion for Python sequences avoids float promotion of mixed-sized IDs.
	array = np.asarray(array) if isinstance(array, np.ndarray) else np.asarray(array, dtype=object)
	if array.ndim == 0: array = array.reshape(1)
	if array.ndim != 1: raise ValueError("parameter must be a scalar or a 1-D array")
	if not array.size: return np.empty(0, dtype=ndtype)
	if array.dtype == np.dtype(ndtype): return array
	limit = int(np.iinfo(ndtype).max)
	if array.dtype.kind in "iu":
		if int(array.min()) < 0: raise ValueError("indices and handles must be nonnegative")
		if int(array.max()) > limit: raise OverflowError("index or handle exceeds storage dtype")
	elif array.dtype.kind == "O":
		values = []
		for value in array:
			if isinstance(value, (bool, np.bool_)): raise TypeError("indices and handles must be integers, not booleans")
			value = integer_index(value)
			if value < 0: raise ValueError("indices and handles must be nonnegative")
			if value > limit: raise OverflowError("index or handle exceeds storage dtype")
			values.append(value)
		return np.asarray(values, dtype=ndtype)
	else:
		raise TypeError("indices and handles must be integers")
	return array.astype(ndtype, copy=False)


def _quantity(value: Any) -> int:
	"""Creation/reservation counts accept indexable integers, excluding booleans."""
	if isinstance(value, (bool, np.bool_)): raise TypeError("quantity must be an integer, not a boolean")
	quantity = integer_index(value)
	if quantity < 0: raise ValueError("quantity must be >= 0")
	return quantity


def _synchronous_callable(func: Callable[..., Any], label: str) -> None:
	if not callable(func): raise TypeError(f"{label} must be callable")
	for target in (func, getattr(func, "__call__", None)):
		if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target) or inspect.isgeneratorfunction(target):
			raise TypeError(f"{label} must execute synchronously")


def _clear_object_references(array: FieldArray, rows: Any) -> None:
	"""Clear object leaves through field views, including nested records/subarrays."""
	if not array.dtype.hasobject: return
	if array.dtype.names:
		for name in array.dtype.names:
			_clear_object_references(array[name], rows)
	else:
		array[rows] = None

def arg_names(func:Callable)->Sequence[str]:
	return tuple(param.name for param in _predicate_parameters(func))

def _predicate_parameters(func: Callable[..., Any]) -> tuple[inspect.Parameter, ...]:
	"""Validate a synchronous predicate that can receive component fields by name."""
	_synchronous_callable(func, "query predicate")
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
			(field.name, field.metadata.get("numpyecs.dtype", hints.get(field.name, field.type)))
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
		dtype = np.dtype(annotation)
	except (TypeError, ValueError):
		raise TypeError(f"Unsupported component annotation {annotation!r}; declare a NumPy dtype or explicit object field") from None
	if dtype.hasobject and annotation is not object and not isinstance(annotation, np.dtype):
		raise TypeError(f"Unsupported component annotation {annotation!r}; use object for Python data")
	return dtype


