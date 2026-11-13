"""Small CPU helpers shared by simulation, math, and rendering."""

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import numpy as np


P = ParamSpec("P")
R = TypeVar("R")


def higher_pow2(n: int | np.uint32 | np.uint64) -> int:
	"""Return the smallest power of two strictly greater than n (zero gives one)."""
	return 1 << int(n).bit_length()


def cache_last(key: Callable[..., object]) -> Callable[[Callable[P, R]], Callable[P, R]]:
	"""Cache an instance method's most recent result separately for each instance.

	The key function receives the same arguments as the method, including self.
	When its value equals the last successful call's key, return the cached result;
	otherwise, run the method and replace that instance's cached key and result.
	Include every argument and piece of state that can affect the result in the key.

	Example::

		class Data:
		    def __init__(self, values):
		        self.values = np.asarray(values)
		        self.version = 0

		    @cache_last(lambda self, scale: (self.version, scale))
		    def scaled(self, scale):
		        return self.values * scale

		data = Data([1, 2, 3])
		result = data.scaled(2)
		assert data.scaled(2) is result
		data.values[0] = 10
		data.version += 1  # The next call recomputes after the data edit.

	Keys must be stable snapshots whose equality produces a single boolean; use
	tuples or version counters instead of NumPy arrays. Results are returned by
	reference, so edits to a cached result are visible on subsequent cache hits.
	"""
	missing = object()

	def decorator(fn: Callable[P, R]) -> Callable[P, R]:
		cache_key = f"__cache_key_{id(fn)}"
		cache_val = f"__cache_val_{id(fn)}"

		@wraps(fn)
		def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
			self = args[0]
			current_key = key(*args, **kwargs)

			cached = getattr(self, cache_key, missing)
			if cached == current_key:
				return getattr(self, cache_val)

			value = fn(*args, **kwargs)
			setattr(self, cache_key, current_key)
			setattr(self, cache_val, value)

			return value
		return wrapped
	return decorator
