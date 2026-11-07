"""Shared utility contracts for capacity growth and per-instance caching."""

from numpyecs.utils import cache_last, higher_pow2


def test_higher_pow2() -> None:
	for i in range(0, 512):
		res = higher_pow2(i)
		assert res >= max(1, i)
		assert res & (res - 1) == 0


class A:
	def __init__(self, state : int = 0):
		self.state = state
		self.calls = 0

	@cache_last(lambda self, x: (self.state, x))
	def compute(self, x):
		self.calls += 1
		return self.state + x


def test_cache_last():
	a = A()

	assert a.compute(1) == 1
	assert a.calls == 1

	# Same key: cached
	assert a.compute(1) == 1
	assert a.calls == 1

	# Parameter changed: recompute
	assert a.compute(2) == 2
	assert a.calls == 2

	# Instance state changed: recompute
	a.state = 10
	assert a.compute(2) == 12
	assert a.calls == 3

	# Same new key: cached again
	assert a.compute(2) == 12
	assert a.calls == 3


def test_cache_last_is_per_instance():

	a = A(10)
	b = A(20)

	assert a.compute(1) == 11
	assert b.compute(2) == 22
	assert b.compute(1) == 21

	assert a.calls == 1
	assert b.calls == 2

	# Each should hit its own cache.
	assert a.compute(1) == 11
	assert b.compute(1) == 21

	assert a.calls == 1
	assert b.calls == 2
