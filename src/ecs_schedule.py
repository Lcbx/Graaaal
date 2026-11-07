from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from .ecs_commands import CommandBuffer
from .ecs_types import _synchronous_callable


if TYPE_CHECKING:
	from .ECS import ECS


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
		_synchronous_callable(callback, "system callback")
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


