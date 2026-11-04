from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ecs_types import EntityArray, EntityLike, as_entities, _quantity

if TYPE_CHECKING:
	from ECS import ECS


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
		quantity = _quantity(quantity)
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


