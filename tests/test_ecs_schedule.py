import numpy as np
import pytest

from ECS import ECS, CommandError, INDEX_MAX, Scheduler, component


@component
class Position:
	x: float


@component
class Selected:
	pass


def world():
	ecs = ECS()
	ecs.register(Position, Selected)
	return ecs


def test_commands_are_fifo_and_do_not_invalidate_selections_before_flush():
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position(1))
	selection = ecs.get(Position)[ids]
	commands = ecs.commands()
	commands.set(ids, Position(2))
	commands.remove(ids[0], Position)
	commands.add(ids[0], Position(3), Selected)
	commands.delete(ids[1])
	assert selection.x.tolist() == [1, 1, 1]
	assert len(commands) == 4
	assert commands.flush() == 4
	assert len(commands) == 0
	assert commands.flush() == 0
	assert ecs.get(Position)[ids[[0, 2]]].x.tolist() == [3, 2]
	assert ecs.where(Selected).tolist() == ids[:1].tolist()
	with pytest.raises(RuntimeError, match="stale"):
		selection.x


@pytest.mark.parametrize("operation", ["add", "set", "remove", "delete"])
def test_queued_stale_handles_never_target_recycled_entities(operation):
	ecs = world()
	stale = ecs.create()
	ecs.add(stale, Position(1))
	commands = ecs.commands()
	args = () if operation == "delete" else (Position,) if operation == "remove" else (Position(99),)
	getattr(commands, operation)(stale, *args)
	ecs.delete(stale)
	current = ecs.create()
	ecs.add(current, Position(2))
	commands.flush()
	assert ecs.get(Position)[current].x.tolist() == [2]
	assert ecs.is_alive(current).all()


def test_command_targets_are_copied_but_payloads_are_retained_by_reference():
	ecs = world()
	ids = ecs.create(2)
	targets, values = ids.copy(), np.array([1., 2.])
	commands = ecs.commands()
	commands.add(targets, Position, values)
	targets[:] = ids[0]
	values[:] = [3, 4]
	commands.flush()
	assert ecs.get(Position)[ids].x.tolist() == [3, 4]


def test_failure_consumes_failed_command_and_preserves_applied_prefix_and_pending_tail():
	ecs = world()
	id_ = ecs.create()
	commands = ecs.commands()
	commands.add(id_, Position(1))
	commands.add(id_, Selected, Position(2))  # Entire command preflight must fail.
	commands.set(id_, Position(3))
	with pytest.raises(CommandError) as caught:
		commands.flush()
	assert caught.value.command_index == 1
	assert caught.value.operation == "add"
	assert isinstance(caught.value.__cause__, ValueError)
	assert ecs.get(Position)[id_].x.tolist() == [1]
	assert not ecs.has(id_, Selected).any()
	assert len(commands) == 1
	assert commands.flush() == 1
	assert ecs.get(Position)[id_].x.tolist() == [3]


def test_deferred_creation_resolution_and_buffer_ownership():
	ecs = world()
	commands = ecs.commands()
	token = commands.create(2)
	commands.add(token, Position, np.array([1., 2.]))
	commands.add(token, Selected)
	with pytest.raises(RuntimeError, match="unresolved"):
		token.resolve()
	with pytest.raises(ValueError, match="another command buffer"):
		ecs.commands().delete(token)
	assert len(ecs) == 0
	commands.flush()
	ids = token.resolve()
	assert ecs.get(Position)[ids].x.tolist() == [1, 2]
	ids[:] = 12345
	assert ecs.is_alive(token.resolve()).all()
	commands.delete(token)
	commands.flush()
	commands.add(token, Position(9))
	recycled = ecs.create(2)
	commands.flush()
	assert not ecs.has(recycled, Position).any()


def test_creation_tokens_survive_later_failures_and_clear_cancels_only_pending_creations():
	ecs = world()
	commands = ecs.commands()
	created = commands.create()
	commands.add(created, Position(1))
	commands.add(created, Position(2))
	with pytest.raises(CommandError):
		commands.flush()
	assert ecs.is_alive(created.resolve()).all()
	cancelled = commands.create()
	commands.add(cancelled, Selected)
	commands.clear()
	with pytest.raises(RuntimeError, match="cancelled"):
		cancelled.resolve()
	assert ecs.is_alive(created.resolve()).all()
	empty = commands.create(0)
	commands.flush()
	assert empty.resolve().size == 0
	with pytest.raises(ValueError):
		commands.create(-1)
	with pytest.raises(TypeError):
		commands.create(1.5)


def test_failed_creation_is_reported_and_never_resolves():
	ecs = world()
	ecs._next_entity_index = INDEX_MAX
	commands = ecs.commands()
	token = commands.create()
	commands.add(token, Selected)
	with pytest.raises(CommandError) as caught:
		commands.flush()
	assert isinstance(caught.value.__cause__, OverflowError)
	with pytest.raises(RuntimeError, match="failed"):
		token.resolve()
	assert len(commands) == 1
	commands.clear()


def test_flush_has_a_fixed_entry_batch_and_rejects_reentrancy():
	class HookWorld(ECS):
		def create(self, quantity=1):
			commands.create(0)
			with pytest.raises(RuntimeError, match="not reentrant"):
				commands.flush()
			with pytest.raises(RuntimeError, match="flushing"):
				commands.clear()
			return super().create(quantity)
	ecs = HookWorld()
	commands = ecs.commands()
	commands.create()
	assert commands.flush() == 1
	assert len(commands) == 1  # Work queued during execution waits for the next flush.
	commands.clear()


def test_scheduler_orders_phases_dependencies_and_registration_ties():
	ecs = world()
	schedule = Scheduler(ecs)
	seen = []
	def record(name):
		return lambda world, commands, dt: seen.append((name, dt))
	schedule.add("render", record("render"), phase="render_prep", after="last")
	schedule.add("last", record("last"), after="first")
	schedule.add("tie", record("tie"))
	schedule.add("first", record("first"), before="last")
	schedule.add("physics", record("physics"), phase="fixed_update")
	assert schedule.compile() == ("physics", "tie", "first", "last", "render")
	schedule.run(0.1, fixed_steps=2, fixed_dt=0.02)
	assert seen == [("physics", 0.02), ("physics", 0.02), ("tie", 0.1), ("first", 0.1), ("last", 0.1), ("render", 0.1)]


@pytest.mark.parametrize("flush_after", [False, True])
def test_visibility_boundaries_and_query_refresh(flush_after):
	ecs = world()
	schedule = Scheduler(ecs)
	query = ecs.query(Selected)
	observed = []
	schedule.commands.add(ecs.create(), Selected)  # Flushed before any system.
	def spawn(world, commands, dt):
		observed.append(("spawn", len(query.fetch())))
		commands.add(commands.create(), Selected)
	def observe(world, commands, dt):
		observed.append(("same-phase", len(query.fetch())))
	schedule.add("spawn", spawn, flush_after=flush_after)
	schedule.add("observe", observe, after="spawn")
	schedule.add("render", lambda w, c, dt: observed.append(("render", len(query.fetch()))), phase="render_prep")
	schedule.run(0)
	assert observed == [("spawn", 1), ("same-phase", 2 if flush_after else 1), ("render", 2)]


def test_each_fixed_tick_flushes_before_next_tick_and_update():
	ecs = world()
	schedule = Scheduler(ecs)
	seen = []
	def tick(world, commands, dt):
		seen.append(len(world))
		commands.create()
	schedule.add("fixed", tick, phase="fixed_update")
	schedule.add("update", lambda w, c, dt: seen.append(len(w)))
	schedule.run(0, fixed_steps=3)
	assert seen == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ["unknown", "cycle", "self", "backwards"])
def test_invalid_dependencies_fail_before_systems_or_queued_commands_run(kind):
	ecs = world()
	schedule = Scheduler(ecs)
	schedule.commands.create()
	if kind == "unknown":
		schedule.add("a", lambda *args: None, after="missing")
	elif kind == "cycle":
		schedule.add("a", lambda *args: None, after="b")
		schedule.add("b", lambda *args: None, after="a")
	elif kind == "self":
		schedule.add("a", lambda *args: None, before="a")
	else:
		schedule.add("a", lambda *args: None, phase="render_prep", before="b")
		schedule.add("b", lambda *args: None, phase="update")
	with pytest.raises(ValueError):
		schedule.run(0)
	assert len(ecs) == 0
	assert len(schedule.commands) == 1


def test_system_failure_keeps_commands_pending_and_stops_later_systems():
	ecs = world()
	schedule = Scheduler(ecs)
	def fail(world, commands, dt):
		commands.create()
		raise ValueError("system failed")
	schedule.add("fail", fail)
	schedule.add("later", lambda *args: pytest.fail("must not run"))
	with pytest.raises(ValueError, match="system failed"):
		schedule.run(0)
	assert len(ecs) == 0
	assert len(schedule.commands) == 1
	schedule.commands.flush()
	assert len(ecs) == 1


def test_schedule_reentrancy_mutation_and_input_validation():
	schedule = Scheduler(world())
	def check(world, commands, dt):
		with pytest.raises(RuntimeError, match="reentrant"):
			schedule.run(0)
		with pytest.raises(RuntimeError, match="running schedule"):
			schedule.add("new", check)
	schedule.add("check", check)
	schedule.run(0)
	for kwargs in ({"dt": -1}, {"dt": np.nan}, {"dt": 0, "fixed_steps": -1}, {"dt": 0, "fixed_dt": 0}):
		with pytest.raises(ValueError):
			schedule.run(**kwargs)
	with pytest.raises(ValueError):
		schedule.add("check", check)
	with pytest.raises(ValueError):
		schedule.add("bad", check, phase="missing")
	with pytest.raises(TypeError):
		schedule.add("bad", None)


def test_phase_flush_failure_stops_render_and_keeps_pending_tail():
	ecs = world()
	id_ = ecs.create()
	schedule = Scheduler(ecs)
	def queue(world, commands, dt):
		commands.add(id_, Position(1))
		commands.add(id_, Position(2))
		commands.add(id_, Selected)
	schedule.add("update", queue)
	schedule.add("render", lambda *args: pytest.fail("failed flush must stop render"), phase="render_prep")
	with pytest.raises(CommandError):
		schedule.run(0)
	assert ecs.get(Position)[id_].x.tolist() == [1]
	assert not ecs.has(id_, Selected).any()
	assert len(schedule.commands) == 1
	schedule.commands.flush()
	assert ecs.has(id_, Selected).all()


def test_compile_can_recover_from_forward_references_and_refreshes_after_add():
	schedule = Scheduler(world())
	schedule.add("last", lambda *args: None, after="first")
	with pytest.raises(ValueError, match="unknown"):
		schedule.compile()
	schedule.add("first", lambda *args: None)
	assert schedule.compile() == ("first", "last")
	schedule.add("middle", lambda *args: None, before="last", after="first")
	assert schedule.compile() == ("first", "middle", "last")
