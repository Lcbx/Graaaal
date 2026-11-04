from functools import partial

import numpy as np
import pytest

from ECS import ECS, component


@component
class Sleeping:
	pass


@component
class Selected:
	pass


@component
class Position:
	x: float


def world():
	ecs = ECS()
	ecs.register(Position, Sleeping, Selected, capacity=1)
	return ecs


def test_tag_forms_batch_parsing_and_owner_only_storage():
	ecs = world()
	ids = ecs.create(4)
	ecs.add(ids[:2], Sleeping, Position(1), Selected())
	ecs.add(ids[2:], Position, np.array([2., 3.]), Sleeping())
	assert ecs.where(Sleeping).tolist() == ids.tolist()
	assert ecs.where(Selected, Sleeping, Position).tolist() == ids[:2].tolist()
	assert ecs.get(Position)[ids].x.tolist() == [1, 1, 2, 3]
	store = ecs.get(Sleeping)._store
	assert store.fields == ()
	assert tuple(store._dense) == ("_entity",)
	assert ecs.get(Sleeping).version() == (store.membership_version,)
	selection = ecs.get(Sleeping)[ids]
	assert selection.vector().shape == (4, 0)
	selection.set_vector(np.empty((4, 0)))
	with pytest.raises(ValueError, match="tag vectors"):
		selection.set_vector(np.empty((4, 1)))


def test_tags_integrate_with_query_presence_offsets_and_cache_invalidation():
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position(1))
	ecs.add(ids[0], Sleeping)
	query = ecs.query(Position, exclude=Sleeping, optional=Selected)
	before = query.fetch()
	assert before.entities.tolist() == ids[1:].tolist()
	assert before[Selected].present.tolist() == [False, False]
	ecs.add(ids[1], Selected)
	after = query.fetch()
	assert after[Selected].present.tolist() == [True, False]
	assert after[Selected].offsets.tolist() == [0, 1, 1]
	assert len(after[Selected].selection) == 1
	with pytest.raises(RuntimeError, match="stale"):
		before[Selected]
	ecs.remove(ids[0], Sleeping)
	assert query.entities().tolist() == ids.tolist()
	assert ecs.query(any_of=(Selected, Sleeping)).entities().tolist() == ids[1:2].tolist()


def test_tag_remove_compaction_delete_and_id_recycling():
	ecs = world()
	ids = ecs.create(4)
	sleeping = ecs.get(Sleeping)
	ecs.add(ids, Sleeping)
	saved = sleeping[ids]
	sleeping.remove(sleeping.query_rows(lambda _entity: _entity == ids[1]))
	assert ecs.has(ids, Sleeping).tolist() == [True, False, True, True]
	with pytest.raises(RuntimeError, match="stale"):
		saved.get_rows()
	ecs.delete(ids[[0, 3]])
	reused = ecs.create(2)
	assert not ecs.has(reused, Sleeping).any()
	ecs.add(ids[[0, 3]], Sleeping)  # Stale handles cannot retag recycled entities.
	assert not ecs.has(reused, Sleeping).any()
	ecs.add(reused, Sleeping())
	assert len(sleeping.rows()) == 3
	assert set(map(int, sleeping.owner_ids(sleeping.rows()))) == {int(ids[2]), *map(int, reused)}
	ecs.remove(ecs.entities(), Sleeping)
	assert not ecs.where(Sleeping).size


@pytest.mark.parametrize("params", [(Sleeping, Position), (Position,), (Position, Sleeping)])
def test_missing_data_payloads_remain_errors_without_partial_tag_insertion(params):
	ecs = world()
	ids = ecs.create(2)
	tag = ecs.get(Sleeping)
	version = tag.version(), tag.layout_version
	with pytest.raises(TypeError, match="missing components"):
		ecs.add(ids, *params)
	assert not ecs.where(Sleeping).size
	assert not ecs.where(Position).size
	assert (tag.version(), tag.layout_version) == version


def test_duplicate_tags_are_rejected_and_set_only_validates_existing_membership():
	ecs = world()
	ids = ecs.create(2)
	ecs.add(ids[:1], Sleeping)
	for owners, args in ((ids, (Sleeping,)), (ids[1:], (Sleeping, Sleeping())), ([ids[1]] * 2, (Sleeping,))):
		with pytest.raises(ValueError):
			ecs.add(owners, *args)
	assert ecs.where(Sleeping).tolist() == ids[:1].tolist()
	tag = ecs.get(Sleeping)
	version = tag.version(), tag.layout_version
	ecs.set(ids[:1], Sleeping)
	assert (tag.version(), tag.layout_version) == version
	with pytest.raises(ValueError, match="missing"):
		ecs.set(ids[1:], Sleeping())


@pytest.mark.parametrize("option", [False, True])
def test_multicomponent_tags_fail_registration_atomically(option):
	@component(multiples=not option)
	class Invalid:
		pass
	ecs = ECS()
	with pytest.raises(TypeError, match="tags do not support"):
		ecs.register(Position, Invalid, allow_same_type_components_per_entity=option)
	assert not ecs._stores
	assert not ecs._comp_bits


def test_inspectable_predicate_functions_partials_bound_methods_and_callable_objects():
	ecs = world()
	ids = ecs.create(3)
	ecs.add(ids, Position, np.array([1., 2., 3.]))
	positions = ecs.get(Position)
	def above(x, *, threshold=1.5):
		return x > threshold
	class Check:
		def __call__(self, x):
			return x > 1.5
		def method(self, *, x):
			return x > 1.5
	for predicate in (above, partial(above, threshold=1.5), Check(), Check().method):
		assert positions.owner_ids(positions.query_rows(predicate)).tolist() == ids[1:].tolist()
	# Bind a non-field positional argument; its parameter disappears from the signature.
	def over(threshold, x):
		return x > threshold
	assert positions.owner_ids(positions.query_rows(partial(over, 2.5))).tolist() == ids[2:].tolist()
	assert len(positions.query_rows(lambda: True)) == 3
	assert len(positions.query_rows(lambda: False)) == 0
	# Fields are supplied even when their parameters have defaults.
	assert len(positions.query_rows(lambda x=0: x > 0)) == 3
	assert len(positions.query_rows(lambda x: np.isfinite(x))) == 3


def positional_only(x, /):
	return x > 0


async def async_predicate(x):
	return x > 0


def generator_predicate(x):
	yield x > 0


class Uninspectable:
	__signature__ = "invalid signature"
	def __call__(self, x):
		return x > 0


class AsyncCheck:
	async def __call__(self, x):
		return x > 0


@pytest.mark.parametrize("predicate, message", [
	(np.isfinite, "inspectable signature|named parameters"),
	(Uninspectable(), "inspectable signature"),
	(AsyncCheck(), "synchronously"),
	(3, "must be callable"),
	(positional_only, "named parameters"),
	(lambda *args: True, "named parameters"),
	(lambda **kwargs: True, "named parameters"),
	(lambda missing: True, "not component fields"),
	(async_predicate, "synchronously"),
	(generator_predicate, "synchronously"),
])
@pytest.mark.parametrize("empty", [True, False])
def test_invalid_predicate_signatures_have_clear_errors_even_on_empty_storage(predicate, message, empty):
	ecs = world()
	if not empty:
		ecs.add(ecs.create(), Position(1))
	with pytest.raises(TypeError, match=message):
		ecs.get(Position).query_rows(predicate)


def test_empty_queries_validate_without_calling_predicates_and_tags_accept_no_field_predicates():
	ecs = world()
	def never_called(x):
		raise AssertionError("an empty storage should not invoke the predicate")
	assert len(ecs.get(Position).query_rows(never_called)) == 0
	ids = ecs.create(2)
	ecs.add(ids, Sleeping)
	tags = ecs.get(Sleeping)
	assert tags.owner_ids(tags.query_rows(lambda: True)).tolist() == ids.tolist()
	assert len(tags.query_rows(lambda _entity: _entity == ids[1])) == 1
