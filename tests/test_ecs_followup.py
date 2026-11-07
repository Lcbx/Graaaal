import gc
import weakref
from functools import partial

import numpy as np
import pytest
from numpyecs import ECS as api
from numpyecs.ECS import ECS, Scheduler, component


@component
class Position:
    x: float
    y: float


@component(multiples=True)
class Item:
    value: float


def make_world():
    w = ECS()
    p, = w.register(Position, capacity=2)
    ids = w.create(3)
    w.add(ids, Position(1, 2))
    return w, p, ids


@pytest.mark.parametrize('count', [0, 1, 3])
def test_shaped_vector_roundtrip(count):
    Shaped = component(type('Shaped', (), {'__annotations__': {
        'v': np.dtype((np.float32, (3,))), 'm': np.dtype((np.float32, (2, 2))), 'x': float}}))
    w = ECS()
    a, = w.register(Shaped)
    ids = w.create(count)
    w.add(ids, Shaped, (np.arange(count*3).reshape(count, 3),
                       np.arange(count*4).reshape(count, 2, 2), 5))
    s = a[ids]
    before = {f: s[f] for f in ('v', 'm', 'x')}
    matrix = s.vector('m', 'x', 'v')
    assert matrix.shape == (count, 8)
    s.set_vector(matrix, 'm', 'x', 'v')
    for f in before:
        np.testing.assert_array_equal(s[f], before[f])


@pytest.mark.parametrize('quantity', [1.5, 1.0, '2', True, np.bool_(False), -1])
@pytest.mark.parametrize('recycle', [False, True])
def test_invalid_creation_preserves_lifecycle(quantity, recycle):
    w = ECS()
    ids = w.create(2)
    if recycle:
        w.delete(ids[:1])
    before = (w.entities().tolist(), len(w), w.entity_version, w._free_entities.copy(), w._next_entity_index)
    for create in (w.create, w.commands().create):
        with pytest.raises((TypeError, ValueError)):
            create(quantity)
        assert (w.entities().tolist(), len(w), w.entity_version, w._free_entities, w._next_entity_index) == before
    assert w.create(np.int64(1)).size == 1


@pytest.mark.parametrize('value,error', [(0.9, TypeError), ([1.0], TypeError), ('0', TypeError),
    (True, TypeError), ([-1], ValueError), ([2**64], OverflowError)])
def test_lossy_handles_rejected_without_mutation(value, error):
    w, p, ids = make_world()
    before = w.entities().copy()
    version = w.entity_version
    commands = w.commands()
    for action in (w.delete, w.is_alive, lambda v: commands.delete(v)):
        with pytest.raises(error):
            action(value)
    np.testing.assert_array_equal(w.entities(), before)
    assert w.entity_version == version
    assert len(commands) == 0


def test_full_uint64_handles_and_row_overflow():
    w, p, ids = make_world()
    with pytest.raises(OverflowError):
        p.remove(np.array([2**32], dtype=np.uint64))
    assert len(p[ids]) == 3
    assert api.as_entities([0, 2**64-1]).tolist() == [0, 2**64-1]
    assert not w.is_alive(2**64-1)[0]
    assert api.as_entities([]).dtype == np.uint64


@pytest.mark.parametrize('fields,value', [
    ((), [[99]]), ((), [[99, 1, 2]]), (('x', 'absent'), [[99, 1]]),
    ((), [['99', 'bad']]), ((), np.ones((2, 2)))])
def test_vector_failure_is_atomic(fields, value):
    w, p, ids = make_world()
    s = p[ids[:1]]
    before, token = s.vector(), p.version()
    with pytest.raises((ValueError, KeyError, TypeError)):
        s.set_vector(np.asarray(value), *fields)
    np.testing.assert_array_equal(s.vector(), before)
    assert p.version() == token


def test_failed_field_assignment_keeps_versions():
    w, p, ids = make_world()
    s = p[ids]
    token = p.version()
    with pytest.raises(ValueError):
        s.x = ['bad'] * 3
    assert p.version() == token
    assert s.x.tolist() == [1, 1, 1]


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('edit', ['delete', 'remove', 'partial', 'grow', 'compact', 'rebuild'])
def test_nested_object_references_are_released(multi, edit):
    dtype = np.dtype([('nested', [('payload', object)]), ('shaped', object, (2,))])
    Record = component(type('Record', (), {'__annotations__': {'record': dtype}}), multiples=multi)
    class Payload:
        pass
    w = ECS()
    a, = w.register(Record, capacity=2)
    ids = w.create(2)
    obj = Payload()
    ref = weakref.ref(obj)
    values = np.zeros(2, dtype=dtype)
    values['nested']['payload'][0] = obj
    values['shaped'][0, 0] = obj
    w.add(ids, Record, values)
    del obj, values
    if edit == 'delete':
        w.delete(ids[:1])
    elif edit == 'partial':
        a.remove(a.rows()[:1])
    else:
        w.remove(ids[:1], Record)
        if edit == 'grow':
            a._store._grow_dense(20)
        elif edit == 'compact' and multi:
            a._store._defragment(a._store._capacity)
        elif edit == 'rebuild' and multi:
            w.add(ids[1:], Record, np.zeros(1, dtype=dtype))
    gc.collect()
    assert ref() is None
    assert len(a[ids[1:]]) >= 1


async def async_system(w, commands, dt):
    pass


async def async_generator_system(w, commands, dt):
    yield 1


def generator_system(w, commands, dt):
    yield 1


class AsyncSystem:
    async def __call__(self, w, commands, dt):
        pass


@pytest.mark.parametrize('callback', [async_system, async_generator_system, generator_system,
    partial(async_system), AsyncSystem(), AsyncSystem().__call__])
def test_scheduler_rejects_non_synchronous_systems(callback):
    schedule = Scheduler(ECS())
    schedule.add('valid', lambda w, c, dt: None)
    order = schedule.compile()
    with pytest.raises(TypeError, match='synchronous'):
        schedule.add('invalid', callback)
    assert schedule.compile() == order


def test_checked_row_edits_target_only_selected_instances():
    w = ECS()
    a, = w.register(Item)
    ids = w.create(2)
    w.add(np.repeat(ids, 2), Item, [1, 2, 3, 4])
    rows = a.query_rows(lambda value: value % 2 == 0)
    s = a.select_rows(rows)
    s.assign(value=[20, 40])
    assert a[ids].value.tolist() == [1, 20, 3, 40]
    s.take([1, 0, 1]).value += 1
    assert a[ids].value.tolist() == [1, 21, 3, 41]
    assert len(s.take([])) == 0
    assert s.take(np.array([False, True])).value.tolist() == [41]
    assert s.take(0).value.tolist() == [21]
    with pytest.raises(TypeError):
        a.select_rows([0])
    foreign, = w.register(Position)
    with pytest.raises(ValueError):
        foreign.select_rows(rows)
    w.delete(ids[:1])
    for action in (lambda: a.select_rows(rows), lambda: s.take(slice(None))):
        with pytest.raises(RuntimeError):
            action()


def test_named_columns_immediate_deferred_and_atomic_assign():
    w, p, ids = make_world()
    batch = api.columns(Position, x=[10, 20, 30], y=5)
    w.set(ids, batch)
    assert p[ids].vector().tolist() == [[10, 5], [20, 5], [30, 5]]
    token = p.version()
    with pytest.raises(ValueError):
        p[ids].assign(x=99, y='bad')
    assert p.version() == token
    assert p[ids].x.tolist() == [10, 20, 30]
    commands = w.commands()
    created = commands.create(2)
    commands.add(created, api.columns(Position, y=2, x=[8, 9]))
    commands.flush()
    assert p[created.resolve()].x.tolist() == [8, 9]
    with pytest.raises(ValueError):
        w.set(ids, api.columns(Position, x=3))
    with pytest.raises(KeyError):
        p[ids].assign(_entity=0)


def test_named_payload_tags_shapes_objects_stale_and_repeated_multi():
    @component
    class Tag:
        pass
    Shape = component(type('Shape', (), {'__annotations__': {
        'v': np.dtype((np.float32, (3,))), 'data': object}}))
    w = ECS()
    shape, tag, items = w.register(Shape, Tag, Item)
    ids = w.create(3)
    w.delete(ids[1:2])
    w.add(ids, api.columns(Shape, v=np.arange(9).reshape(3, 3), data=['a', 'b', 'c']),
          api.columns(Tag), api.columns(Item, value=1), api.columns(Item, value=[4, 5, 6]))
    assert shape[ids].v.tolist() == [[0, 1, 2], [6, 7, 8]]
    assert shape[ids].data.tolist() == ['a', 'c']
    assert len(tag[ids]) == 2
    assert items[ids].value.tolist() == [1, 4, 1, 6]
    fresh = w.create()
    with pytest.raises(ValueError):
        w.add(fresh, api.columns(Tag), api.columns(Shape, v=0, data='bad', extra=0))
    assert not w.has(fresh, Tag)[0]


def test_diagnostics_reservation_compaction_and_defensive_masks():
    w, p, ids = make_world()
    items, = w.register(Item, capacity=8)
    w.add(np.repeat(ids, 2), api.columns(Item, value=np.arange(6)))
    w.remove(ids[:1], Item)
    w.check_invariants()
    info = w.inspect()
    assert info['entities'] == 3
    assert items.inspect()['holes'] == 2
    expected_bytes = w._entity_masks.nbytes + w._current_entities.nbytes
    expected_bytes += sum(s._sparse.nbytes + sum(a.nbytes for a in s._dense.values()) for s in w._stores.values())
    assert info['array_bytes'] == expected_bytes
    info['components'][0]['fields']['x']['shape'] = 'changed'
    w.entity_masks[:] = 0
    assert w.has(ids, Position).all()
    selection = p[ids]
    token = (w.entity_version, p.version(), p.layout_version)
    w.reserve(100)
    p.reserve(0, sparse_capacity=100)
    assert (w.entity_version, p.version(), p.layout_version) == token
    assert len(selection) == 3
    p.reserve(100)
    assert p.version() == token[1]
    with pytest.raises(RuntimeError):
        selection.x
    before = items[ids].value.copy()
    saved = items.rows()
    membership = items.version()
    items.compact()
    assert items.inspect()['holes'] == 0
    assert items.version() == membership
    np.testing.assert_array_equal(items[ids].value, before)
    with pytest.raises(RuntimeError):
        len(saved)
    fresh = items.rows()
    items.compact()
    assert len(fresh) == 4
    w.check_invariants()


@pytest.mark.parametrize('corruption', ['mask', 'sparse', 'count', 'free', 'generation'])
def test_invariant_checker_detects_corruption(corruption):
    w, p, ids = make_world()
    w.check_invariants()
    if corruption == 'mask':
        w._entity_masks[0] = 0
    elif corruption == 'sparse':
        p._store._sparse[0] = 1
    elif corruption == 'count':
        w._entity_count += 1
    elif corruption == 'free':
        w._free_entities.append(ids[0])
    else:
        w._current_entities[0] += np.uint64(1 << 32)
    with pytest.raises(AssertionError):
        w.check_invariants()


def test_seeded_named_mutations_and_row_writes_match_reference():
    rng = np.random.default_rng(71)
    w = ECS()
    items, = w.register(Item, capacity=1)
    reference = {}
    for step in range(160):
        if not reference or rng.random() < 0.2:
            owner = int(w.create()[0])
            reference[owner] = []
        else:
            owner = int(rng.choice(list(reference)))
        operation = int(rng.integers(5))
        if operation == 0:
            w.delete(owner)
            del reference[owner]
        elif operation == 1:
            w.add(owner, api.columns(Item, value=step))
            reference[owner].append(float(step))
        elif operation == 2:
            selected = items.select_rows(items.query_rows(lambda value: value % 2 == 0, owner))
            selected.assign(value=selected.value + 1)
            reference[owner] = [v + 1 if v % 2 == 0 else v for v in reference[owner]]
        elif operation == 3:
            items.remove(items.query_rows(lambda value: value % 3 == 0, owner))
            reference[owner] = [v for v in reference[owner] if v % 3 != 0]
        else:
            items.compact()
        w.check_invariants()
        for owner, values in reference.items():
            assert items[owner].value.tolist() == values
