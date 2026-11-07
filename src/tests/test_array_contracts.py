import numpy as np
import pytest
from numpyecs.ECS import ECS, array_field, component, columns


@component
class Position:
    value: np.ndarray = array_field(np.float32, shape=(3,))


@component
class Other:
    value: float


def world_data():
    world = ECS()
    positions, other = world.register(Position, Other)
    entities = world.create(3)
    world.add(entities, columns(Position, value=np.arange(9).reshape(3, 3)), columns(Other, value=1))
    return world, positions, other, entities


def test_shaped_metadata_and_reusable_gather_preserve_order_and_duplicates():
    world, positions, _, ids = world_data()
    selected = positions[ids[[2, 0, 2]]]
    out = np.empty((3, 3), dtype=np.float32)
    version = positions.version()
    assert selected.read("value", out=out) is out
    np.testing.assert_array_equal(out, [[6, 7, 8], [0, 1, 2], [6, 7, 8]])
    assert positions.version() == version
    selected.assign(value=out + 10)
    np.testing.assert_array_equal(positions[ids].value, [[10, 11, 12], [3, 4, 5], [16, 17, 18]])
    with pytest.raises(ValueError): selected.read("value", out=np.empty((3, 3)))
    with positions.edit_dense("value") as fields:
        with pytest.raises(ValueError, match="alias"):
            selected.read("value", out=fields["value"])


def test_dense_scope_updates_columns_and_invalidates_after_exceptions():
    world, positions, _, ids = world_data()
    version = positions.version("value")
    with pytest.raises(LookupError):
        with positions.edit_dense("value") as fields:
            fields["value"] += 2
            retained = fields["value"]
            raise LookupError()
    assert positions.version("value") != version
    assert not retained.flags.writeable
    np.testing.assert_array_equal(positions[ids].value, np.arange(9).reshape(3, 3) + 2)
    world.check_invariants()


@pytest.mark.parametrize("operation", ["delete", "remove", "add", "reserve", "nested"])
def test_borrow_blocks_structural_edits_before_other_stores_change(operation):
    world, positions, other, ids = world_data()
    fresh = world.create()
    with positions.edit_dense("value"):
        with pytest.raises(RuntimeError, match="edit_dense"):
            if operation == "delete": world.delete(ids)
            elif operation == "remove": world.remove(ids, Other, Position)
            elif operation == "add": world.add(fresh, columns(Other, value=2), columns(Position, value=0))
            elif operation == "reserve": positions.reserve(10000)
            else:
                with positions.edit_dense("value"): pass
        assert world.has(ids, Other).all()
        assert not world.has(fresh, Other).any()
        world.check_invariants()


def test_unsupported_annotations_need_explicit_object_storage():
    @component
    class Invalid:
        value: list

    with pytest.raises(TypeError, match="object"):
        ECS().register(Invalid)

    @component
    class Objects:
        value: object

    world = ECS()
    objects, = world.register(Objects)
    ids = world.create()
    world.add(ids, Objects([1, 2]))
    assert objects[ids].value[0] == [1, 2]
    with pytest.raises(TypeError):
        with objects.edit_dense(): pass


def test_multicomponent_dense_edit_requires_compaction_and_invalidates_selections():
    @component(multiples=True)
    class Multi:
        value: float

    world = ECS()
    values, = world.register(Multi)
    ids = world.create(3)
    world.add(ids, columns(Multi, value=[1, 2, 3]))
    world.remove(ids[1], Multi)
    selected = values[ids]
    with pytest.raises(RuntimeError, match="Compact"):
        with values.edit_dense(): pass
    values.compact()
    with pytest.raises(RuntimeError, match="stale"): selected.read("value")
    with values.edit_dense() as fields: fields["value"] *= 2
    np.testing.assert_array_equal(values[ids].value, [2, 6])


def test_default_instance_packing_produces_the_shader_bit_layout():
    from numpyecs.InstanceData import make_instances, MESH_INSTANCE_DTYPE
    position = dict(x=[1, 2], y=[3, 4], z=[5, 6])
    rotation = dict(x=[0, 0], y=[0, 0], z=[0, 0], w=[1, 1])
    scale = dict(x=[1, 2], y=[1, 3], z=[1, 4])
    records = make_instances(position, rotation, scale, {"value": np.array([123, 456], dtype=np.uint32)})
    assert records.dtype == MESH_INSTANCE_DTYPE
    np.testing.assert_array_equal(records["iPosition"], [[1, 3, 5], [2, 4, 6]])
    np.testing.assert_array_equal(records["iRotation"].copy().view(np.int16), [[0, 0, 0, 32767]] * 2)
    np.testing.assert_array_equal(records["iScale"].copy().view(np.float16), [[1, 1, 1, 0], [2, 3, 4, 0]])
    np.testing.assert_array_equal(records["iTint"][:, 0], [123, 456])
