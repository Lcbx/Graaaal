"""Procedural API contracts; GPU commands are covered by the rendering suite."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

import DrawBatches as batches
import RenderContext as rc
import TransparencyPass as transparency
from ECS import ECS, component, columns


@pytest.fixture
def draws(monkeypatch, tmp_path):
    device = NS(queue=Mock(), create_shader_module=Mock(),
        create_buffer=lambda **kw: NS(size=kw["size"]),
        create_buffer_with_data=lambda **kw: NS(size=kw["data"].nbytes))
    monkeypatch.setattr(rc.render_context, "device", device, raising=False)
    # Default engine shaders must also load from outside the repository.
    monkeypatch.chdir(tmp_path)
    draws = batches.DrawBatches(np.dtype([("value", "u4")]))
    pipeline = rc.RenderPipeline.__new__(rc.RenderPipeline)
    pipeline.shader = NS(bind_group=lambda group, **resources: NS(group=group, resources=resources))
    for mesh_id in (1, 2):
        mesh = NS(index_count=3, index_range=(0, 3), vertex_range=(0, 3))
        draws.register_mesh(mesh_id, mesh, bounds=([0, 0, 0], [1, 1, 1]))
        draws.register_lod_group(mesh_id * 10, (mesh_id,))
    draws.register_shader(0, batches.RenderShader(batches.ShaderPass(pipeline)))
    return draws


def test_array_grouping_keeps_duplicate_labels_and_aligns_instance_data(draws):
    labels = np.array([7, 7, 8, 9], dtype=np.uint64)
    lod_ids = np.array([20, 10, 20, 10])
    values = np.array([100, 200, 300, 400], dtype=np.uint32)
    assert draws.sync_arrays(labels, lod_ids, [0, 0, 0, 0])
    np.testing.assert_array_equal(draws.instance_order, [1, 3, 0, 2])
    np.testing.assert_array_equal(draws.entities, labels[draws.instance_order])
    draws.write_instance_fields(value=values[draws.instance_order])
    np.testing.assert_array_equal(draws.buffers["instances"].content["value"][:4], [200, 400, 100, 300])
    version = draws.instance_order_version
    rc.render_context.device.queue.write_buffer.reset_mock()
    assert not draws.sync_arrays(labels, lod_ids, [0, 0, 0, 0])
    assert draws.instance_order_version == version
    rc.render_context.device.queue.write_buffer.assert_not_called()
    # Mutating caller-owned keys must invalidate the retained snapshot.
    lod_ids[0] = 10
    assert draws.sync_arrays(labels, lod_ids, [0, 0, 0, 0])
    np.testing.assert_array_equal(draws.instance_order, [0, 1, 3, 2])
    assert draws.instance_order_version == version + 1
    with pytest.raises(RuntimeError, match="sync_batches"):
        draws.mesh_field("value")


def test_array_empty_invalidation_and_registry_changes(draws):
    assert draws.sync_arrays([1], [10], [0])
    version = draws.instance_order_version
    draws.register_lod_group(10, (1, 2), (30,))
    assert not draws.sync_arrays([1], [10], [0])
    assert len(draws.draw_batches) == 2
    assert draws.instance_order_version == version
    draws.invalidate_batches()
    assert draws.sync_arrays([1], [10], [0])
    assert draws.sync_arrays([], [], [])
    assert not draws.sync_arrays([], [], [])
    assert not len(draws.entities) and not draws.workgroup_count
    assert not draws.source_batches and not draws.draw_batches


@pytest.mark.parametrize("lod_ids,shader_ids,error", [
    ([10, 20], [0], ValueError), ([[10]], [0], ValueError),
    ([10.5], [0], TypeError), ([10], [0.5], TypeError),
    ([30], [0], KeyError), ([10], [3], KeyError),
])
def test_array_validation_preserves_current_selection(draws, lod_ids, shader_ids, error):
    draws.sync_arrays([5], [10], [0])
    version = draws.instance_order_version
    with pytest.raises(error):
        draws.sync_arrays([6], lod_ids, shader_ids)
    np.testing.assert_array_equal(draws.entities, [5])
    assert draws.instance_order_version == version
    assert not draws.sync_arrays([5], [10], [0])


def test_switching_between_arrays_and_ecs_invalidates_order(draws):
    @component
    class Transform:
        x: float

    @component(multiples=True)
    class MeshRef:
        lod_id: np.int32
        shader_id: np.int32

    world = ECS()
    transforms, refs = world.register(Transform, MeshRef)
    entities = world.create(2)
    world.add(entities, columns(Transform, x=0), columns(MeshRef, lod_id=[20, 10], shader_id=0))
    assert draws.sync_batches(entities, transforms, refs)
    np.testing.assert_array_equal(draws.mesh_field("lod_id"), [10, 20])
    version = draws.instance_order_version
    assert draws.sync_arrays(entities, [20, 10], [0, 0])
    assert draws.instance_order_version == version + 1
    assert draws.sync_batches(entities, transforms, refs)
    assert draws.instance_order_version == version + 2
    np.testing.assert_array_equal(draws.instance_order, [1, 0])
    np.testing.assert_array_equal(draws.mesh_field("lod_id"), [10, 20])


def frame_state():
    oit = transparency.TransparencyPass.__new__(transparency.TransparencyPass)
    oit.size = (32, 16)
    oit.uniforms = NS(content={}, upload=Mock())
    return oit


def test_frame_values_need_no_camera_and_preserve_matrix_convention():
    oit = frame_state()
    view, projection = np.eye(4), np.diag([2, 3, 4, 1])
    view[3, 0] = 5
    result = oit.update_frame(view=view, projection=projection,
        camera_position=[1, 2, 3], near=0.1, far=100)
    np.testing.assert_array_equal(result, view @ projection)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(oit.uniforms.content["camera_position"], [1, 2, 3, 1])
    oit.uniforms.upload.assert_called_once()
    camera = NS(view=lambda: view, projection=Mock(return_value=projection),
        position=[1, 2, 3], near=0.1, far=100)
    np.testing.assert_array_equal(oit.update_camera(camera), result)
    camera.projection.assert_called_once_with(2)


@pytest.mark.parametrize("overrides", [
    {"view": np.eye(3)}, {"projection": np.full((4, 4), np.nan)},
    {"camera_position": [0, 0]}, {"light_dir": [0, np.inf, 0]},
    {"near": 0}, {"near": 100}, {"far": np.inf},
])
def test_invalid_frame_values_do_not_write_or_upload(overrides):
    oit = frame_state()
    values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
    values.update(overrides)
    with pytest.raises(ValueError):
        oit.update_frame(**values)
    assert not oit.uniforms.content
    oit.uniforms.upload.assert_not_called()


@pytest.mark.parametrize("method,args", [
    ("opaque_pass", (None,)), ("guide_pass", (None,)),
    ("transmittance_pass", (None,)), ("selection_pass", (None,)),
    ("shade_winner", (None, None)), ("compose", (None, None)),
])
def test_passes_explain_missing_targets(method, args):
    oit = frame_state()
    oit.size = None
    with pytest.raises(RuntimeError, match="Resize"):
        getattr(oit, method)(*args)


def test_cull_camera_accepts_both_keyword_spellings(draws):
    draws.update_cull_camera(camera_position=[0, 0, 0], view_projection=np.eye(4))
    before = draws.camera_params_buffer.content.copy()
    draws.update_cull_camera(cameraPosition=[0, 0, 0], viewProjectionMatrix=np.eye(4))
    np.testing.assert_array_equal(draws.camera_params_buffer.content, before)
    with pytest.raises(TypeError):
        draws.update_cull_camera(camera_position=[0, 0, 0], cameraPosition=[1, 1, 1], view_projection=np.eye(4))
    with pytest.raises(TypeError):
        draws.update_cull_camera(camera_position=[0, 0, 0], view_projection=np.eye(4), typo=1)


def test_legacy_names_share_identity_and_writable_window_state():
    assert rc.RenderContext is rc.render_context
    assert rc.GC_Manager is rc.GCManager
    assert rc.Shader.UniformBuffer is rc.Shader.uniform_buffer
    assert batches.standard_RenderShader is batches.standard_render_shader
    context = rc._RenderContext()
    context.windowDimensions = (40, 30)
    assert context.window_dimensions == (40, 30)
    context.window_dimensions = (20, 10)
    assert context.windowDimensions == (20, 10)


def test_gpu_array_batches_cull_in_compact_order_from_another_directory(monkeypatch, tmp_path):
    from Utils import mesh_instance_dtype, pack_quaternion, pack_scale

    rc.render_context.setup_graphics_backend(True)
    monkeypatch.chdir(tmp_path)
    draws = batches.DrawBatches(mesh_instance_dtype)
    # Read back the actual GPU indirect commands, including frustum rejection.
    name = "prepass_draw_cmd"
    buffer = draws.buffers[name]
    draws.buffers[name] = rc.GpuBuffer(buffer.content,
        buffer.usage | rc.BufferUsage.COPY_SRC, upload=False)
    shader = rc.Shader(filepath=str(transparency.SHADERS / "stochastic_geometry.shader"))
    pipeline = rc.RenderPipeline(shader, vertex_entry="vertex", fragment_entry="opaque")
    spec = batches.ShaderPass(pipeline)
    draws.register_shader(0, batches.RenderShader(spec, spec))
    mesh = NS(index_count=3, index_range=(0, 3), vertex_range=(0, 3))
    draws.register_mesh(1, mesh, bounds=([0, 0, 0], [0.1, 0.1, 0.1]))
    for group_id in (10, 20):
        draws.register_lod_group(group_id, (1,))
    draws.sync_arrays([7, 7], [20, 10], [0, 0])
    positions = np.array([[100, 0, 0.5], [0, 0, 0.5]], dtype=np.float32)
    draws.write_instance_fields(iPosition=positions[draws.instance_order], iTint=0,
        iRotation=pack_quaternion(np.array([0, 0, 0, 1], dtype=np.float32)),
        iScale=pack_scale(np.ones(3, dtype=np.float32)))
    dummy = rc.Texture((1, 1), format="r32float", usage=rc.TextureUsage.TEXTURE_BINDING)
    draws.refresh_bindings(dummy.view())
    draws.update_cull_camera(camera_position=[0, 0, 0], view_projection=np.eye(4))
    commands = rc.render_context.commands("array_culling")
    draws.reset_draw_counts(commands)
    draws.cull_instances(commands, "frustum")
    rc.render_context.submit(commands.finish())
    raw = rc.render_context.device.queue.read_buffer(draws.buffers[name].handle)
    indirect = np.frombuffer(raw, dtype=batches.indirect_dtype, count=2)
    np.testing.assert_array_equal(indirect["instance_count"], [1, 0])
    # Compile both HZB entries from the engine path as well.
    hzb = batches.HZB()
    assert hzb.resize((8, 8))


def test_resize_reports_unchanged_and_minimized_targets():
    oit = frame_state()
    assert oit.resize((32, 16)) is False
    assert oit.resize((0, 0)) is False
    assert oit.size == (32, 16)
    with pytest.raises(ValueError, match="integer"):
        oit.resize((32.5, 16))
