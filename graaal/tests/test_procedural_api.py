"""Procedural API contracts; GPU commands are covered by the rendering suite."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from graaal import DrawBatches as batches
from graaal import RenderContext as rc
from graaal import TransparencyPass as transparency
from graaal.ECS import ECS, component, columns
from graaal.draw_resources import _DrawResources
from graaal.draw_culling import _CullingState, default_cull_shader
from graaal.HZB import HZB


@pytest.fixture
def draws(monkeypatch, tmp_path):
    device = NS(queue=Mock(), create_shader_module=Mock(),
        create_buffer=lambda **kw: NS(size=kw["size"]),
        create_buffer_with_data=lambda **kw: NS(size=kw["data"].nbytes))
    monkeypatch.setattr(rc.RenderContext, "device", device, raising=False)
    # Default engine shaders must also load from outside the repository.
    monkeypatch.chdir(tmp_path)
    draws = _DrawResources(np.dtype([("value", "u4")]))
    for mesh_id in (1, 2):
        mesh = NS(index_count=3, index_range=(0, 3), vertex_range=(0, 3))
        draws.register_mesh(mesh_id, mesh, bounds=([0, 0, 0], [1, 1, 1]))
        draws.register_lod_group(mesh_id * 10, (mesh_id,))
    draws.register_shader(0)
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
    rc.RenderContext.device.queue.write_buffer.reset_mock()
    assert not draws.sync_arrays(labels, lod_ids, [0, 0, 0, 0])
    assert draws.instance_order_version == version
    rc.RenderContext.device.queue.write_buffer.assert_not_called()
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


def frame_state(frame_fields=None):
    oit = transparency.TransparencyPass.__new__(transparency.TransparencyPass)
    oit.size = (32, 16)
    oit.noise_scale = 1.0
    oit.sample_count = 1
    oit.noise_mode = "uv_blue_lerp"
    oit.frame_fields = frame_fields or {}
    if frame_fields:
        shader = rc.Shader.__new__(rc.Shader)
        source, declarations = shader._render_source(None, str(transparency.SHADERS / "stochastic_geometry.shader"), None, None,
            {"frame_fields": {name: spec[0] for name, spec in frame_fields.items()}})
        dtype = rc.ShaderSource(source, declarations=declarations).uniforms["frame"].dtype
        oit.uniforms = NS(content=np.zeros(1, dtype=dtype), dtype=dtype, upload_changed=Mock())
    else:
        oit.uniforms = NS(content={}, upload_changed=Mock())
    lighting_dtype = np.dtype([("light_dir", np.float32, (4,))])
    oit.lighting_uniforms = NS(content=np.zeros(1, dtype=lighting_dtype), dtype=lighting_dtype, upload_changed=Mock())
    return oit


def test_frame_values_need_no_camera_and_preserve_matrix_convention():
    oit = frame_state({"time": ("f32", 0.0)})
    view, projection = np.eye(4), np.diag([2, 3, 4, 1])
    view[3, 0] = 5
    result = oit.update_frame(view=view, projection=projection,
        camera_position=[1, 2, 3], near=0.1, far=100, frame={"time": 2.5})
    np.testing.assert_array_equal(result, view @ projection)
    assert result.dtype == np.float32
    assert oit.uniforms.content["time"] == 2.5
    np.testing.assert_array_equal(oit.uniforms.content["camera_position"][0], [1, 2, 3, 1])
    oit.uniforms.upload_changed.assert_called_once()
    camera = NS(view=lambda: view, projection=Mock(return_value=projection),
        position=[1, 2, 3], near=0.1, far=100)
    np.testing.assert_array_equal(oit.update_camera(camera), result)
    camera.projection.assert_called_once_with(2)


@pytest.mark.parametrize("overrides", [
    {"frame": {"time": np.nan}}, {"frame": {"time": np.inf}}, {"frame": {"time": 1e40}},
    {"frame": {"time": 1j}}, {"frame": {"time": True}}, {"frame": {"other": 1}},
    {"view": np.eye(3)}, {"projection": np.full((4, 4), np.nan)},
    {"camera_position": [0, 0]},
    {"near": 0}, {"near": 100}, {"far": np.inf},
])
def test_invalid_frame_values_do_not_write_or_upload(overrides):
    oit = frame_state({"time": ("f32", 0.0)})
    values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
    values.update(overrides)
    with pytest.raises(ValueError):
        oit.update_frame(**values)
    assert not np.any(oit.uniforms.content.view(np.uint8))
    oit.uniforms.upload_changed.assert_not_called()


def test_lighting_updates_persist_and_camera_forwards_values():
	oit = frame_state({"time": ("f32", 0.0)})
	values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
	oit.update_frame(**values, lighting={"light_dir": (1, 2, 3)})
	np.testing.assert_array_equal(oit.lighting_uniforms.content["light_dir"][0], [1, 2, 3, 0])
	oit.lighting_uniforms.upload_changed.assert_called_once()
	oit.lighting_uniforms.upload_changed.reset_mock()
	oit.update_frame(**values)
	oit.lighting_uniforms.upload_changed.assert_not_called()
	np.testing.assert_array_equal(oit.lighting_uniforms.content["light_dir"][0], [1, 2, 3, 0])
	camera = NS(view=lambda: np.eye(4), projection=lambda _: np.eye(4), position=[0, 0, 0], near=0.1, far=100)
	oit.update_camera(camera, lighting={"light_dir": (0, 1, 0)})
	np.testing.assert_array_equal(oit.lighting_uniforms.content["light_dir"][0], [0, 1, 0, 0])
	oit.lighting_uniforms.upload_changed.assert_called_once()


def test_lighting_mapping_accepts_other_reflected_uniform_fields():
	oit = frame_state()
	dtype = np.dtype([("light_dir", np.float32, (4,)), ("color", np.float32, (3,))])
	oit.lighting_uniforms = NS(content=np.zeros(1, dtype=dtype), dtype=dtype, upload_changed=Mock())
	values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
	oit.update_frame(**values, lighting={"color": (0.7, 0.5, 0.3)})
	np.testing.assert_allclose(oit.lighting_uniforms.content["color"][0], [0.7, 0.5, 0.3])
	oit.lighting_uniforms.upload_changed.assert_called_once()


@pytest.mark.parametrize("updates", [
	{"light_dir": (1, 2)}, {"light_dir": (1, 2, 3, 4)},
	{"light_dir": (1, np.nan, 0)}, {"light_dir": (1e40, 0, 0)},
	{"light_dir": (True, False, True)}, {"light_dir": (1j, 2, 3)},
	{"light_dir": "abc"}, {"unknown": 1}, [("light_dir", (1, 2, 3))],
])
def test_invalid_lighting_does_not_update_either_uniform(updates):
	oit = frame_state({"time": ("f32", 0.0)})
	values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
	before = oit.lighting_uniforms.content.copy()
	with pytest.raises(ValueError):
		oit.update_frame(**values, frame={"time": 2.0}, lighting=updates)
	np.testing.assert_array_equal(oit.lighting_uniforms.content, before)
	assert not np.any(oit.uniforms.content.view(np.uint8))
	oit.uniforms.upload_changed.assert_not_called()
	oit.lighting_uniforms.upload_changed.assert_not_called()


def test_custom_frame_fields_share_a_uniform_layout_and_preserve_values():
    spec = {"wind": ("vec3f", (0, 0, 0)), "tick": ("u32", 7)}
    oit = frame_state(spec)
    for name, (_, default) in spec.items():
        oit.uniforms.content[name] = oit._frame_value(name, default)
    assert oit.uniforms.dtype.names[-2:] == ("wind", "tick")
    assert oit.uniforms.content["tick"] == 7
    values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
    oit.update_frame(**values, frame={"wind": (1.0, 2.0, 3.0), "tick": np.uint32(9)})
    np.testing.assert_array_equal(oit.uniforms.content["wind"][0], [1, 2, 3])
    assert oit.uniforms.content["tick"] == 9
    oit.update_frame(**values, frame={"wind": (0.5, 0.0, 0.0)})
    assert oit.uniforms.content["tick"] == 9


@pytest.mark.parametrize("updates", [
    {"wind": (1, 2)}, {"wind": (1, 2, np.nan)}, {"wind": (1e40, 2, 3)},
    {"wind": (True, False, True)}, {"tick": -1}, {"tick": 1.5}, {"tick": 2**32},
    {"no_such_field": 0},
])
def test_invalid_custom_frame_fields_leave_uniforms_unchanged(updates):
    oit = frame_state({"wind": ("vec3f", (0, 0, 0)), "tick": ("u32", 0)})
    values = dict(view=np.eye(4), projection=np.eye(4), camera_position=[0, 0, 0], near=0.1, far=100)
    before = oit.uniforms.content.copy()
    with pytest.raises(ValueError):
        oit.update_frame(**values, frame=updates)
    np.testing.assert_array_equal(oit.uniforms.content, before)
    oit.uniforms.upload_changed.assert_not_called()


def test_custom_frame_schema_rejects_reserved_names_and_malformed_specs():
    for spec in ({"view": ("f32", 0)}, {"time": "f32"}, {"other-field": ("f32", 0)}):
        with pytest.raises((TypeError, ValueError)):
            transparency.TransparencyPass(materials=None, base_textures=None, vertices=None, frame_fields=spec)


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


def test_cull_camera_uses_explicit_keywords(draws):
    draws = _CullingState(default_cull_shader(), draws.buffers)
    draws.update_cull_camera(camera_position=[0, 0, 0], view_projection=np.eye(4))
    assert np.isfinite(draws.camera_params_buffer.content["view_proj"]).all()
    with pytest.raises(TypeError):
        draws.update_cull_camera(camera_position=[0, 0, 0], view_projection=np.eye(4), typo=1)


def test_gpu_array_batches_cull_in_compact_order_from_another_directory(monkeypatch, tmp_path):
    from graaal.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale

    rc.RenderContext.setup_graphics_backend(True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batches, "GpuBuffer", lambda content, usage, **kw:
        rc.GpuBuffer(content, usage | rc.BufferUsage.COPY_SRC, **kw))
    draws = batches.DrawBatches(MESH_INSTANCE_DTYPE)
    shader = rc.Shader(filepath=str(transparency.SHADERS / "stochastic_geometry.shader"))
    pipeline = rc.RenderPipeline(shader, vertex_entry="vertex", fragment_entry="opaque")
    spec = batches.ShaderPass(pipeline)
    draws.register_shader(0, batches.RenderShader({"main": spec, "prepass": spec}))
    mesh = NS(index_count=3, index_range=(0, 3), vertex_range=(0, 3))
    draws.register_mesh(1, mesh, bounds=([0, 0, 0], [0.1, 0.1, 0.1]))
    for group_id in (10, 20):
        draws.register_lod_group(group_id, (1,))
    draws.sync([20, 10], [0, 0], labels=[7, 7])
    positions = np.array([[100, 0, 0.5], [0, 0, 0.5]], dtype=np.float32)
    draws.write_instance_fields(iPosition=positions[draws.instance_order], iTint=0,
        iRotation=pack_quaternion(np.array([0, 0, 0, 1], dtype=np.float32)),
        iScale=pack_scale(np.ones(3, dtype=np.float32)))
    visibility = draws.create_visibility()
    commands = rc.RenderContext.commands("array_culling")
    batches.FrustumCuller().record(commands, visibility, camera_position=[0, 0, 0], view_projection=np.eye(4))
    rc.RenderContext.submit(commands.finish())
    raw = rc.RenderContext.device.queue.read_buffer(visibility.commands.handle)
    indirect = np.frombuffer(raw, dtype=batches.INDIRECT_DTYPE, count=2)
    np.testing.assert_array_equal(indirect["instance_count"], [1, 0])
    # Compile both HZB entries from the engine path as well.
    hzb = HZB()
    assert hzb.resize((8, 8))


def test_resize_reports_unchanged_and_minimized_targets():
    oit = frame_state()
    assert oit.resize((32, 16)) is False
    assert oit.resize((0, 0)) is False
    assert oit.size == (32, 16)
    with pytest.raises(ValueError, match="integer"):
        oit.resize((32.5, 16))
