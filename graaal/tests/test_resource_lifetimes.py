from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from graaal import RenderContext as rc
from graaal.MeshArena import MeshArena


@pytest.fixture
def device(monkeypatch):
    device = NS(queue=Mock(), create_shader_module=Mock(), create_buffer=lambda **kw: NS(size=kw["size"]),
        create_buffer_with_data=lambda **kw: NS(size=max(4, kw["data"].nbytes)))
    monkeypatch.setattr(rc.RenderContext, "device", device, raising=False)
    return device


def test_buffer_capacity_active_count_and_gpu_only_mirror(device):
    buffer = rc.GpuBuffer(np.arange(4, dtype=np.uint32), rc.BufferUsage.COPY_DST)
    assert buffer.count == 4 and buffer.binding_size == 16
    assert buffer.reserve(20)
    assert buffer.capacity_bytes >= 80 and buffer.count == 4
    np.testing.assert_array_equal(buffer.content, [0, 1, 2, 3])
    buffer.set_count(2)
    assert buffer.binding["size"] == 8
    assert buffer.resource_key[1] == 8
    gpu = rc.GpuBuffer(None, rc.BufferUsage.STORAGE, dtype=np.float32, count=16)
    assert gpu.content is None and gpu.count == 16 and gpu.capacity_bytes == 64
    with pytest.raises(RuntimeError, match="GPU-only"): gpu.upload()
    with pytest.raises(ValueError, match="COPY"): gpu.reserve(32)


def test_buffers_reject_other_device_and_generation_even_without_growth(device, monkeypatch):
    buffer = rc.GpuBuffer(np.zeros(4, dtype=np.uint32), rc.BufferUsage.COPY_DST)
    monkeypatch.setattr(rc.RenderContext, "device_generation", rc.RenderContext.device_generation + 1)
    with pytest.raises(RuntimeError, match="generation"): buffer.upload()
    with pytest.raises(RuntimeError, match="generation"): buffer.resize(1)
    with pytest.raises(RuntimeError, match="generation"): _ = buffer.binding


def test_resolve_attachment_rejects_expired_device_generation(device, monkeypatch):
    monkeypatch.setattr(device, "create_texture",
        lambda **kwargs: NS(create_view=lambda: object()), raising=False)
    target = rc.Texture((16, 16), format="rgba8unorm", usage=rc.TextureUsage.RENDER_ATTACHMENT)
    attachment = target.color_attachment()
    monkeypatch.setattr(rc.RenderContext, "device_generation", rc.RenderContext.device_generation + 1)
    source = rc.Texture((16, 16), format="rgba8unorm",
        usage=rc.TextureUsage.RENDER_ATTACHMENT, sample_count=4)
    with pytest.raises(RuntimeError, match="expired device generation"):
        source.color_attachment(resolve_target=attachment)


def test_mesh_release_reuses_capacity_and_compaction_updates_live_ranges(device):
    arena = MeshArena()
    vertices = np.zeros(3, dtype=[("position", "f4", 3)])
    indices = np.arange(3, dtype=np.uint32)
    first = rc.Mesh(vertices, indices, arena=arena)
    second = rc.Mesh(vertices, indices, arena=arena)
    before = arena.inspect()["capacity_bytes"]
    old_buffer = first.vertex_buffer
    first.release()
    first.release()
    with pytest.raises(RuntimeError, match="released"): _ = first.vertex_range
    third = rc.Mesh(vertices, indices, arena=arena)
    assert arena.inspect()["capacity_bytes"] == before
    assert third.vertex_range == (0, 3)
    assert third.vertex_buffer is not old_buffer
    third.release()
    version = second.version
    arena.compact()
    assert second.vertex_range == (0, 3) and second.version != version
    assert arena.inspect()["allocations"] == 2
    arena.close()
    arena.close()
    with pytest.raises(RuntimeError, match="closed"): _ = second.vertex_buffer


def test_resource_close_is_idempotent_and_rejects_use(device):
    buffer = rc.GpuBuffer(np.zeros(1, dtype=np.uint32), rc.BufferUsage.COPY_DST)
    buffer.close()
    buffer.close()
    with pytest.raises(RuntimeError, match="closed"): buffer.upload()


def test_unload_unused_mesh_and_reuse_retained_draw_registry(device):
    from graaal.DrawBatches import DrawBatches, RenderShader, ShaderPass
    arena = MeshArena()
    vertices = np.zeros(3, dtype=[("position", "f4", 3)])
    mesh = rc.Mesh(vertices, np.arange(3, dtype=np.uint32), arena=arena)
    draws = DrawBatches(np.dtype([("value", "f4")]))
    shader = rc.Shader(source="@vertex fn vertex() -> @builtin(position) vec4f { return vec4f(); }")
    draws.register_shader(0, RenderShader({"depth": ShaderPass(rc.RenderPipeline(shader, fragment_entry=None))}))
    draws.register_mesh(1, mesh)
    draws.register_lod_group(10, (1,))
    draws.sync([10], [0])
    with pytest.raises(ValueError): draws.unregister_lod_group(10)
    with pytest.raises(ValueError): draws.unregister_mesh(1)
    draws.sync([], [])
    mesh.release()
    draws.sync_resources()  # A retained unused registration must not dereference released ranges.
    assert not draws.sync([], [])
    draws.unregister_lod_group(10)
    draws.unregister_mesh(1)
    replacement = rc.Mesh(vertices, np.arange(3, dtype=np.uint32), arena=arena)
    draws.register_mesh(1, replacement)
    draws.register_lod_group(10, (1,))
    assert draws.sync([10], [0])
    assert len(draws.draw_batches) == 1


@pytest.mark.parametrize("offset,count,expected", [(0, None, 3), (1, None, 2), (1, 1, 1)])
def test_draw_mesh_accepts_gpu_only_instance_buffers(device, monkeypatch, offset, count, expected):
    arena = MeshArena()
    mesh = rc.Mesh(np.zeros(3, dtype=[("position", "f4", 3)]), np.arange(3, dtype=np.uint32), arena=arena)
    instances = rc.GpuBuffer(None, rc.BufferUsage.VERTEX, dtype=np.dtype([("offset", "f4", 2)]), count=3)
    render = rc.RenderPass(NS())
    render.handle = Mock()
    handle = Mock()
    monkeypatch.setattr(render, "_prepare_pipeline", lambda: handle)
    render.draw_mesh(mesh, instances=instances, instance_offset=offset, instance_count=count)
    instance_binding = next(call for call in render.handle.set_vertex_buffer.call_args_list if call.args[0] == 1)
    assert instance_binding.args[2] == offset * instances.dtype.itemsize
    handle.draw_indexed.assert_called_once_with(3, expected, 0, 0, 0)
    assert instances.content is None
