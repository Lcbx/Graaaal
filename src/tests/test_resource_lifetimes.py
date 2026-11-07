from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from numpyecs import RenderContext as rc
from numpyecs.MeshArena import MeshArena


@pytest.fixture
def device(monkeypatch):
    device = NS(queue=Mock(), create_buffer=lambda **kw: NS(size=kw["size"]),
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
