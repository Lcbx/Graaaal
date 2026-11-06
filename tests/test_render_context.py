"""Rendering regressions using CPU metadata and mocked window/GPU resources."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import weakref

import numpy as np
import pytest
import RenderContext as rc


def test_entry_inference_comments_and_depth_only():
    shader = rc.Shader.__new__(rc.Shader)
    shader.info = rc.ShaderSource('''
        // @compute @workgroup_size(1) fn ghost() {}
        /* @group(0) @binding(0) var<uniform> ghost: Missing; */
        @vertex fn vertex() -> @builtin(position) vec4f { return vec4f(); }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(); }
        @compute @workgroup_size(1) fn main() {}
    ''')
    assert shader.entry("compute") == "main"
    assert not shader.info.bindings
    assert rc.RenderPipeline(shader).fragment_entry == "fragment"
    assert rc.RenderPipeline(shader, fragment_entry=None).fragment_entry is None
    shader.info = rc.ShaderSource('''
        @compute @workgroup_size(1) fn a() {}
        @compute @workgroup_size(1) fn b() {}
    ''')
    with pytest.raises(ValueError, match="multiple"):
        shader.entry("compute")
    assert shader.entry("compute", "b") == "b"


@pytest.mark.parametrize("dtype", [np.uint64, np.int64, np.float64, (np.float32, (2, 3))])
def test_reject_unsupported_vertex_layouts(dtype):
    with pytest.raises(TypeError):
        rc.dtype_to_vertex_format(np.dtype(dtype))


def test_buffer_partial_upload_and_replacement(monkeypatch):
    queue = Mock()
    device = NS(queue=queue, create_buffer=lambda **kw: NS(size=kw["size"]))
    monkeypatch.setattr(rc.RenderContext, "device", device, raising=False)
    buffer = rc.GpuBuffer(np.array([1, 2, 3], dtype=np.uint16), rc.BufferUsage.COPY_DST, upload=False)
    buffer.upload_range(1, 2)
    _, offset, data = queue.write_buffer.call_args.args
    assert offset == 0
    assert data.nbytes == 8
    assert data[:6].view(np.uint16).tolist() == [1, 2, 3]
    buffer.write_range(np.array([9], dtype=np.uint16), offset=2)
    assert buffer.content.tolist() == [1, 9, 3]
    assert buffer.resource_key[1] == 6
    with pytest.raises(ValueError):
        buffer.upload_range(2, 2)
    with pytest.raises(ValueError):
        buffer.write_range(np.ones(9, dtype=np.uint32))
    buffer.write(np.arange(10, dtype=np.uint32))
    assert buffer.handle.size >= 40
    assert buffer.resource_key[1] == 40
    assert buffer.generation == 1


@pytest.mark.parametrize("count", [0, 3, 4, 5])
def test_buffer_write_preserves_handle_until_capacity_exceeded(monkeypatch, count):
    queue = Mock()
    create_buffer = Mock(side_effect=lambda **kw: NS(size=kw["size"]))
    device = NS(queue=queue, create_buffer=create_buffer)
    monkeypatch.setattr(rc.RenderContext, "device", device, raising=False)
    buffer = rc.GpuBuffer(np.zeros(4, dtype=np.uint32), rc.BufferUsage.COPY_DST, upload=False)
    original_handle = buffer.handle
    create_buffer.reset_mock()

    replacement = np.arange(count, dtype=np.uint32)
    buffer.write(replacement)

    if count <= 4:
        assert buffer.handle is original_handle
        assert buffer.generation == 0
        create_buffer.assert_not_called()
    else:
        assert buffer.handle is not original_handle
        assert buffer.handle.size >= replacement.nbytes
        assert buffer.generation == 1
        create_buffer.assert_called_once()
    if count:
        uploaded_handle, offset, data = queue.write_buffer.call_args.args
        assert uploaded_handle is buffer.handle
        assert offset == 0
        np.testing.assert_array_equal(data.view(np.uint32), replacement)
    else:
        queue.write_buffer.assert_not_called()


def test_buffer_growth_rounds_capacity_once(monkeypatch):
    device = NS(create_buffer=lambda **kw: NS(size=kw["size"]))
    monkeypatch.setattr(rc.RenderContext, "device", device, raising=False)
    buffer = rc.GpuBuffer(np.zeros(1, dtype=np.uint32), rc.BufferUsage.COPY_DST, upload=False)
    assert buffer.resize(4)
    assert buffer.handle.size == 16
    assert buffer.resize(5)
    assert buffer.handle.size == 32


def test_render_pass_material_switch_drops_previous_groups():
    render_pass = rc.RenderPass(NS())
    render_pass.handle = Mock()
    first = NS(handle=object(), bind_group=Mock(side_effect=lambda binding: object()))
    second = NS(handle=object(), bind_group=Mock(side_effect=lambda binding: object()))
    shared, extra = NS(group=0, cache_key=lambda: (0,)), NS(group=2, cache_key=lambda: (2,))
    render_pass.set_pipeline(NS(get_variant=lambda *args: first))
    render_pass.set_bind_groups(shared, extra)
    render_pass.draw(3)
    assert first.bind_group.call_count == 2
    render_pass.set_pipeline(NS(get_variant=lambda *args: second))
    render_pass.set_bind_groups(shared)
    render_pass.draw(3)
    second.bind_group.assert_called_once_with(shared)
    assert list(render_pass.bind_groups) == [0]
    with pytest.raises(ValueError, match="only once"):
        render_pass.set_bind_groups(shared, shared)
    assert list(render_pass.bind_groups) == [0]


def test_gc_budget_and_restore(monkeypatch):
    enabled, thresholds = rc.gc.isenabled(), rc.gc.get_threshold()
    manager = rc.GC_Manager()
    try:
        monkeypatch.setattr(rc.RenderContext, "target_frame_time", 0.02)
        monkeypatch.setattr(rc.RenderContext, "frame_start", 1.0)
        monkeypatch.setattr(rc, "get_time", lambda: 1.001)
        monkeypatch.setattr(rc.gc, "get_count", lambda: (1000, 100, 100))
        collect = Mock()
        monkeypatch.setattr(rc.gc, "collect", collect)
        manager()
        collect.assert_called_once_with(2)
        collect.reset_mock()
        monkeypatch.setattr(rc, "get_time", lambda: 1.02)
        manager()
        collect.assert_not_called()
    finally:
        manager.close()
    manager.close()
    assert rc.gc.isenabled() == enabled
    assert rc.gc.get_threshold() == thresholds


def test_gc_busy_frames_cannot_starve_old_generation(monkeypatch):
    clock = [0.0]
    counts = [1000, 100, 100]
    monkeypatch.setattr(rc, "get_time", lambda: clock[0])
    monkeypatch.setattr(rc.RenderContext, "target_frame_time", 0.01)
    monkeypatch.setattr(rc.RenderContext, "frame_start", -1.0)
    monkeypatch.setattr(rc.gc, "get_count", lambda: tuple(counts))
    collect = Mock()
    monkeypatch.setattr(rc.gc, "collect", collect)
    manager = rc.GC_Manager(max_deferral=1.0)
    try:
        manager()
        clock[0] = 0.5
        # Only generation 0 fits; this must not reset generation 2's deadline.
        monkeypatch.setattr(rc.RenderContext, "frame_start", 0.4906)
        manager()
        collect.assert_called_once_with(0)
        collect.reset_mock()
        clock[0] = 1.0
        manager()
        collect.assert_called_once_with(2)
        collect.reset_mock()
        # A completed collection starts a fresh deferral period.
        manager()
        collect.assert_not_called()
    finally:
        manager.close()


def test_gc_idle_reclaims_a_real_cycle(monkeypatch):
    class Cycle:
        pass

    monkeypatch.setattr(rc.RenderContext, "target_frame_time", None)
    manager = rc.GC_Manager(thresholds=(1, 10000, 10000))
    try:
        rc.gc.collect(2)
        obj = Cycle()
        obj.link = obj
        ref = weakref.ref(obj)
        del obj
        assert ref() is not None
        manager()
        assert ref() is None
    finally:
        manager.close()


def test_gc_helpers_reclaim_promoted_cycles_without_freezing(monkeypatch):
    class Cycle:
        def __init__(self):
            self.link = self

    def abandoned_cycle():
        obj = Cycle()
        return weakref.ref(obj)

    freeze, unfreeze = Mock(), Mock()
    monkeypatch.setattr(rc.gc, "freeze", freeze)
    monkeypatch.setattr(rc.gc, "unfreeze", unfreeze)
    manager = rc.GC_Manager()
    refs = []
    try:
        old = abandoned_cycle()
        rc.GC_Manager.full_sweep()
        assert old() is None

        def load():
            obj = Cycle()
            refs.append(weakref.ref(obj))
            # Promote while still reachable; gen-0 sweeps alone cannot reclaim it.
            rc.gc.collect(0)
            rc.gc.collect(1)

        rc.GC_Manager.scene_transition(load)
        assert refs[-1]() is None

        def failed_load():
            load()
            raise RuntimeError("scene failed")

        with pytest.raises(RuntimeError, match="scene failed"):
            rc.GC_Manager.scene_transition(failed_load)
        assert refs[-1]() is None
    finally:
        manager.close()
    freeze.assert_not_called()
    unfreeze.assert_not_called()


def test_gc_closed_manager_does_not_collect(monkeypatch):
    manager = rc.GC_Manager()
    manager.close()
    collect = Mock()
    monkeypatch.setattr(rc.gc, "collect", collect)
    manager()
    collect.assert_not_called()


def test_pacing_reports_wait_time(monkeypatch):
    context = rc._RenderContext()
    context.frame_start = 1.0
    context.target_frame_time = 0.02
    times = iter([1.005, 1.02])
    monkeypatch.setattr(rc, "get_time", lambda: next(times))
    monkeypatch.setattr(rc, "sleep", Mock())
    context._frame_pacing()
    assert context.frame_time == pytest.approx(0.02)


def test_window_vsync_reinitialization_and_failure(monkeypatch):
    context = rc._RenderContext()
    monkeypatch.setattr(rc.glfw, "init", lambda: True)
    monkeypatch.setattr(rc.glfw, "window_hint", Mock())
    monkeypatch.setattr(rc.glfw, "get_primary_monitor", lambda: object())
    monkeypatch.setattr(rc.glfw, "get_monitor_workarea", lambda m: (0, 0, 800, 600))
    monkeypatch.setattr(rc.glfw, "get_video_mode", lambda m: NS(refresh_rate=60))
    monkeypatch.setattr(rc.glfw, "create_window", lambda *args: object())
    monkeypatch.setattr(rc.glfw, "destroy_window", Mock())
    terminate = Mock()
    monkeypatch.setattr(rc.glfw, "terminate", terminate)
    monkeypatch.setattr(rc.glfw, "wait_events_timeout", Mock())
    monkeypatch.setattr(rc, "get_time", Mock(side_effect=range(100)))
    monkeypatch.setattr(context, "setup_callbacks", Mock())
    graphics = Mock()
    monkeypatch.setattr(context, "setup_graphics", graphics)
    context.init_window(100, 100, "test", custom_gc=False)
    assert graphics.call_args.kwargs["vsync"] is True
    context.cleanup()
    context.cleanup()
    terminate.assert_called_once()
    context.init_window(100, 100, "test", target_fps=-1, custom_gc=False)
    assert context.target_frame_time is None
    assert graphics.call_args.kwargs["vsync"] is False
    context.cleanup()
    monkeypatch.setattr(context, "setup_graphics", Mock(side_effect=RuntimeError("GPU failed")))
    with pytest.raises(RuntimeError, match="GPU failed"):
        context.init_window(100, 100, "test", custom_gc=False)
    assert context.window is None
    assert not context._glfw_initialized
    rc.atexit.unregister(context.cleanup)
