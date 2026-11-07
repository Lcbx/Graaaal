# Source and ECS documentation

| Document | Purpose |
| --- | --- |
| [ECS_API.md](ECS_API.md) | Public API reference and executable usage examples |
| [ECS_TODO.md](ECS_TODO.md) | Completed implementation phases, validation record, and deferred work |
| [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md) | Recorded measurements, methodology, and optimization decisions |
| [Scene guide](../scenes/README.md) | Running examples and bounded rendering checks |
| [Benchmark guide](../scenes/benchmarks/README.md) | Benchmark commands, dependencies, and result organization |

Import ECS application types from `ECS`. Implementation modules are grouped by
responsibility:

| Module | Responsibility |
| --- | --- |
| `ECS.py` | Public facade, component pools/selections, joins, and world lifecycle |
| `ecs_types.py` | Numeric types, handles, decorators, and input validation |
| `ecs_schema.py` | Field metadata and payload conversion |
| `ecs_commands.py` | Deferred commands and entity creation tokens |
| `ecs_schedule.py` | Sequential update phases and dependencies |
| `ecs_diagnostics.py` | Explicit inspection and invariant auditing |

`RenderContext.py`, `DrawBatches.py`, and `Utils.py` implement rendering and its
supporting utilities. Generated compiler output belongs under `build/`; recorded
benchmark observations belong under `scenes/benchmarks/results/`.

`Imgui.py` provides an optional Dear ImGui plugin backed by WGPU's
`ImguiWgpuBackend`, inspired by its
[canvas renderer](https://github.com/pygfx/wgpu-py/blob/main/wgpu/utils/imgui/imgui_renderer.py).
Install `imgui-bundle==1.92.5` alongside the pinned WGPU version. Create it after
`RenderContext.init_window()` and use a context manager to detach handlers and
release its ImGui context before `RenderContext.cleanup()`:

```python
from Imgui import Imgui
from imgui_bundle import imgui

def draw_ui():
    imgui.begin("Settings")
    imgui.text("Hello")
    imgui.end()

with Imgui() as gui:
    while RenderContext.window_loop():
        # Render the scene first, then overlay its controls.
        gui.render(draw_ui)
```

For a UI-only scene, pass `clear=(r, g, b, a)` to `render()`. The plugin shares
`RenderContext.screen()` and leaves presentation to `window_loop()`. It updates
logical size, framebuffer scale and elapsed time each frame, skips minimized
windows, and feeds GLFW keyboard, Unicode, pointer, scroll and focus events to
ImGui before scene handlers. Captured input stops propagation. Widget callbacks
run with the plugin's context selected; an existing caller context is restored.
Applications that do not create the plugin do not import `imgui-bundle`.

`GpuBuffer.write(data)` replaces CPU content and uploads it, growing capacity
when needed. Use `write_range(data, offset=byte_offset)` for partial updates;
it preserves CPU content outside the range and the binding size. Uploads require
`COPY_DST`; sub-word ranges include neighboring CPU bytes to satisfy GPU upload
alignment. `resize(count)` only grows GPU capacity and discards previous GPU
contents, so re-upload after growth.

`RenderPipeline` infers a sole fragment entry by default. Pass
`fragment_entry=None` for depth-only rendering. `Mesh.reset_pools()` releases
pool ownership at scene transitions; discard old meshes first. Context cleanup
and device setup also reset the pools. Recreate meshes after changing devices.

`GC_Manager` temporarily disables automatic cyclic GC and performs at most one
collection per idle call. Its time budgets estimate how much spare frame time a
collection needs; they cannot cap the pause. A generation that remains due for
`max_deferral` seconds (default: 1 second) is collected even in a busy frame.
Younger collections do not postpone an older generation's deadline. `close()`
restores the previous GC settings. Use only one active manager at a time.
The manager and its sweep helpers never freeze or unfreeze objects.
Release old scene references before `scene_transition(loader)`; the loader must
retain the new scene. Full sweeps run before loading and after it, including
when loading raises an exception.

`DrawBatches` requires render pipelines in its `ShaderPass` specifications.
It replaces the complete bind-group set on each material draw, so supply every
required group in `ShaderPass.bindings` except group 1 (owned by `DrawBatches`).
For direct rendering, `RenderPass.set_bind_group()` updates one group, while
`set_bind_groups(*bindings)` replaces the entire set.
