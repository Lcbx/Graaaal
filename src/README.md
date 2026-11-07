# Source and ECS documentation

| Document | Purpose |
| --- | --- |
| [ECS_API.md](ECS_API.md) | Public API reference and executable usage examples |
| [PROCEDURAL_API.md](PROCEDURAL_API.md) | API review, naming conventions, array batching, and explicit rendering workflow |
| [Improvements.md](Improvements.md) | Phased plan for reusable NumPy APIs and customizable rendering, preserving RenderContext's central role |
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

`Utils.load_gltf_scene(path)` loads static glTF/GLB scenes on the CPU and returns
`(primitives, materials, images)`. Each `GltfPrimitive` has world-space,
triangle-list `position`, `normal`, and `uv` vertices plus a material index.
Node transforms and reflected winding are baked; missing normals are generated
and missing UVs become zero. Materials retain their glTF PBR/alpha properties;
images are RGBA NumPy arrays indexed by material. A final `None` material and
white image provide glTF's default material. The loader has no GPU layouts or
example paths. Skinning, sparse accessors, non-triangle primitives, and texture
coordinate transforms are unsupported.

`read_gltf_accessor`, `gltf_buffer_bytes`, `gltf_image_bytes`, and
`gltf_node_matrix` expose the underlying loading helpers. Buffer/image resources
can be embedded or external relative to the asset directory. The first-mesh
loaders share the accessor reader while preserving their local-space outputs.
`load_blue_noise(path)` reads single-channel 16-bit PNGs as linear thresholds.

`Utils.load_scene(path)` builds on `load_gltf_scene()` to return
`(primitives, materials, images, shading_vertices)` in the transparency shader
layouts. Its `Primitive` records include per-mesh triangle offsets and stable
surface IDs. It adds barycentrics, packs PBR/opacity records, and prepares the
deferred shading vertex buffer, all on the CPU. The asset path is explicit.

`Utils.build_material_texture_array(images, materials, size)` uploads base-color
images and generates their texture-array mip chains on the GPU. It preserves
MASK alpha coverage and BLEND mean opacity while filtering color in linear
space. It requires an initialized rendering device and material records with
`base_color` and `surface` fields; it is independent of the transparency passes.

`TransparencyPass.py` owns the reusable stochastic transparency pipelines,
filter settings, and render targets. It accepts
caller-owned material/vertex buffers, a texture array view and a linear noise
texture. It does not load glTF, create a camera or window, manage an ECS world,
or submit frame commands. Scene loading and material preparation live in `Utils`;
the example's ImGui controls remain in `scenes/transparency_example.py`, and its
main loop records the pass sequence directly with DrawBatches. `init_scene()`
sets up module-level render data and camera state; it creates no scene container.

Create `TransparencyPass(materials=..., base_textures=..., vertices=...,
blue_noise=..., target_format=...)` after device setup, then call
`create_geometry_passes(metadata, transparent=True/False)` for each instance
metadata buffer. Register the returned `render_shader` with DrawBatches.
Metadata records are four `uint32`s: deferred triangle offset, material index,
stable surface ID, reserved. Metadata follows compact DrawBatches instance
order. All transparent draws in a winner-selection pass must share one metadata
address space; pass that geometry's bindings to `shade_winner`.

Use `update_frame(view=..., projection=..., camera_position=..., near=..., far=...)`
with plain arrays, or `update_camera(camera)` as a convenience adapter. Both
return the culling view-projection matrix. The frame can be recorded procedurally
with an existing command context:

```python
oit.resize((width, height))
view_projection = oit.update_camera(camera, light_dir=(0.4, 0.8, 0.3))
cmd = RenderContext.commands("frame")
# Prepare/cull DrawBatches using view_projection before drawing.
with oit.opaque_pass(cmd) as rp:
    opaque.draw(rp, "prepass")
with oit.guide_pass(cmd) as rp:
    transparent.draw(rp, "main", pass_override=transparent_passes.guide)
with oit.transmittance_pass(cmd) as rp:
    transparent.draw(rp, "prepass")
with oit.selection_pass(cmd) as rp:
    transparent.draw(rp, "main")
oit.shade_winner(cmd, transparent_passes)
oit.compose(cmd, output.color_attachment())  # Or RenderContext.screen().
RenderContext.submit(cmd.finish())
```

Skip frames with nonpositive dimensions. `resize` retains valid targets for
minimized sizes and rebuilds dependent bindings only when dimensions change.
The example copies frustum visibility/commands from prepass to main, enabling
`COPY_SRC` on those two DrawBatches buffers. Other callers can supply their own
culling and visibility preparation. Direct drawing works with the same passes
when group 1 supplies instance and visibility buffers.

`VERTEX_DTYPE`, `SHADING_VERTEX_DTYPE`, and `MATERIAL_DTYPE` describe the current
shader layouts. The material shaders expect baked world-space vertices/normals;
packed rotation/scale are not applied. If raster instances use `iPosition`, the
deferred vertex data must include the same translation. The sampler defaults to
repeat/trilinear filtering. Supply `target_format` for offscreen output so
composition selects the correct sRGB encoding. Live `set_bilateral_filter()`
and `set_debug_view()` calls update uniforms without rebuilding targets.

Renderer shaders and shared shader libraries live under `src/shaders`.
File-backed `Shader` instances resolve includes beside the shader first, then
in that engine directory; an explicit `basedir` retains its supplied search path.

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

`GCManager` temporarily disables automatic cyclic GC and performs at most one
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
