# Graaal game engine

Install the checkout from its repository root with `pip install -e .`.
The Python package is `numpyecs`; core simulation needs only NumPy. GPU rendering,
glTF/image loading, UI, and bundled scenes use the `render`, `assets`, `ui`, and
`examples` extras. Install development dependencies with
`pip install -e ".[render,assets,ui,examples,test]"`.

RenderContext manages windows, events, presentation, and GPU resources through
one shared device. Systems are ordinary functions; command order is explicit.
Import the manager with `from numpyecs.RenderContext import RenderContext`.

Public subsystem modules and classes use PascalCase; internal helpers, functions,
and attributes use snake_case. Constants use UPPER_SNAKE_CASE.

| Guide | Contents |
| --- | --- |
| [ECS API](src/ECS_API.md) | Components, arrays, queries, lifetimes, commands, and scheduling |
| [Scenes](scenes/README.md) | Running interactive examples and their controls |
| [Transparency](src/stochastic_transparency.md) | Pass order, reconstruction, and technique limits |
| [Dictionary materials](src/materials.md) | Plain WGSL materials, shared parameters, instance overrides, and Jinja inheritance |
| [Benchmarks](scenes/benchmarks/README.md) | Reproduction commands and diagnostic options |
| [Implementation work](IMPLEMENTATION_WORK.md) | Backlog, implementation decisions and historical measurements (status record) |

## Modules and useful APIs

Import modules as `numpyecs.<Module>`. This non-exhaustive overview covers useful
classes, methods, and helpers; the linked guides explain their contracts.

| CPU module | Useful APIs and purpose |
| --- | --- |
| [ECS](src/ECS_API.md) | `component`, `array_field`, `columns` define data; `ECS.register/create/add/set/remove/delete` manage it; `ECS.query` joins stores. `ComponentAccessor.version/touch` track edits; `ComponentSelection.read/assign` gather and write; accessor `edit_dense` borrows store columns. |
| `ecs_commands`, `ecs_schedule` | Re-exported by `ECS`: `CommandBuffer.flush/clear` apply or discard deferred edits; `DeferredEntities.resolve` retrieves created IDs; `Scheduler.add/compile/run` orders synchronous systems. |
| `Math` | `Vec2/Vec3/Vec4` are NumPy vectors; `Camera.view/projection/direction` supplies camera values; `extract_frustum_planes` derives culling planes. |
| `Assets` | `load_gltf_scene` returns static triangle lists, materials and images; `read_gltf_accessor` reads attributes; `load_gltf_first_mesh_interleaved` loads one local-space primitive; `load_blue_noise` returns CPU thresholds. |
| `InstanceData` | `MESH_INSTANCE_DTYPE` defines transforms/tint; `pack_quaternion`, `pack_scale`, `pack_rgba8_srgb` encode fields; `make_instances` packs separate component columns. |
| `TransparencyData` | `load_transparency_scene` returns primitives, material records and images; `VERTEX_DTYPE`, `SHADING_VERTEX_DTYPE`, `MATERIAL_DTYPE` define the rendering ABI. |
| `DrawList` | `DrawList.build(lod_ids, shader_ids)` returns `.order` and `SourceBatch` ranges for CPU grouping. |
| `Geometry` | `make_cube_mesh` creates a GPU mesh when called; importing the module needs no graphics backend. |
| `Diagnostics` | `WatchTimer` times scopes; `WatchTimer.capture` formats reports; `setup_gc_monitor` logs collections; `setup_memory_monitor` returns a report callback and requires separately installed `psutil`. |
| `utils` | `cache_last` caches an instance method's most recent keyed call; `higher_pow2` returns the next strictly greater power of two. |

| Rendering module | Useful APIs and purpose |
| --- | --- |
| `RenderContext` | Singleton `RenderContext.init_window/setup_graphics_backend`, `window_loop`, `screen`, `commands/submit`, `subscribe_event`, `cleanup` manage the backend and frames; `enable_statistics/inspect_resources` expose counters. |
| `RenderContext` resources | `GpuBuffer.write/write_changed/upload_range/reserve/set_count`, `Texture.view/color_attachment/depth_attachment`, `Sampler`, and `Mesh.release` manage GPU data. |
| `RenderContext` recording/shaders | `CommandContext.render_pass/compute_pass/finish`, `RenderPass.draw_mesh`, `ComputePass.dispatch`, `Shader.uniform_buffer/bind_group`, `RenderPipeline`, `ComputePipeline`, `BindGroupLayout`, and `PipelineLayout` compose work. Jinja `field()`/`binding()` declarations provide shader metadata; `ShaderInterface` supplies explicit metadata. |
| `MeshArena` | `MeshArena.allocate/compact/inspect/close` pools mesh storage; `MeshAllocation.release` returns a range for reuse. |
| `DrawBatches` | `DrawBatches.register_mesh/register_lod_group/register_shader`, `sync`, `write_instances`, `create_visibility`, `draw` manage shared draws. `RenderShader` maps names to `ShaderPass`; `Visibility.all_visible/write` prepares visibility; `FrustumCuller.record` and `HiZCuller.record` record GPU culling. |
| `EcsDrawing` | `EcsDrawAdapter.sync` extracts ECS mesh references; `mesh_field` returns attributes in draw order. |
| `HZB` | `HZB.resize` allocates `.depth_texture` and pyramid; `build` records reductions; `.view` feeds Hi-Z culling. |
| [TransparencyPass](src/stochastic_transparency.md#useful-apis) | `TransparencyPass.resize/update_frame/create_geometry_passes` prepares inputs; `opaque_pass/guide_pass/transmittance_pass/selection_pass/shade_winner/compose` records stages. `TransparencyTargets`, `BilateralFilter`, `GrazingFade` configure resources and settings. |
| `TextureProcessing` | `get_default_blue_noise` shares a GPU tile; `build_material_texture_array` generates alpha-aware mips; `build_correlated_noise_hierarchy` builds ranked noise levels. These helpers upload and submit GPU work when creating textures. |
| [Materials](src/materials.md#useful-apis) | `MaterialShader` defines shader families; `MaterialBinding` selects dictionaries/overrides; `MaterialLibrary.pack` produces CPU tables, while renderer Jinja2 templates generate hook dispatch. |
| `DefaultMaterial` | `DefaultMaterialShader` supplies lit opaque/transparent defaults, validation, and shader hooks. |
| `MaterialRenderer` | `MaterialMesh` describes geometry; `DrawInstance` is a direct-draw input; `MaterialRenderer.update/update_transforms/record/set_sample_count` uploads arrays and records rendering. |
| `EcsMaterials` | `MaterialInstance` and `MaterialTransform` are components; `EcsMaterialAdapter.sync/touch_materials` extracts changes and invalidates edited dictionaries. |
| `ImGui` | `ImGui.render(callback)` draws a UI overlay and routes input; `close` removes handlers and releases its context. |

Simulation, math, asset-data, packing, grouping, and utility modules import without
graphics dependencies. Asset decoders load their optional dependencies when called.
Engine shaders are included in the installed package. `Geometry.make_cube_mesh()`
creates a GPU mesh and requires an initialized backend.

## Arrays and ordinary systems

```python
import numpy as np
from numpyecs.ECS import ECS, component, array_field, columns

@component
class Position:
	value: np.ndarray = array_field(np.float32, shape=(3,))

world = ECS()
positions, = world.register(Position)
ids = world.create(3)
world.add(ids, columns(Position, value=np.zeros((3, 3), dtype=np.float32)))
selected = positions[ids]
scratch = np.empty((3, 3), dtype=np.float32)
selected.read("value", out=scratch)
np.add(scratch, 1, out=scratch)
selected.assign(value=scratch)
with positions.edit_dense("value") as fields:
	fields["value"] *= 0.5
```

Normal field reads copy. `read(out=...)` rejects incorrect dtype/shape, read-only
outputs, and storage aliases. Repeated rows retain NumPy's last-write scatter
semantics. Numeric metadata is explicit; declare `object` for Python data rather
than relying on an unsupported annotation fallback.

Dense edits borrow columns in storage order. Views must not escape the scope.
Structural changes to the borrowed store are rejected before any multi-store
request can partially apply. Declared fields are touched on exit, including after
exceptions; writes are not rolled back. Compact multicomponent stores containing
holes first. General joins between independently ordered stores still gather.

See the [ECS reference](src/ECS_API.md) for full contracts and the
[array measurements](IMPLEMENTATION_WORK.md#array-access) for scratch-memory costs.

## Shared draws and independent visibility

Initialize RenderContext first and request `indirect-first-instance` through
`init_window(..., required_gpu_features=[...])` or
`setup_graphics_backend(..., required_features=[...])` when using indirect draws.
The following recording example assumes a compatible `mesh`, `pipeline`,
`input_instances`, camera values, and output texture already exist. Configure the
pipeline for the target attachments; this example uses a color-only pass.

```python
from numpyecs.RenderContext import RenderContext
from numpyecs.DrawBatches import DrawBatches, RenderShader, ShaderPass, FrustumCuller
from numpyecs.InstanceData import MESH_INSTANCE_DTYPE

draws = DrawBatches(MESH_INSTANCE_DTYPE)
draws.register_mesh(1, mesh)
draws.register_lod_group(10, (1,))
draws.register_shader(0, RenderShader({"color": ShaderPass(pipeline)}))
draws.sync([10, 10], [0, 0], labels=["first", "second"])
draws.write_instances(input_instances[draws.instance_order])
main_view = draws.create_visibility()

commands = RenderContext.commands("frame")
FrustumCuller().record(commands, main_view, camera_position=eye, view_projection=view_projection)
with commands.render_pass(color=output.color_attachment()) as render:
	draws.draw(render, main_view, "color")
RenderContext.submit(commands.finish())
```

`DrawList.build(lod_ids, shader_ids)` provides pure CPU grouping. `DrawBatches`
owns shared geometry/instances; `Visibility` owns each view's counters, visible
indices, and indirect commands. Create a separate visibility for each camera or
shadow view. `sync()` snapshots keys and labels to detect
in-place changes and returns whether instance order was invalidated. Upload all
attributes and metadata in the same `instance_order`. Pass names are arbitrary.

Prepare visibility again after regrouping or invalidating instance order; stale
or unprepared results raise `RuntimeError` when drawn or accessed.
Use `visibility.all_visible(lod=0)` for CPU/direct preparation, or
`visibility.write(uint32_indices, indirect_records)` for a custom culler;
`INDIRECT_DTYPE` describes those command records. For Hi-Z culling, create `HZB`
(from `numpyecs.HZB`), call `resize(size)`, render occluder depth into
`hzb.depth_texture`, call `hzb.build(commands)`, then call
`HiZCuller().record(commands, main_view, hzb_view=hzb.view)` to refine that view's
frustum results.

The default culling ABI is `MESH_INSTANCE_DTYPE`; another dtype requires matching
shaders/culling or caller-produced visibility. Group 1 supplies `instances` and
`visible_instances` by default. `ShaderPass.instance_group` selects another group
or `None` for caller-owned bindings. `draw()` accepts an `instance_binding`
function for custom layouts. A pass supplies its complete extra binding set.

`EcsDrawAdapter(draws, mesh_refs, lod_field="lod_id", shader_field="shader_id")`
handles ECS extraction/version tracking. Call `adapter.sync(entities)` and use
`adapter.mesh_field(name)` for packed per-component attributes. Repeated owners
expand once; no transform component type is prescribed by the renderer.

## Resources and pipelines through RenderContext

`GpuBuffer(content, usage, copy=False)` retains contiguous CPU content by default;
noncontiguous input is copied to make it contiguous, and `copy=True` always copies.
`GpuBuffer(None, usage, dtype=..., count=...)` allocates
GPU-only storage. `count`, `binding_size`, and `capacity_bytes` distinguish active
records/scalars, bound bytes, and GPU capacity. Element-range uploads require 1-D
content; `write_range()` offsets are bytes.

`reserve(count)` preserves active data without changing count. GPU-only
preservation requires `COPY_SRC | COPY_DST`; CPU-backed preservation needs
`COPY_DST`. `resize(count)` grows capacity and discards GPU contents without
changing the active count. `set_count()` changes the active count within GPU
capacity and the existing CPU mirror's size; it does not resize that mirror.

Resources retain their device/generation and reject reuse after device replacement
or cleanup. Raw WGPU handles remain available. `close()` releases wrapper ownership
without destroying handles referenced by recorded GPU work.

`Mesh` uses `RenderContext.mesh_arena` by default. Supply `MeshArena` for an asset
lifetime boundary. `mesh.release()` returns allocations for reuse;
`arena.compact()` updates live ranges/generations. Draw synchronization detects
relocation. Copy-on-reuse/reallocation preserves previously recorded handles.

To unload registered geometry, synchronize away the group's instances, call
`draws.unregister_lod_group(group_id)`, then `draws.unregister_mesh(mesh_id)` once
all referencing groups are removed. Registry removal does not release the mesh;
call `mesh.release()` when no other owner needs it. Releasing an unused mesh
before removing its registration also permits subsequent synchronization.
Active groups and referenced meshes cannot be unregistered.

`RenderPipeline` accepts explicit `layout`, `depth_stencil`, per-attachment
`targets`, `sample_count`, and normalized `vertex_formats`. Color attachments
accept a texture, texture view, or `RenderContext.screen()` attachment as
`resolve_target`. Attachment formats/counts and sample compatibility are
validated when preparing a pipeline.

Shader templates accept complete-line WGSL-like declarations for struct fields and
uniform/storage bindings. These work in both `Shader(source=...)` and
`Shader(filepath=...)`, including Jinja `#include`, `#if` and macros:

```wgsl
<Frame> view_proj: mat4x4f;
uniform<0, 0, read> frame: Frame;
<VertexInput> position: vec3f;

@vertex fn vertex(input: VertexInput) -> @builtin(position) vec4f {
    return frame.view_proj * vec4f(input.position, 1);
}
```

`<Struct> @location(2) field: type;` also accepts `@builtin`, `@interpolate`,
`@align` and `@size`. An optional `= value` records a Jinja2-expression
default; regular uniform buffers are still initialized to zero. Omit `read`
for uniforms if preferred; storage bindings accept `read` or `read_write`.
`Shader.uniform_buffer()` uses the declared struct to derive NumPy layouts.
For textures, samplers or dynamic declarations, the existing Jinja2
`binding(name, type, group, binding, storage=..., access=...)` and
`field(struct, name, type, ...)` callbacks remain available.
Use `ShaderInterface` when supplying your own layouts.

`ShaderInterface` provides explicit bindings, entries,
vertex inputs, and `UniformInfo` NumPy offsets/strides. This supports padded
matrices without treating a packed matrix as WGSL-aligned. Automatic layouts
traverse entry-point function dependencies. Explicit interfaces' `active_bindings`
handle shader-specific binding elimination; explicit WGPU layouts retain the
caller's full binding ABI.

## Composable transparency

`TransparencyPass` records opaque, guide, transmittance, winner-selection,
deferred-shading, and composition stages into caller-owned commands. Call
`resize(size)` before `update_frame(...)` or `update_camera(camera)`, then record
the stages and submit. It does not submit or present frames itself.

Use `create_geometry_passes(metadata, transparent=True, instances=buffer)` to
create transparent `transmittance` and `main` shader passes plus a guide override.
Keep metadata in packed instance order, and share its vertex/material address
space across all transparent draws in a selection target. Raster and deferred
vertices use the same local coordinates and packed instance transforms.

`TransparencyTargets` accepts caller-owned attachments. Existing sampleable
opaque color/depth can be supplied through `opaque_color` and `opaque_depth`;
they must match the target dimensions and sample count. Otherwise targets are
allocated internally. Set `target_format` to match the output attachment.

Defaults are single-sample rendering, anchored-blue noise at a one-pixel grain
size, two averaged winner draws, grazing fade from 83 to 86 degrees, and 3x3 bilateral reconstruction with
spatial sigma 0.9. `set_sample_count(1 or 4)`, `set_noise_mode`, `set_noise_scale`,
`set_grazing_fade`, `set_bilateral_filter`, and `set_debug_view` update live
settings. A 1x1 kernel bypasses reconstruction; 4x MSAA remains available.
No temporal accumulation is used.

The [transparency guide](src/stochastic_transparency.md) covers pass order,
custom targets, noise choices, shader contracts and limitations. The
[material guide](src/materials.md) covers dictionary materials and vertex hooks;
[transparency_example.py](scenes/transparency_example.py) shows a full frame.

## Examples and diagnostics

From any directory after installation:

```powershell
python -m numpyecs
python -m numpyecs --offscreen --frames 1
python -m numpyecs --window --frames 120
```

The first command needs only NumPy. GPU modes load packaged shaders and use
RenderContext. Repository scenes run through `python main.py scenes/<name>.py`;
add `--frames N` for a bounded run. Imports do not create windows or devices.

Create `ImGui` after initializing a window and close it before context cleanup.
It overlays the current surface and forwards uncaptured input to application handlers.
`GCManager` is an optional idle-time collection policy; its budget estimates spare
time rather than capping a collection pause. `close()` restores process settings.

Enable counters with `RenderContext.enable_statistics()` and inspect them through
`inspect_resources()`. Counters cover resource creation, uploads, cache behavior,
grouping, and visibility; arenas report active/capacity bytes. Counters do not
measure GPU execution time or synchronize the GPU.

From the repository root run `python -m pytest tests src/tests -p no:cacheprovider`.
Tests cover CPU-only imports, mutation/lifetimes, multiple views, explicit layouts,
MSAA, custom materials, and offscreen GPU readback.
