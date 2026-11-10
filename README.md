# NumPy ECS and procedural rendering

Install the checkout from its repository root with `pip install -e .`.
The Python package is `numpyecs`; core simulation needs only NumPy. GPU rendering,
glTF/image loading, UI, and bundled scenes use the `render`, `assets`, `ui`, and
`examples` extras. Install development dependencies with
`pip install -e ".[render,assets,ui,examples,test]"`.

RenderContext remains the convenient window, event-loop, presentation, and GPU
resource manager. There is one shared device; callers do not need to pass it
through each API. Systems remain ordinary functions; command order is explicit.
Import the manager with `from numpyecs.RenderContext import RenderContext`.

Public subsystem modules and classes use PascalCase; internal helpers, functions,
and attributes use snake_case. Constants use UPPER_SNAKE_CASE.

| Guide | Contents |
| --- | --- |
| [ECS API](src/ECS_API.md) | Components, arrays, queries, lifetimes, commands, and scheduling |
| [Scenes](scenes/README.md) | Running interactive examples and their controls |
| [Transparency](src/stochastic_transparency.md) | Pass order, reconstruction, and technique limits |
| [Dictionary materials](src/materials.md) | Plain WGSL materials, shared parameters, instance overrides, and Jinja inheritance |
| [Benchmarks](scenes/benchmarks/README.md) | Reproduction commands and dated measurements |

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `ECS`, `ecs_*` | Worlds, schemas, checked selections, queries, commands, scheduling |
| `Math` | NumPy vectors, cameras, projection and frustum helpers |
| `Assets` | CPU glTF/image loading; optional dependencies load when called |
| `InstanceData` | Default instance dtype, quaternion/scale/color packing |
| `TransparencyData` | Transparency layouts and `load_transparency_scene()` |
| `DrawList` | Pure NumPy grouping and instance permutations |
| `Geometry` | Geometry helpers; GPU construction imports the renderer when called |
| `Diagnostics` | Optional timing and memory helpers |
| `TextureProcessing` | GPU material mip generation |
| `EcsDrawing` | Optional ECS-to-array drawing adapter |
| `DrawBatches` | Shared draw resources, named shader passes, and per-view visibility |
| `HZB` | Hierarchical depth-pyramid generation |
| `RenderContext`, `MeshArena` | Window/event loop, command recording, GPU resources, pooled mesh allocation |
| `TransparencyPass` | Composable stochastic transparency passes |
| `Materials`, `DefaultMaterial`, `MaterialRenderer`, `EcsMaterials` | Generic material layouts, default shader inheritance, DrawBatches integration and ECS extraction |
| `ImGui` | Optional Dear ImGui integration with the RenderContext window/surface |
| `utils` | Small shared CPU helpers: `cache_last` and `higher_pow2` |

Simulation, math, asset-data, packing, grouping, and utility modules import without
graphics dependencies. Asset decoders load their optional dependencies when called.
Engine shaders are included in the installed package. `Geometry.make_cube_mesh()`
creates a GPU mesh and requires an initialized backend.

Internally, `draw_compiler` generates layouts with NumPy, `draw_layout` defines
their records, `draw_resources` manages registries/uploads, and `draw_culling`
records per-view compute work. Compilation can run without a GPU.

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
[array measurements](scenes/benchmarks/README.md#array-access) for scratch-memory costs.

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
FrustumCuller().record(commands, main_view,
    camera_position=eye, view_projection=view_projection)
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
`INDIRECT_DTYPE` describes those command records. After
drawing depth, build `HZB` (imported from `numpyecs.HZB`) and call
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
`COPY_DST`. `resize(count)` discards GPU contents on growth. `set_count()` changes
the active range within capacity and any CPU mirror.

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

`ShaderInterface` replaces Python reflection with explicit bindings, entries,
vertex inputs, and `UniformInfo` NumPy offsets/strides. This supports padded
matrices without treating a packed matrix as WGSL-aligned. Automatic layouts
traverse entry-point function dependencies. Explicit interfaces' `active_bindings`
handle shader-specific binding elimination; explicit WGPU layouts retain the
caller's full binding ABI.

## Composable transparency

Create `TransparencyPass` with material/vertex/texture/noise resources and an
explicit offscreen `target_format` where appropriate. Existing opaque inputs can
be supplied as `opaque_color=...`, `opaque_depth=...`, or by `set_opaque_inputs()`.
They must be sampleable textures with matching dimensions and the configured sample count.
`resize(size, targets=TransparencyTargets(...))` accepts caller-owned targets;
otherwise reconstruction targets are allocated internally.
The guide pass writes both `transparent_depth` and an rgba16float
`transparent_guide` containing nearest acceptance and reciprocal-depth gradients.
Supply both when using caller-owned `TransparencyTargets`.

MSAA is optional: `sample_count=4` enables it, and `set_sample_count(1 or 4)`
switches the default shader set live. The example uses **M** or its MSAA dropdown;
the default is off. At 4x, winner payloads use two `rgba16uint` MSAA targets and
shaded colors use a four-layer array. Composition resolves linear HDR before tone
mapping; no temporal accumulation is used. See the
[target requirements](src/stochastic_transparency.md#optional-msaa) for custom targets.

Create geometry with
`oit.create_geometry_passes(metadata, transparent=True, instances=instance_buffer)`.
Transparent draws expose `transmittance` and `main` shader passes; they have no
opaque `prepass` entry. Draw `transmittance` inside `oit.transmittance_pass()`.
Metadata is a vec4u per packed instance: triangle offset, material index, stable
surface ID, reserved. Transparent draws in a selection pass share the metadata/
vertex address space passed to `shade_winner()`. Invalid deferred addresses yield
empty samples; geometry bindings cannot be mixed between technique owners.

Raster and deferred vertices share local coordinates. Both apply packed instance
translation, rotation, and scale; normals use inverse scale and rotation. CPU
loading can still bake glTF node transforms before instance transforms apply.
Noise defaults to point-sampled UV blue-noise octaves with CDF-corrected lerp.
`noise_mode` / `set_noise_mode(name)` also selects stochastic octave picking,
object-space hashing or diagnostic Bayer noise. UV modes fall back to object
hashing for missing/degenerate UVs; no cards or triangle reseeding are assumed.
`noise_scale=1.0` / `set_noise_scale(pixels)` controls grain size. Guide and
transmittance passes are independent of these selection-only options.
`uv_blue_correlated`, `triplanar_blue`, and `triplanar_correlated` add a GPU-built
rank hierarchy and UV-independent object-space projections. Choose them through
`set_noise_mode(...)` or the example's noise selector; sampling remains static.
`GrazingFade(enabled=True, start_angle=70, end_angle=85)` optionally fades
transparent-surface opacity between two angles from the geometric normal.
Pass it as `grazing_fade=...` or call `set_grazing_fade(...)`; the current defaults
enable fading between 83 and 86 degrees.
MASK materials use sharpened alpha when routed through the transparent
passes. Reconstruction uses a fixed 3x3, 5x5 or 9x9 bilateral footprint (default
3x3 with spatial sigma 0.9), with slope correction always enabled. A 1x1 kernel
bypasses neighborhood reconstruction and retains the unfiltered stochastic estimator.

`material_source` replaces `shade(material, uv, position, normal, lod) -> vec3f`
without changing selection/reconstruction. A matched `geometry_shader` and
`shading_shader` pair can customize deformation under the same winner ABI.
`create_geometry_passes()` accepts additional group-2 resources and higher-numbered
binding groups. Animation data preparation remains caller-owned; no animation or
skinning system is supplied.

Use `update_frame(view=..., projection=..., camera_position=..., near=..., far=...)`
or the `update_camera(camera)` adapter. The row-vector matrix convention remains
`view @ projection`. Record opaque (skip when inputs already exist), guide,
transmittance, selection, shading, and composition explicitly in caller commands.
Call `resize()` before updating camera uniforms or recording passes. The technique
does not submit or present. See the [transparency notes](src/stochastic_transparency.md)
and [scene implementation](scenes/transparency_example.py) for complete pass order.

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
