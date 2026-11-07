# Procedural ECS and rendering API

Keep persistent resources in explicit variables and pass them to ordinary
functions. The application owns update order, command recording, submission,
and presentation. Classes represent state and resource handles; procedural
systems do not need a base class, scheduler, or scene container.

## Review and refinements

| API | Finding | Supported workflow |
| --- | --- | --- |
| `ECS` | Already provides explicit worlds, array operations, and ordinary function systems. | Keep `world = ECS()`, pass accessors/queries into functions, and use `assign()` for writes. Scheduler and deferred commands remain optional. |
| `RenderContext` | The exported CamelCase name is an instance, and some factories/fields have mixed names. Resource constructors share its device. | Use the `render_context` singleton, `window_dimensions`, `Shader.uniform_buffer()`, and the `GCManager` class. |
| `DrawBatches` | ECS accessors were required even when the caller already had arrays. | Use `sync_arrays(entities, lod_ids, shader_ids)` and `instance_order` to pack instances and metadata. `sync_batches()` retains the ECS version-based path. |
| `TransparencyPass` | Frame uniform updates required a camera object; the module name was misspelled. | Import `TransparencyPass` and use `update_frame()` with explicit matrices and values. Record each GPU pass with caller-owned commands. |

Default culling and HZB shaders now live beside the renderer's other shaders;
creating these resources does not depend on the working directory or scene files.
Transparency pass recording raises a clear `RuntimeError` until targets have
been allocated. `resize()` returns `True` when targets change and `False` for
unchanged or minimized dimensions, matching `HZB.resize()`.

## Naming and compatibility

Use PascalCase for classes/types, snake_case for functions, methods, parameters,
and state variables, and UPPER_SNAKE_CASE for new constants. Acronyms remain
readable in established type names such as `ECS` and `HZB`.

| Preferred spelling | Compatibility spelling |
| --- | --- |
| `from RenderContext import render_context` | `from RenderContext import RenderContext` |
| `render_context.window_dimensions` | `RenderContext.windowDimensions` |
| `shader.uniform_buffer()` | `shader.UniformBuffer()` |
| `GCManager` | `GC_Manager` |
| `standard_render_shader(...)` | `standard_RenderShader(...)` |
| `update_cull_camera(camera_position=..., view_projection=...)` | `cameraPosition=...`, `viewProjectionMatrix=...` |

Aliases share the same resources and implementation; both window dimension
spellings remain writable. Passing both spellings of a camera parameter raises
`TypeError`. Existing module filenames (`ECS`, `RenderContext`, `DrawBatches`)
remain stable; case-only renames cannot coexist portably on Windows. The
transparency implementation lives directly in `TransparencyPass.py`, with no
compatibility module. Existing lower-case dtype exports remain available.

## Ordinary ECS systems

```python
from ECS import ECS, component, columns

@component
class Position:
    x: float

@component
class Velocity:
    x: float

world = ECS()
world.register(Position, Velocity)
ids = world.create(3)
world.add(ids, columns(Position, x=0), columns(Velocity, x=[1, 2, 3]))
positions, velocities = world.get(Position), world.get(Velocity)
moving = world.query(Position, Velocity)

def move(positions, velocities, entities, dt):
    selected = positions[entities]
    selected.assign(x=selected.x + velocities[entities].x * dt)

move(positions, velocities, moving.entities(), 1 / 60)
```

Numeric selection reads are copies. Assign changed arrays back using `assign()`
or field assignment. Reacquire selections after structural edits. The complete
[ECS reference](ECS_API.md) covers multicomponent selection and query joins.

## Draw instances from arrays

After device setup, register meshes, LoD groups, and render shaders as usual.
Each input row represents exactly one draw instance; repeated entity labels do
not expand the selection:

```python
import numpy as np
from DrawBatches import DrawBatches
from Utils import mesh_instance_dtype

draws = DrawBatches(mesh_instance_dtype)
draws.register_mesh(1, mesh)
draws.register_lod_group(10, (1,))
draws.register_shader(0, geometry.render_shader)

labels = np.array([42, 42, 99], dtype=np.uint64)
lod_ids = np.array([10, 10, 10])
shader_ids = np.array([0, 0, 0])
changed = draws.sync_arrays(labels, lod_ids, shader_ids)
order = draws.instance_order
draws.write_instances(packed_instances[order])
metadata.write(input_metadata[order])
```

`packed_instances` must match `instance_dtype`; `input_metadata` holds the
transparency metadata rows described in [README.md](README.md). Both arrays
above are in input order before indexing. All instance data, including custom
attributes and deferred metadata, must follow the same compact output order.
`draws.entities == labels[draws.instance_order]`. Stable grouping preserves input
order within each shader/LoD pair.

`sync_arrays()` returns whether instance inputs were invalidated. It snapshots
and compares input values, detecting in-place edits in O(n) time. Unchanged
input skips regrouping and uploads. Registry changes are still synchronized.
Value-only instance/metadata edits remain the caller's responsibility to upload.
Empty arrays clear the draw list. Integer IDs are required.

For ECS-backed data, keep `sync_batches(entities, transforms, mesh_refs)` and
read per-component metadata through `mesh_field()`. It avoids expanding repeated
owners twice and uses version tokens to skip field reads. `instance_order` then
indexes the selected MeshRef rows before grouping. `mesh_field()` is only
available after `sync_batches()`. Switching between the array and ECS paths, or
calling `invalidate_batches()`, invalidates all instance inputs on the next sync.

## Record transparency with explicit frame values

Initialize `TransparencyPass` with caller-owned material, vertex, texture, and
noise resources. Create geometry passes for each metadata address space and
register their `render_shader` with the corresponding batches. A complete
direct-draw example is exercised by `tests/test_tranparency_pass.py` in the
repository root.

```python
from RenderContext import render_context
from TransparencyPass import TransparencyPass

oit = TransparencyPass(materials=materials, base_textures=base_textures,
    vertices=vertices, blue_noise=blue_noise, target_format="rgba8unorm")
transparent_geometry = oit.create_geometry_passes(metadata, transparent=True)
# Register transparent_geometry.render_shader with transparent_draws.

oit.resize((width, height))  # Skip nonpositive frame sizes in the caller.
view_projection = oit.update_frame(view=view, projection=projection,
    camera_position=eye, near=0.1, far=1000)
commands = render_context.commands("frame")
# Synchronize and upload instances/metadata before recording GPU passes.
# Prepare visibility and indirect commands for both opaque and transparent draws.

with oit.opaque_pass(commands) as render:
    opaque_draws.draw(render, "prepass")
with oit.guide_pass(commands) as render:
    transparent_draws.draw(render, "main", pass_override=transparent_geometry.guide)
with oit.transmittance_pass(commands) as render:
    transparent_draws.draw(render, "prepass")
with oit.selection_pass(commands) as render:
    transparent_draws.draw(render, "main")
oit.shade_winner(commands, transparent_geometry)
oit.compose(commands, output.color_attachment())
render_context.submit(commands.finish())
```

`update_frame()` uses the existing row-vector matrix convention and returns
`view @ projection`. It validates finite matrices, vec3 camera/light values,
and `0 < near < far` before writing uniforms. `update_camera(camera)` delegates
to the same implementation and obtains projection aspect from the target size.

Visibility preparation remains explicit: call `update_cull_camera()`,
`refresh_bindings(hzb_view)`, `reset_draw_counts(commands)`, and
`cull_instances(commands, "frustum")` after synchronization. For the normal Hi-Z
workflow, draw the depth prepass, build the HZB, then cull with `"hiz"` before the
main draw. For the transparency example's frustum-only workflow, copy prepass
commands and visibility to their main buffers before drawing; enable `COPY_SRC`
on the source buffers. `oit.dummy_hzb` can satisfy the unused HZB binding.

The context still owns one shared device/window. This change does not introduce
multiple-device contexts. Transparency geometry retains the existing baked
world-space vertex ABI, and every transparent draw in a winner-selection pass
must share the metadata/vertex address space used by `shade_winner()`.
