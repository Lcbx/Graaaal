# Dictionary materials

The generic [Materials.py](Materials.py) handles shader-defined schemas,
dictionary resolution, packing and linking. It contains no lighting, alpha or
grazing policy. [DefaultMaterial.py](DefaultMaterial.py) supplies the default
lit schema, validation and opaque/stochastic interface. The default WGSL
template interprets those parameters.

Run the ECS example:

```powershell
python main.py scenes/material_example.py
```

The six labeled spheres use two rows: three opaque examples above, and three
transparent examples below. Only the bottom-left and bottom-middle spheres share
the roughness/specular sliders. D replaces the top-left default material; the
top-right sphere has an explicit green override. M toggles MSAA; Space orbits.
The top-middle sphere uses a procedural striped shader written in the Python
script; its stripes and time-driven vertex displacement have live controls for
frequency, color, displacement amplitude and speed, plus an animation pause.

## Useful APIs

This non-exhaustive map covers the material modules. Import from
`numpyecs.<Module>`; CPU packing needs no initialized GPU device.

| Module | Classes and methods |
| --- | --- |
| `Materials` | `MaterialShader(interface=..., source/filepath/template=...)` declares a family; `.parameters` exposes its schema; `.validate(values)` checks dictionary fields. `MaterialBinding(shader, material, overrides)` selects values. |
| `Materials` | `MaterialLibrary(shaders).pack(bindings)` returns headers, shared slots, instance slots, material IDs and instance offsets; `.source` contains linked WGSL; `.parameter_layouts` exposes slot/lane offsets. |
| `Materials` | `MaterialInterface`, `ShaderHook`, and `FrameParameter` describe a rendering adapter's dispatch and frame-data contract; `ParameterLayout.pack` applies a declared slot layout. |
| `DefaultMaterial` | `DefaultMaterialShader` provides the lit schema, shader hooks, alpha routing and semantic validation used in this guide. |
| `MaterialRenderer` | `MaterialMesh(vertices, shader, ...)` registers geometry; `DrawInstance` describes optional direct inputs. `MaterialRenderer.record` records a frame; `update` accepts explicit arrays; `update_transforms` skips material packing; `set_sample_count` switches 1x/4x passes. |
| `EcsMaterials` | `MaterialInstance` stores mesh/material references and overrides; `MaterialTransform` stores numeric transforms. `EcsMaterialAdapter.sync` extracts changed ECS data; `touch_materials(*dicts)` marks nested edits for repacking. |

The default adapter uses `TransparencyPass` for rendering and `DrawBatches` for
grouping/culling. See the [engine API overview](../README.md#modules-and-useful-apis)
for lower-level resources and commands.

## ECS usage

This integration excerpt assumes an initialized RenderContext device with
`indirect-first-instance` enabled, triangle-list `vertices` with `position`,
`normal`, and `uv` fields, and matching `indices`. The frame inputs `size`, `camera`,
and `output_attachment` belong to the application; the output format must match
the renderer's `target_format`. Run from the repository root to resolve the glass
template. See [material_example.py](../scenes/material_example.py) for complete setup.

```python
from numpyecs.ECS import ECS
from numpyecs.Math import Vec3
from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.MaterialRenderer import MaterialMesh, MaterialRenderer
from numpyecs.RenderContext import RenderContext
from numpyecs.EcsMaterials import (
    MaterialInstance, MaterialTransform, EcsMaterialAdapter,
)

glass = DefaultMaterialShader(
    transparent=True,
    template="scenes/shaders/glass_material.shader",
    default_material={"roughness": 0.3},
)
shared = {"base_color": (0.2, 0.6, 0.9, 0.5)}
mesh = MaterialMesh(vertices, glass, indices=indices, material=shared)

world = ECS()
world.register(MaterialInstance, MaterialTransform)
[left, right] = world.create(2)
world.add(left, MaterialInstance(mesh, overrides={"grazing_fade": True}),
          MaterialTransform(position=Vec3(-1, 0, 0)))
world.add(right, MaterialInstance(mesh, overrides={"tint": (1, 0.6, 0.3, 1)}),
          MaterialTransform(position=Vec3(1, 0, 0)))

renderer = MaterialRenderer(meshes=[mesh])
adapter = EcsMaterialAdapter(renderer, world)

# Before recording each frame:
adapter.sync()
commands = RenderContext.commands()
renderer.record(commands, size, camera, output_attachment)
RenderContext.submit(commands.finish())
```

`MaterialInstance` uses `@component(multiples=True)`: an entity can own multiple
renderables. Mesh, shader, dictionary and override references use explicit object
fields; transforms use separate numeric columns. The adapter also accepts
application component types exposing the same fields through its material_type
and transform_type arguments.

Material selection resolves the entity's dictionary, then the mesh's dictionary,
then the shader default. Values resolve declaration defaults, shader defaults,
selected dictionary, then instance overrides. The entity may select another
registered shader with its shader field.

The adapter watches component membership, layout and field versions. It refreshes
selections after storage compaction, handles spawning/deletion and empty worlds,
and preserves a component's surface_id through repacking. Zero surface IDs are
allocated once and written into ECS storage. Explicit nonzero IDs must be unique
within the renderer. A new renderable gets a new ID; deleting another entity
does not reseed its noise. Preserve surface_id when replacing an existing component
if it represents the same physical surface.

## Editing dictionaries

Plain dictionaries do not track nested writes. Notify the adapter explicitly:

```python
shared["roughness"] = 0.8
adapter.touch_materials(shared)

# Replacing a shader's default also needs notification:
glass.default_material = {"roughness": 0.5}
adapter.touch_materials(glass.default_material)

# Alternatively, notify through the ECS accessor:
refs = world.get(MaterialInstance)
refs[left].overrides[0]["grazing_fade"] = False
refs.touch("overrides")

# Ordinary ECS field writes already advance versions:
world.get(MaterialTransform)[left].position = Vec3(-2, 0, 0)
```

`touch_materials(*dictionaries)` invalidates the named shared/default/override
dictionaries. Unaffected shared records reuse their resolved values and packed
data. Editing a shader-default dictionary invalidates all records inheriting it;
replacing that dictionary also invalidates those records. `touch_materials()`
with no arguments remains a full invalidation, as do ECS material field edits.
Notify every dictionary edited since the previous sync; dictionaries remain
ordinary mutable objects, without automatic nested change tracking.

Unchanged ECS frames perform no material packing or uploads. Transform-only edits
update transforms without packing materials. Material updates compare GPU buffer
contents by words and upload coalesced changed ranges; nearby changes may share
an upload to reduce queue overhead. Membership changes can rebuild record
addresses. Dictionary edits and default replacement require no shader compilation.
Instance values do not split draw batches.

## Shader-defined materials

`MaterialShader` accepts exactly one of `source` (inline Jinja2 template),
`filepath` or `template` (path to a Jinja2 file), plus a `MaterialInterface`.
`DefaultMaterialShader` supplies the built-in lighting, alpha and vertex hooks.
Custom materials extend its template and override only the hooks they need.

For example, this is the striped material used in
[material_example.py](../scenes/material_example.py), with a minimal lighting override:

```python
striped = DefaultMaterialShader(source='''
#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> stripe_color: vec3f = (1, 0.15, 0.03);
<MaterialParameters> stripe_frequency: f32 = 8.0;
#endblock
#block lighting
    let band = smoothstep(-0.2, 0.2, sin(uv.y*p.stripe_frequency*6.2831853));
    var striped = material;
    striped.base_color = vec4f(mix(material.base_color.rgb, p.stripe_color, band), material.base_color.a);
    return default_lighting_{{ PREFIX }}(striped, p, uv, position, normal, lod);
#endblock
''')
```

In a material's `parameters` block, `<MaterialParameters>` declares shared
values, `<InstanceParameters>` declares per-instance overrides, and
`<FrameParameters>` selects renderer-owned frame values. Each declaration
requires a default, expressed as a Jinja2 value (scalar or tuple).
The available inherited blocks are `parameters`, `helpers`, `surface`, `vertex`,
`opacity`, `grazing` and `lighting`. `super()` retains inherited parameter
schema and defaults. Parameters are accessed as `p.name`; the template owns
material-family naming through `{{ PREFIX }}`. `TRANSPARENT` provides the
routing choice and `HOOKS` identifies the hooks needed by the current pass.
The `MaterialRenderer` supplies the frame, lighting and texture bindings;
material templates must not declare GPU entry points.

A generic `MaterialShader` can extend `material_surface.shaderlib` and implement
its own hooks without depending on the default lit material. The
`MaterialInterface` specifies the required hook contract.

## Time and vertex displacement

Time is an optional frame-scope parameter, declared in a Jinja parameters block:

```jinja2
#block parameters
{{ super() }}
<FrameParameters> time: f32 = 0.0;
#endblock

#block vertex
    let offset = 0.1*sin(p.time*2.0);
    return MaterialVertex(position+vec3f(offset, 0.0, 0.0), normal, uv);
#endblock
```

`p.time` is elapsed seconds since the MaterialRenderer was created. Supply
`renderer.record(..., time=seconds)` to use an application clock, seek, pause or
render reproducibly. The example accumulates RenderContext.frame_time and stops
advancing its clock while paused. Low-level TransparencyPass.update_frame() and
update_camera() accept `time=seconds`, defaulting to zero.

Frame parameters are read directly from the frame uniform. They do not occupy
shared/instance parameter slots or require material dictionaries to be repacked,
and cannot be assigned through those dictionaries. `time: f32` is currently the
frame parameter exposed by the opaque/stochastic MaterialInterface. The generic
MaterialLibrary takes frame expressions from its adapter's FrameParameter metadata.

The vertex hook receives object-space position/normal and UVs, and returns
`MaterialVertex(position, normal, uv)` before the instance transform. Update the
normal when deformation changes the surface orientation. The example computes
an analytic normal for its radial ripple. Static materials retain the identity
hook. A Jinja child can override the `vertex` block for custom deformation.

All raster passes use the hook, and transparent winner shading applies it to
each original triangle vertex before barycentric interpolation. This keeps
positions, normals and UVs consistent with rasterization, including MSAA.
Object-space noise anchors retain the original rest positions.

GPU displacement does not change CPU mesh bounds. Pass conservative local
`bounds=(center, extents)` to MaterialMesh when vertices can move outside the
original bounds. Bounds must include every allowed amplitude and time; the
example reserves a 1.3 radius for its maximum 0.3 displacement.

## Template parameter declarations

Use `source=`, `filepath=` or `template=` to supply Jinja2. Declare types,
defaults and scope inside a Jinja parameters block:

```jinja
#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> rim_strength: f32 = 0.8;
<MaterialParameters> rim_color: vec3f = (0.3, 0.7, 1.0);
<InstanceParameters> rim_power: f32 = 3.0;
#endblock
```

The scope defaults to material. Use super() to retain inherited declarations.
A child may add fields, replace defaults, or change scope while preserving the
inherited type. Omitting super() replaces the inherited schema. Parameters are
available as p.name in WGSL; declarations produce no WGSL text themselves.

```python
rim = DefaultMaterialShader(
    transparent=True,
    template="scenes/shaders/rim_material.shader",
    default_material={"rim_strength": 1.5},
)
```

The `parameters` attribute exposes read-only declaration metadata. Changing
declarations requires a new shader and renderer; ordinary dictionary edits use the
existing explicit invalidation path. TRANSPARENT is available in templates for
route-dependent defaults, as used by the default alpha_mode declaration.

Supported types are f32, vec2f, vec3f, vec4f, u32 and bool. Each parameter declares
material, instance or frame scope. Shared dictionaries are deduplicated by identity per
shader. Fields share lanes in aligned 16-byte `vec4u` slots, preserving
float/integer bits. Scalars use one lane, bools use uint32 zero/one, and vectors
remain within one slot. `MaterialLibrary.parameter_layouts[shader][scope]`
exposes `offsets` (field to slot/lane) and `slot_count`; record offsets are in
slots, not bytes. Field declaration order does not imply GPU offsets.
Overrides of shared fields are rejected; declare instance scope when needed.

The default_material.shader template declares base_color, emission, roughness, metallic, opacity_rgb,
alpha_mode, alpha_cutoff, texture_layer, double_sided, ambient, diffuse, specular,
grazing_fade, grazing_start and grazing_end. Tint and opacity_multiplier have
instance scope by default. Built-in scopes may change while retaining their
types. These requirements and semantic checks belong to DefaultMaterialShader.

Its alpha modes are 0 OPAQUE, 1 MASK, and 2 BLEND. Transparent routing defaults
to BLEND; opaque routing defaults to OPAQUE. MASK uses sharpened coverage on the
transparent path. Grazing fade defaults off at 70/85 degrees, independently of
the legacy TransparencyPass frame override.

Texture selection uses texture_layer in a renderer-owned texture array, supplied
through base_textures. A white array is supplied when omitted. Camera and
directional lighting remain frame data.

The blue_noise argument is optional on MaterialRenderer and TransparencyPass.
By default, the packaged 64x64 tile is uploaded lazily and shared among active
renderers on the same GPU device. Device changes or closed textures trigger a
fresh upload. The weak cache does not retain old devices. Loading requires no
Pillow or scene assets. Supply blue_noise explicitly to use a caller-owned tile.

## Template inheritance

The example's [rim shader](../scenes/shaders/rim_material.shader) extends
default_material.shader. That template extends material_surface.shaderlib,
the minimal opaque/stochastic surface contract.

Available blocks:

- parameters: declares typed fields, default values and sharing scope.
- surface: converts typed parameters into the rendering technique's Material.
- vertex: returns object-space position, normal and UVs as MaterialVertex.
- lighting: returns straight linear HDR color.
- opacity: modifies evaluated RGB opacity consistently in guide, transmittance
  and selection, before clamping and grazing fade.
- grazing: supplies start/end angles and enabled state.
- helpers: defines additional WGSL functions.

```jinja
#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> rim_strength: f32 = 0.8;
<MaterialParameters> rim_color: vec3f = (0.3, 0.7, 1);
#endblock
#block lighting
    let view = normalize(frame.camera_position.xyz-position);
    let rim = pow(1.0-abs(dot(normalize(normal), view)), 3.0);
    return default_lighting_{{PREFIX}}(material, p, uv, position, normal, lod)
        + p.rim_color*p.rim_strength*rim;
#endblock
```

Declare extra fields in the shader schema. PREFIX makes family-specific functions
unique. The loader searches application and engine directories. Lighting must be
callable from fragment and compute shading: use explicit texture LOD and avoid
fragment-only derivatives.

A custom shader can instead extend material_surface.shaderlib, override surface
to fill the technique's surface record, and use its default unlit lighting and
disabled grazing. ShaderHook/MaterialInterface describe the adapter's function
contract and metadata addressing; MaterialLibrary generates dispatch without
interpreting any parameter names.

All transparent families share winner selection, reconstruction and MSAA targets.
Transparent draws have no opaque prepass. MaterialRenderer uses frustum culling
and caller-owned commands/output; presentation and submission remain external.

Supply existing scene targets as opaque_color and opaque_depth to MaterialRenderer.
Their contents are preserved: transparent-only rendering skips the opaque pass,
and local opaque geometry loads the existing color/depth before adding its draws.
Both textures must be sampleable and match the selected sample count and size;
depth requires render-attachment usage, as does color when local opaque draws
are present. With internal opaque targets, each frame clears them normally.

## Resource lifetime and direct array use

Register mesh resources and shader families at renderer construction; spawning
entities using registered resources needs no renderer rebuild. New shader
layouts, families or geometry require constructing a new renderer.
set_sample_count(1 or 4) retains material data and rebuilds compatible passes.

MaterialRenderer.update() also accepts explicit mesh IDs, bindings, packed
transform columns, owner IDs and persistent surface IDs. This is the array API
used by the ECS adapter. DrawInstance and the optional constructor instance list
remain a direct-draw convenience; that path scans its plain Python inputs on each
record and does not provide ECS change tracking.

MaterialLibrary.pack() is CPU-only and returns material headers, shared slots,
instance slots, material IDs and instance offsets. Headers store shader ID and
shared offset; they contain no default-material record. The opaque/stochastic
adapter binds shared slots at group 2 binding 6, instance slots at binding 7,
and headers at binding 8. Its metadata y/w fields select material/instance data.

## Standard WGSL struct names and blocks

These names describe the built-in opaque/stochastic adapter's ABI. Shared types
use PascalCase; bound values use snake_case. Custom material code uses the
provided values and types without redeclaring bindings. Family-owned declarations
are namespaced automatically, so local helper/type names can be reused between
families.

| Struct | Bound value / use | Ownership and updates |
| --- | --- | --- |
| `Frame` | `frame`, uniform at group 0 binding 0 | View/projection, camera position, clip/noise settings, legacy grazing and time; changed ranges only |
| `Lighting` | `lighting`, uniform at group 0 binding 1 | Scene directional light, currently `light_dir: vec4f`; uploaded only when changed |
| `Material` | Shader-local evaluated surface value; direct materials also use `array<Material>` | Common color, emission, surface, opacity and flags contract |
| `Materials` | `material_parameters`, storage at group 2 binding 6 | Runtime `values: array<vec4u>` containing compact shared dictionary records |
| `InstanceOverrides` | `instance_parameters`, storage at group 2 binding 7 | Runtime `values: array<vec4u>` containing compact per-instance parameters |
| `MaterialHeaders` | `material_headers`, storage at group 2 binding 8 | Runtime `values: array<vec4u>`; shader ID and shared-slot offset per material |
| `MeshInstance` | Raster `instances` and deferred `shading_instances` tables | Existing 32-byte packed position/rotation/scale ABI; shared by render and culling |
| `MaterialVertex` | Vertex-hook return value | Object-space position, normal and UV before instance transformation |
| `ShadingVertex` | Deferred `vertices` table | Original triangle vertices for winner reconstruction |
| `BilateralSettings` | `bilateral`, composition uniform | Reconstruction sigmas, kernel and debug mode; changed ranges on edit |
| `CameraUniforms` | Culling `camera_params` uniform | Derived view-projection/planes/camera plus culling counts; unchanged cameras skip updates |
| `ParamsN` | Generated shader-local family value | Typed access to shared fields, overrides and opted-in frame parameters |

`MaterialParameters`, `InstanceParameters` and `FrameParameters` are material
authoring schemas, not extra bound GPU blocks. The linker preserves declarations
after namespacing; it does not analyze reachability or remove unused code.
Control emitted shader size by splitting reusable WGSL into granular Jinja
includes or macros, importing only the helpers a template needs. Jinja-authored
family helpers should use `PREFIX` to keep their names unique, as in the template
examples above. Default lighting math lives once in `material_lighting.shaderlib`;
family-specific `default_lighting` adapters retain each family's coefficients.

`Frame` and `Lighting` are declared once in `frame.shaderlib` and included by
raster, deferred and composition modules. `Frame` is currently 192 bytes and
`Lighting` 16 bytes. Supply lighting from Python with:

```python
renderer.record(commands, size, camera, output, time=seconds,
                light_dir=(0.4, 0.8, 0.3))
```

The low-level `TransparencyPass.update_frame()` and `update_camera()` accept the
same `light_dir` argument. Material response coefficients (`ambient`, `diffuse`,
`specular`) remain material parameters; the light direction is scene data.
Shaders that previously read `frame.light_dir` should read
`lighting.light_dir`. Rebuild custom uniform dtypes from the shared include or
reflection rather than retaining the earlier Frame byte offsets.

Lighting currently supports a scene directional light; no local-light distribution
compute pass is implemented.

## Bind-group convention and lifetime

Built-in material passes derive explicit layouts from reflected shader bindings.
The same group-layout objects own bounded GPU bind-group caches across compatible
pipelines and across 1x/4x MSAA. Cache keys include buffer handles and bound sizes,
so growth or resource replacement selects a new bind group; changing contents
preserves it. Custom shader sets keep the automatic-layout path.

| Group | Raster geometry | Deferred shading | Composition |
| --- | --- | --- | --- |
| 0 | `Frame` + `Lighting` | Same shared group | Same shared group |
| 1 | Instance/visibility indexing | Winner input/output textures | Color/depth/guide textures and `BilateralSettings` |
| 2 | Material tables, metadata, texture array and sampler | Material resources plus vertex/transform tables | Unused |
| 3 | Pass-specific depth/noise inputs | Unused | Unused |

Group 1 retains the existing DrawBatches convention. Pass groups can differ;
compatible low-index layout prefixes allow bindings to survive pipeline changes.
Bindings have explicit resource types and stage visibility; pass texture groups
include only resources needed by that pass. Default depth geometry is compiled
lazily when requested.

Buffer update cadence is separate from binding cadence: frame contents change
each frame while the group remains stable throughout the draws. Continue using
indexed storage for per-instance values. Do not overwrite one frame buffer for
multiple views recorded before the same submission; each such view needs its
own buffer or aligned range. `upload_changed()` on reflected uniform buffers
tracks raw bytes, including padding; ordinary `upload()` remains available.

The [dated material audit](materials_audit.md) records storage and upload
measurements for the six-sphere example. These describe that setup, not general
performance guarantees.
