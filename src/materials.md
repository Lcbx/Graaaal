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

## ECS usage

```python
from numpyecs.ECS import ECS
from numpyecs.Math import Vec3
from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.MaterialRenderer import MaterialMesh, MaterialRenderer
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

MaterialInstance is a registered multicomponent: an entity can own multiple
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
adapter.touch_materials()

# Replacing a shader's default also needs notification:
glass.default_material = {"roughness": 0.5}
adapter.touch_materials()

# Alternatively, notify through the ECS accessor:
refs = world.get(MaterialInstance)
refs[left].overrides[0]["grazing_fade"] = False
refs.touch("overrides")

# Ordinary ECS field writes already advance versions:
world.get(MaterialTransform)[left].position = Vec3(-2, 0, 0)
```

Unchanged frames perform no material packing or uploads. Transform-only edits
update transforms without packing materials. A material update packs the selected
rows once; unchanged GPU buffers retain their contents. Dictionary edits and
default replacement require no shader compilation. Instance values do not split
draw batches.

## Shader-defined schemas

MaterialShader requires an interface and exactly one of `source` (plain WGSL),
`filepath` (a plain WGSL file), or `template` (a Jinja template file).
Python supplies material values and instance overrides; declarations stay in
the shader. DefaultMaterialShader supplies default surface and lighting behavior.
A minimal unlit shader can use a color-only schema; it does not need roughness,
alpha_mode or grazing parameters.

## Plain WGSL materials

The example's `STRIPED_SHADER` is ordinary WGSL embedded in Python. It has no
Jinja statements or substitutions:

```python
striped = DefaultMaterialShader(source='''
struct MaterialParameters {
    stripe_color: vec3f,
    stripe_frequency: f32,
}
const default_stripe_color = vec3f(1.0, 0.15, 0.03);
const default_stripe_frequency = 8.0;

fn shade(material: Material, p: Params, uv: vec2f,
         position: vec3f, normal: vec3f, lod: f32) -> vec3f {
    let band = smoothstep(-0.2, 0.2, sin(uv.y*p.stripe_frequency*6.2831853));
    var striped = material;
    striped.base_color = vec4f(mix(material.base_color.rgb, p.stripe_color, band),
                              material.base_color.a);
    return default_lighting(striped, p, uv, position, normal, lod);
}
''')
```

The existing `ShaderSource` reflection reads `MaterialParameters` for shared
fields and `InstanceParameters` for per-instance fields. Both structs are
optional. `const default_NAME` provides a field's default; supported defaults
are scalar literals or float-vector constructors with literal components
(including splats), rather than arbitrary constant expressions. Without an
explicit default, inherited fields keep their defaults and new fields start at
zero/false. Supported types match template parameters below. A declaration can
change a built-in field's scope while retaining its type. Duplicate fields
across parameter structs are rejected. `FrameParameters` additionally opts into
renderer-owned values such as `time`, as described below.

The renderer provides the combined `Params` type, including inherited fields,
and passes the selected shared and instance values to each function. Custom
types, constants, private globals and helper functions are namespaced for each
shader family so multiple materials can use the same names. Struct member names
are preserved. The resulting module uses the existing `Shader` compilation,
uniform reflection and pipeline machinery.

Source analysis is shared with the renderer: `ShaderSource.declarations` exposes
module declarations and initializers, and `rename_symbols(mapping)` performs
scope-aware rewriting. Reflected parameter/member `name_span` values and function
`body_start` positions address the expanded `ShaderSource.source`, after compact
struct declarations are expanded. Material linking uses this public API rather
than parsing WGSL again. Comments are masked without moving positions; numeric
literals and attribute enums are kept separate from symbol references.

With DefaultMaterialShader, implement any of these functions; omitted functions
keep their default behavior. Argument names may vary; types and order must match:

```wgsl
fn surface(p: Params) -> Material
fn shade(material: Material, p: Params, uv: vec2f,
         position: vec3f, normal: vec3f, lod: f32) -> vec3f
fn opacity(material: Material, p: Params, alpha: vec3f,
           uv: vec2f, position: vec3f, lod: f32) -> vec3f
fn grazing(p: Params) -> vec3f
fn vertex(p: Params, position: vec3f, normal: vec3f, uv: vec2f) -> MaterialVertex
```

`shade` returns straight linear HDR color. `opacity` returns RGB opacity used
consistently in guide, transmittance and selection. `grazing` returns start angle,
end angle and enabled state. The existing implementations are callable as
`default_surface`, `default_shade`, `default_opacity`, `default_grazing`, `default_vertex`, and
`default_lighting`, with their corresponding signatures. Function names starting
with `default_` are reserved. Generic MaterialShader has no implicit defaults:
its source must supply the functions expected by its MaterialInterface. The
vertex hook is optional on this interface and falls back to unchanged geometry.

Frame uniforms (`frame.view`, `frame.proj`, `frame.light_dir`,
`frame.camera_position`, `frame.clip`, `frame.grazing`, `frame.time`), the Material type,
`base_color(material, uv, lod)`, `base_textures`, and `material_sampler` are
provided by the existing opaque/stochastic shaders. Reference these resources
directly; material source must not redeclare bindings or GPU entry points.
Geometry entry points and pass outputs remain renderer-owned. Shading functions
must work in fragment and compute stages: sample textures with explicit LOD and
avoid fragment-only derivatives. Transparency, reconstruction and MSAA remain
shared across custom and default materials.

## Time and vertex displacement

Time is an optional frame-scope parameter, declared in the shader:

```wgsl
struct FrameParameters { time: f32, }

fn vertex(p: Params, position: vec3f, normal: vec3f, uv: vec2f) -> MaterialVertex {
    let offset = 0.1*sin(p.time*2.0);
    return MaterialVertex(position+vec3f(offset, 0.0, 0.0), normal, uv);
}
```

The compact form `<FrameParameters> time: f32;` is also supported by the existing
ShaderSource struct syntax. Jinja templates declare the same value with
`parameter("time", "f32", 0.0, "frame")` in their parameters block.

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
hook. A Jinja child can override the `vertex` block with the same behavior.

All raster passes use the hook, and transparent winner shading applies it to
each original triangle vertex before barycentric interpolation. This keeps
positions, normals and UVs consistent with rasterization, including MSAA.
Object-space noise anchors retain the original rest positions.

GPU displacement does not change CPU mesh bounds. Pass conservative local
`bounds=(center, extents)` to MaterialMesh when vertices can move outside the
original bounds. Bounds must include every allowed amplitude and time; the
example reserves a 1.3 radius for its maximum 0.3 displacement.

## Template parameter declarations

Existing Jinja files remain supported through `template=`. Declare types,
defaults and scope inside a Jinja parameters block:

```jinja
#extends "default_material.shader"
#block parameters
{{ super() }}
{{ parameter("rim_strength", "f32", 0.8) }}
{{ parameter("rim_color", "vec3f", (0.3, 0.7, 1.0)) }}
{{ parameter("rim_power", "f32", 3.0, "instance") }}
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

The parameters attribute exposes read-only declaration metadata. Python no longer
accepts a parameters constructor argument. Adding/changing declarations requires
creating a new shader and renderer; ordinary dictionary value edits use the
existing explicit invalidation path. TRANSPARENT is available in templates for
route-dependent defaults, as used by the default alpha_mode declaration.

Supported types are f32, vec2f, vec3f, vec4f, u32 and bool. Each parameter declares
material, instance or frame scope. Shared dictionaries are deduplicated by identity per
shader. Each material/instance field occupies an aligned 16-byte slot preserving float/integer bits.
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
{{ parameter("rim_strength", "f32", 0.8) }}
{{ parameter("rim_color", "vec3f", (0.3, 0.7, 1)) }}
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
