# Material WGSL audit

Audited 2026-11-10 using `scenes/material_example.py`, Python 3.14,
wgpu-py 0.31.0, and an NVIDIA RTX 5060 Ti through Vulkan. The findings below record
the original baseline; the implementation status follows here. Sizes describe generated source,
logical buffer contents, or proposed layouts, not measured GPU execution time.

The existing sharing model is sound. Keep shader-indexed material and instance
tables. The clearest improvements are compact parameter packing, partial material
updates, shared explicit bind-group layouts, and less repeated generated code.

## Implemented improvements

The renderer now packs fields into shared vec4u lanes, uploads changed word
ranges, and accepts targeted `adapter.touch_materials(*dictionaries)` invalidation.
It derives explicit layouts from reflected bindings, shares bind-group caches
across compatible passes/MSAA, and extracts one lighting implementation. Material
template render results are cached per prefix/routing. Auxiliary depth geometry
is compiled lazily. Culling skips unchanged camera data.

Validation: the implementation previously passed **602 tests** in the full suite.
After removing declaration processing, **79 focused material/shader tests passed**,
including GPU rendering at 1x/4x MSAA, mixed float/vector/uint32/bool packing,
default inheritance, targeted invalidation, raw-byte uniform snapshots, shared
bind groups after buffer growth, shader declaration preservation, and resource lifetime checks.

| Measurement on the same six-sphere setup | Baseline | Implemented |
| --- | ---: | ---: |
| Shared material storage | 1,280 bytes | 512 bytes |
| Instance parameter storage | 224 bytes | 192 bytes |
| Single shared roughness edit | 1,280 bytes uploaded | 4 bytes uploaded |
| Warm frame, advancing time only | 400 bytes / 2 uploads | 4 bytes / 1 upload |
| Unchanged paused frame | 400 bytes / 2 uploads | 0 uploads |
| Cold bind-group cache misses, including culling | 22 | 12 |
| Generated material library source | 19,898 bytes | 17,973 bytes |

The shared `Frame` struct is now 192 bytes. Scene light direction moved into
`Lighting`, a separate 16-byte uniform. Both live in the same reusable group 0.
The standard names `Materials`, `MaterialHeaders`, `InstanceOverrides`,
`Material`, `MeshInstance`, and the authoring schemas are documented in the
[material guide](materials.md#standard-wgsl-struct-names-and-blocks).
Future per-instance light distribution is reserved for a separate compute/storage
path; it has not been implemented.

Declaration reachability processing and hook-body inspection have been removed.
The linker preserves supplied declarations after namespacing. Authors control
source size with granular Jinja includes, imports and macros; GPU optimization
belongs to the shader compiler. Reflection remains for parameter schemas,
namespacing, signatures, optional-hook detection and bind-group layout usage.

Local baseline snapshots are in `build/material_wgsl_audit_before/`; current
snapshots and the reproducible probe are in `build/material_wgsl_audit/` and
`build/material_wgsl_audit_probe.py`. The original findings below retain their
historical measurements and recommendations. Unimplemented items include fused
surface/opacity/grazing dispatch, stable material allocations across membership
changes, finer frame/settings separation, and broader device-scoped texture,
sampler and module caches. Those need separate profiling/design work.

## Original generation path and baseline

The path is:

1. [The example](../scenes/material_example.py) creates four shader families:
   opaque default, glass, rim, and plain-WGSL striped. Six draws resolve to five
   distinct material records because the two glass draws share their dictionary.
2. [MaterialShader](Materials.py) renders declarations and default functions from
   Jinja. Plain WGSL still uses the default template as its implementation base.
3. [MaterialSource](MaterialSource.py) reflects `MaterialParameters`,
   `InstanceParameters`, and `FrameParameters`, checks overrides, and namespaces
   symbols using the existing `ShaderSource`/`ShaderSyntax` machinery.
4. `MaterialLibrary._source()` emits three storage bindings, one `ParamsN` and
   decoder per family, the family functions, and five hook dispatch switches.
5. [TransparencyPass](TransparencyPass.py) inserts the library into geometry and
   winner-shading modules. [Shader](RenderContext.py) renders the outer templates,
   expands compact struct declarations, reflects the result, and creates WGSL
   modules. Pipeline layouts default to automatic inference.

`ParamsN` is a shader-local value, not another GPU uniform buffer. Likewise,
`MaterialParameters_3` and `FrameParameters_3` in the emitted module are schema
declarations; no buffers are allocated for them. The additional `time` field
decodes as `p.time = frame.time` and consumes no material or instance slot.

| Captured item | Size / count |
| --- | ---: |
| Material library | 19,898 bytes, 319 lines, 36 functions |
| Geometry module, 1x | 40,477 bytes, 781 lines |
| Winner-shading module, 1x | 27,447 bytes, 525 lines |
| Composition module, 1x | 8,739 bytes, 225 lines |
| Material headers | 80 bytes, five records |
| Shared material slots | 1,280 bytes |
| Instance parameter slots | 224 bytes |
| Frame uniform | 208 bytes, shared by raster/shading/composition |
| Culling camera uniform | 192 bytes |
| Bilateral settings | 32 bytes |

Local snapshots and the measurement script are in ignored `build/` output:
[baseline library](../build/material_wgsl_audit_before/material_library.wgsl),
[baseline geometry](../build/material_wgsl_audit_before/geometry.wgsl),
[baseline shading](../build/material_wgsl_audit_before/shade.wgsl),
[baseline composition](../build/material_wgsl_audit_before/compose.wgsl),
[baseline measurements](../build/material_wgsl_audit_before/summary.json), and
[probe](../build/material_wgsl_audit_probe.py). These are local audit artifacts,
not tracked package assets. From the repository root, reproduce with
`python build/material_wgsl_audit_probe.py` while the probe is available.

## Findings, in recommended implementation order

### 1. Material pooling exists, but updates invalidate the entire pool

`MaterialLibrary.pack()` deduplicates by `(shader, dictionary identity)`. The two
glass spheres correctly share one material record even though they have distinct
instance overrides. Separate dictionaries with equal values remain separate;
that preserves their independent mutation semantics.

`EcsMaterialAdapter.touch_materials()` invalidates the whole material selection.
`MaterialRenderer.update()` repacks every binding, and `_write()` compares entire
buffers by making byte strings. Changing only shared glass roughness produced
one 1,280-byte upload. Headers, instance data, and metadata were unchanged and
were correctly not uploaded. Transform-only edits and time animation already
avoid material repacking through the ECS path.

Keep the pools, but give records stable allocations and track dirty dictionaries
or material handles. Resolve/validate affected records and upload coalesced dirty
ranges using the existing `GpuBuffer.write_range()` or `upload_range()`.
Keep `touch_materials()` as a full-invalidation fallback for ordinary dictionaries.
A shader-default replacement must invalidate records that inherit those values.
Membership changes may still require rebuilding metadata and instance addresses.

This offers a larger scaling benefit than creating more uniform buffers. For
the six-sphere example, the current upload is small in absolute terms.

### 2. One 16-byte slot per parameter is unnecessary padding

`Parameter.pack()` always returns four uint32 words. Every scalar, uint, and bool
therefore occupies 16 bytes. This is an engine convention, not a requirement of
WGSL storage buffers. Integer bits and bool-as-u32 encoding are worth preserving.

| Family | Shared fields | Current shared bytes / record | Typed storage struct, declaration order |
| --- | ---: | ---: | ---: |
| Opaque default | 15 | 240 | 112 |
| Glass | 14 | 224 | 96 |
| Rim | 17 | 272 | 128 |
| Striped | 19 | 304 | 144 |

With two opaque records and one of each other family, declaration-order typed
packing would use **592 shared bytes instead of 1,280**, a 54% reduction. This is
a layout calculation, not an implemented optimization. The same conversion does
not reduce instance bytes in this example: alignment leaves the total at 224.

Prefer a schema-generated layout that packs multiple scalar fields into lanes
of a pooled word/vec4 arena and gives both CPU packing and WGSL accessors the same
offset metadata. This preserves a small number of bindings across heterogeneous
families. Alternatively, generate correctly aligned typed records, with a clear
strategy for storing different family layouts in the shared allocation.

A tightly packed `array<u32>` arena could hold the shared scalar/vector payloads
in 476 bytes and instance payloads in 128 bytes, but vector reconstruction and
load behavior need benchmarking. Those are payload sizes, not typed struct sizes.
Do not assign those byte counts to an `array<vec3f>` or ordinary WGSL struct.
WGSL defines both member alignment and array stride; see the
[WGSL memory layout rules](https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size).

### 3. Buffers are shared; GPU bind groups remain pipeline-specific

`TransparencyPass` allocates one frame buffer and passes it to all three stages.
Material/instance buffers are similarly shared. There is no per-material copy
of the camera uniform to remove.

However, `RenderPipeline` and `ComputePipeline` use `layout="auto"`, and
`_PipelineVariant` owns its own GPU bind-group cache. Raster opaque, guide,
transmittance, winner selection, winner shading, and composition consequently
create separate GPU bind-group objects around overlapping resources. Warm frames
had 22 bind-group cache hits and no misses, including culling. This is persistent
duplication and a cross-pipeline reuse limitation, not GPU object creation every
frame. Python `BindGroup` descriptors are still constructed per batch in
`DrawBatches.draw()`.

Introduce explicit reusable layouts for the material renderer, then cache GPU
bind groups by device, layout, and bound resource/range. `RenderPipeline` already
accepts an explicit layout. Keep compatible low-index layouts identical across
pipelines; composition currently mixes the frame uniform with its textures in
group 0, preventing a common frame bind group. The render wrapper also clears
all its bound-group state on a pipeline variant change; compatibility-aware
state retention can follow once shared layouts exist.

Define binding numbers, resource kinds, visibility, and buffer ABI in one
adapter-owned resource schema that emits both WGSL declarations and explicit
layouts. Today the generic `MaterialLibrary._source()` hardcodes group 2,
bindings 6/7/8, while the transparency templates declare the remaining bindings
separately. Adding manually maintained layout descriptors would create another
representation to keep synchronized. `MaterialInterface` is a natural place to
reference adapter binding metadata alongside hook/frame metadata.

An explicit-layout implementation must also update `_PipelineVariant`'s binding
filtering: today the explicit path treats all declared shader bindings as active,
whereas the auto path filters by entry-point use. It needs the actual supplied
layout's slots and requirements. Set appropriate stage visibility and minimum
binding sizes. Preserve buffer-growth/resource-replacement invalidation and
bounded caches.

Automatic layouts are pipeline-exclusive; see the
[WebGPU default layout rules](https://gpuweb.github.io/gpuweb/#default-pipeline-layout).
The [bind-group guidance](https://toji.dev/webgpu-best-practices/bind-groups.html)
also explains explicit layouts, reuse, and change-frequency grouping.

### 4. Default lighting is emitted four times

`default_material.shader` emits one `default_lighting_N` per family. Only the
three lighting scalars require the family-specific `ParamsN` type. Extract the
lighting math into one helper accepting `ambient`, `diffuse`, and `specular`, or
a small common lighting value. Retain `default_lighting_N` as a thin adapter so
plain WGSL's `default_lighting(...)` and Jinja overrides remain compatible.

`stochastic_material.shaderlib` also contains essentially the same math with
fixed coefficients for the direct/legacy material route. Route both through the
common helper. Keep that route: other scenes and composition tests use it.

The common surface representation already exists as the 80-byte `Material`
value. Standardize default-surface and lighting helpers around that contract,
with family-specific extension accessors for rim/stripes. Keep built-in field
policies inside `DefaultMaterialShader`; the generic material interface supports
minimal color-only shaders and fields whose scope changes in child schemas.

The library also retains schema-only structs, literal default constants, and
renamed overridden default hooks. For the striped family, `default_shade_3` and
`default_vertex_3` are uncalled. They are deliberately preserved: custom code can
call default helpers and reference constants. Authors can select smaller code
units with Jinja includes/imported macros rather than relying on declaration
reachability processing. GPU instruction savings from removing source declarations
were not established here.

### 5. Every hook dispatch reconstructs the whole parameter value

Each `library_material`, `library_shade`, `library_opacity`, `library_grazing`,
and `library_vertex` selects a family and calls `parameters_N(...)`. That decoder
assigns every declared field, even for an identity vertex or opacity hook.
Opaque fragments call material and shade dispatch separately; transparent
opacity evaluation additionally calls grazing and opacity dispatch. Deferred
shading calls vertex deformation three times before material and shade dispatch.

Preserve the three deformation evaluations: deforming original vertices before
interpolation is needed to match rasterization. Possible improvements are to
emit direct identity cases without a decoder, or fuse related surface/opacity/
grazing evaluation within one family dispatch. Hook-specific decoders require
actual field-use analysis; identifying only which top-level hooks exist is not
enough. Custom helpers and aliases must remain supported.

The emitted repetition is observable, but inlining, dead-code elimination, and
common-subexpression elimination can remove loads. Do not count source
assignments as executed memory traffic. Profile this after simpler packing and
CPU update changes; keep family dispatch if it supports efficient batching.

### 6. Frame declarations and update ownership can be standardized

`stochastic_common.shaderlib` and `stochastic_compose.shader` duplicate the Frame
declaration. `draw_common.shaderlib` has another smaller view/projection/light
block; culling has view-projection, planes, and camera position in a separate
block. Share declarations through includes and define the ABI once. Derive view,
projection, view-projection, camera position, and culling planes once per view;
keep culling workgroup/batch/command counts in culling-owned settings.

Do not immediately merge all of these buffers. Frame currently mixes fast
view/time data with slow light/noise/grazing settings. The frame grazing values
are used by the legacy route; dictionary materials take grazing from their own
parameters. Bilateral settings already have a separate 32-byte buffer and update
on edits. Preserve that useful separation of ownership and update cadence.

The probe measured two uploads totaling 400 bytes on both a warm animated frame
and an unchanged paused frame: 208 frame bytes plus 192 culling-camera bytes.
`update_frame()` always uploads; `update_cull_camera()` always marks its buffer
dirty. Add value/version checks, allow changed-range updates, and accumulate
settings changes before recording instead of immediately uploading the full
frame block from every setter. These small uploads alone are unlikely to justify
many extra buffers or bind groups.

The constructor also performs a full material render/link to validate its
declarations, then the library renders/links again. The library reparses emitted
functions to discover optional hooks. Return/cache linked code, declarations,
and hook metadata together instead of re-reflecting output where possible.
The material Jinja pass and outer pass-template assembly serve different
purposes; both can remain. Reuse the existing lexical rewriting implementation.

One additional startup inefficiency occurs when constructing a 4x pass:
`depth_shader = self._shader_set(1)[0]` eagerly creates the entire 1x shader set,
including 1x shading/composition modules even if only the geometry module is
needed. Make that auxiliary set lazy or request only the required module.

## Suggested standardized blocks and groups

Separate **data update cadence** from **binding changes between draws**. Writing
a buffer's contents does not require creating or replacing its bind group.
Frame data changes every frame but its resource binding can remain stable
through every draw. An indexed instance table also stays bound while instances
vary. Low group numbers should favor bindings shared across pipelines/draws,
as described in the [WebGPU pipeline layout guidance](https://www.w3.org/TR/webgpu/#gpupipelinelayout).

| Block | Ownership and cadence | Storage |
| --- | --- | --- |
| FrameGlobals | Application clock, frame number if needed; once per frame | Small uniform |
| ViewGlobals | Matrices, camera, clip distances, frustum planes; once per view/change | Uniform |
| LightingGlobals | Directional light/environment; when scene lighting changes | Uniform; storage for large light lists |
| OITSettings / BilateralSettings | Noise, legacy fade, reconstruction/debug; on edit | Small uniforms |
| MaterialTable + extensions | One allocation per shared dictionary/family; dirty ranges on edit | Read-only pooled storage |
| InstanceTable + overrides | Transforms, material addressing, surface identity, instance values; on change | Indexed storage |
| PassResources | Depth/guide/winner inputs and outputs; on resize/pass change | Textures/storage bindings |

These are semantic blocks; several can be bindings within one group. A clean
longer-term four-group convention could be:

| Group | Resources |
| --- | --- |
| 0 | Common frame/view/lighting/settings uniforms |
| 1 | Persistent material/scene tables, transform/geometry tables, texture array and sampler |
| 2 | Per-view draw/visibility indexing |
| 3 | Pass-specific textures, winner inputs, and output resources |

This ordering is a proposed migration, not an API requirement. The current
`DrawBatches` convention reserves group 1 for instance/visibility bindings;
renumbering needs coordinated changes. Start with a shared group-0 layout and
frame declaration while retaining groups 1/2 elsewhere. Standalone culling can
retain a separate pipeline layout with its own writable work buffers. Check
per-stage binding limits when using union layouts; avoid making every resource
visible in every stage or requiring a fifth group on baseline devices.

Keep per-instance storage indexing for this renderer. A dynamic-uniform arena
is useful for a small per-draw block when separate draw calls need different
values, but does not replace instance indexing within one instanced draw. If
introduced, query `minUniformBufferOffsetAlignment`; this adapter reported 64
bytes, so hardcoding a universal alignment would be wrong. The commonly portable
256-byte assumption should not be confused with struct member alignment.

Pooling frame/view storage across multiple cameras also needs distinct ranges
for views recorded before one submission. Queue writes do not snapshot values
at each encoded draw; the current example updates once and then submits. See
[wgpu queue write semantics](https://docs.rs/wgpu/latest/wgpu/struct.Queue.html#method.write_buffer).

Additional safe device-scoped pooling candidates are the fallback white texture,
default material sampler, and shader modules with identical source. The blue
noise texture already has such a device-aware cache in `TextureProcessing`.
Preserve device-generation checks and resource lifetimes in any broader cache.

## Verification and limits

The six-sphere scene was recorded headlessly at 96x72 at 1x and after switching
to 4x MSAA. Statistics covered initial, warm, paused, and single-material-edit
updates. No visual or GPU timing comparison of proposed changes was performed.

The existing focused suite passed: **43 tests**, covering materials, plain WGSL,
animation, and ECS extraction, including GPU cases. Command:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_materials.py tests/test_material_source.py tests/test_material_animation.py tests/test_ecs_materials.py
```

Implement compact layouts and dirty record updates first, then shared explicit
layouts and default-helper extraction. Evaluate fused dispatch and more elaborate
frame-block splitting with profiling evidence. Keep the existing dictionary
identity semantics, per-instance batching, namespace isolation, frame-time path,
and raster/deferred deformation agreement throughout.
