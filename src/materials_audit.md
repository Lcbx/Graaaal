# Material WGSL audit (2026-11-10)

This dated review used [material_example.py](../scenes/material_example.py),
Python 3.14, wgpu-py 0.31.0, and an NVIDIA RTX 5060 Ti through Vulkan. It compares
the original baseline with the implemented changes on the same six-sphere setup.
Sizes describe generated source or logical buffer contents; upload counts measure
CPU submissions. No GPU execution-time or visual-quality improvement was measured.

For current APIs and binding conventions, use the [material guide](materials.md).

## Recorded measurements

Four shader families (opaque default, glass, rim, and plain-WGSL striped) produce
five shared material records for six draws. The glass pair shares one dictionary;
its distinct instance overrides do not split the batch or duplicate that record.

| Measurement | Baseline | After implementation |
| --- | ---: | ---: |
| Material headers (five records) | 80 bytes | 80 bytes |
| Shared material storage | 1,280 bytes | 512 bytes |
| Instance parameter storage | 224 bytes | 192 bytes |
| Single shared roughness edit | 1,280 bytes uploaded | 4 bytes uploaded |
| Warm frame, advancing time only | 400 bytes / 2 uploads | 4 bytes / 1 upload |
| Unchanged paused frame | 400 bytes / 2 uploads | 0 uploads |
| Cold bind-group cache misses, including culling | 22 | 12 |
| Generated material library source | 19,898 bytes | 17,973 bytes |
| Geometry module, 1x | 40,477 bytes | 39,554 bytes |
| Winner-shading module, 1x | 27,447 bytes | 26,605 bytes |
| Composition module, 1x | 8,739 bytes | 8,971 bytes |
| Frame uniform | 208 bytes | 192 bytes |
| Separate lighting uniform | None | 16 bytes |

Culling-camera and bilateral uniforms remained 192 and 32 bytes respectively.
The frame/lighting split changes update ownership, not their combined byte size.
The baseline already shared frame/material buffers; its bind-group caches were
pipeline-specific. Warm baseline frames reused those groups rather than creating
new GPU objects each frame.

## Implemented behavior

- `ParameterLayout` packs multiple fields into lanes of aligned `vec4u` slots;
  CPU packing and generated WGSL use the same slot/lane offsets. Integer bits and
  bool-as-u32 encoding are preserved.
- `EcsMaterialAdapter.touch_materials(*dictionaries)` targets edited dictionaries.
  Unaffected records reuse resolved/packed values; no arguments requests full
  invalidation. Default-dictionary replacement invalidates inheriting records.
- Material buffers compare words and upload coalesced changed ranges. Transform
  edits avoid material packing; instance values retain shared draw batching.
- Built-in material passes derive explicit layouts from reflected bindings.
  Compatible passes and 1x/4x MSAA share bounded bind-group caches; buffer-handle or
  binding-size changes select new entries.
- `frame.shaderlib` defines the common `Frame` and `Lighting` ABI. Uniform updates
  compare raw bytes, and unchanged culling camera data skips uploads.
- `material_lighting.shaderlib` supplies one lighting implementation, with thin
  family-specific adapters. Material template renders are cached by prefix and
  routing; auxiliary depth geometry compiles lazily.
- The linker namespaces and preserves supplied declarations. Reflection supports
  schemas, signatures, optional hooks, namespacing and binding usage; declaration
  reachability and hook-body pruning are not performed. Authors control source
  size with Jinja includes/imports/macros; shader optimization belongs to the
  GPU compiler.

Dictionary identity remains the sharing key within a shader family. Equal values
in distinct dictionaries retain independent mutation semantics. `ParamsN` is a
shader-local decoded value; `MaterialParameters`, `InstanceParameters` and
`FrameParameters` are authoring schemas, not additional bound buffers. Opted-in
`time` reads the frame uniform and consumes no material/instance slot.

## Remaining work and limits

Potential follow-up work needs profiling before implementation:

- Stable material allocations across membership changes; current edits can rebuild
  addresses when the renderable set changes.
- Fused surface/opacity/grazing dispatch or smaller decoders. Repeated generated
  assignments do not establish actual memory traffic after shader optimization.
- Finer frame/view/settings separation and broader device-scoped texture, sampler
  or shader-module caches. The default blue-noise texture already has a weak cache.
- Local-light distribution through separate compute/storage data; only directional
  scene lighting exists today.

Three deferred vertex-hook evaluations must remain consistent with rasterization:
deform the original triangle vertices before barycentric interpolation. Multiple
views recorded before one submission also need distinct uniform buffers/ranges;
queue writes do not snapshot a buffer separately for each encoded draw.

The current group convention keeps frame/lighting at group 0, DrawBatches instance
indexing at group 1, material tables at group 2, and pass depth/noise at group 3.
The original proposal to renumber these groups was not implemented. Do not infer a
new public binding ABI from an optimization proposal.

## Reproduction and validation record

Local, ignored artifacts used for this review are `build/material_wgsl_audit_before/`
and `build/material_wgsl_audit/`, including `summary.json` and generated WGSL.
They are not distributed package assets. If the local probe is available, run
`python build/material_wgsl_audit_probe.py` from the repository root to measure
the current checkout; reproducing the baseline also requires its original source.

The probe recorded headless frames at 96x72, then switched from 1x to 4x MSAA.
It counted cold, warm, paused and single-material-edit updates. Historical
validation reported 602 full-suite passes and a later 79-test focused run covering
material/shader behavior, 1x/4x rendering, mixed parameter packing, targeted
invalidation, uniform byte snapshots, buffer growth and resource lifetimes.
Those are results from that implementation session, not a fresh test run.

Run the current focused checks with:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_materials.py tests/test_material_source.py tests/test_material_animation.py tests/test_material_pooling.py tests/test_ecs_materials.py tests/test_shader_source.py src/tests/test_resource_lifetimes.py
```
