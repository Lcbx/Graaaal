# Engine API and internals improvement plan

Status: core implementation completed on 2026-11-07. The phase descriptions
below remain the design/acceptance checklist; the implementation record identifies
the delivered APIs and the deliberately deferred, workload-dependent work.

Review results:
- [P2] Allow released meshes to leave the draw registry — C:/Users/User/Desktop/tests/engines/numpyEcs/src/draw_compiler.py:267-271
  After removing a mesh's instances and calling mesh.release(), subsequent synchronization still visits the registered mesh: its changed generation marks it dirty, and this unconditional range access raises RuntimeError("Mesh allocation has been released"), even when _mesh_commands contains no destinations for it. There is no public unregister operation, so unloading an unused mesh breaks the retained DrawBatches object. Provide registry removal or skip range access for meshes without active destinations.
- [P2] Remove the remaining CPU-mirror dependency from draw_mesh — C:/Users/User/Desktop/tests/engines/numpyEcs/src/RenderContext.py:719-720
  When a newly supported GPU-only buffer is passed as instances, draw_mesh() still calculates the vertex-buffer offset using instances.content.dtype.itemsize. Since content is None, this raises AttributeError before reaching the updated count calculation, including when instance_offset is zero. Use instances.dtype.itemsize for the offset so GPU-generated instance buffers work through this drawing API.


## Implementation record

| Phase | Delivered |
| --- | --- |
| 1 | Installable `numpyecs` package with shader resources; relative imports; CPU-only math/assets/packing modules; import-safe scenes; removed naming aliases and `Utils.py` |
| 2 | `array_field()` dtype/shape metadata; explicit object fields; reusable `selection.read(out=...)`; scoped `edit_dense()` with structural preflight and exception-path invalidation |
| 3 | Captured device/generation ownership; active buffer count/binding size/capacity; GPU-only storage and preserving reserve; RenderContext-owned default mesh arena with release/reuse and compaction |
| 4 | Pure NumPy `DrawList`; optional `EcsDrawAdapter`; public drawing separated from its private compiler; independent `Visibility`; frustum/Hi-Z and CPU/custom preparation; named passes and configurable instance bindings |
| 5 | Explicit shader interfaces/layouts, normalized vertex formats, per-attachment state, MSAA/resolve, depth/stencil configuration, feature validation, and pipeline/binding cache handling |
| 6 | External opaque inputs and caller-owned targets; shared raster/deferred transforms and inverse-scale normals; bounded winner addressing; custom material source and matched deformation shader/resource interfaces |
| 7 | Installed CPU/offscreen/window examples; migrated scene integrations; opt-in counters and resource inspection; GPU readback tests; recorded NumPy access measurements |

Validation: the complete suite passes (415 tests), including CPU-only dependency
checks, scene import checks, independent GPU visibility, padded explicit uniforms,
MSAA, custom materials, transformed reconstruction, and resource lifetimes. A wheel
was independently installed in a temporary directory and its packaged shader
example rendered successfully from an unrelated working directory.
The batched-draw, direct-draw, transparency, and prototype scenes each completed
a bounded three-frame window run. The direct-draw shadow pipeline now explicitly
disables its fragment entry for the depth-only pass. `git diff --check` passes.

The current public contracts are in [PROCEDURAL_API.md](PROCEDURAL_API.md).
Measured array access costs and retained scratch-memory tradeoffs are in
[ARRAY_BENCHMARKS.md](ARRAY_BENCHMARKS.md). These are local CPU measurements,
not GPU execution timings or whole-frame speedup claims.

Workload-dependent changes remain deferred: dirty-row/chunk tracking, paged sparse
maps, alternate multicomponent storage, wider component masks, and parallel
scheduling. Existing schedule phases remain sufficient for the integrations;
custom phase configuration was not needed. Custom deformation interfaces do not
constitute an animation/skinning system. The shared winner-address-space contract
remains explicit. RenderContext remains the central window/event-loop/GPU manager.

The goal is a reusable, customizable NumPy-based game engine with a small,
procedural API. Applications should be able to use the ECS, asset preparation,
GPU resources, draw batching, and individual rendering techniques independently.

## Design decisions to preserve

- **RenderContext stays the central entry point.** Its window creation, event
  handling, frame loop, presentation, and GPU work/resource management are
  deliberate conveniences. Keep the simple default setup and shared device.
  This plan does not require splitting windowing into another public service,
  introducing multiple contexts, or passing a device through every API call.
- RenderContext also supports headless GPU work through its existing backend
  setup path. Resource management improvements live behind this same entry point.
- Systems remain ordinary functions. Schedulers, deferred ECS commands, cameras,
  and bundled rendering techniques are optional conveniences.
- NumPy arrays are the primary interchange format. Types and small resource
  handles describe ownership and layout; they should not hide array behavior.
- Applications control simulation order and render-pass order. Keep explicit
  command recording, submission, and the raw WGPU escape hatches.
- Backward compatibility is not required. Migrate callers, documentation, and
  tests directly when replacing an API; avoid compatibility aliases and adapters.

## Delivery approach

Implement each phase as a reviewable change with an updated working example.
Preserve the existing ECS storage algorithms initially. Use the measurements in
[ECS_BENCHMARKS.md](ECS_BENCHMARKS.md) when evaluating performance changes.

| Phase | Deliverable | Dependencies |
| --- | --- | --- |
| 1 | Importable library boundaries and consistent naming | None |
| 2 | Explicit NumPy schemas, reads, writes, and allocation control | 1 |
| 3 | Buffer and mesh lifetime management through RenderContext | 1 |
| 4 | Independent draw lists and per-view visibility | 2, 3 |
| 5 | Shader layout and pipeline customization | 3; coordinate with 4 |
| 6 | Composable transparency with shared geometry reconstruction | 4, 5 |
| 7 | Integration examples, diagnostics, and measured optimization | Validation throughout; complete after 6 |

The API names used below are proposed names. Establish the final signatures in
the relevant phase before migrating callers.

## Phase 1: establish library boundaries and naming

### Changes

1. Replace `src/__init__.py`'s `sys.path` modification and internal top-level
   imports with package-relative imports. Add installable package metadata and
   include engine shader files as package resources.
2. Separate `Utils.py` by responsibility: NumPy math/camera helpers, CPU asset
   loading, renderer-specific packing, GPU texture preparation, and diagnostics.
   Keep these modules small; move existing code before adding new abstractions.
3. Keep the CPU dependency path independent of WGPU, GLFW, ImGui, and glTF
   tooling. Import optional integrations only where needed. Importing GPU
   modules must not create a window or initialize a device.
4. Keep `RenderContext` as the public convenience manager. Use snake_case for
   functions, methods, parameters, and attributes; PascalCase for types; and
   UPPER_SNAKE_CASE for constants. Preserve the chosen PascalCase public module
   names, including `TransparencyPass.py`. Remove superseded naming aliases.
5. Put every scene's initialization and loop behind `run_scene()` or a main
   guard. Importing scene helpers must be safe during test collection.
6. Keep general glTF loading independent of transparency layouts. Move the
   current transparency-specific `load_scene()` preparation next to that
   technique's packing code, with a name that describes its output.

### Acceptance

- Import the ECS and math helpers in a NumPy-only environment.
- Load shaders and run an installed-package example from another working directory.
- Collect the full test suite without opening a window or starting a frame loop.
- A normal application still starts with RenderContext window setup and its loop.

## Phase 2: make NumPy memory and mutation contracts explicit

### Changes

1. Extend component schema declarations with explicit NumPy dtype and shape
   metadata, retaining dataclasses as convenient declarations. Shaped numeric
   fields should not require engine vector subclasses. Preserve deliberate
   float32/float64 choices rather than imposing an implicit conversion.
2. Reject unsupported annotations instead of silently falling back to object
   storage. Keep an explicit object-field path for Python application data.
3. Retain checked selections and validated assignment. Add an explicit field
   read operation with optional caller-provided `out` storage, so repeated
   gather operations can reuse arrays. Document which operations copy.
4. Add scoped writable access to dense columns for systems that can operate
   directly in storage order. Forbid structural edits to the borrowed store
   during the scope. Views must not escape the scope; NumPy views cannot be
   safely revoked after export.
5. On leaving a writable scope, invalidate the declared fields, including when
   an exception may have left partial writes. Do not imply transactional rollback.
6. Keep arbitrary joins as gather/scatter operations when stores have different
   row orders. Reuse cached mappings and scratch arrays; do not promise zero-copy
   joins without a storage arrangement that actually supports them.
7. Specify dtype conversion, output shape, broadcasting, aliasing, duplicate-row
   writes, and selection lifetime consistently. Preserve the separate meanings
   of membership versions, layout versions, and field-write versions.

### Acceptance

- A movement system can reuse its input/output arrays across unchanged queries.
- A dense numeric system can update storage without gathering its own columns.
- Tests cover stale selections, structural edits during a borrow, exception-path
  invalidation, aliased arrays, shaped fields, and repeated selected rows.
- Benchmark allocations and CPU time for cached movement against the current
  gather/scatter implementation; record results without assuming a speedup.

## Phase 3: clarify buffers, mesh pools, and resource lifetime

### Changes

1. Keep convenient resource constructors backed by RenderContext. Capture the
   creating device and a context device-generation token inside each resource.
   Uploads and deferred pipeline creation must use that captured device.
2. Advance the generation on device replacement/cleanup. Reject use of resources
   from an expired generation instead of mixing old handles with a new device.
   Headless setup and window setup follow the same ownership rules.
3. Distinguish GPU capacity, active element count, and bound byte range. Give
   reservation, resizing, replacement writes, and partial writes explicit
   preservation rules. State whether offsets are bytes or elements.
4. Support GPU-only buffers without requiring a full CPU mirror. CPU-backed
   upload buffers retain a clear copy/borrow policy for their NumPy content.
5. Replace class-level append-only mesh pools with arenas managed through
   RenderContext. Provide a default arena for the simple path and optional
   application-owned arenas for scene/asset lifetime boundaries.
6. Give allocations releasable handles and generations. Reuse freed space;
   make compaction explicit and invalidate or refresh affected draw ranges.
7. Ensure buffer replacement and arena relocation refresh dependent bindings
   and indirect commands. Retain resources needed by recorded/in-flight GPU
   work until they can be released safely. Make explicit cleanup idempotent.

### Acceptance

- Repeated load/unload cycles reuse arena capacity rather than growing forever.
- Buffer growth preserves exactly the data promised by its API.
- Device reinitialization produces clear stale-resource errors.
- The default mesh/resource path remains usable without manual arena setup.

## Phase 4: separate draw grouping from per-view visibility

### Changes

1. Extract pure CPU grouping into a draw-list representation: mesh/material
   groups, source ranges, and the permutation from input rows to packed rows.
   Array input is the core API; entity labels are optional application metadata.
2. Move ECS extraction/version tracking into an adapter. Component names and
   transform accessors are not requirements of draw-list construction.
3. Separate instance packing from grouping. Let callers supply packed arrays
   or ordinary packing functions with explicit input/output layouts. Apply the
   same permutation to every instance attribute and metadata stream.
4. Introduce a visibility result holding visible indices and indirect commands
   for one view. Share geometry and instance data across independently retained
   main-camera, shadow, reflection, or picking visibility results.
5. Define a small culling contract around draw lists, bounds, view values, and
   output visibility. Supply frustum and Hi-Z implementations. Permit direct
   drawing or caller-produced visibility without dummy HZB resources.
6. Make draw recording consume a draw list, visibility result, and explicit
   shader-pass specification. Replace fixed `prepass`/`main` storage with named
   passes or explicit pass handles. Preserve convenient depth/main presets.
7. Document the instance/visibility binding ABI. Keep a default convention and
   provide an explicit layout route for custom pipelines. Changing only the
   NumPy dtype must not pretend to generate compatible culling/shading code.

### Acceptance

- One instance dataset renders through two cameras with independent visibility.
- A custom instance layout works with a matching custom culler and shader.
- CPU/direct rendering works without Hi-Z allocation or frustum-result copying.
- Regrouping, repeated entity labels, empty lists, and multicomponent ECS inputs
  preserve instance/metadata alignment. Unchanged data avoids rebuilds/uploads.

## Phase 5: expose shader and pipeline customization

### Changes

1. Keep reflection and shader preprocessing as conveniences. Add an explicit
   layout path for shaders whose interfaces exceed supported reflection.
2. Specify NumPy/WGSL field offsets, alignment, array strides, padding, matrix
   convention, and normalized vertex formats. Extend layout handling for padded
   matrices/arrays where practical; otherwise report the unsupported layout.
3. Expose per-color-attachment state, sample count, depth/stencil state, and
   explicit bind-group layouts. Include all relevant configuration in pipeline
   cache keys and validate attachment compatibility before drawing.
4. Make resource requirements discoverable: required GPU features, limits,
   binding requirements, and supported target formats. Validate at construction
   or preparation time. Choose an available explicit fallback where implemented.
5. Keep raw WGPU handles usable through documented escape hatches. Avoid a
   requirement to subclass engine objects to supply a custom shader or pass.

### Acceptance

- A custom shader works with explicit layouts and NumPy buffers.
- Multiple color outputs can use different blend/write settings.
- Sample-count and target-format changes create the correct pipeline variants.
- ABI errors identify the field/binding and expected layout.

## Phase 6: make transparency compose with other renderers

### Changes

1. Accept existing opaque color/depth inputs and caller-owned output targets.
   Retain convenient internal target allocation for the default workflow.
2. Separate target allocation/binding, geometry passes, winner shading, and
   reconstruction configuration while retaining the explicit pass-recording API.
3. Define a winner-addressing contract that identifies the geometry, instance,
   triangle, material, and reconstruction attributes needed for deferred shading.
   Preserve the current shared-address-space restriction until the new contract
   and its bounds checks are implemented and tested.
4. Apply matching transforms to raster geometry, culling bounds, and deferred
   reconstruction. Support shared local-space geometry with instance translation,
   rotation, scale, and correct normal transformation.
5. Define how custom deformation supplies matching reconstruction data. Establish
   this interface before claiming animation/skinning support.
6. Separate material evaluation from stochastic selection and bilateral
   reconstruction. Keep the current material/shading implementation as a default;
   accept compatible custom shader modules or pass specifications.
7. Validate target dimensions, formats, sampling assumptions, and color encoding.
   Make resize/target replacement update dependent bindings predictably.

### Acceptance

- Transparency renders over color/depth produced by another renderer.
- Multiple transformed instances reuse the same local-space geometry and match
  between rasterization and deferred shading, including nonuniform scale.
- Custom material shading can replace the default without editing selection or
  reconstruction code. Empty samples, occlusion, and resize remain correct.

## Phase 7: integration, diagnostics, and optimization

1. Maintain small examples for NumPy-only simulation, procedural direct drawing,
   batched drawing with custom attributes, two-view rendering, and transparency
   over externally produced opaque targets. Use the normal RenderContext setup.
2. Keep scheduling optional and systems callable directly. If phase customization
   is needed by these integrations, allow caller-declared phases while retaining
   the existing default schedule and explicit command-flush boundaries.
3. Add opt-in counters for allocations, active/capacity bytes, upload bytes/calls,
   regrouping, visibility preparation, and pipeline/binding cache behavior.
   Distinguish CPU submission timing from GPU execution timing.
4. Validate CPU behavior with the existing ECS reference/invariant tests, and GPU
   behavior with bounded offscreen tests and readback. Keep import/collection
   checks separate from interactive examples.
5. Compare fully moving and sparsely changing scenes before introducing dirty-row
   or chunk tracking. Measure bookkeeping, packing, upload calls, and total time.
6. Revisit paged sparse maps, multicomponent storage changes, wider component
   masks, and parallel scheduling only when concrete workloads justify them.

### Completion criteria

The simple default application remains short and procedural. ECS/math/asset
preparation can be used independently. Custom instance layouts, culling,
materials, and passes are supported through documented inputs and outputs.
Resource and array ownership are explicit, multi-view rendering does not duplicate
scene state, and all bundled examples/tests use the final API without aliases.

Rendering features listed in the repository TODO (lighting models, shadows,
post-processing, animation, and other systems) can then build on these contracts
as separate work rather than expanding this API refactor indefinitely.
