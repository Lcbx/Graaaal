# Running the scenes

Run from the repository root with a Python environment containing the engine's
NumPy, WGPU, GLFW, and supporting dependencies:

```powershell
python main.py scenes/drawbatches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
```

| Scene | Purpose |
| --- | --- |
| `drawbatches_example.py` | Cached movement/render queries, deferred cube creation/deletion, scheduler phases, LoD and Hi-Z rendering. Space switches cameras; scrolling adjusts the view. |
| `direct_draw_example.py` | Direct instanced drawing and shadows, with cached movement joins and versioned instance packing. Scrolling adjusts the view. |
| `prototype.py` | Static combat actor panels and health bars using the native WGPU renderer. Rebuilds/uploads only when displayed data changes. The player is blue, enemies red; actor names and health are in the window title. Combat action selection/resolution remains a placeholder. |
| `transparency_example.py` | Single-winner stochastic order-independent transparency using DrawBatches and `resources/small_scene.glb`. Up/Down selects a filter setting, Left/Right edits it, Shift gives fine steps, and F hides/shows the list. Space toggles orbit, scrolling zooms, R reverses transparent draw order, A toggles adaptation, and D cycles debug views. |

The prototype preserves the original actor data, including the player's initial
150 health versus 100 maximum; the health bar clamps to its full width. It uses
`create()`/`add()` and `get()` rather than the removed `create_entity()`/`get_store()`
interfaces. It no longer depends on the absent `shader_util` module.

Direct drawing refreshes render membership on structural edits. Its CPU instance
cache repacks only changed position, rotation, scale, or tint fields, preserving
already packed static fields during movement. The interleaved GPU buffer still
uploads in full when any field changes; this does not introduce partial GPU writes.
Completely unchanged instances skip both packing and upload. Cache output is owned
by the scene-local helper and shared with the GPU buffer; do not modify it externally.
Shader-specific packing stays inside `direct_draw_example.py`; there is no shared
scene cache module. The scene exposes `run_scene()` so importing its CPU helpers
does not initialize graphics. Reusable queries and version tokens remain ECS APIs.

The prototype tracks health, names, maximum health, metadata, and actor membership.
Changes to unrelated attributes such as damage do not rebuild its panels. As with
other ECS version consumers, in-place object or raw-array edits require `touch()`.

## Bounded GPU checks

Use `--frames` to render a fixed number of frames and exit:

```powershell
python -u main.py scenes/prototype.py --frames 120
python -u main.py scenes/direct_draw_example.py --frames 120
python -u main.py scenes/drawbatches_example.py --frames 1500
```

The launcher presents the final submitted frame and prints the actual frame count.
Closing the window early can produce fewer frames. `--frames` applies to scenes
using `RenderContext.window_loop()`; legacy Raylib benchmark loops do not use it.
The option requires a positive integer and cannot be combined with testing or
compilation modes.

The transparency scene implements [the stochastic transparency pipeline](stochastic_transparency.md):
opaque HDR color/depth, multiplicative RGB/scalar transmittance, hardware-depth
winner selection, deferred compute shading, and 3×3 or 5×5 cross-bilateral reconstruction
guided by transparent/opaque depth and transmittance. Selection uses static, surface-decorrelated
blue noise and a separate depth buffer; no temporal accumulation or sorting is used.
The original tileable 64×64, 16-bit texture is bundled in
[`resources/blue_noise`](resources/blue_noise/README.md), together with Christoph
Peters' CC0 dedication. It is loaded as linear thresholds without resizing or
sRGB conversion, and repeated with deterministic per-surface pixel offsets.
Sampling still uses pixel coordinates: these offsets decorrelate surfaces but
do not anchor the noise pattern to moving surfaces.
RGB opacity is supported by the shader and material buffer; glTF alpha maps to
scalar opacity by default. Its RGB estimator has the approximation described in
the pipeline notes for overlapping surfaces with different RGB opacities.

The scene's static glTF loader bakes node hierarchies and honors OPAQUE, MASK,
BLEND, and double-sided materials. It uses base-color textures resized to 512×512
array layers with repeat/trilinear sampling, base color/roughness/metallic factors, emissive
factors, and simple directional lighting. Normal/metallic/roughness texture maps,
sampler overrides, material extensions, animation, skinning, sparse accessors,
and additive materials are outside this example's material model. Explicit
triangle IDs and interpolated barycentrics avoid optional fragment-barycentric
GPU features. Window resizing recreates all dependent targets and bindings.

Material textures have a full mip chain, generated by compute shaders once during
scene setup. CPU work decodes and uploads the source images; GPU kernels handle
source resizing, mip filtering, coverage reduction/search, and output encoding,
without reading textures back to the CPU. RGB is
filtered in linear space with alpha weighting for MASK/BLEND materials, then
stored as straight sRGB color. MASK mip alpha is scaled to retain source
alpha-test coverage, including the material alpha factor and the initial resize
to the array size. Coverage is estimated with bilinear samples and repeat
wrapping, following the [alpha coverage preservation approach](https://www.ludicon.com/castano/blog/2026/02/runtime-mipmap-generation/).
The smallest mips use the closest representable coverage. BLEND mip alpha keeps
its area mean, preserving stochastic acceptance without inflating opacity.
Uncorrected premultiplied linear RGBA32F scratch mips prevent coverage correction
or 8-bit quantization from feeding back into subsequent downsampling. Area
filtering handles non-power-of-two sizes; upsampling respects repeat boundaries.
Compute encodes straight sRGB bytes into RGBA8 storage, then GPU copies populate
the sRGB sampling array. This supports the installed WGPU backend without
requiring mutable texture views. Cutoff
coverage rescaling applies only to MASK materials; it is not used for stochastic
BLEND transparency. Mean-alpha mipmaps address texture aliasing while stochastic
winner variance remains a separate source of shimmer.

When encoding MASK/BLEND mips, fully transparent texels receive RGB from the
nearest nonzero-alpha neighbors within two texels, including across repeat
boundaries. This prevents straight-color bilinear/trilinear sampling from mixing
black into visible cutout fringes. Padding changes only encoded RGB: alpha,
coverage measurements, and the premultiplied mip chain remain unchanged. Opaque
black texels retain their actual color.

Raster passes choose a mip from screen-space UV derivatives before any discard.
The winning payload stores that LOD alongside face orientation, so deferred
compute shading samples the same trilinear mip blend as transmittance and
selection. This filters distant texture detail while keeping coverage and winner
color consistent. Static stochastic selection still has sampling noise; this
does not add temporal accumulation.

Press **D** to cycle `final`, `transmittance`, `winner`, `reconstruction`, and
`filter_support`.
The transmittance view shows `T_scalar` as grayscale with display encoding but
without tone mapping or bilateral reconstruction. Winner and reconstruction
views show tone-mapped HDR color; missing winners are black. The reconstruction
view omits the opaque background. Use these views while moving the camera: if
`T_scalar` is stable while reconstructed color flickers, investigate selection
variance and the filter's effective support before changing mip generation.
The filter-support view displays `(sum(w)**2 / sum(w*w)) / kernel_area`, so
brighter pixels indicate more effective spatial samples. It includes zero-valued
no-winner samples and is not a survivor count; bright support can still contain
few winners at low opacity.

Mip selection uses the screen-space texture footprint directly, with no
transparency LOD bias. Code can call `renderer.set_debug_view("transmittance")`
without rebuilding targets or pipelines.

Press **3** or **5** to change reconstruction footprints while running (default:
5×5). Distance adaptation is enabled by default; press **A** to compare it with
fixed near-distance settings. The default profile transitions smoothly from
spatial sigma approximately 1.41 to 2 pixels, relative depth sigma 0.02 to 0.05,
and transmittance sigma approximately 0.204 to 0.5. The near spatial and
transmittance sigmas retain the original reconstruction's support; smaller
values can leave too few contributing samples along stochastic foliage edges.
Transmittance edge weighting fades to zero at
the far endpoint; `far_transmittance_weight` can retain some of it.

Adaptation uses `smoothstep(filter_near, filter_far, view_depth)` of a stable
nearest-transparent depth guide, independent of stochastic acceptance. A cheap
additional depth pass reuses DrawBatches visibility and indirect commands. It
rejects fully transparent texels and opaque occlusion, while retaining arbitrarily
low nonzero opacity. Neighbors use squared relative depth differences with a
Gaussian weight and a world-unit minimum tolerance. Both transparent-guide and
opaque depth remain in the weights as transmittance constraints relax.

When no settings are supplied to `TransparencyRenderer`, the transition spans
one to four times the framed objects' radius in world units. Explicit
`BilateralFilter()` settings default to distances 10 through 50. Distance is a
proxy for projected feature size; the filter does not measure individual leaves.
The footprint remains fixed, so empty neighborhoods remain zero and permissive
weights can blend colors across overlapping surfaces. No temporal accumulation
is introduced.

The settings list is drawn in the render window and is visible at startup.
**Up/Down** selects a row; **Left/Right** changes its value immediately; hold
**Shift** for one-tenth increments. The list exposes every filter parameter,
including the adaptation toggle and 3×3/5×5 kernel. Values are bounded so sigmas
stay positive, near distance stays below far distance, and edge weight stays in
[0,1]. **Home** restores startup settings. **F** hides/shows the list; arrow edits
are disabled while hidden. **F5** saves all settings to
`scenes/transparency_filter.json`; **F9** loads that file. Load errors appear in
the list without changing active settings. No separate window or Tkinter is used.
The text texture is rebuilt only when selection, values, status, or size changes.

For code-level tuning, pass settings to the renderer or change them live:

```python
renderer = TransparencyRenderer(bilateral_filter=BilateralFilter(
    kernel_size=5,
    adaptive=True,
    filter_near=10.0, filter_far=50.0,   # view-space distances in scene units
    spatial_sigma=2**0.5, far_spatial_sigma=2.0,
    depth_sigma=0.02, far_depth_sigma=0.05,  # relative depth standard deviations
    transmittance_sigma=(1/24)**0.5, far_transmittance_sigma=0.5,
    far_transmittance_weight=0.0,         # fade transmittance edge rejection out
    min_depth_sigma=0.01,                # world-unit depth tolerance floor
))
renderer.set_bilateral_filter(BilateralFilter(kernel_size=5, adaptive=False))
```

To check whether bilateral edge rejection makes the effective footprint too
small, compare the reconstruction view with looser tolerances, for example
`BilateralFilter(kernel_size=5, adaptive=False, depth_sigma=0.1, transmittance_sigma=1.0)`.
This is a diagnostic comparison: loosening weights can blur across real edges.

Smaller tolerances reject neighbors across depth/transmittance edges more
strongly; larger spatial spreads smooth more broadly within the selected
footprint. All tolerances must be finite and positive. Settings update a uniform
buffer without recreating render targets or pipelines. `linear_depth`,
`linear_to_srgb`, and `hash_u32` live in `shaders/utils.shaderlib`;
`linear_depth(depth, near, far)` uses standard forward perspective WebGPU depth.

CPU and offscreen GPU read-back regressions (requires a WGPU adapter):

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_stochastic_transparency.py -q -p no:cacheprovider
python -u main.py scenes/transparency_example.py --frames 120
```

At the earlier scene-integration checkpoint, all three commands above completed successfully. The
draw-batch check also logged a growing instance count after a deferred update.
These checks exercised graphics initialization, shader/pipeline creation, frame
submission, and clean shutdown. A Windows policy blocked the separate window
capture script, so no screenshot-based visual verification is claimed.

## Scene entry points and CPU checks

New scenes can expose `run_scene()`: the launcher calls it after importing the
module. The prototype follows this convention, so its ECS model helpers can be
imported without opening a window. Existing scenes that run during import remain
supported. Direct execution of the prototype is also supported when `src` is on
`PYTHONPATH`.

The model/launcher regressions need no graphics device:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_scenes.py -q -p no:cacheprovider
python -m pytest tests/test_scene_caches.py -q -p no:cacheprovider
```

## Benchmarks and ECS documentation

See the [benchmark guide](benchmarks/README.md) for current CPU, query comparison,
and bounded upload measurements, plus the optional legacy Raylib scripts.

- [ECS usage and API reference](../src/ECS_API.md)
- [ECS implementation status and backlog](../src/ECS_TODO.md)
- [ECS measurements and optimization decisions](../src/ECS_BENCHMARKS.md)

The current implementation checkpoint passed 266 ECS/follow-up/scene/draw-batch
cases; the measured direct-draw check completed 120 frames. Earlier 120/180/1,500
frame observations above describe separate historical smoke checks. No screenshot
verification is claimed.
