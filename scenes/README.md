# Interactive scenes

Install the package and example dependencies from the repository root:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python main.py scenes/draw_batches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
python main.py scenes/material_example.py
python main.py scenes/msaa_example.py
```

| Scene | Purpose and controls |
| --- | --- |
| `draw_batches_example.py` | ECS movement, deferred spawning/deletion, LoD, and Hi-Z culling. Space switches cameras; scrolling adjusts the view. |
| `direct_draw_example.py` | Direct instanced drawing and shadows. Scrolling adjusts the view. |
| `prototype.py` | ImGui actor/health panels backed by ECS data; combat resolution remains a placeholder. |
| `transparency_example.py` | Stochastic transparency using `resources/small_scene.glb`; controls below. |
| `material_example.py` | Shared dictionary materials, instance overrides, rim lighting, and an embedded WGSL shader with animated vertex displacement. Controls adjust stripes, displacement amplitude/speed, and pause animation. M: MSAA; D: replace default; Space: orbit. |
| `msaa_example.py` | Rotating triangle with resize handling. Number keys: 1 disables MSAA; 4 selects 4x MSAA (default). |

All six expose `run_scene()` and can be imported without creating a window or
GPU device. The launcher initializes scene execution and cleans up RenderContext
on exit. Paths and asset files resolve relative to the scene modules.

Add `--frames N` for a bounded run:

```powershell
python -u main.py scenes/draw_batches_example.py --frames 120
```

The launcher presents the final submitted frame and prints the frame count.
Closing the window early can produce fewer frames. The limit applies to
`RenderContext.window_loop()` scenes, including the six above; it does not apply
to the legacy Raylib benchmark loop.

## Useful scene APIs

These are examples to adapt, not a stable library API. This list is non-exhaustive;
import paths start with `scenes.` when running from the repository root.

| Module | Useful classes and functions |
| --- | --- |
| All six scenes | `run_scene()` starts the interactive example; importing the module creates no window or device. |
| `draw_batches_example` | `Transform`, `Velocity`, `MeshRef` define components; `movement_system(world, dt)` updates motion; `update_instances(world, data)` packs drawing columns. Other scene systems use state initialized by `run_scene()`. |
| `direct_draw_example` | `CubeInstances.update()` refreshes `.content` only when component versions or query membership change. |
| `prototype` | `create_world()` returns sample actors; `actor_statuses(world, query)` returns `ActorStatus` records; `.health_fraction` clamps UI health; `ActorCache.update()` refreshes displayed values on change. |
| `transparency_example` | `init_scene(...)` builds GPU resources; `render_frame()` records/submits a frame using scene state; `FilterSettings.set_value/reset` edits/restores reconstruction controls. |
| `material_example` | `sphere_vertices()`, `make_instances()`, and `make_world()` build CPU geometry/material/ECS inputs without opening a device. |
| `msaa_example` | `run_scene()` contains the minimal window, resize and sample-count switching loop. |

## Transparency controls

The settings window is visible at startup. **Winner draws** accepts a positive
integer, independently of MSAA. Average mode supports one or two (default);
Fill accepts any positive count. Extra attempts reduce missing winners but add
selection/shading work and do not supersample edges.
Diagnostic views include each stream and the valid-stream fraction. See the
[two-winner plan and findings](../src/transparency_multiple_winners.md).
The diagnostic dropdown and **D** shortcut follow the active technique. A/B
views appear only for two-draw Average mode; Fill labels its binary mask
**winner_validity**. Single-draw mode hides stream/count views, disabled forcing
hides **forced**, and 1x1 hides **filter_support**. A view that becomes unavailable
returns to **final** when editing settings.
**Filter implementation** selects the default Fragment path or experimental
Compute reconstruction. They share the same bilateral estimator. See the
[profiling report](../src/transparency_reconstruction.md) for measurements.
**Force single contributor** deterministically accepts exactly one visible
contributor at its opacity, including fractional MASK edges; it adds a small
count target and is off by default. **Winner mix** offers experimental hole
filling with default-source Object hash or Anchored blue noise. Select two draws,
then Fill, before raising the count further. Reduce to two before returning to Average.
Other noise modes retain averaging. The **forced** view marks deterministic
samples; weighted color views show color with its weight removed.
**Reconstruction passes** stacks the selected kernel, filtering weighted pairs
and applying opacity once at the end. Two 3x3 passes reach up to 5x5 support;
four reach 9x9, with different weights from a single large kernel. A 1x1 kernel
bypasses all stages. The FPS/frame-time overlay remains visible when settings hide.
Drag values or Ctrl+click to type;
hover controls for units and tuning guidance. ImGui captures input while editing
widgets, so those actions do not also move the camera or trigger scene shortcuts.

| Input | Action |
| --- | --- |
| Space / scroll | Toggle orbit / zoom |
| R | Reverse transparent draw order |
| D | Cycle final, transmittance, winner, reconstruction, and filter-support views |
| N | Cycle noise modes, including UV correlated hierarchy and both triplanar options |
| [ / ] | Decrease/increase surface noise-cell size |
| G | Toggle grazing-angle opacity fade |
| M | Toggle 4x MSAA (off by default); also available in the settings dropdown |
| 1 / 3 / 5 / 9 | Disable reconstruction / select its fixed footprint |
| F | Hide/show settings |
| Up/Down, Left/Right | Select/edit a settings row; Shift gives finer increments |
| Home | Restore startup filter settings |

Arrow edits are disabled while the settings window is hidden.
Startup uses the defaults in code; controls update live without a profile file.
Anchored blue with two averaged winner draws is the default; UV CDF lerp, stochastic picking and Bayer
remain experimental comparisons. The settings window also exposes these controls.

The noise selector includes correlated UV hierarchies and triplanar object-space
projection for meshes with unsuitable UVs. Hierarchies build once on first use.
**MSAA (M)** smooths silhouettes and adds four stochastic samples at extra GPU and
memory cost. It works with every reconstruction kernel, including 1x1.
Sampling is static; there is no TAA or frame history.
The noise-cell slider stores the 1x width; 4x MSAA halves the effective target
(1 px becomes 0.5 px), and disabling MSAA restores the chosen width. The effective
target is shown below the slider. Sizes remain approximate at oblique angles.
**UV footprint balance** is an experimental compromise: 0 keeps maximum-derivative
sizing (default), 0.5 uses a geometric mean, and 1 uses the minimum estimate.
Increasing it makes tilted grain finer along both axes, including subpixel cells
along the compressed axis. It applies to UV modes and retains existing fallback
and triplanar behavior. See the [tilt diagnostic](../IMPLEMENTATION_WORK.md#uv-footprint-compromise-experiment)
for measured screen-space autocorrelation widths and limitations.
The balancing shader feature is disabled by default. Check **Enable UV footprint
balancing** to compile/cache it and show the editable UV slider. Its value can be
adjusted in any noise mode, but only affects UV modes. Settings are separated by
feature; disabled features hide their parameters. Selecting the 1 x 1 kernel
hides the reconstruction tolerances, and disabling grazing fade hides its angles.

**Anchored blue: pixel lattice** is available through the noise selector or N.
It reads pixel-sized grain around rest-space coarse anchors, without requiring
UVs. Its anchor target defaults to 24 pixels and shift subdivisions to 4. Higher
subdivisions reduce close-layer correlation at the cost of more shift seams.
The mode keeps grain much more isotropic at tilt, but can reshuffle more during
orbit/dolly. It is the example's default noise mode. See the
[anchored noise audit](../IMPLEMENTATION_WORK.md#pixel-lattice-anchored-blue).

"Fade grazing surfaces" starts enabled. Its two editable angles default to
83 degrees (start) and 86 degrees (completely faded). Angles are measured from
the geometric surface normal: 0 is face-on, 90 is edge-on. Opacity fades smoothly
between them on both front and back faces. The interval must satisfy
`0 <= start < end <= 90`; drag either value or Ctrl+click to type it. The fade
affects MASK/BLEND surfaces consistently in guide, transmittance and selection.

Default reconstruction uses a fixed 3x3 footprint, spatial sigma 0.9, relative
depth sigma 0.05, transmittance sigma 0.3, and minimum depth tolerance 0.01.
Slope correction is always enabled. The 1x1 option shows the unfiltered stochastic
result; certain fully opaque nearest winners also bypass reconstruction.
See the [transparency notes](../src/stochastic_transparency.md) for the pass order,
filter parameters, and interpretation of diagnostic views.

The example loads static glTF node transforms, OPAQUE/MASK/BLEND modes,
double-sided materials, base-color textures, roughness/metallic factors, and
emissive factors. The tree's `Simple_foliage.001` material is treated as MASK.
The grain target defaults to 1 pixel at 1x;
`transparency.set_noise_scale(2)` makes it coarser. Repeated UV islands can correlate
overlapping layers; compare object-hash or triplanar modes on those assets.
See the [sampling notes](../IMPLEMENTATION_WORK.md#acceptance-and-winner-data)
for threshold construction and its limits.
Normal/material maps, sampler overrides,
material extensions, animation, and skinning are outside this example.
[Blue-noise source and license](../src/data/blue_noise/README.md).

## CPU caches and checks

For untracked edits to ECS object fields, call `touch()` to notify readers.

Run scene-model/import/cache tests without opening windows:

```powershell
python -m pytest tests/test_scenes.py tests/test_scene_caches.py src/tests/test_library_boundaries.py -q -p no:cacheprovider
```

The [engine guide](../README.md) describes procedural rendering and resource
lifetimes; the [ECS reference](../src/ECS_API.md) covers simulation contracts.
See [benchmarks](benchmarks/README.md) for measurement commands and
[implementation work](../IMPLEMENTATION_WORK.md#ecs-benchmark-history) for recorded data.
