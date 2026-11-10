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
| `material_example.py` | Shared dictionary materials, per-instance overrides and inherited rim lighting. M: MSAA; D: replace default; Space: orbit. |
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

## Transparency controls

The settings window is visible at startup. Drag values or Ctrl+click to type;
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
UV blue CDF lerp is the default; stochastic picking and Bayer
remain experimental comparisons. The settings window also exposes these controls.

The noise dropdown additionally offers **UV blue: correlated hierarchy**,
**Triplanar blue: CDF lerp**, and **Triplanar: correlated hierarchy**. The hierarchy
is generated once on the GPU when first selected. Its levels share a spatial
origin and are re-ranked after downsampling to retain threshold coverage.
Triplanar modes use object-space projections with smooth local-normal weights
and work independently of UV islands. All sampling is static, without TAA.
The previous default and existing modes remain available.

**MSAA (M)** switches between off and 4x. Four independent samples cover geometry,
opacity and winner selection. Composition averages their linear HDR contributions
before tone mapping. This smooths silhouettes and supplies extra stochastic samples,
at an additional GPU/memory cost. It works with every reconstruction kernel,
including 1x1, and does not use TAA or frame history.

"Fade grazing surfaces" starts enabled. Its two editable angles default to
83 degrees (start) and 86 degrees (completely faded). Angles are measured from
the geometric surface normal: 0 is face-on, 90 is edge-on. Opacity fades smoothly
between them on both front and back faces. The interval must satisfy
`0 <= start < end <= 90`; drag either value or Ctrl+click to type it. The fade
affects MASK/BLEND surfaces consistently in guide, transmittance and selection.

Default reconstruction uses a fixed 3x3 footprint, spatial sigma 0.9, relative
depth sigma 0.05, transmittance sigma 0.3, and minimum depth tolerance 0.01.
The five controls are kernel size, spatial sigma, relative depth sigma,
transmittance sigma, and minimum depth tolerance. Slope correction is always
enabled: tilted cards keep their filter support while depth breaks remain protected.
Certain nearest winners bypass the blur; all fractional pixels use the complete
normalized filter. The 1x1 option skips neighborhood reconstruction and shows the
unfiltered stochastic result without an extra opacity multiplication.
See the [transparency notes](../src/stochastic_transparency.md) for the pass order,
filter parameters, and interpretation of diagnostic views.

The example loads static glTF node transforms, OPAQUE/MASK/BLEND modes,
double-sided materials, base-color textures, roughness/metallic factors, and
emissive factors. Static primitives go directly into DrawBatches arrays; metadata
follows its packed instance order. It builds 512x512 material texture layers with repeat/trilinear
sampling and alpha-aware mipmaps. The tree's `Simple_foliage.001` material is
explicitly treated as MASK, with coverage-preserving mips and sharpened alpha
shared across guide, transmittance and selection. Thresholds are anchored to UV
octaves, with object-space fallback for missing or rank-one UVs. A single UV
octave preserves the tile spectrum; fractional-octave mixtures do not guarantee
it. There is no card inference or projected-size cutoff. The independent surface
selector, CDF lerp, and Bayer options are compared in the
[sampling notes](../src/stochastic_transparency.md#acceptance-and-winner-data).
`transparency.set_noise_scale(2)` trades coarser grain for greater stability;
the default scale is 1 pixel. Repeated UV islands within one instance can
correlate overlapping layers, so object-hash mode remains useful for those assets.
Normal/material maps, sampler overrides,
material extensions, animation, and skinning are outside this example.
[Blue-noise source and license](../src/data/blue_noise/README.md).

## CPU caches and checks

Direct drawing retains packed instance data and refreshes it when relevant ECS
fields or membership change. Unchanged instances skip packing/upload; changed
interleaved data still uploads in full. Prototype panels likewise refresh from
the fields they display. Notify otherwise untracked object edits with `touch()`.

Run scene-model/import/cache tests without opening windows:

```powershell
python -m pytest tests/test_scenes.py tests/test_scene_caches.py src/tests/test_library_boundaries.py -q -p no:cacheprovider
```

The [engine guide](../README.md) describes procedural rendering and resource
lifetimes; the [ECS reference](../src/ECS_API.md) covers simulation contracts.
See [benchmarks](benchmarks/README.md) for measurement commands and recorded data.
