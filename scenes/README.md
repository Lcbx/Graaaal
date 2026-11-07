# Interactive scenes

Install the package and example dependencies from the repository root:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python main.py scenes/draw_batches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
```

| Scene | Purpose and controls |
| --- | --- |
| `draw_batches_example.py` | ECS movement, deferred spawning/deletion, LoD, and Hi-Z culling. Space switches cameras; scrolling adjusts the view. |
| `direct_draw_example.py` | Direct instanced drawing and shadows. Scrolling adjusts the view. |
| `prototype.py` | ImGui actor/health panels backed by ECS data; combat resolution remains a placeholder. |
| `transparency_example.py` | Stochastic transparency using `resources/small_scene.glb`; controls below. |

All four expose `run_scene()` and can be imported without creating a window or
GPU device. The launcher initializes scene execution and cleans up RenderContext
on exit. Paths and asset files resolve relative to the scene modules.

Add `--frames N` for a bounded run:

```powershell
python -u main.py scenes/draw_batches_example.py --frames 120
```

The launcher presents the final submitted frame and prints the frame count.
Closing the window early can produce fewer frames. The limit applies to
`RenderContext.window_loop()` scenes, including the four above; it does not apply
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
| 3 / 5 | Select reconstruction footprint |
| A | Toggle distance adaptation |
| F | Hide/show settings |
| Up/Down, Left/Right | Select/edit a settings row; Shift gives finer increments |
| Home | Restore startup settings |
| F5 / F9 | Save/load `scenes/transparency_filter.json` |

Arrow edits are disabled while the settings window is hidden. Save/load errors
appear in the window and leave active settings unchanged.

Default reconstruction uses a 5x5 footprint. When `init_scene()` is called without
filter settings, adaptation distances span one to four times the framed scene's
radius. Explicit `BilateralFilter()` defaults use distances 10 through 50 instead.
See the [transparency notes](../src/stochastic_transparency.md) for the pass order,
filter parameters, and interpretation of diagnostic views.

The example loads static glTF node transforms, OPAQUE/MASK/BLEND modes,
double-sided materials, base-color textures, roughness/metallic factors, and
emissive factors. It builds 512x512 material texture layers with repeat/trilinear
sampling and alpha-aware mipmaps. Normal/material maps, sampler overrides,
material extensions, animation, and skinning are outside this example.
[Blue-noise source and license](resources/blue_noise/README.md).

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
