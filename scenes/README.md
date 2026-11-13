# Interactive scenes

Run the examples from the repository root after installing the optional dependencies:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python main.py scenes/draw_batches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
python main.py scenes/material_example.py
python main.py scenes/msaa_example.py
```

Add `--frames N` to stop a RenderContext scene after N displayed frames, for
example `python main.py scenes/material_example.py --frames 120`. Closing the
window can end a run earlier. The six modules export `run_scene()` and can be
imported without opening a GPU device. `main.py` invokes that entry point and
cleans up the context afterward.

| Scene | What to try |
| --- | --- |
| `draw_batches_example.py` | ECS movement and entity changes, LOD, indirect draws and Hi-Z culling; Space switches camera, scroll zooms. |
| `direct_draw_example.py` | Direct drawing with instance changes and shadows; scroll zooms. |
| `prototype.py` | ECS-backed actor/health panels in ImGui. Combat resolution is a placeholder. |
| `transparency_example.py` | OPAQUE/MASK/BLEND glTF geometry, transparency modes and diagnostics; see below. |
| `material_example.py` | Shared material dictionaries, per-instance overrides, rim lighting and animated stripes. Space orbits; M toggles MSAA; D changes a default material; F opens settings. |
| `msaa_example.py` | Minimal full-window rotating triangle; 1 selects single sampling and 4 selects 4x MSAA. |

These files are runnable examples rather than a stable scene API. For reusable
components, see the [engine guide](../README.md), [ECS reference](../graaal/ECS_API.md),
[material guide](../graaal/materials.md), and
[transparency guide](../graaal/stochastic_transparency.md).

## Transparency settings and controls

In `transparency_example.py`, the ImGui settings panel is visible at startup.
`material_example.py` opens it with **F**. Both use
`FilterSettings` in `transparency_settings.py`; its `handle_key` callback
handles keyboard editing of visible controls.
Both examples start with `TransparencyPass` defaults.

The default **Banded + dual Kawase** mode captures the nearest transparent
layer exactly and uses a complementary banded pair for the remaining tail.
**Down/up pairs** controls the blur (0–8, default 1). **Proxy residual**
subtracts a local albedo estimate before blurring to reduce color halos.
It is **off by default**. **Shade one pixel
per 2 x 2 block** is an optional lower-cost, noisier approximation. Tail
estimation defaults to 1x even when the scene uses MSAA; exact nearest,
transmittance and opaque/MASK geometry still use the selected native sample count.
There is no tail-MSAA control in this UI (see `set_banded_options()` for the API).

Select **Stochastic nearest wins** in the panel to compare other options:

| Setting | Usage |
| --- | --- |
| **Exact nearest layer** | Enable/disable the exact foremost transparent contribution; required in banded mode. |
| **Winner draws / mix** | Average supports 1–2 draws, with complemented thresholds by default; Fill supports any positive count with supported noise modes. Reduce the count to 2 or less before switching back to Average. |
| **Noise mode / grain size** | Compare UV, object-hash, anchored and triplanar noise. UV footprint balancing applies only to UV modes, is enabled by default and has a default balance of **1.0** (0 uses maximum-derivative sizing). |
| **Filter implementation** | Fragment or compute bilateral (1x1, 3x3, 5x5 or 9x9), a-trous cascade, or confidence pyramid. A 1x1 footprint skips reconstruction. |
| **Reconstruction passes** | Repeat the bilateral kernel on weighted color/validity pairs. Only fixed bilateral modes use this setting. |
| **Support budget / cascade passes** | Control smoothing in a-trous reconstruction; the four-pass cascade reaches a maximum radius of 15 pixels. |
| **Force single contributor** | Optionally force an isolated contributor to be selected; disabled by default. |
| **Grazing fade** | Fade BLEND opacity toward silhouettes. The low-level pass defaults to enabled from 83° to 86°; material shaders also have per-material grazing parameters. |

Some settings and diagnostic views disappear when the active technique cannot
use them. Debug views include **final**, **transmittance**, **winner**,
**reconstruction**, **nearest** and stream-specific views where applicable.
The **winner** view shows signed residuals when the banded proxy is active.
The settings panel describes each control on hover.

| Input | Action in `transparency_example.py` |
| --- | --- |
| Space / mouse scroll | Toggle orbit / zoom |
| R | Reverse transparent draw order |
| D | Cycle the currently available diagnostic views |
| N | Cycle noise modes (used by stochastic allocation) |
| [ / ] | Decrease/increase the stochastic noise-cell size |
| G | Toggle grazing fade |
| M | Toggle 4x MSAA (starts off) |
| 1 / 3 / 5 / 9 | In banded mode: select 0 / 1 / 2 / 3 Kawase pairs; in stochastic bilateral mode: select filter size |
| F | Show/hide the settings panel |
| Arrow keys | Select/edit settings while the panel is visible; Shift makes smaller changes |
| Home | Reset the banded blur to one pair, or reset stochastic bilateral-filter values |

No TAA or temporal history is used. MSAA improves opaque/MASK and nearest-layer
edges, but the 1x banded tail can miss subpixel details. Stochastic modes can
still show spatial grain and motion shimmer. For controls from Python, use
`set_tail_allocation()`, `set_kawase_passes()`, `set_banded_options()`,
`set_winner_count()`, `set_noise_mode()`, `set_reconstruction()` and
`set_debug_view()` as documented in the [transparency guide](../graaal/stochastic_transparency.md).

The transparency scene loads `resources/small_scene.glb`, with
`Simple_foliage.001` treated as MASK. It supports static glTF node transforms,
base color, roughness/metallic and emission. Normal maps, skinning, animation,
material extensions and sampler overrides are not covered by this example.
The [blue-noise data and license](../graaal/data/blue_noise/README.md) are bundled.

## Scene tests and benchmarks

For untracked in-place edits to ECS object fields, call `touch()` so dependent
readers can observe them. Run the import, scene-state and cache tests with:

```powershell
python -m pytest tests/test_scenes.py tests/test_scene_caches.py graaal/tests/test_library_boundaries.py -q -p no:cacheprovider
```

See [benchmarks](benchmarks/README.md) for CLI usage of diagnostic scripts.
