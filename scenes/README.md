# Running the scenes

Run from the repository root with a Python environment containing the engine's
NumPy, WGPU, GLFW, and supporting dependencies:

```powershell
python main.py scenes/drawbatches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
```

| Scene | Purpose |
| --- | --- |
| `drawbatches_example.py` | Cached movement/render queries, deferred cube creation/deletion, scheduler phases, LoD and Hi-Z rendering. Space switches cameras; scrolling adjusts the view. |
| `direct_draw_example.py` | Direct instanced drawing and shadows, with cached movement joins and versioned instance packing. Scrolling adjusts the view. |
| `prototype.py` | Static combat actor panels and health bars using the native WGPU renderer. Rebuilds/uploads only when displayed data changes. The player is blue, enemies red; actor names and health are in the window title. Combat action selection/resolution remains a placeholder. |

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
