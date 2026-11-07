# NumPy ECS and procedural rendering

The installed package is `numpyecs`. RenderContext remains the convenient window,
event-loop, presentation, and GPU resource/work manager. Simulation and math can
be used independently with only NumPy.

Install from the repository root:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python -m numpyecs
python -m numpyecs --offscreen --frames 1
python main.py scenes/transparency_example.py --frames 120
python -m pytest tests src/tests -p no:cacheprovider
```

For CPU-only use, `pip install -e .` is sufficient. Engine shaders ship in the
package, so the packaged examples also work from another working directory.

| Document | Purpose |
| --- | --- |
| [PROCEDURAL_API.md](PROCEDURAL_API.md) | Current package/API guide and customization contracts |
| [ECS_API.md](ECS_API.md) | Detailed ECS operations, selections, queries, and examples |
| [Improvements.md](Improvements.md) | Implementation plan and progress |
| [ARRAY_BENCHMARKS.md](ARRAY_BENCHMARKS.md) | Measured gather/dense access costs |
| [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md) | Historical storage/query measurements |
| [ECS_TODO.md](ECS_TODO.md) | Historical implementation record and deferred work |
| [Scene guide](../scenes/README.md) | Interactive examples and bounded runs |

`ECS` and `ecs_*` implement simulation. `Math`, `Assets`, `InstanceData`,
`DrawList`, and `TransparencyData` expose independently importable CPU helpers.
`TextureProcessing` prepares GPU material textures. `EcsDrawing` adapts application
components to the array-based renderer. `DrawBatches` shares geometry/instances
across independent visibility results and named shader passes.

`HZB` builds hierarchical depth pyramids independently of the batching API.
Internally, `draw_compiler` generates layouts with NumPy, `draw_resources` retains
registries and uploads shared data, `draw_layout` defines NumPy records, and
`draw_culling` records per-view GPU work. Compilation can run without a GPU.
Public subsystem modules use PascalCase; implementation helpers use snake_case.
Classes use PascalCase and functions/methods use snake_case.
`utils` contains the small CPU helpers shared across subsystems: `cache_last`
and `higher_pow2`.

`TransparencyPass` accepts caller-owned opaque inputs/targets and applies matching
instance transforms in rasterization and deferred shading. Its material shader
and matched geometry/reconstruction shaders can be customized under an explicit
winner ABI. It records work into caller commands and leaves submission/presentation
to the application. The procedural API guide documents the layouts and pass order.

`ImGui` is an optional integration with the RenderContext window and surface.
Create it after window setup and close it before context cleanup. It overlays the
current surface and forwards uncaptured input to application handlers.

`GCManager` is an optional idle-time collection policy. Budgets estimate suitable
spare frame time rather than capping collection pauses; `close()` restores the
previous process settings. Use one manager at a time and release scene references
before a full scene-transition sweep.

Generated compiler/build output belongs under `build/`; benchmark results belong
with their benchmark scripts. Superseded module/name aliases are removed; callers
use package-qualified imports and the current names directly.
