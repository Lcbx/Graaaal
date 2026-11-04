# Source and ECS documentation

| Document | Purpose |
| --- | --- |
| [ECS_API.md](ECS_API.md) | Public API reference and executable usage examples |
| [ECS_TODO.md](ECS_TODO.md) | Completed implementation phases, validation record, and deferred work |
| [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md) | Recorded measurements, methodology, and optimization decisions |
| [Scene guide](../scenes/README.md) | Running examples and bounded rendering checks |
| [Benchmark guide](../scenes/benchmarks/README.md) | Benchmark commands, dependencies, and result organization |

Import ECS application types from `ECS`. Implementation modules are grouped by
responsibility:

| Module | Responsibility |
| --- | --- |
| `ECS.py` | Public facade, component pools/selections, joins, and world lifecycle |
| `ecs_types.py` | Numeric types, handles, decorators, and input validation |
| `ecs_schema.py` | Field metadata and payload conversion |
| `ecs_commands.py` | Deferred commands and entity creation tokens |
| `ecs_schedule.py` | Sequential update phases and dependencies |
| `ecs_diagnostics.py` | Explicit inspection and invariant auditing |

`RenderContext.py`, `DrawBatches.py`, and `Utils.py` implement rendering and its
supporting utilities. Generated compiler output belongs under `build/`; recorded
benchmark observations belong under `scenes/benchmarks/results/`.
