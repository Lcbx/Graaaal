# Benchmarks

Run commands from the repository root. The ECS CPU scripts require NumPy;
the upload benchmark also requires the rendering dependencies and opens a window.
Write fresh observations to a chosen output path. The checked-in `results/`
directories contain historical measurements and should not be overwritten by
routine runs.

## Current scripts

| Script | Measures |
| --- | --- |
| `benchmark_ecs.py` | Query lookup/cache behavior, gather/scatter, removals, membership churn, appends, and array-memory estimates |
| `benchmark_ecs_query_pairs.py` | Alternating baseline/current query scans to investigate timing noise |
| `benchmark_ecs_uploads.py` | Direct-draw instance packing and CPU time submitting buffer uploads during a bounded scene |

```powershell
python scenes/benchmarks/benchmark_ecs.py --entities 100000 --repeat 11 --loops 50 --output ecs_cpu_run.json
python scenes/benchmarks/benchmark_ecs_query_pairs.py --implementation path/to/exported/ECS.py --output ecs_query_pairs_run.json
python scenes/benchmarks/benchmark_ecs_uploads.py --frames 120 --output ecs_uploads_run.json
```

`benchmark_ecs.py --implementation path/to/exported/ECS.py` can benchmark a compatible
standalone implementation snapshot. The paired script requires such a baseline.
Do not treat the current source file copied alone as a standalone snapshot: it
imports the extracted `ecs_*` modules. Preserve those dependencies when saving a
modern implementation for comparison.

CPU timing excludes setup and uses medians of repeated samples; allocation peaks
come from a separate tracemalloc pass. Peaks exclude preexisting storage and are
not process RSS. Upload timings measure CPU submission, not GPU completion. The
upload script may finish early if the window is closed; inspect its actual frame
count. Measurements depend on hardware, installed versions, and host activity.

## Recorded result series

| Directory | Observations | Provenance |
| --- | --- | --- |
| [results/phase6](results/phase6/) | `baseline.json`, `optimized.json` | Original candidate-query/removal/tail-append optimization comparison |
| [results/review_2026-11-04](results/review_2026-11-04/) | `baseline.json`, `current.json`, `query_pairs.json`, `uploads.json` | Validation/API/refactoring follow-up; baseline source revision is documented in the report |

Raw observations are preserved unchanged, including historical implementation
names/hashes and environment information. The two baseline files belong to
different comparisons. Source-hash caveats and interpretation of outliers are
documented in [the benchmark report](../../src/ECS_BENCHMARKS.md).

## Legacy Raylib comparison

`benchmark_python_only.py` is the original non-ECS comparison;
`benchmark_optimized.py` uses the ECS and cached movement/render queries.
Both use `benchmark_loop.py` and require the optional `pyray` backend. Their
historical names are retained because callers/tests import them. They are separate
from the standalone NumPy CPU benchmark and the current rendering-upload benchmark.

See [the scene guide](../README.md) for renderer examples and bounded smoke checks.
