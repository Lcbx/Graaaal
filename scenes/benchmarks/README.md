# Benchmarks

Run from the repository root after installing `numpyecs`. CPU scripts require
NumPy; the upload benchmark also requires rendering dependencies and opens a
window. Write new results to a chosen output path rather than overwriting the
checked-in [results](results/).

## Reproduction

| Script | Measures |
| --- | --- |
| `benchmark_ecs.py` | Queries, gather/scatter, removals, membership churn, appends, and array-memory estimates |
| `benchmark_ecs_query_pairs.py` | Alternating baseline/current scans to investigate between-run noise |
| `benchmark_ecs_uploads.py` | CPU instance packing and buffer-upload submission during a bounded direct-draw scene |
| `../../src/tests/benchmark_array_access.py` | Copying, reusable gathers, and dense editing for equivalent movement |

```powershell
python scenes/benchmarks/benchmark_ecs.py --entities 100000 --repeat 11 --loops 50 --output ecs_cpu_run.json
python scenes/benchmarks/benchmark_ecs_query_pairs.py --implementation path/to/exported/ECS.py --output ecs_query_pairs_run.json
python scenes/benchmarks/benchmark_ecs_uploads.py --frames 120 --output ecs_uploads_run.json
python src/tests/benchmark_array_access.py
```

The paired script requires a compatible saved baseline. `benchmark_ecs.py` also
accepts `--implementation path/to/exported/ECS.py`. Both loaders expect a standalone
module. Current `ECS.py` uses package-relative `ecs_*` imports, so it cannot be
copied alone and passed directly; use a self-contained historical snapshot or an
export adapted to the loader.

CPU timing excludes setup and reports medians after warmup. Read/update actions
repeat within samples; mutation samples start with fresh worlds. Tracemalloc
peaks come from separate action-only runs, exclude retained storage, and are not
process RSS. Array `nbytes` estimates exclude Python objects/overhead. Upload
durations measure CPU submission, not GPU completion or frame-time guarantees;
check the actual frame count if the window closes early.

## Recorded ECS snapshots

These are historical observations, not measurements of the current checkout.
Raw JSON retains interpreter, NumPy, platform, sampling configuration, and source
hashes. The recorded CPU environment was Windows 11, AMD64 Family 25 Model 97,
Python 3.14.6, and NumPy 2.4.4, with 100,000 entities, 11 samples, and 50 loops
for repeated actions. Small differences across runs are not evidence of a win.

| Series | Artifacts |
| --- | --- |
| Original storage optimizations | [phase6 baseline](results/phase6/baseline.json), [optimized](results/phase6/optimized.json) |
| 2026-11-04 validation/API review | [baseline](results/review_2026-11-04/baseline.json), [measured implementation](results/review_2026-11-04/current.json), [paired scans](results/review_2026-11-04/query_pairs.json), [uploads](results/review_2026-11-04/uploads.json) |

The review baseline was exported from revision
`72b28d57baa79e7ee9c294570d562aa2429bb38d`. The two baseline files belong to
different comparisons. Match source/dependency hashes before attributing results
to an implementation; later package/API changes are not represented by these runs.

Selected original optimization results (milliseconds per action):

| Operation | Before | After |
| --- | ---: | ---: |
| Required query, 1,000 of 100,000 matches | 0.0677 | 0.0116 |
| Remove one component row | 0.0694 | 0.0422 |
| 1,024 appends to a tail owner | 114.12 | 70.41 |

Those changes selected sparse query candidates, avoided pool-sized small-removal
masks, and extended tail-owner blocks directly. They did not remove gather/scatter
costs or make alternating-owner appends cheap.

Selected 2026-11-04 review results (milliseconds per action):

| Operation | Baseline | Measured implementation |
| --- | ---: | ---: |
| Movement with uncached join/gather/scatter | 4.3518 | 4.5624 |
| Movement with cached join/gather/scatter | 2.0056 | 1.4188 |
| 1,024 alternating-owner appends | 125.0547 | 130.5569 |
| 1,024 rows appended in one batch | 0.1481 | 0.1430 |

An apparent sparse-query slowdown in separate runs did not recur in alternating
scans: 1,000-match medians were 0.01126/0.01128 ms. Batch appends remain much
cheaper than repeated calls. The 120-frame upload snapshot recorded 4.259 ms total
packing time and 2.709 ms CPU submission time for 3,843,840 instance bytes; these
values exclude setup and do not measure GPU completion.

## Array access

Measured locally on 2026-11-07: 100,000 entities, one `float32[3]` position field,
seven samples, and 30 movement steps per sample. Setup, reusable scratch allocation,
and warmup were excluded. Peaks measure one warmed step, excluding retained buffers.

| Cached movement path | Median ms/step | Peak traced bytes/step |
| --- | ---: | ---: |
| Copy selected values, compute, assign | 2.0284 | 3,600,288 |
| `read(out=...)`, NumPy `out`, assign | 1.1621 | 69,424 |
| `edit_dense()`, NumPy `out` | 0.0447 | 945 |

Reusable gathering retained two 1.2 MB work arrays and a cached 0.8 MB index
conversion; dense editing retained one 1.2 MB scratch array. Dense editing applies
when work follows one store's row order. Independent stores still need joins/gathers
for alignment. These access-path measurements are not whole-frame speedups.
The array script prints fresh JSON to stdout; no raw artifact accompanies this table.

## Legacy Raylib comparison

`benchmark_raylib_objects.py` uses Python objects; `benchmark_raylib_ecs.py` uses
the ECS. Both share `raylib_benchmark_loop.py` and require the optional `pyray`
backend, which is not included in the package extras. These interactive comparisons
are separate from the NumPy CPU and current GPU-upload benchmarks.

See the [ECS reference](../../src/ECS_API.md) for API contracts and the
[scene guide](../README.md) for renderer examples and bounded runs.
