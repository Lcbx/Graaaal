# Benchmarks and diagnostics

Run these commands from the repository root after `pip install -e .` (and the
`render`, `assets`, `ui`, and `examples` extras for GPU tests). CPU scripts need
NumPy only. GPU scripts need a working WGPU backend; `benchmark_ecs_uploads`
opens a window, whereas the transparency diagnostics render offscreen.
Add `--help` to an entry point to see all supported options. Output files are
chosen by the caller; benchmarking does not modify the engine's defaults.

## ECS and CPU data access

```powershell
python -m scenes.benchmarks.benchmark_ecs --entities 100000 --repeat 11 --loops 50 --output ecs_cpu.json
python -m scenes.benchmarks.benchmark_ecs_uploads --frames 120 --output ecs_uploads.json
python src/tests/benchmark_array_access.py
```

`benchmark_ecs` reports query and mutation timings, gather/scatter costs and
array sizes. `benchmark_ecs_uploads` measures CPU-side uploads in a bounded
`direct_draw_example` run; its durations are not GPU frame times.
`benchmark_array_access` compares ordinary reads, reusable output arrays and
dense edits. Use `--implementation /path/to/compatible/ECS.py` with
`benchmark_ecs` to compare against a *standalone* ECS implementation. The
`benchmark_ecs_query_pairs` script similarly accepts a required
`--implementation` and `--output`; a standalone historical snapshot is needed
because the current `ECS.py` has package-relative imports.

## Transparency and noise

| Script | Output / purpose |
| --- | --- |
| `diagnose_banded_tail` | CPU-only comparison of nearest-wins and disjoint-band selection; JSON errors and sampling metrics. |
| `diagnose_noise_tilt` | GPU UV/anchored threshold noise at oblique angles; optional CSV and NPZ. |
| `diagnose_anchored_noise` | GPU anchored-noise layer correlations and motion samples; JSON. |
| `diagnose_multiple_winners` | GPU two-winner comparisons, validity/noise and timing; JSON. |
| `diagnose_winner_fill` | GPU winner-frequency checks for different retry counts and noise modes; JSON. |
| `diagnose_confidence_transparency` | GPU error/noise comparisons of a-trous and push-pull filtering; JSON and a PNG plot. Plotting requires Matplotlib. |
| `profile_transparency` | Per-pass offscreen GPU timing for stochastic reconstruction variants; JSON. |
| `profile_banded_tail` | GPU timing and image errors for banded-tail proxy/subset/sample-count variants; JSON. |

Examples:

```powershell
python -m scenes.benchmarks.diagnose_banded_tail --trials 64 --seed 2026 --output banded_tail.json
python -m scenes.benchmarks.diagnose_noise_tilt --mode uv_blue_lerp --output noise_tilt.csv --images noise_tilt.npz
python -m scenes.benchmarks.diagnose_anchored_noise --output anchored_noise.json
python -m scenes.benchmarks.diagnose_multiple_winners --frames 30 --output multiple_winners.json
python -m scenes.benchmarks.diagnose_winner_fill --attempts 1 2 4 8 --modes object_hash anchored_blue --output fill.json
python -m scenes.benchmarks.diagnose_confidence_transparency --size 256 --reconstruction atrous --output confidence.json
python -m scenes.benchmarks.profile_transparency --size 1280x720 --frames 30 --implementations fragment compute atrous --output transparency_profile.json
python -m scenes.benchmarks.profile_banded_tail --size 1280x720 --frames 24 --variants baseline proxy subset single --output banded_profile.json
```

`profile_transparency` accepts `--configs`, `--rounds`, `--warmup`, `--seed`,
`--stencil` (optional device feature) and `--exact-nearest`. Its timestamp
readbacks synchronize frames; compare configurations rather than treating the
numbers as sustained throughput. `profile_banded_tail` accepts `--pairs`,
`--samples`, `--frames`, `--warmup` and `--rounds`; it requires GPU timestamp
queries. Diagnostic images/metrics are static comparisons, not measurements of
camera-motion stability. Results depend on GPU, drivers, resolution and scene.

## Optional legacy Raylib comparisons

`benchmark_raylib_objects.py` and `benchmark_raylib_ecs.py` share
`raylib_benchmark_loop.py`. Install `pyray` separately to use them; it is not
part of the package extras. These interactive scripts are separate from the
WGPU scene benchmarks.

For the underlying ECS and rendering APIs, see [ECS](../../src/ECS_API.md),
[transparency](../../src/stochastic_transparency.md) and
[interactive scenes](../README.md).
