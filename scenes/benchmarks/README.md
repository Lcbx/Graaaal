# Benchmarks

Run from the repository root after installing `numpyecs`. CPU scripts require
NumPy; the upload benchmark also requires rendering dependencies and opens a
window. Write new results to a chosen output path; historical JSON
artifacts referenced by the reports are not included in this archive.

## Reproduction

`python -m scenes.benchmarks.diagnose_confidence_transparency --size 256 --output results/confidence.json`
compares complemented/independent threshold error and 2/4/8 sorted faint layers
at several confidence budgets. It writes JSON and a PNG; plotting requires
Matplotlib. `--reconstruction atrous` (default) or `push_pull` selects the filter;
`--support-seeds 64` adds a fixed-final-footprint variance audit for the cascade.
The audit reports empirical variance reduction relative to D; it does not assert
independence of spatial blue noise. `profile_transparency.py --implementations fragment compute push_pull atrous`
compares renderer timings. A-trous and push-pull use different footprints from
the bilateral modes; their output differences are comparisons, not parity checks.

The table lists main entry points and selected reusable helpers, not every
function. `main()` parses command-line arguments; the CPU helpers need no window.

| Script | Measures and useful functions |
| --- | --- |
| `benchmark_ecs.py` | `main()` measures queries, gather/scatter, removals, churn, appends and array bytes. `measure(setup, action, repeat, loops=1)` times actions; `make_world(n, rare_count=None)` builds test data. |
| `benchmark_ecs_query_pairs.py` | `main()` alternates baseline/current scans to investigate between-run noise; uses private query internals. |
| `benchmark_ecs_uploads.py` | `main()` instruments `CubeInstances.update` and `GpuBuffer.upload` in a bounded direct-draw scene, including initialization. |
| `diagnose_noise_tilt.py` | GPU threshold-image x/y autocorrelation at 0/60/75/83 degrees for UV footprint balances 0/0.5/1. Reports spatial FWHM in screen pixels, not timing. |
| `diagnose_anchored_noise.py` | Same-ID rear-layer win rates at 0.1–2 coarse-cell separations, plus point-to-pixel-to-anchor pan/orbit/dolly/wind comparisons. |
| `diagnose_multiple_winners.py` | Joint A/B validity, low-frequency error power, same-ID layer correlation, camera sweeps, and synthetic GPU timing versus 5x5 and 4x MSAA. |
| `diagnose_winner_fill.py` | Million-sample, five-sigma layer-frequency gate for single/average/fill under all noise modes, 1–4 layers, shared/distinct IDs and 1x/4x sample seeding. |
| `profile_transparency.py` | Offscreen actual-scene per-pass GPU timings, sample/tile classification, and fragment/compute parity at configurable resolution/zoom. |
| `../../src/tests/benchmark_array_access.py` | `benchmark(count=100000, samples=7, iterations=30)` returns JSON-compatible measurements of copies, reusable gathers and dense edits. |

```powershell
python scenes/benchmarks/benchmark_ecs.py --entities 100000 --repeat 11 --loops 50 --output ecs_cpu_run.json
python scenes/benchmarks/benchmark_ecs_query_pairs.py --implementation path/to/exported/ECS.py --output ecs_query_pairs_run.json
python scenes/benchmarks/benchmark_ecs_uploads.py --frames 120 --output ecs_uploads_run.json
python scenes/benchmarks/diagnose_noise_tilt.py --output noise_tilt.csv --images noise_tilt.npz
python scenes/benchmarks/diagnose_noise_tilt.py --mode anchored_blue --output anchored_tilt.csv
python scenes/benchmarks/diagnose_anchored_noise.py --output anchored_audit.json
python -m scenes.benchmarks.diagnose_multiple_winners --frames 30 --output multiple_winners_audit.json
python -m scenes.benchmarks.diagnose_multiple_winners --improvements --frames 30 --output force_fill_audit.json
python -m scenes.benchmarks.diagnose_winner_fill --output winner_fill_gate.json
python -m scenes.benchmarks.diagnose_winner_fill --attempts 1 2 4 8 --modes object_hash anchored_blue --output fill_iterations.json
python -m scenes.benchmarks.profile_transparency --size 1920x1080 --frames 30 --output tree_profile.json
python -m scenes.benchmarks.profile_transparency --size 1280x720 --frames 60 --rounds 4 --stencil --implementations fragment --output selection_profile.json
python src/tests/benchmark_array_access.py
```

The transparency profiler crosses force acceptance on/off with one/two winners,
average/fill mixing and 1x/4x MSAA. `--configs` selects names such as `single_3`,
`single_force_3`, `average_3`, `fill_3`, `msaa_fill_force_3`; single-winner kernels
also include 5 and 9. `--stencil` requests the optional `depth32float-stencil8`
feature and adds fill cases ending in `_stencil`. Hi-Z policy is unchanged.
Iteration comparisons include `fill4_3x1`, `fill8_3x1`, `fill2_3x2`, `fill2_3x3`,
`fill2_3x4`, `fill4_3x2`, `fill2_5x1` and `fill2_9x1`: the number after `fill` is
the winner budget, and `KxP` means a K-by-K kernel repeated P times. Prefix
`msaa_` and insert `_force` before the kernel for the matched MSAA/force cases.

Configurations are randomized each round using `--seed`; fragment/compute order
alternates between rounds. `--frames` is the total measured count for each
configuration/implementation, divided among rounds, with `--warmup` frames per
block. JSON includes execution order, raw per-frame/per-pass measurements,
median/p10/p90/p95/IQR, classification and output hashes. Selection A/B have
separate `3_select_winner_0` / `3_select_winner_1` labels. Timestamp readback
synchronizes each frame, so this is diagnostic timing rather than sustained
asynchronous throughput. Static parity does not measure camera-motion quality.
See the [selection report](../../src/transparency_reconstruction.md#selection-optimization-and-controlled-matrix)
for results and limitations.

The tilt diagnostic uses an analytic orthographic projection and the production
noise shader. It needs rendering dependencies but opens no window. `--oversample`
sets diagnostic samples per screen pixel (default 8); widths below two such
samples are flagged. `--samples 4` halves the effective noise size without
emulating hardware MSAA sample positions. `--mode` selects a UV noise mode.
`--mode anchored_blue` selects pixel-lattice anchored sampling instead, with
`--anchor-pixels` and `--subdivisions` exposing its grid controls. The UV balance
diagnostic explicitly compiles the balancing shader feature; the engine default
has it disabled.
NPZ image storage is optional; increasing oversampling helps resolve compressed
grain but raises GPU/readback/FFT costs. It does not measure camera-motion swim.

The multiple-winner diagnostic uses the production noise helpers with separate
default tiles/hierarchies. It reports all noise modes at p=0.1/0.5/0.9, including
joint validity and error power in 0 < frequency <= 1/8 cycles/pixel. A null power
ratio means the reference band has no measurable power. Same-ID overlap and
point-to-pixel camera sweeps expose remaining correlation and acceptance changes;
these are not rendered motion-quality measurements. GPU timestamps surround the
complete material renderer in a synthetic two-colored-sheet scene. If timestamp
queries are unavailable, synchronized CPU+GPU wall time is reported separately.
The four configurations have measured costs, not an assumed equal-time budget;
use `--size` and `--frames` to repeat at useful resolutions and sample counts.

The paired script requires a compatible saved baseline. `benchmark_ecs.py` also
accepts `--implementation path/to/exported/ECS.py`. Both loaders expect a standalone
module. Current `ECS.py` uses package-relative `ecs_*` imports, so it cannot be
copied alone and passed directly; use a self-contained historical snapshot or an
export adapted to the loader.

CPU ECS/array timing excludes setup and reports medians after warmup. Read/update
actions repeat within samples; mutation samples start with fresh worlds. Tracemalloc
peaks come from separate action-only runs, exclude retained storage, and are not
process RSS. Array `nbytes` estimates exclude Python objects/overhead. The upload
script includes initialization and wraps full `upload()` calls only; direct
`write_range()`/`upload_range()` calls are not separately counted. Its byte counts
use full CPU array sizes. Durations measure CPU submission, not GPU completion or
frame-time guarantees; check the actual frame count if the window closes early.

## Legacy Raylib comparison

`benchmark_raylib_objects.py` uses Python objects; `benchmark_raylib_ecs.py` uses
the ECS. Both share `raylib_benchmark_loop.py` and require the optional `pyray`
backend, which is not included in the package extras. These interactive comparisons
are separate from the NumPy CPU and current GPU-upload benchmarks.

Historical results are recorded in the [implementation work log](../../IMPLEMENTATION_WORK.md#ecs-benchmark-history).

See the [ECS reference](../../src/ECS_API.md) for API contracts and the
[scene guide](../README.md) for renderer examples and bounded runs.
