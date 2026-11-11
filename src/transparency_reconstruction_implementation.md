# Transparency profiling and compute reconstruction

## Status

The original fragment reconstruction remains the default. A tiled compute
prototype is available through `reconstruction="compute"` or
`set_reconstruction("compute")` on `TransparencyPass` and `MaterialRenderer`.
The example exposes **Filter implementation**. It preserves the bilateral
estimator and has no temporal history, projection jitter, or extra noise streams.

The prototype did not improve whole-frame GPU time on the measured scene, so it
is explicitly experimental. The profiling tools identify where further work is
likely to pay off rather than assuming reconstruction is the largest cost.

## Shared estimator and compute implementation

`shaders/stochastic_reconstruction.shaderlib` contains the common inputs,
bilateral weights, slope correction, certainty/force bypasses, expected-acceptance
1x1 estimator and normalized reconstruction. Both fragment and compute call this
code. Weighted RGB and validity are pooled before normalization, including fill's
weight correction. Background composition, MSAA HDR resolve, tone mapping and
sRGB encoding remain in the final presentation pass and happen once.

Compute uses 8x8 workgroups, one workgroup per hardware sample at 4x MSAA. It caches
the tile plus the kernel halo (10x10 for 3x3, 12x12 for 5x5, 16x16 for 9x9).
Separate shared arrays hold transmittance, winner RGB/weight, guide and two linear
depths: 14 KiB maximum, plus two small workgroup flags. This stays below the
16 KiB portable workgroup-storage limit. Actual kernel size determines halo loads;
the maximum storage allocation is fixed.

Threads first classify whether any center in the tile needs filtering. Uniform
workgroup decisions keep barriers safe for partial tiles at image boundaries.
Empty tiles return together; certain/forced tiles avoid shared halo loads.
Within mixed tiles, certain/forced pixels retain the same direct bypass.

The optional `TransparencyTargets.reconstruction_color` is `rgba16float`, sampled
+ storage, one layer per hardware sample (array view at 4x). RGB contains the
linear transparent contribution; alpha contains filter support. This costs
8 bytes per hardware sample and exists only in compute mode. It is deliberately
undefined/stale where current scalar acceptance is zero: presentation checks
current transmittance before reading it, so no clear or empty-pixel write is
needed. Caller code must respect that contract when inspecting the intermediate.

1x1 and winner/transmittance/forced diagnostics skip compute dispatch entirely.
Filter-support and reconstruction diagnostics use the same compute estimator.
Kernel/debug changes keep targets and pipelines; sample-count/implementation
changes rebuild the appropriate resources. Low-level custom shader sets remain
on their own fragment path; default material hooks are supported.

An additional half-float round occurs at the intermediate. All measured scene
comparisons differed by at most one 8-bit output value, including 4x MSAA; the
prototype does not promise bitwise equality to the float32 fragment reduction.

## GPU profiler

`GpuProfiler` records timestamps at native render/compute pass boundaries:

```python
from numpyecs.GpuProfiler import GpuProfiler

# Request timestamp-query with the device's required features first.
profiler = GpuProfiler(max_passes=128)
commands = profiler.wrap(RenderContext.commands())
renderer.record(commands, size, camera, output)
RenderContext.submit(commands.finish())
passes_ms = profiler.read()  # Synchronizes: intended for diagnostics.
frame_ms = profiler.last_frame_ms
profiler.close()
```

Repeated labels are summed. Frame duration spans the first pass start through
the last pass end, including intervening work/gaps; it excludes uploads before
the first pass, query resolution/readback and later diagnostic classification.
The sum of individual pass durations is reported separately and is not the whole
frame. One wrapped command context must finish, submit and be read before the
next is wrapped. `close()` releases the query/buffer resources after completion.

The profiling benchmark falls back to separately labeled synchronized wall time
if timestamp queries are unavailable; it never labels CPU timings as GPU time.
`scene.render_frame(profiler=profiler)` also records the example's depth prepass,
Hi-Z construction and culling, which `MaterialRenderer.record()` does not perform.

## Reproduction and measured findings

```powershell
python -m scenes.benchmarks.profile_transparency --size 1280x720 --frames 30 --output tree_720p.json
python -m scenes.benchmarks.profile_transparency --size 1920x1080 --frames 30 --output tree_1080p.json
python -m scenes.benchmarks.profile_transparency --size 1920x1080 --zoom 2 --frames 30 --output tree_close.json
```

Runs are offscreen, static-camera captures of the actual example asset, using
anchored blue and five warmup frames per block. `--configs` selects a
subset of the matched force/winner/MSAA/kernel configurations. GPU classification happens after
the timed frames and reports empty, exact-single forced, certain, stochastic,
missing-winner samples and tiles requiring filtering. Classification is not timed
as rendering. Counts refer to hardware samples, and tiles are per hardware sample.

NVIDIA RTX 5060 Ti, Vulkan; median full-frame GPU milliseconds:

| Configuration | 720p fragment / compute | 1080p fragment / compute | 1080p zoom 2 fragment / compute |
| --- | --- | --- | --- |
| 1 winner, force, 3x3 | 0.463 / 0.496 | 0.684 / 0.763 | 1.059 / 1.166 |
| 1 winner, force, 5x5 | 0.480 / 0.501 | 0.721 / 0.762 | 1.128 / 1.199 |
| 1 winner, force, 9x9 | 0.520 / 0.527 | 0.804 / 0.821 | 1.314 / 1.321 |
| 2 averaged winners, 3x3 | 0.583 / 0.634 | 0.917 / 1.008 | 1.523 / 1.611 |
| 4x MSAA, 1 winner, 3x3 | 0.939 / 1.040 | 1.810 / 1.989 | 3.219 / 3.660 |

Saved raw per-pass timings/classification/parity:
[720p](../scenes/benchmarks/results/tree_reconstruction_720p_2026-11-11.json),
[1080p](../scenes/benchmarks/results/tree_reconstruction_1080p_2026-11-11.json),
[closer 1080p](../scenes/benchmarks/results/tree_reconstruction_close_1080p_2026-11-11.json).

At the default 1080p framing, only ~3.1% of samples contain transparency and ~1.8%
need reconstruction with forcing enabled. Only 972/32,400 sample-tiles require
filtering. At zoom 2, coverage rises to ~11.4%, with ~5.8% needing filtering.
This explains why an additional dispatch/intermediate is hard to amortize.

In the first 1080p force+5x5 profile, selection cost ~0.134 ms versus ~0.075 ms for
fragment reconstruction/composition. Two winners approximately doubled selection
cost. At 9x9, the filter becomes more significant, but shared-memory reuse still
did not yield a material whole-frame win after presentation/dispatch overhead.
Small differences should be repeated under stable clocks and on other hardware.

Next priority from these measurements: reduce selection/deformation/opacity work,
especially B's unnecessary raster work, and consider scheduling only active tiles
if pursuing compute further. The current workgroup empty-tile bypass still
launches workgroups across the screen. No claim is made about rendered camera
motion quality; the estimator and sample positions are intentionally unchanged.

## Selection optimization and controlled matrix

Deterministic rejection of zero opacity, culled backfaces and opaque occlusion
now precedes contributor/guide reads and noise evaluation in selection. All
material/LOD/alpha calculations and local/UV derivatives remain before discard;
noise functions consume those explicit gradients. Nine before/after actual-tree
captures were bitwise identical. At 720p, separate 40-frame runs measured 4x MSAA
selection at 0.237 -> 0.229 ms without forcing and 0.245 -> 0.235 ms with forcing.
Other selection differences were small; these separate runs do not establish a
whole-frame speedup. Raw [before](../scenes/benchmarks/results/selection_baseline_720p_2026-11-11.json)
and [after](../scenes/benchmarks/results/selection_early_reject_720p_2026-11-11.json).

The profiler now randomizes configuration order over seeded rounds, alternates
fragment/compute order, and records all frame/pass samples plus median, p10, p90,
p95 and IQR. Force on/off pairs exist for each winner/MSAA setting, including
real-tree fill. Selection A/B receive distinct timestamp labels. `--frames` is
the total measured count per case, divided over `--rounds`; each block warms up
again. Readback still synchronizes every frame and cannot establish asynchronous
throughput. The asset framing remains sparse, so a high-overdraw workload is
still needed for general performance conclusions.

The first expanded 1080p run exposed stale target retention: native bind groups
in the shared layout caches held earlier full-resolution textures across setting
switches, pushing GPU memory usage to 15,156 MiB of 16,311 MiB and
producing large timing outliers. `BindGroupLayout.clear_cache()` now releases
cached bindings without destroying the layout or invalidating recorded work.
Transparency clears its layout caches after replacing targets; same-size no-op
resizes retain caches. This also fixes retention during ordinary resize/MSAA/
feature switches. The saved matrices below were rerun with this fix; the noisy
run was replaced rather than used to claim a speedup.

`early_b_rejection=True` / `set_early_b_rejection(True)` is an experimental
fill-only stencil gate. Request the optional `depth32float-stencil8` GPU feature.
A clears stencil and marks accepted depth-passing samples; B clears selection
depth/payload but loads A's stencil read-only and rejects marked samples. Winner
colors and the fill estimator remain unchanged. Both raster streams must use
`selection_pass(commands, stream=stream)`; MaterialRenderer and the example do
this automatically. Switching to one winner or average disables the gate.

This keeps full float32 selection depth and does not change opaque/guide depth.
It offers an opportunity for hardware rejection, not a guarantee that the driver
skips fragment execution: discard/helper invocations can limit early testing.
It cannot skip vertex processing. The ungated shader check remains in place.
Hi-Z generation and culling policy are unchanged.

Matched tree results, RTX 5060 Ti/Vulkan, fragment reconstruction, 60 frames
per case over four randomized rounds (five warmups per block), in milliseconds:

| Setting, 3x3 | 720p frame median (IQR) | 1080p frame median (IQR) | Fill B, 720p / 1080p |
| --- | --- | --- | --- |
| Single | 0.446 (0.025) | 0.674 (0.028) | — |
| Single + force | 0.470 (0.036) | 0.692 (0.030) | — |
| Average | 0.573 (0.024) | 0.875 (0.028) | — |
| Average + force | 0.592 (0.026) | 0.912 (0.025) | — |
| Fill | 0.518 (0.023) | 0.793 (0.027) | 0.0344 / 0.0628 |
| Fill + stencil | 0.525 (0.029) | 0.800 (0.023) | 0.0348 / 0.0650 |
| Fill + force | 0.544 (0.025) | 0.811 (0.028) | 0.0355 / 0.0642 |
| Fill + force + stencil | 0.543 (0.027) | 0.815 (0.027) | 0.0363 / 0.0654 |
| 4x MSAA single | 0.857 (0.024) | 1.662 (0.028) | — |
| 4x MSAA single + force | 0.878 (0.024) | 1.685 (0.029) | — |
| 4x MSAA average | 1.327 (0.035) | 2.616 (0.053) | — |
| 4x MSAA average + force | 1.428 (0.024) | 2.827 (0.062) | — |
| 4x MSAA fill | 1.112 (0.024) | 2.176 (0.031) | 0.1285 / 0.2563 |
| 4x MSAA fill + stencil | 1.120 (0.034) | 2.204 (0.141) | 0.1291 / 0.2578 |
| 4x MSAA fill + force | 1.144 (0.027) | 2.213 (0.045) | 0.1288 / 0.2557 |
| 4x MSAA fill + force + stencil | 1.157 (0.026) | 2.244 (0.056) | 0.1296 / 0.2560 |

All gated/ungated output hashes match exactly. There is no consistent B-pass
speedup, and whole-frame differences overlap measured spread. Keep stencil off;
force acceptance remains a quality/cost choice rather than a free optimization.
Fill costs less than averaging here, but its independence restrictions still
apply and static captures cannot rank motion quality.
Of 28,731 active 720p samples, single-winner missing counts fall from 3,410 to
1,163 with forcing; two streams reduce these to 1,652 and 385 respectively.
Average and fill have identical empty-sample counts, as expected for the same
two threshold streams. These are deterministic capture diagnostics, not an
independence test or a direct measure of perceived crawling.
[Raw 720p matrix](../scenes/benchmarks/results/selection_stencil_matrix_720p_2026-11-11.json),
[raw 1080p matrix](../scenes/benchmarks/results/selection_stencil_matrix_1080p_2026-11-11.json).

## Arbitrary iterations

Fill accepts any positive uint32 `winner_count`; average remains limited to one
or two. Both backends accept a positive uint32 `reconstruction_passes`, default
one. For example, with an existing material renderer:

```python
renderer.set_noise_mode("anchored_blue")
renderer.set_winner_mix("fill")
renderer.set_winner_count(4)
renderer.set_reconstruction_passes(3)  # Three stages of the chosen kernel.
```

The example exposes editable **Winner draws** and **Reconstruction passes**.
For its UI, select two draws and Fill before raising the budget further. A 1x1
kernel bypasses all reconstruction stages. Current defaults are two averaged
winners using anchored blue and one 3x3 filter pass.

Each retry preserves existing winners and uses its own immutable stream settings
and hashed seed. Shade weights use the total geometric sum
`1/(1+T+...+T**(N-1))`, computed in O(log N), not a loop over all N terms per
shaded fragment. Selection/depth/winner textures are reused; only uniform and
binding/draw storage grows with N. The optional stencil gate preserves A's mask
for every retry; shading-buffer checks reject wins from later attempts. Writing
new stencil bits during a retry would incorrectly block a nearer winner within
that same raster pass, so that shortcut is deliberately avoided.

Reconstruction stages average weighted color/validity pairs before final ratio
normalization. Each stage uses the original deterministic guide and the original
raw winners for certain/forced bypasses. Opacity, background, MSAA resolve, tone
mapping and sRGB conversion happen once. Two stages allocate one extra 8-byte
RGBA16F target per hardware sample; three or more use two reusable targets.
Stage count does not increase target count. Fragment stages render separately
to each hardware-sample layer; compute stages use tiled dispatches. Intermediate
empty regions are explicitly zeroed for subsequent neighborhood reads. Final
compute's presentation-only intermediate retains its existing empty-tile skip.

Repeated filtering has different weights from one wider kernel. Four 3x3 stages
reach 9x9 support, but do not reproduce a single 9x9 bilateral pass. Half-float
stores may round differently from NumPy's nearest conversion; weighted-pair CPU
checks allow one half-float ULP per intermediate store. Large pass counts can
accumulate quantization error. No fixed performance or quality improvement is
promised by raising either count.

The [million-sample audit](../scenes/benchmarks/results/winner_fill_iterations_2026-11-11.json)
passes at 1/2/4/8 attempts, 1–4 layers and 1x/4x seed rules for both supported
noise modes with distinct IDs: maximum z is 3.04 for object hash, 3.03 for anchored
blue. Actual raster tests additionally pass the four-attempt layer-frequency gate
at native 1x/4x MSAA. More than eight attempts are supported but unaudited;
these tests do not prove joint independence for every asset, and shared-ID
overlap failures remain. A retains its tile; later attempts reuse the second
tile with separate hashed seeds.

Actual tree, RTX 5060 Ti/Vulkan, anchored blue, force off, 40 measured frames per
case over four randomized rounds, five warmups per block. Fragment/compute order
alternates. Whole-frame median GPU milliseconds at 1080p:

| Fill attempts | Kernel × stages | Fragment | Compute | Missing raw active winners |
| --- | --- | --- | --- | --- |
| 2 | 3x3 × 1 | 0.801 | 0.863 | 3,328 |
| 4 | 3x3 × 1 | 1.008 | 1.090 | 1,528 |
| 8 | 3x3 × 1 | 1.470 | 1.517 | 935 |
| 2 | 3x3 × 2 | 0.852 | 0.943 | 3,328 |
| 2 | 3x3 × 3 | 0.918 | 1.034 | 3,328 |
| 2 | 3x3 × 4 | 0.952 | 1.107 | 3,328 |
| 2 | 5x5 × 1 | 0.818 | 0.873 | 3,328 |
| 2 | 9x9 × 1 | 0.908 | 0.930 | 3,328 |
| 4 | 3x3 × 2 | 1.071 | 1.163 | 1,528 |

There are 64,175 active samples. Filtering does not change raw winner counts;
retrying does. At native 4x MSAA, two attempts/one 3x3 stage measured 2.222/2.492
ms fragment/compute; four attempts/two stages measured 3.519/3.883 ms. All actual
tree fragment/compute comparisons differ by at most one output byte. Full raw
distributions/classification are saved for
[720p](../scenes/benchmarks/results/transparency_iterations_720p_2026-11-11.json) and
[1080p](../scenes/benchmarks/results/transparency_iterations_1080p_2026-11-11.json).
These static captures establish cost/parity, not a motion-quality ranking. At this
sparse framing, larger single kernels remain cheaper than stacking stages with
the same maximum reach; keep both choices available for visual comparison.

## Validation

After arbitrary iteration support: **809 passed in 174.05 seconds** in the full
regression suite, plus the final custom-shader constructor rejection check run
separately. Direct scene startup and nine offscreen frames verified constructor
options and runtime switches across 1x/4x MSAA, both implementations, 3/4/7 fill
draws and 2/3/4 reconstruction stages. New tests cover CPU weighted-pair
convolution with image boundaries, native million-sample four-attempt frequencies,
first-winner retention, exact weights/source tags, odd/even ping-pong, clearing,
forced bypasses, raw diagnostics and 1x1 bypass. Expanded stencil checks compare
both two- and four-attempt fill bit-for-bit, including reversed submission.

After selection/stencil and target-cache changes: **765 passed in 148.42 seconds**
in the full suite; the focused API/cache/stencil run passed **48 tests**.
The new checks compare exact per-sample winner storage and displayed output
with stencil on/off at 1x/4x, force on/off and 1x1/3x3, reversed submission,
new-frame clearing, resize and runtime mix/count/MSAA switching. Cache checks
verify release on target replacement, reuse on no-op resize and safety of
already-recorded work. A two-round fragment/compute scheduler smoke test verified
alternating order and output parity within one byte.

Focused checks cover tilted depth discontinuities, force/fill weights, 1x/4x MSAA,
3x3/5x5/9x9, non-multiple image sizes, empty/certain tiles, stale intermediates,
resizing, runtime switches, all debug views, and native timestamp queries.
Before the selection follow-up, full regression: **755 passed in 141.30 seconds**. Final focused compute/UI checks
also cover allocating timestamps on scope entry rather than construction (unused
scopes and different entry ordering do not corrupt queries). Live example:
**150 frames** completed across implementation/MSAA/force/fill/kernel/debug
switches. These are integration checks, not a rendered motion-quality verdict.
