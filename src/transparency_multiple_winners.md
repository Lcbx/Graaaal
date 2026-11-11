# Multiple-winner stochastic transparency

## Current defaults

The example and `TransparencyPass` now start with anchored blue noise and two
averaged winners. `MaterialRenderer` inherits those defaults. Explicit one-winner
and alternate-noise configurations remain supported; historical measurements
below retain their original recorded settings.

## 2026-11-11: arbitrary fill budgets and repeated reconstruction

- Implemented any positive uint32 winner count in fill mode; averaging stays at one
  or two. Reuse payload/depth/winner targets, with immutable per-attempt settings
  and source tags recording the actual first successful attempt.
- Generalized fill weights to `1 / sum(T**i, i=0..N-1)` using a stable logarithmic
  geometric sum. Keep force acceptance and raw diagnostics unchanged.
- Added `reconstruction_passes`, default one, to the pass/material renderer and
  example. Filter weighted color/validity pairs through at most two reusable
  intermediate targets; normalize/apply opacity only at final composition.
  Support fragment and tiled compute paths, per hardware sample. 1x1 bypasses all.
- Validated higher-count layer frequencies, weights, retention/clearing, runtime
  switches and MSAA; repeated filtering matches a CPU weighted-pair reference,
  allowing one half-float ULP per store. Both implementations preserve empty
  regions and deterministic bypasses. Image comparisons stay within one byte.
- Extended timing/diagnostic tooling. The million-sample gate passed at 1/2/4/8
  attempts for distinct IDs, 1–4 layers and 1x/4x seed rules, for both supported
  modes (max 3.04 sigma). Shared-ID failures remain diagnostic, not fixed by retries.
  Native-raster million-sample tests also pass at four attempts, 1x/4x.
  [Audit](../scenes/benchmarks/results/winner_fill_iterations_2026-11-11.json).
- A stencil that marks winners inside each retry blocks closer candidates in
  that same draw. Retained A-only read-only stencil; later retries also test the
  shaded winner buffer. No changed winner ordering or depth semantics.

Whole-frame tree timings and the repeated-filter tradeoffs are in the
[iteration report](transparency_reconstruction.md#arbitrary-iterations).

Validation: full regression **809 passed in 174.05 seconds**. The final
one-attempt/custom-shader rejection check passed separately after adding its
constructor guard. A direct example startup with four winners/three filters
completed nine offscreen frames while switching 1x/4x MSAA, fragment/compute,
3/4/7 winners and 2/3/4 reconstruction stages. `git diff --check` is clean apart
from repository line-ending warnings. Defaults and Hi-Z policy are unchanged.

## 2026-11-11: selection optimization and controlled comparisons

Requested follow-ups 1, 2 and 4 are implemented. Hi-Z remains unchanged for
high-mesh-count scenes.

- Moved zero-opacity, backface and opaque-depth rejection ahead of expensive
  selection noise and optional target reads, after every derivative-dependent
  calculation. Nine actual-tree before/after image hashes match exactly.
- Expanded the profiler to matched force on/off cases across winner count,
  average/fill and 1x/4x MSAA, including real-tree fill. Added seeded randomized
  rounds, alternating fragment/compute order, separate A/B timestamps, output
  hashes, raw frame/pass samples and timing distributions.
- Investigated early per-sample B rejection with an optional fill-only stencil
  gate. A writes accepted coverage, B preserves that stencil while clearing its
  own depth/payload. Uses optional `depth32float-stencil8`, retaining float32
  selection depth; does not alter opaque inputs, probabilities or the estimator.
  Runtime setters on pass/material renderer reset it when leaving two-winner fill.
- Keep stencil gating off by default: matched 720p measurements show no reliable
  B-pass benefit. Drivers may defer stencil testing around discard, and vertex
  processing still runs. Gated/ungated image hashes match at 720p and 1080p.
- The expanded 1080p matrix exposed old targets retained by native bind-group
  caches after repeated switches. Added explicit cache release on target
  replacement, preserving layouts and in-flight work. Both matrices were rerun
  after the fix; the contaminated timings were replaced.

Detailed timings, API contract, reproduction and limitations are maintained in
the [selection report](transparency_reconstruction.md#selection-optimization-and-controlled-matrix).

Validation: **765 passed in 148.42 seconds** in the full regression suite.
The focused API/cache/stencil suite passed **48 tests**. Per-sample parity,
reversed submission, clearing, resize, mix/count/MSAA switching and cache
invalidation are covered. The alternating fragment/compute scheduler smoke test
preserved output within one byte; both saved full-resolution matrices preserve
gated/ungated fill hashes exactly. `git diff --check` is clean apart from line-ending warnings.

## 2026-11-11: pass profiling and reconstruction prototype

Added native per-pass GPU timestamps, untimed sample/tile classification and a
shared-estimator compute reconstruction prototype. The default remains fragment
reconstruction. At 720p/1080p and a closer 1080p view, compute did not provide a
material whole-frame speedup; its displayed results matched within one byte.
The actual scene is mostly empty at default framing, and selection costs more
than 5x5 filtering. Details, raw results, reproduction commands and the next
optimization priorities are in [transparency_reconstruction.md](transparency_reconstruction.md).

Validation: **755 full-suite tests passed in 141.30 seconds**. The compute/UI
checks were also repeated after tightening profiler scope-entry ordering. The
live example completed **150 frames** while switching implementation, MSAA,
force/fill, kernels and debug views. `git diff --check` is clean apart from
repository line-ending conversion warnings.

## 2026-11-11: force acceptance and conditional fill investigation

Implemented and validated. Averaging remains the default. Before changing the
production shaders, `diagnose_winner_fill.py` evaluates 1,048,576 samples per
configuration using the production threshold fields, 1–4 layers, all modes,
distinct/shared IDs, and 1x/4x sample seeding. Its five-sigma gate checks both
streams and inferred hole-filler layer frequencies. Shared-ID failures diagnose
existing overlap correlation rather than establish a valid independent-layer model.

Applicable changes selected:

- Exact single-contributor detection in the transmittance pass, with an optional
  additive per-sample count target. This replaces the approximate `p >= A_s-0.002`
  rule, which can force one of two equally important p=0.001 layers. Counts ignore
  zero-opacity/backfacing/opaque-occluded fragments. No probability floor is added.
- Reserve payload.w bit 16 (a zero mantissa bit of the +/-1 face half) for the
  forced flag; mask it before unpacking the face/LOD pair. Payload size unchanged.
- Forced shading stores `(A_s*C, A_s)`. Composition bypass uses the exact count,
  positive stored acceptance and a nonempty winner, not an ambiguous
  `abs(weight-A_s)` comparison. A sole
  grazing-faded contributor can correctly be forced at its reduced opacity.
- Force acceptance starts disabled. Conditional fill will only be offered for
  noise configurations that pass the statistical gate. Failing modes remain
  available with averaging; independence will not be inferred from correlation.

The gate completed before production shader edits. For distinct IDs, only
`object_hash` (maximum 2.48 sigma) and `anchored_blue` (2.30 sigma) passed every
tested 1x/4x, 1–4 layer configuration. UV CDF/pick, Bayer, triplanar, and both
hierarchy modes failed at least one configuration (maximum z respectively
24.54/9.91, 407.53, 14.80, and 116.80/189.41). These are empirical restrictions,
not a proof of independence for all assets/footprints. Full results:
[million-sample gate](../scenes/benchmarks/results/winner_fill_gate_2026-11-11.json).

Implemented:

- `force_single_contributor=False` and runtime setter on `TransparencyPass` and
  `MaterialRenderer`, with an example checkbox and a forced-mask diagnostic.
  The optional count target is per-sample `r16float`, additive in the existing
  transmittance draw. Integer increments distinguish exactly one from two or more;
  half-float saturation at very large counts cannot turn them back into one.
- Payload bit 16 records forcing, preserving the 128-bit payload and 4x packing.
  Weighted shading uses the same stored transmittance as composition. Composition
  uses exact count==1, positive stored acceptance and a nonempty winner to identify the bypass, never a
  rounded probability/weight comparison. B skips forced samples and average mixing
  copies A. A sole grazing-faded surface is deliberately eligible at its faded
  opacity; zero-opacity fragments and any second nonzero contributor are not.
- `winner_mix="average"|"fill"` and runtime setters. Fill requires two winners,
  the audited default noise pair, and object hash or anchored blue. Other modes
  and custom tiles are rejected before mutation. Selecting one winner resets to
  average. The example returns to average when cycling into an unsupported mode.
- Fill uses `select A -> shade A -> clear payload/depth -> select B -> shade B`.
  A's full-screen shading writes zeros for empties each frame; B discards where A
  has positive weight and its compute shader never writes an empty payload. No
  read-write storage texture or HDR color scratch is needed. Non-forced A/B
  winners both store `(C,1)/(1+T_scalar)`; forced ones retain `(A_scalar*C,A_scalar)`.
- Fill retains an `r32uint` source tag per hardware sample for A/B/count diagnostics
  (4 bytes/sample instead of the averaging color scratch's 8). Together with the
  optional force count, this adds 6 bytes/sample; fill without force adds 4.
  Single+force adds 2; single without force retains its original target footprint.
- Debug color divides by stored weight for weighted modes. Validity and forced
  views remain distinct. The old averaging diagnostic convention is retained
  when forcing is disabled.
- Four actual-raster tests passed the five-sigma `(0.625,0.3125,0.0625)` layer
  frequency check over 1,048,576 samples each: both allowed modes at 1x and 4x.
  Focused tests also verified A retention, weight compensation, exact/near-zero
  contributor distinctions, live switches, RGB opacity, grazing, MASK interiors,
  and clearing. Regression results are recorded below after completion.

### Performance decision

[Measured results](../scenes/benchmarks/results/force_fill_2026-11-11.json), using
`diagnose_multiple_winners --improvements --frames 30`, NVIDIA RTX 5060 Ti/Vulkan,
256x256, anchored blue, p=0.1, 30 timed frames after five warmups. RMSE is measured
in tone-mapped linear display color against the analytic mean; no sRGB-space error
metric or subjective motion verdict is substituted.

| Layers | Draws/mix | MSAA | Kernel | Force | Median GPU ms | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1/average | 1x | 3x3 | off | 0.1586 | 0.04821 |
| 2 | 2/average | 1x | 3x3 | off | 0.2116 | 0.03492 |
| 2 | 1/average | 1x | 5x5 | off | 0.1864 | 0.03739 |
| 2 | 1/average | 4x | 3x3 | off | 0.5174 | 0.02426 |
| 2 | 2/fill | 1x | 3x3 | off | 0.2033 | 0.03658 |
| 2 | 2/fill | 1x | 3x3 | on | 0.2236 | 0.03658 |
| 1 | 1/average | 1x | 3x3 | off | 0.1510 | 0.02766 |
| 1 | 1/average | 1x | 3x3 | on | 0.1281 | 0.00069 |

For isolated surfaces, forcing removed stochastic holes and reduced cost in this
scene. Fully overlapping layers cannot force and incur count-target overhead.
Fill was slightly cheaper but noisier than averaging. It did not establish a
clear cost/error win over both averaging and the larger single-winner kernel, so
**averaging stays default; forcing stays opt-in**. These are discrete measured
costs, not an exact matched-time sweep or a tree-scene performance claim.
Measurements preceded the final zero-acceptance guard; sub-0.01 ms differences
between configurations should not be treated as a decisive performance win.

Still needed for a default-policy change: target-scene, equal-budget comparisons
and human pan/orbit/dolly assessment. The old acceptance-motion sweep remains an
averaging proxy; no new rendered fill-motion-quality claim is made here.

### Integration validation

- Contributor count, weighted single/dual paths, 1x/4x MSAA, payload flag and
  canonical face decoding passed focused readback tests.
- Actual fill frequencies passed four million-sample raster cases. A retention,
  source tags and empty-frame clearing passed; defaults remain unaltered.
- External opaque color/depth tests were expanded across forcing on/off, both
  winner counts and both MSAA counts.
- Final complete suite: **732 passed in 157.85 seconds**. This includes the
  positive-acceptance check in both forced identification and the average-mode
  B-copy path. Even with count==1, a stochastic winner must not be mistaken for a
  forced one if stored T_scalar is exactly one; a custom-threshold p=1e-8
  regression verifies that A/B still average and the forced mask remains clear.
- Live tree example: **150 frames** completed with force/mix/count/MSAA/kernel
  and diagnostic changes. This confirms resource/binding/command integration,
  not a subjective motion-quality or equal-time quality verdict.
- `git diff --check` passed (line-ending conversion warnings only).

## 2026-11-10: original averaging implementation

Status: implemented, regression-tested, and measured. Single winner remains the
default. Rendered motion quality on the target scene still needs human comparison.

## Scope and estimator

Implement `winner_count=1|2`, default **1**. Each stream independently performs the
existing nearest-accepted-fragment selection, with replacement: both streams may
select the same surface. Do not take the two nearest accepted layers.

Each stream shades to `(straight_color * valid, valid)`. Mix both components by
one half. Mixed alpha is **0, 0.5, or 1**, representing valid-stream fraction,
not opacity or certainty. Pool these samples in the existing bilateral ratio:

```
A_rgb * sum(weight * mixed.rgb) / max(sum(weight * mixed.a), epsilon)
```

The unfiltered 1x1 estimator retains division by expected acceptance `A_scalar`.
Transmittance remains deterministic and is accumulated once. Existing RGB-opacity
approximation and normalized-filter bias remain; adding streams does not fix them.

## Expected benefit and limits

Two streams primarily provide another chance at a valid sample and another layer
sample. At the same raster position, a single surface has the same shaded color
in both streams. This adds no texture detail or silhouette supersampling.
Independent streams give no-winner probability `(1-A_scalar)^2`; ideal variance
reduction is one half. Same-ID layer correlation, UV seams, octave transitions,
geometric aliasing, and camera-motion crawling can remain. No TAA, temporal seeds,
projection offsets, or extra changes to footprint handling are included.

For constant p and uniform spatial weights on a single surface, counting shared
per-pixel color samples gives an effective-support improvement of `2/(1+p)`:
about 1.82x at p=0.1, 1.33x at p=0.5, and 1x at certainty. This is different from
claiming twice as many distinct shaded surface samples.

## Planned implementation

- [x] Audit packaged noise provenance and available independent channels. Preserve
  stream A's thresholds. Hash surface identity and stream identity together;
  never use `id + stream`. Use independently generated noise channels where
  available, otherwise translated masks and measured limitations. Hierarchies
  require separate channel hierarchies for independent source fields.
- [x] Add validated construction/runtime winner-count settings and shader caching
  to `TransparencyPass`; expose through `MaterialRenderer` and example controls.
  Keep count independent of 1x/4x MSAA and preserve external opaque inputs.
- [x] Record `select A -> shade A -> clear/reuse selection targets -> select B ->
  shade B/mix`. Reuse integer payload and selection-depth textures; allocate one
  extra HDR scratch texture only for dual mode, per sample under MSAA.
- [x] Skip B when the current per-sample guide encodes exact `pNearest == 1`;
  reuse A for both streams during mixing. Required derivatives precede discard.
  The 1x1 path continues to omit the guide and disables this shortcut.
- [x] Audit all winner-alpha readers. Preserve the certainty shortcut based on
  guide probability, not mixed validity. Add stream/count diagnostics.
- [x] Update both scene and material recording paths, runtime switching, resizing,
  caller-owned target validation, and custom-shader compatibility rules.
- [x] Add correctness tests, stream diagnostics, performance comparisons, and
  documentation. Retain the single-winner behavior and resource footprint.

## Validation

- Scalar/RGB opacity, two differently colored layers, submission order, empty
  pixels, exact/near certainty, opaque occlusion, animation and material hooks.
- Counts 1/2 crossed with 1x/4x MSAA and 1x1/filtered reconstruction; switching,
  resize, external opaque resources, and native sample positions.
- At constant probability p: joint validity against `(1-p)^2`, `2p(1-p)`, `p^2`;
  compare low-frequency error power as well as union coverage for all noise modes.
- Same stable ID on closely separated layers: report rear-layer rate per stream
  to expose correlation that adding a stream does not resolve.
- Compare dual winners against single-winner 5x5 and 4x MSAA 3x3. Report GPU time
  where timestamp queries are available, error/support and pan/orbit/dolly behavior.
  Automated image/statistical checks cannot establish subjective motion quality.

## Progress and findings

- Initial inspection: the current payload is 128 bits; 4x MSAA packs it into two
  `rgba16uint` attachments. Reusing these avoids another 20 bytes per hardware
  sample for a second payload plus depth buffer.
- The existing shaded target stores `(color, 0/1 validity)`. Its bilateral
  denominator already uses alpha numerically. Certainty comes from a separately
  encoded guide probability; the 1x1 path uses expected acceptance.
- Packaged a second uint16 tile from the supplied archive's
  `Data/64_64/HDR_L_1.png`; A retains `HDR_L_0.png`. Two separate scalar textures
  allow existing noise helpers to stay unchanged instead of adding channel
  selection to every helper. Each stream lazily builds its own hierarchy.
  Custom single-tile overrides reuse that source with a hashed stream phase;
  callers can supply `blue_noise_second` for a separate source.
- Added static per-stream uniform buffers, avoiding writes between recording
  two draws (which would both observe the last queue upload).
- Added `winner_a`, `winner_b`, and `winner_count` diagnostics. The existing
  `winner` view shows the unnormalized average. B's diagnostic color is recovered
  from `2*mixed-A` and is subject to half-float rounding; it does not feed back
  into reconstruction. Count displays valid fraction, averaged over MSAA samples.
- B's certainty rejection happens after all material/noise derivatives and before
  noise evaluation. Geometry/vertex processing and derivative work still cost
  time; this optimization saves acceptance/payload work and deferred shading.
- Initial GPU checks compiled/executed both counts and both MSAA settings. Readback
  confirmed mixed validity includes 0.5, A/B marginal and joint probabilities,
  exact-certainty copying, colored-layer means, and submission-order invariance.
  Broader regressions and diagnostic measurements are recorded below.

## Measurements

Command: `python -m scenes.benchmarks.diagnose_multiple_winners --frames 20
--output audit.json`. [Saved results](../scenes/benchmarks/results/multiple_winners_2026-11-10.json).
NVIDIA GeForce RTX 5060 Ti, Vulkan, 256x256 synthetic scene,
20 measured frames after 5 warmups per configuration. GPU timestamps include the
complete material renderer; synchronized wall times are recorded separately in
the diagnostic output. These are measured-cost comparisons, **not equal-time
claims**, and should be repeated on the target scene/resolution.

| Winner draws | MSAA | Kernel | GPU median ms | Linear display RMSE |
| --- | --- | --- | --- | --- |
| 1 | 1x | 3x3 | 0.1033 | 0.04961 |
| 2 | 1x | 3x3 | 0.1321 | 0.03657 |
| 1 | 1x | 5x5 | 0.1181 | 0.03858 |
| 1 | 4x | 3x3 | 0.3052 | 0.02687 |

The scene is two constant-colored opacity-0.1 sheets. Error is measured against
the analytic alpha-over mean, after the renderer's tone map and before display
encoding. This checks variance/bias together, not texture detail or silhouettes.
Dual 3x3 improved this error relative to single 5x5 at a slightly higher measured
cost; this one microbenchmark cannot establish the best setting for the tree.

For UV blue CDF lerp at p=0.5, joint frequencies (neither, one, both) were
`(0.24902, 0.50195, 0.24902)` versus independent `(0.25, 0.5, 0.25)`. Anchored blue
gave `(0.25035, 0.50034, 0.24931)`. In the 0 < frequency <= 1/8 cycles/pixel band,
mixed power divided by the independent reference `(power_A+power_B)/4` was
1.041 and 0.9995 respectively. UV at p=0.1/0.9 gave 0.935/0.929. These are one
seed, planar footprint and scale; they do not prove independence in general.

The diagnostic also exposed deliberate/structural correlation: Bayer at p=0.5
was antithetic for the tested seed, with exactly one winner everywhere. The UV
hierarchy at p=0.9 had 6.25% empty samples versus the independent prediction of
1%. Separate source tiles therefore do **not** make all hierarchy levels
independent. Where a reference frequency band has zero power, its ratio is now
reported as null rather than a misleading zero. No Bayer/hierarchy algorithm
changes are included in this work.

Same-ID anchored layers at separation 0.1 coarse cell had rear-win rates 10.08%
in A and 10.22% in B, both far below the independent-layer target of 25%.
Coincident same-ID layers gave zero in both streams. At separation 0.25 cell,
rates were 25.18% and 25.08%. This confirms the close-layer limitation remains.

Approximate point -> pixel -> anchor pan/orbit/dolly sweeps at p=0.1 reduced
unfiltered acceptance-change RMS to 70–72% of single-stream values for both UV
CDF and anchored blue, consistent with reduced amplitude rather than removed
crawling. They do not measure reconstructed rendered motion or silhouette quality.

## Integration validation

- Focused stochastic/material/UI suite: **163 passed**.
- Expanded all-mode, near-certainty, external opaque and animated-material suite:
  **64 passed** (before adding the final stream-A preservation/empty/same-ID cases).
- Complete dual-winner test file: **40 passed**, including the final additional
  RGB-opacity tests for raw and normalized filtered reconstruction, bitwise A
  preservation, clearing, target validation, and the same-ID correlation case.
- Live tree example: **120 frames** completed while switching winner counts,
  MSAA, reconstruction, noise modes and footprint feature. This verifies command
  recording and live pipelines; it is not a subjective image-quality comparison.
- Full regression suite: **690 passed in 114.64 seconds**. The two final RGB tests
  were added afterward and passed in the focused 40-case run; no production shader
  changes were made after this full run.
- `git diff --check`: clean (only repository line-ending conversion warnings).

## Remaining evaluation and optional follow-ups

- Visually compare the tree under pan/orbit/dolly at useful resolutions and
  matched GPU budgets. The automated acceptance-motion sweep is only a proxy.
- Test more surface IDs, anisotropic footprints, and hierarchy levels before
  treating any measured stream pair as independent in general.
- Native MSAA integration and unchanged A thresholds are covered; the spectral
  diagnostic itself evaluates a regular single-sample lattice, not native 4x
  sample patterns. The timing comparison includes actual 4x MSAA rendering.
- No second-depth peel, per-frame reseeding, projection jitter, TAA, or changes to
  the existing Bayer/hierarchy algorithms were introduced.
