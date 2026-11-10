# Two-winner stochastic transparency

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
--output audit.json`. [Saved results](scenes/benchmarks/results/multiple_winners_2026-11-10.json).
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
