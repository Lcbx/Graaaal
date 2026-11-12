# Exact nearest transparent layer

## Design and scope

Source: `C:/Users/User/Downloads/Stochastic OIT with an Exact Nearest Layer.md`.
Add an opt-in `exact_nearest=False` option to TransparencyPass and its material
adapter. Existing defaults stay unchanged. Split each native hardware sample
into one deterministic nearest fragment and a stochastic tail. Composite
`a1*c1 + (1-a1)*(opaque*T_tail + (1-T_tail)*tail_color)` before tone mapping
and before the existing linear-HDR MSAA resolve.

### Corrections to the guide

- Select every visible fragment with nonzero evaluated opacity. An A_MIN above
  zero can move faint foreground fragments behind the selected layer, so it is
  not part of the exact baseline. RGB tail color mixing retains the existing
  max-channel stochastic approximation; only the nearest contribution is exact.
- Do not introduce the suggested 1/64 cull or 1/256 denominator clamp: both
  intentionally drop contributions. Early depth initialization, faint routing,
  reversed Z, extra additive routing and lossy cutoffs are later work.
- Keep standard-Z LESS and the current explicit per-sample opaque-depth check.
  Clearing a separate nearest depth to one avoids copying external/MSAA depth.
- There is no dedicated additive-material route today. Preserve the caller's
  existing external opaque inputs and composition responsibilities.

### Integration with current code

- Add one deterministic raster and one deferred shading dispatch. Reuse the
  existing packed visibility ABI, barycentrics, half-float LOD/front-face payload,
  material library, deformation hooks, frame fields and lighting.
- Retain the current transparent guide raster, excluding the nearest fragment.
  It describes the nearest *tail* fragment, so certainty shortcuts, force counts,
  fill retries and all existing reconstruction implementations remain coherent.
  This deliberately costs an extra raster compared with the guide's fused plan.
- Exclude nearest by `(packed instance + 1, local triangle)` in tail guide,
  transmittance/count and every stochastic selection pass, after derivatives.
  Do not compare depths or re-evaluate which fragment belongs to the nearest set.
- Save nearest RGB opacity and shade its visibility through a separate
  single-winner, unweighted shading variant. Tail A/B retain current seeds,
  complement-last behavior, payload reuse and optional fill/force semantics.
- For this option, initialize reconstruction pairs with `(winner.rgb, A_tail)`;
  repeated bilateral passes propagate these pairs. Keep original validity-ratio
  behavior when the option is disabled. Confidence filters already use A_tail.
- Native 4x uses the current two-RGBA16UI payload layout. Add nearest alpha and
  depth per sample; shaded nearest color is a four-layer storage texture.
  Allocate nothing extra with the option disabled. Extend caller-owned target
  validation and clear/rebind all optional resources on resize/runtime switches.
- Low-level custom shader sets must explicitly reject the option; material
  template hooks continue to work without copying transparency code.

## Implementation sequence

1. Reference tests and plan: scalar/RGB analytic composites for isolated layers
   and contrasting 2/4/8 stacks. Record the existing tail approximation separately.
2. Nearest capture/shading: matching opacity function, exact IDs/LOD/deformation,
   native 1x/4x, opaque occlusion, clear/resize and external inputs.
3. Tail exclusion and deterministic-denominator reconstruction: 1/2 averaged
   winners, complement/independent, force/fill and fragment/compute/pyramid/cascade.
4. Plumbing: constructor, live setter, MaterialRenderer adapter, both examples,
   shared settings, diagnostics and API documentation.
5. Validation: single-layer output is noise-mode/seed/kernel independent; scalar
   stacks match analytic means across realizations; p below 0.02 remains in correct
   order; distinct coplanar IDs survive exclusion; reversed submission, material
   animation, MSAA, mode switching, resizing, empty frames and external depth.
   Check raw nearest/tail targets and final output, then full regression and
   native GPU timings against the existing method. Do not assert scalar-exact
   colored-tail mixing or an empirical blue-noise independence bound.

## Status

Implemented the opt-in constructor/setter, deterministic visibility/alpha capture,
unweighted material shading, ID exclusion, and per-channel composition. The
material adapter and direct scene record the new stages; shared settings expose
the switch and nearest contribution/mask plus tail-transmittance diagnostics.
Native 1x/4x targets, explicit external target validation, resize and live shader
cache switching are wired. Feature-off recording still allocates no nearest
targets and records no additional passes.

The single-layer matrix passed at alpha 0.001/0.1/0.5/1 across point, fragment,
compute, pyramid and cascade paths, native 1x/4x. A sole 0.5-alpha tail with
complemented draws matches its analytic composite. Focused integration then
passed 152 tests including animation, external opaque inputs, scalar stack
statistics, force/fill and runtime switching. A dedicated initial-pair shader
variant ensures repeated bilateral filtering starts with A_tail rather than
observed winner validity; later stages keep their accumulated denominators.
RGB/coplanar/owned-target and repeated-denominator checks passed. Full regression:
**1,014 passed in 333.81 seconds**. Nine final checks additionally cover the
one-opaque-channel RGB case, explicit faint-nearest identity, and rejection of
low-level custom shaders before resource allocation. Live runs completed ten
material frames with F, exact toggles, animation, 4x MSAA and all four filters,
and eight direct-scene frames with nearest enabled and 1x/4x switching.

### Measurements

RTX 5060 Ti/Vulkan, actual tree, 1280x720, anchored blue, two complemented
winners, one 3x3 fragment filter, force off, 40 measured frames across four
randomized configuration rounds, five warmups per block. Separate feature-off
and feature-on runs (not an interleaved paired experiment):

| Sampling | Feature off | Exact nearest | Added GPU time |
| --- | --- | --- | --- |
| 1x | 0.580 ms | 0.655 ms | 0.076 ms |
| 4x | 1.341 ms | 1.684 ms | 0.343 ms |

[Off timings](../scenes/benchmarks/results/exact_nearest_off.json),
[on timings](../scenes/benchmarks/results/exact_nearest_on.json).
The baseline adds 36 bytes per pixel per sample for nearest depth, opacity,
visibility and shaded color (the 4x visibility pair has the same total size).
This is a measured quality/cost option; no performance win is claimed.

The same-build sorted scalar-stack sweeps use UV CDF with footprint balance 1,
four-pass a-trous, alpha 0.05/0.1/0.15, 2/4/8 contrasting layers, independent and
complemented winners, five support budgets. At budget four, complemented RMSE
is lower in all nine tested cases. At alpha 0.1, off/on RMSE is 0.02697/0.01425
(two layers), 0.04759/0.03993 (four), and 0.05779/0.04959 (eight). The tail's
changed acceptance can select a wider footprint, so these are equal-setting,
not equal-support comparisons. This is static image evidence, not a camera-motion
quality verdict.

[Off sweep](../scenes/benchmarks/results/exact_nearest_quality_off.json),
[on sweep](../scenes/benchmarks/results/exact_nearest_quality_on.json),
[nearest-on RMSE plot](../scenes/benchmarks/results/exact_nearest_quality_on.png).

The baseline implementation and integration sequence is complete. Approximate
alpha eligibility thresholds, lossy tail culling/clamps, depth-init early-Z,
fused nearest/guide raster, pure-absorber routing and faint-group accumulation
remain deliberately deferred. In particular, the exact pass does not fix the
pre-existing colored-tail or shared-ID stochastic correlation limitations.
