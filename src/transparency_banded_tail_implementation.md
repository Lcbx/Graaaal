# Banded tail allocation

Source: `C:/Users/User/Downloads/Banded Tail Allocation for Stochastic OIT.md`.

## Scope and status

Implemented the guide's Phase 0 offline reference and decision sweep in
`scenes/benchmarks/diagnose_banded_tail.py`. Initial work stopped at its gate:
the guide says, "If the gain is under 2×, stay on the nearest-wins tail and stop
here." The user's subsequent explicit request authorizes an experimental
exact-nearest/banded-tail/dual-Kawase version despite that result. That version
is now implemented as the opt-in `tail_allocation="banded"`; it does not change
the default stochastic technique.

This decision does not assert that banding never helps. A few five-layer cases
approach 2×, and a second random seed puts two cases slightly above it. The
broader faint-stack sweep does not support a robust 2× improvement. No particular
stack distribution or acceptable motion/bias tolerance was supplied.

## Experiment

Use the actual packaged 64x64 tile, remapped to distinct uniform rank midpoints
in (0,1), preserving its spatial layout. Both methods use complementary draws;
the nearest-wins reference gives each fragment a random integer toroidal shift.
The banded reference gives all fragments the same threshold and disjoint bands
with widths p/max(1,sum(p)). Both keep the same exact RGB transmission (scalar
opacity in this experiment).

Sweep 2-6 tail fragments at alpha 0.03/0.05/0.1/0.15/0.2 with 64 random color and
shift draws per setting, seeds 2026 and 2027. Each realization uses all 4096 tile
pixels. Compare the guide's wrapped 3x3 box and a Gaussian with sigma 0.9, matching
the default bilateral's spatial weights when all guides are constant. Bias,
spatial noise standard deviation, RGB RMSE and low-frequency error energy are
recorded separately. The reduction below is a standard-deviation ratio, not a
variance ratio.

Representative seed-2026 Gaussian results:

| Tail layers | Alpha | Noise reduction | Predicted banded bias |
| --- | --- | --- | --- |
| 2 | 0.05 | 1.26× | 0.00063 |
| 2 | 0.10 | 1.20× | 0.00278 |
| 4 | 0.05 | 1.42× | 0.00194 |
| 4 | 0.10 | 1.54× | 0.00823 |
| 5 | 0.10 | 1.96× | 0.01020 |
| 6 | 0.10 | 1.55× | 0.01401 |
| 6 | 0.20 | 1.88× | 0.04123 |

Bias is the trial-average largest absolute channel error in linear tail
radiance for random colors in [0,1], predicted analytically by p-weighting versus
sorted over. It is not normalized by color contrast and is not a displayed sRGB
error. It exists even with arbitrarily many noise samples.

For alpha <=0.1, seed 2026 gives 1.19-2.00× with the box (maximum rounds up),
1.16-1.96× with the Gaussian. None of its 25 settings per filter reaches 2×.
Seed 2027 gives 1.18-2.05× and 1.14-2.02× respectively, with only two of 25
settings per filter reaching 2×. For an entering alpha-0.1 layer and an existing
alpha-0.1 layer, 20% of pixels change the existing layer's acceptance state,
as predicted. Their two 10%-coverage masks are disjoint.

Raw local results (ignored benchmark artifacts):
[seed 2026](../scenes/benchmarks/results/banded_tail_gate.json),
[seed 2027](../scenes/benchmarks/results/banded_tail_gate_2027.json).

These are idealized static CPU simulations, not GPU or motion measurements.
They exclude CDF octave distortion, depth/alpha edges, MSAA, atomic scheduling,
force acceptance and realized-validity normalization. In particular, the
reference models the deterministic-denominator exact-nearest tail, not every
existing renderer configuration. They establish neither a GPU speedup nor a
universal quality ranking.

## Integration constraints identified before implementation

- WebGPU exposes no ordered fragment read-modify-write mechanism. The guide's
  atomic fallback can reorder bands between frames; it cannot promise the static
  allocation or motion stability of the ordered version. No claim of an ordered
  one-pass implementation is made.
- The current exact-nearest option is off by default. A tail option would have
  to require it or explicitly define behavior without a captured nearest layer.
  Tail exclusion must keep the existing packed instance/primitive identity,
  shared alpha evaluation, deformation and manual opaque-depth checks.
- Native 4x requires separate atomic sums and packed winner addresses for every
  hardware sample. Both total and running sums must use identical fixed-point
  quantization; the existing RGBA16F multiplicative transmittance cannot supply
  that integer total. Winner-buffer writes would need an explicit conversion
  into the existing deferred-shading payload layout or new buffer bindings.
- Allocation probability must become min(1,sum(p)) everywhere the reconstruction
  uses expected coverage, while RGB composition still uses 1-T_rgb. Current
  scalar transmittance, guide certainty, force acceptance, fill retries and
  diagnostics cannot silently change meaning. Fill/force and arbitrary draw
  counts need explicit restrictions or a separate derivation for banding.
- Noise is shared by all tail fragments at a sample. Existing per-surface
  anchoring cannot be reused unchanged. Screen anchoring is the guide's first
  baseline, with its admitted motion tradeoff; nearest anchoring is separate work.
- The guide's lossy denominator/opacity cutoffs remain unsuitable for an exact
  baseline, for the same reason they were omitted from exact-nearest capture.

## Validation completed

Eight analytic tests passed: interval tiling/marginals above and below total
opacity one, zero-width intervals, both complementary draws, p-weighted means
under submission reversal, the exact half-alpha complement case, and the
layer-entry mask replacement. Two complete simulation sweeps completed.
The Phase 0 work changed no renderer resources or defaults. The follow-up below
adds the explicitly requested experimental rendering path.

## Follow-up implementation

- Capture/shade exact nearest with the existing payload/material/deformation ABI.
- Clear per-sample integer counters and a two-slice-per-sample RGBA32UI payload
  array. Add quantized p to a total counter in the transmittance raster while
  keeping existing RGB/scalar multiplicative transmission. Both passes explicitly
  reject opaque occlusion and exact nearest by ID before atomic side effects.
- One allocation raster uses atomicAdd offsets and writes both complemented
  payloads. There is no tail winner depth write and no ordered-interlock claim.
  Both passes quantize identically to 1/65536, so running and total sums agree.
- Deferred shading shades the two payloads, reusing an identical full payload,
  and writes `(0.5*(CA+CB), min(1,S))`. Empty samples are explicitly overwritten.
- Full pair down/up filtering uses dual Kawase with bilinear clamped reads.
  Divide once, apply RGB tail opacity, then composite under the exact nearest
  contribution before per-sample HDR resolve/tone mapping. Pass count 0-8 sets
  down/up pairs; zero bypasses blur. Depth/alpha edge rejection is deliberately
  absent from the requested blur; cross-boundary color bleed is possible.
- Native MSAA reads the shared field at actual hardware sample positions.
  Additional per-sample phase shifts were removed after a test exposed sampling
  of only two tile parities at half-pixel grain. The four native positions now
  collectively cover the fine lattice. Per-sample subsampled tile histograms
  are not independently guaranteed uniform on a finite tile.
- Keep one eight-byte counter record per sample in storage, plus 32 bytes of
  payloads per sample in a storage texture. This avoids a large payload SSBO
  exceeding binding-size or storage-buffer-count limits with material resources.
  Do not allocate unused conventional winner payloads or selection depth.
- Per-pass material bind-group visibility now follows actual entry-point usage
  for groups 3+, so fragment atomics do not request vertex writable storage.
- Both examples expose a Tail allocation selector. Banded mode hides irrelevant
  controls, exposes Kawase pair count, and keeps only its available diagnostics.
  MaterialRenderer records the entire sequence and retains ECS/material ownership.

Defaults stay unchanged. Force/fill/independent pairing and low-level custom
shaders are explicitly excluded; per-material shader hooks and animation remain
supported. Surface anchoring, ordered allocation, faint-only routing, over-weighted
bands, content-aware blur and lossy culling are not implemented. The statistical
gate findings above still apply; no motion-quality or performance win is claimed.

### Follow-up validation

The GPU matrix checks native 1x/4x, zero/one/multiple blur pairs, odd and small
sizes, exact single-layer output, equal half-opacity complementary output,
integer total/running equality, per-fragment winner marginals, no empty slots
when S>=1, contrasting p-weighted tail colors under reversed submission, RGB
nearest composition, sparse-tail variance reduction, empty-frame clearing,
coplanar distinct IDs, external opaque inputs and fully occluded counters.
Animated forward/deferred parity and the real settings UI also passed.

The broad suite completed with 1,078 passing tests and six intentionally skipped
unsupported parameter combinations. It exposed eleven validation/mock regressions:
two setters inspected mode state before rejecting invalid input, and an old
keyboard stub lacked the new mode flag. These were fixed; the affected matrix,
including the final GPU and read-only-depth descriptor checks, then passed all
140 tests in 55.11 seconds. Read-only depth descriptors now omit load/store
operations, eliminating backend warnings from the new sampled-depth passes.

Both actual examples completed nine-frame window runs with the banded settings
panel, zero/three blur pairs, 1x-to-4x switching, window resize, stochastic/banded
switching, and material animation in the material example. These are smoke and
correctness checks, not a motion-quality ranking or GPU performance measurement.
`git diff --check` is clean. No dependencies were added.

## Proxy residual and tail cost follow-up

Before modifying the renderer, added `scenes/benchmarks/profile_banded_tail.py`
using the existing GpuProfiler pass-boundary queries. Captured the unchanged
banded method at 1280x720, native 1x/4x, one/two/three blur pairs, 24 measured
frames per configuration across three randomized rounds, five warmups per block.
[Initial timestamps](../scenes/benchmarks/results/banded_before_proxy.json)
include transmission, allocation, lighting, each Kawase pass, final resolve,
composition, clearing and nearest capture/lighting. No CPU timing substitutes
are used; quality readbacks are outside the timed interval.

### Implemented

- Banded proxy residual (`banded_proxy=True`). Fetch base color once
  in transmission, reuse its alpha for the shared opacity function, and add
  `Q = sum(p_quantized * base.rgb)` into an RGBA16F ONE/ONE attachment. Keep
  matching integer S for normalization. Albedo is a cheap proxy, not an exact
  representation of arbitrary lighting or sorted transparency.
- Shade `(n-Q/max(1,S), min(1,S))`. Retain signed RGBA16F throughout the blur;
  restore the full-resolution Q/S term before applying physical RGB opacity.
  No clamp touches residuals. Empty or unshaded pairs are zero; proxy restoration
  is independent of whether the blurred confidence is zero. No additional lossy
  faint-tail cutoff or nearest-opacity cull was introduced.
- Optional `banded_subset=True`. A shared integer hash chooses exactly one pixel
  per 2x2 block. Allocation skips other pixels after alpha derivatives; deferred
  lighting dispatches one thread per block and zeros its other pixels. Compact
  half-width/half-height payloads also quarter payload storage and clearing.
  S, Q and transmission stay full resolution. WGSL derivative uniformity prevents
  the guide's proposed discard before all alpha evaluation; that cost is retained.
- Optional `banded_tail_msaa=False`. At native 4x, transmission/nearest remain 4x;
  add a single-sample statistics raster, and allocate/light/filter a 1x tail.
  Its resolve stores mean color; composition multiplies it by each native sample's
  physical RGB opacity before applying the corresponding exact nearest layer.
- Mixed sampling checks opaque depth at native sample zero and excludes the union
  of all native nearest IDs. Initially using just nearest sample zero exposed a
  false front-layer contribution along internal triangle diagonals; the union
  fixes that without mesh topology assumptions. Genuine subpixel tail coverage
  can still differ or be absent from the 1x raster. This is an explicit quality
  approximation, not a multisample-equivalent estimator.
- Per-material opacity/deformation/lighting and frame uniforms keep the existing
  contracts. Added live setters and scene controls. The winner diagnostic shows
  signed residuals around neutral grey when proxy is enabled. Other techniques
  and their defaults are unchanged.

Both performance switches remain opt-in (`subset=False`, `tail_msaa=True`).
Proxy is enabled by default only within the optional banded method. The shader
does not assume cards, use temporal accumulation, add an exact second layer,
perform edge-aware filtering or add adaptive footprint gathers.

### Same-build measurements

[Full timestamp and quality matrix](../scenes/benchmarks/results/banded_proxy_final.json):
RTX 5060 Ti/Vulkan, same run protocol as above. Three pixel-aligned, staggered
emissive quads with red/green/blue albedo matching their constant lighting. Static
linear-HDR transparent contribution is compared against both sorted over and
the analytic p-weighted banded reference. Noise is spatial variation after
removing each column's mean in central flat rows. Halo is maximum absolute
column-mean error within +/-2^pairs pixels of the middle quad's vertical edges.
These are not independent temporal realizations or a general lighting benchmark.

Two blur pairs, native 4x. Stage medians are milliseconds; frame is GPU timestamp
span, so it includes gaps/transitions and is not the sum of stage medians.

| Variant | Frame | T | Separate S/Q | Allocation | Lighting | Blur | Resolve | Composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.990 | .192 | 0 | .496 | .200 | .298 | .090 | .190 |
| Proxy | 2.532 | .259 | 0 | .502 | .316 | .313 | .417 | .191 |
| Proxy + subset | 2.344 | .254 | 0 | .435 | .205 | .295 | .487 | .192 |
| Proxy + subset + 1x tail | 1.076 | .166 | .045 | .063 | .037 | .083 | .014 | .188 |
| Proxy + full-pixel 1x tail | 1.092 | .166 | .046 | .085 | .028 | .079 | .014 | .189 |

| Variant | Halo versus sorted | Halo versus banded | Noise std |
| --- | --- | --- | --- |
| Baseline | .068829 | .056219 | .000392 |
| Proxy | .020747 | .000884 | .000388 |
| Proxy + subset | .020893 | .001705 | .005687 |
| Proxy + subset + 1x tail | .029440 | .010761 | .013684 |
| Proxy + full-pixel 1x tail | .021362 | .001802 | .001065 |

Residual error versus sorted includes the pre-existing deterministic p-weighting
bias. Proxy's halo against the banded reference decreases with pair count rather
than growing: native 1x pairs 1/2/3 measure .00410/.00172/.00089, versus baseline
.05526/.05662/.05726. The proxy solves the measured contrast-driven blur halo,
not the sorted-over bias or all halos for materials with poor albedo proxies.

Full-pixel 1x tail with **three** pairs measures 1.092 ms, halo .000967 versus
banded and noise .000349: approximately the baseline 4x/two-pair noise at about
45% less frame time on this workload. This is the useful starting comparison.
Adding subset sampling saves little more here and increases noise much more
than the simple 2x standard-deviation estimate; thinning a blue field also
changes its filtered spectrum. Cheap constant lighting can even cost more in
the block dispatch due to its clearing/scatter overhead. Complex lighting and
different geometry need their own timings before choosing that option.

Tile early-out is deferred. The benchmark records 60% active pixels and 39.6%
empty 8x8 tiles. After the 1x-tail change, lighting plus blur/resolve are about
.13 ms, while exact nearest and composition are about .42 ms. These results do
not demonstrate that an extra occupancy hierarchy and correctly dilated blur
mask would pay off. No tile optimization was added on an unmeasured assumption.

### Validation

Added signed-residual, exact-proxy reconstruction, quarter-density confidence,
compact payload, thick-S normalization and mixed-attachment checks. Point-mode
proxy cancellation agrees with baseline under custom animated lighting even
when albedo differs from shaded color. Opaque-input/occlusion tests now cover
proxy and subset/single-sample variants. Live examples completed ten frames
each with proxy/subset/native-versus-single-tail toggles, 4x MSAA, resize, mode
switching, visible settings and animated material rendering. Full regression:
**1,113 passed, six skipped in 375.95 seconds**. The skips are unsupported
banded/winner parameter combinations from the existing animation matrix.
`git diff --check` is clean; no dependencies were added.

## Default BLEND path and deterministic MASK

The default allocation is now banded in TransparencyPass, MaterialRenderer and
the transparency example: one Kawase down/up pair, proxy residuals, full-pixel
lighting and 1x tail estimation. Scene MSAA still applies to physical transmission,
the exact nearest layer and opaque/MASK geometry. Removed the tail-MSAA UI toggle
as requested; the explicit native-tail API variant remains for comparison tools.

DefaultMaterialShader routes bindings by resolved alpha_mode: OPAQUE/MASK write
opaque depth, BLEND uses the transparency technique. The generic MaterialShader
keeps its transparent flag and exposes is_blended() for custom routing policies.
MaterialLibrary contains no alpha-mode policy; its material_values() view exposes
the actual last-packed shared values. This prevents routing from disagreeing with
GPU data when unchanged cached materials are intentionally retained. Instance
overrides still take precedence. ECS alpha-mode edits use the same update path.

The direct glTF scene likewise groups only mode 2 into transparent batches.
Removed sharpened MASK probabilities from the stochastic opacity function; MASK
fragments are excluded from every transparent raster pass. Opaque rendering and
its prepass use the existing fixed alpha-cutoff discard. Native sample-interpolated
UVs ensure cutoff/color/depth agree per hardware sample. Pixel-center Hi-Z remains
disabled at 4x. MASK ignores stochastic noise, grazing fade and tail reconstruction.

Classic estimator tests and diagnostics now choose stochastic allocation explicitly.
Updated obsolete MASK-winner expectations to require opaque cutoff instead. A
real-canopy check verifies MASK geometry belongs to opaque batches and its image
is independent of blue-noise offsets and tail blur count. New GPU cases cover
below/equal/above cutoff, MASK occluding BLEND, dynamic material/ECS routing,
cached shared-value consistency, and a texture boundary with two of four native
MSAA samples accepted. The low-level opaque shader remains usable for explicit
caller-selected forward rendering, including forward/deferred reference checks.

Fixed an unused correlated-noise option attempting to build a second hierarchy
from None in banded mode. Hierarchies are now lazy until stochastic allocation is
selected, including when switching from a banded configuration with a stored
correlated-noise choice. Both examples ran with the new defaults; six-frame
live runs exercised native 4x MSAA and window resize while retaining a 1x tail.

Validation: the broad run passed 1,118 tests and skipped six unsupported cases;
seven failures exposed the correlated-noise issue and obsolete implicit-mode or
MASK-winner expectations. After fixing those and preserving intentional low-level
forward-shading use, the affected mask/MSAA/winner/composition matrix passed all
115 tests in 50.55 seconds. Earlier focused scene/material matrices also passed
130, 109 and 27 tests respectively. `git diff --check` is clean.
