# Implementation work, measurements, and decisions

This is not an API guide. It preserves implementation
rationale, diagnostic methods, measured results, and outstanding engine work.
Update the relevant section as implementation work progresses; do not put
implementation history into the public usage guides.

For application-facing contracts and examples, see [README](README.md),
[ECS API](src/ECS_API.md), [materials](src/materials.md),
[transparency](src/stochastic_transparency.md), and
[benchmarks](scenes/benchmarks/README.md).

Other focused work/status records remain in
[material audit](src/materials_audit.md),
[multiple-winner transparency](src/transparency_multiple_winners.md), and
[transparency profiling/reconstruction](src/transparency_reconstruction.md).
These dated records are not necessarily descriptions of the current defaults.

## DrawBatches helper module layout

Internally, `draw_compiler` generates layouts with NumPy, `draw_layout` defines
their records, `draw_resources` manages registries/uploads, and `draw_culling`
records per-view compute work. Compilation can run without a GPU.


## ECS internal module layout

Import the public API from `numpyecs.ECS`. Schema conversion lives in `ecs_schema`,
input types/validation in `ecs_types`, deferred edits in `ecs_commands`, scheduling in
`ecs_schedule`, and inspection in `ecs_diagnostics`. Storage and joins remain
together in `ECS.py` because they share row lifetime and cache contracts.

## Scene cache implementation notes

Direct drawing retains packed instance data and refreshes it when relevant ECS
fields or membership change. Unchanged instances skip packing/upload; changed
interleaved data still uploads in full. Prototype panels likewise refresh from
the fields they display. Notify otherwise untracked object edits with `touch()`.


## Interactive scene implementation notes

Static primitives go directly into DrawBatches arrays; metadata
follows its packed instance order. It builds 512x512 material texture layers with repeat/trilinear
sampling and alpha-aware mipmaps. The tree's `Simple_foliage.001` material is
explicitly treated as MASK, with coverage-preserving mips and sharpened alpha
shared across guide, transmittance and selection. UV noise falls back to object
hashing for missing or degenerate UVs. The grain target defaults to 1 pixel at 1x;

## Material uniform layout snapshot

At the time of the source audit, `Frame` was 192 bytes and `Lighting` 16 bytes.

## Material binding internals and lifetime


Built-in material passes derive explicit layouts from reflected shader bindings.
The same group-layout objects own bounded GPU bind-group caches across compatible
pipelines and across 1x/4x MSAA. Cache keys include buffer handles and bound sizes,
so growth or resource replacement selects a new bind group; changing contents
preserves it. Custom shader sets keep the automatic-layout path.

| Group | Raster geometry | Deferred shading | Composition |
| --- | --- | --- | --- |
| 0 | `Frame` + `Lighting` | Same shared group | Same shared group |
| 1 | Instance/visibility indexing | Winner input/output textures | Color/depth/guide textures and `BilateralSettings` |
| 2 | Material tables, metadata, texture array and sampler | Material resources plus vertex/transform tables | Unused |
| 3 | Pass-specific depth/noise inputs | Unused | Unused |

Group 1 retains the existing DrawBatches convention. Pass groups can differ;
compatible low-index layout prefixes allow bindings to survive pipeline changes.
Bindings have explicit resource types and stage visibility; pass texture groups
include only resources needed by that pass. Default depth geometry is compiled
lazily when requested.

Buffer update cadence is separate from binding cadence: frame contents change
each frame while the group remains stable throughout the draws. Continue using
indexed storage for per-instance values. Do not overwrite one frame buffer for
multiple views recorded before the same submission; each such view needs its
own buffer or aligned range. `upload_changed()` on reflected uniform buffers
tracks raw bytes, including padding; ordinary `upload()` remains available.

The [dated material audit](src/materials_audit.md) records storage and upload
measurements for the six-sphere example. These describe that setup, not general
performance guarantees.

## ECS benchmark history

The historical result-file paths below are retained for provenance; their JSON
files are **not included in this archive**. The transcribed measurements remain
available here, but cannot be independently reproduced from the saved artifacts.

Results below are historical measurements; rerun the benchmark scripts
before using them to make performance decisions.

### Recorded ECS snapshots

These are historical observations, not measurements of the current checkout.
CPU suite JSON records interpreter, NumPy, platform, sampling configuration and
the selected ECS file's hash. Paired-scan JSON records baseline hash and samples;
upload JSON records environment and counters. The recorded CPU environment was
Windows 11, AMD64 Family 25 Model 97, Python 3.14.6, and NumPy 2.4.4,
with 100,000 entities, 11 samples, and 50 loops
for repeated actions. Small differences across runs are not evidence of a win.

| Series | Artifacts |
| --- | --- |
| Original storage optimizations | `scenes/benchmarks/results/phase6/baseline.json`, `scenes/benchmarks/results/phase6/optimized.json` |
| 2026-11-04 validation/API review | `scenes/benchmarks/results/review_2026-11-04/baseline.json`, `scenes/benchmarks/results/review_2026-11-04/current.json`, `scenes/benchmarks/results/review_2026-11-04/query_pairs.json`, `scenes/benchmarks/results/review_2026-11-04/uploads.json` |

The review baseline was exported from revision
`72b28d57baa79e7ee9c294570d562aa2429bb38d`. The two baseline files belong to
different comparisons. Match recorded sources and dependency versions before
comparing results. The current CPU script hashes `ECS.py` alone, so its hash does
not identify changes in imported `ecs_*` helpers. Later package/API changes are
not represented by these runs.

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
packing time and 2.709 ms CPU submission time for 3,843,840 instance bytes;
the packing counter includes an initialization call (121 calls for 120 frames).
These counters do not measure GPU completion.

### Array access

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

## Stochastic transparency: implementation and validation record

Some referenced JSON result files are **not included in this archive**.
The tables and descriptions below are preserved from the original notes.

Retained from `src/stochastic_transparency.md` on 2026-11-11.
This is an implementation record, not current API documentation.

### Pass order

`blue_noise` is optional. The default is a lazily uploaded, packaged 64x64 tile,
shared among active users of the current GPU device. An explicit texture overrides
it. The default asset and CC0 attribution are bundled under `data/blue_noise`.

| Stage | API | Result |
| --- | --- | --- |
| Opaque color/depth | `opaque_pass()` or another renderer | HDR opaque color and depth |
| Transparent depth guide | `guide_pass()` | Nearest nonzero-opacity depth, acceptance probability and reciprocal-depth gradient, independent of stochastic acceptance |
| Transmittance | `transmittance_pass()` | RGB/scalar transmittance, initialized to one |
| Winner selection | `selection_pass()` | Nearest accepted fragment and its payload, using separate writable depth |
| Deferred shading | `shade_winner()` | Straight HDR winner color; empty pixels become zero |
| Reconstruction/composition | `compose()` | Filtered transparency over the opaque color, tone-mapped for output |

The guide pass rejects opaque occlusion and fully transparent texels while
retaining low nonzero opacity. Transmittance tests opaque depth without writing
it. Selection clears its own depth and explicitly rejects fragments behind
opaque geometry. Transparent geometry is rasterized three times: guide,
transmittance, and selection. The 1x1 reconstruction mode can skip the guide.
The technique requires no transparent sorting.

Transparent `RenderShader` entries are `transmittance` and `main` (winner
selection); they have no `prepass` entry, so an opaque depth-prepass draw skips
them. The culler's internal prepass visibility buffer remains a list of frustum
candidates, independent of which named rendering pass is drawn.

### Winner draws and repeated reconstruction

Reconstruction can independently use `reconstruction="compute"`. Both paths share
the same bilateral estimator; compute uses an intermediate and keeps composition,
MSAA resolve and presentation at the end. 1x1 bypasses compute. See the
[profiling and reconstruction report](src/transparency_reconstruction.md) for API,
target formats, exactness limits, native pass profiling and measured results.

`reconstruction_passes=N` stacks the chosen kernel N times. The first N-1 stages
filter weighted `(C*w,w)` pairs, averaging all four channels with deterministic
bilateral weights; only final composition divides RGB by validity and applies
RGB acceptance. Opacity, background composition, MSAA resolve and tone mapping
are not repeated. Original raw winners remain available for diagnostics and
certain/forced bypasses. Every stage uses the original depth/transmittance guide.

Both fragment and compute paths support this. Fragment stages render each MSAA
layer separately; compute stages dispatch per layer. Two passes allocate one
extra `rgba16float` target, three or more allocate two reusable targets, regardless
of the count. They are sampled/storage/render-attachment textures, array-layered
at MSAA. Caller-provided targets must supply `reconstruction_pair` and, above two
passes, `reconstruction_pair_scratch`. Empty regions are written as zero because
later stages may sample them. Half-float conversion error accumulates with passes.
The filter-support diagnostic describes the final local kernel, not total support.

Repeated 3x3 filtering is not equivalent to one larger square kernel. N passes
can reach a `(2*N+1)`-wide footprint, but the weights form a repeated diffusion
and retain bilateral boundaries/bypasses. A 1x1 kernel still skips every stage.

`winner_mix="average"` remains the default. Experimental `winner_mix="fill"`
retries only samples with no winner so far. It accepts any positive uint32 count,
default noise sources, and `noise_mode="object_hash"` or `"anchored_blue"`.
The other modes failed a million-sample five-sigma layer-frequency gate and are
rejected for fill. Empirical passing results do not prove independence for every
asset or footprint; same-ID overlap correlation remains. See the
`scenes/benchmarks/results/winner_fill_gate_2026-11-11.json`.

Use `set_winner_mix("fill"|"average")` on the pass or material renderer. Selecting
one winner restores average mixing. Above two winners, select fill before raising
the count; returning to average requires reducing the count first. The example
exposes editable **Winner draws** and **Reconstruction passes** integers. It
reduces winner count to two and returns to average when cycling into an
unsupported noise mode.
Custom noise overrides are not eligible for fill without a separate audit.

Optional `early_b_rejection=True` uses A's per-sample stencil coverage to reject
already-filled samples in B. Request `depth32float-stencil8` when creating the
device. Only the selection depth changes to that format; float32 depth precision
and opaque inputs are preserved. It requires multi-winner fill and switches off
when selecting average mixing or one winner. The original shader rejection
remains as well. Drivers may defer stencil testing around fragment discard;
measured timings did not establish a benefit, so this experiment is off by
default and exposed through the pass/material renderer API and benchmark.

Fill repeats selection/shading N times. Every stochastic winner stores straight
color and validity scaled by `1/(1+T_scalar+...+T_scalar**(N-1))`, evaluated with
a logarithmic geometric-sum algorithm without subtraction near transmission one.
Each retry samples the current per-sample shaded target and discards at nonempty
samples; its compute stage never overwrites empty payloads. The correction uses
the total requested budget for every attempt so each neighbour retains
expected weight `A_scalar`; a raw filled 0/1 validity would upweight low-coverage
neighbours. Shading A overwrites all pixels, including zeros for empty samples,
once per frame, so no additional color-clear pass is required. Retries read no storage
texture: the shaded target is sampled in raster and written only in compute.

Fill allocates no `winner_first` HDR scratch. Instead, `winner_source` is an
`r32uint` storage/sampled texture (one layer per MSAA sample) recording zero for
empty and the winning attempt's one-based index. This costs 4 bytes per hardware sample versus the
averaging scratch's 8. Caller-supplied `TransparencyTargets` must include it in
fill mode. `winner_a`/`winner_b` isolate the stream that supplied the retained
sample from attempts one/two; later attempts remain visible in `winner`.
`winner_count` displays valid/empty, rather than the weighted alpha.

Attempts use immutable per-stream uniforms and hashed stream seeds; attempt A
retains its original thresholds. A uses the first packaged tile, later attempts
use the second tile with their own hashed seed. Extra tile shifts do not prove
joint independence: the formula assumes each attempt has the correct conditional
layer distribution. Arbitrary counts remain experimental and same-ID overlaps
can remain correlated. Target memory is independent of the winner count; uniform
and draw/binding storage scale with the count. Each retry still runs geometry
and a shading dispatch. The optional stencil gate retains A's coverage across
all retries; shader checks reject winners found by subsequent attempts.

#### Exact single-contributor acceptance

`force_single_contributor=True` enables a separate optional optimization, also
available with one winner. It is off by default and configurable through
`set_force_single_contributor(bool)` on the pass/material renderer, or the
example's **Force single contributor** checkbox.

The transmittance draw additionally accumulates a per-sample `r16float`
`contributor_count`, with one additive increment for every nonzero-opacity
fragment that passes opaque occlusion and sidedness tests. Only count **exactly
one**, with positive stored scalar acceptance, can force selection. A fixed
absolute probability tolerance is not used: two p=0.001 layers remain stochastic,
and a tiny secondary layer is not silently suppressed. Large counts may saturate
half-float addition but cannot turn back into one. This attachment costs 2 bytes
per hardware sample; no additional geometry pass or readback is required.

Selection records forcing in `payload.w & 0x00010000u`, a spare mantissa bit of the
face-orientation half. Mask it before decoding LOD/face:

```wgsl
let forced = (payload.w & 0x00010000u) != 0u;
let lod_face = unpack2x16float(payload.w & 0xfffeffffu);
```

The payload remains 128 bits, including its 4x MSAA split. Forced shading stores
`(A_scalar*C, A_scalar)`; it overrides the fill correction. Average B skips forced
samples and copies A. Composition identifies forced results with the same exact
count and a positive winner weight, bypassing reconstruction with `A_rgb*C`.
It never infers forcing from `abs(winner.a-A_scalar)`, which would misclassify a
half-valid average at A_scalar=0.5. The 1x1 estimator's formula is unchanged.

A single grazing-faded surface and fractional MASK edge are eligible at their
reduced coverage. Fully transparent fragments are not. No probability floor was
introduced; this does not recover opacity already rounded away by the existing
half-float transmittance target. Compose epsilon 1e-6 is below the smallest
positive acceptance represented by that target near transmission one (~4.88e-4).
The **forced** debug view shows the exact-single bypass mask. Weighted debug
colors divide RGB by weight; count views do not interpret weight as validity.

Both improvements use default geometry/shading/compose shaders and material
hooks. Low-level custom shader overrides are rejected; custom payload readers
must respect the reserved flag and weighted `(C*w,w)` representation if adopted.

#### Original averaging mode

Construct with `winner_count=2`, or call `set_winner_count(2)` (and
`MaterialRenderer.set_winner_count(2)` when using its registered passes). The
example exposes **Winner draws**. Default is 2. Each stream chooses its nearest
accepted fragment with replacement: both may choose the same surface. It does
not average the two nearest accepted layers, which would change the estimator.

Record the guide and transmittance once, then:

```python
for stream in range(oit.winner_count):
    with oit.selection_pass(commands, stream=stream) as render:
        draws.draw(render, visibility, oit.selection_name(stream))
    oit.shade_winner(commands, geometry, stream=stream)
oit.compose(commands, output)
```

Selection clears/reuses its depth and integer payload targets between streams.
A shades into `winner_first`, B shades and mixes into `winner_color`. The scratch
texture exists only in two-winner mode: `rgba16float`, storage + texture binding,
single-sample, with `sample_count` layers (a 2D array at 4x). Caller-owned
`TransparencyTargets` must supply this additional field in dual mode. The extra
storage is 8 bytes per hardware sample, excluding driver alignment.

Each shaded stream is `(straight_color * valid, valid)`. The stored average has
alpha **0, 0.5, or 1**, a valid-stream fraction rather than opacity or certainty.
Reconstruction pools its RGB numerator and alpha denominator; it does not average
two separately normalized filtered results. The 1x1 path retains division by
expected scalar acceptance. Existing RGB-opacity approximation and ratio-estimator
bias remain. Exact certainty is still encoded by guide probability, never inferred
from winner validity.

With reconstruction enabled, B skips samples whose guide has exact probability
one, after material/noise derivatives and before noise lookup. Mixing copies A
there. At 1x1 the guide remains skipped and B evaluates normally. This saves
acceptance/payload and shading work at certain samples; it does not skip vertex
processing, draw submission, or derivative work.

A's noise is unchanged. B hashes the stream identity with the surface/sample
seed, keeping footprint and anchoring parameters unchanged. Defaults use separate
Peters `HDR_L_0`/`HDR_L_1` tiles and lazily generated separate hierarchies. For a
custom `blue_noise`, provide `blue_noise_second` for a separate source; otherwise
B uses hashed shifts of the same tile. Separate masks do not guarantee statistical
independence after projection or hierarchical resampling. Bayer ignores the tile
and can correlate strongly. Same-ID nearby surfaces can still share thresholds
within each stream.

`winner` displays the stored average. Additional diagnostics are `winner_a`,
`winner_b`, and `winner_count` (valid fraction: black/gray/white for 0/1/2 winners,
averaged across MSAA samples). B's diagnostic is derived from `2*mixed-A`, so small
half-float rounding differences are possible; reconstruction reads the average
directly. In single mode both stream views show A.

The default shader set supports both counts, including plain-WGSL material hooks
and deformation. Low-level custom geometry/shading/compose shader overrides remain
single-winner; dual construction/runtime switching rejects those overrides.
Runtime switches preserve settings and registered geometry, reallocate the targets,
and retain caller-supplied opaque color/depth.

Two winners reduce missing samples and layer-selection variance. At the same
raster position they add no texture detail on a single surface, do not supersample
silhouettes, and do not remove crawling or within-stream layer correlation. See
the [implementation plan and measured findings](src/transparency_multiple_winners.md)
and run `python -m scenes.benchmarks.diagnose_multiple_winners --output audit.json`
for joint validity, low-frequency spectra, same-ID overlap, camera sweeps and
synthetic GPU timing comparisons.

### Geometry and caller-owned targets

The default geometry uses `VERTEX_DTYPE` and packed `MESH_INSTANCE_DTYPE`
transforms. Raster and deferred vertices share local coordinates; both apply
translation, rotation and scale, with inverse-scale normal transformation.
`metadata` contains one `vec4u` per packed instance: triangle offset, material
index, stable surface ID, and a reserved word. The dictionary-material adapter
uses that fourth word as its instance-parameter slot offset. All transparent draws
sharing a selection target must share metadata/vertex addressing; pass the matching
`GeometryPasses` to `shade_winner()`. Geometry from another technique owner is rejected.

`material_source` can replace the direct-material
`shade(material, uv, position, normal, lod) -> vec3f` implementation. A matched
`geometry_shader`/`shading_shader` pair can customize deformation under the winner
ABI. `create_geometry_passes()` accepts extra group-2 resources through
`geometry_resources`/`shading_resources`; additional binding groups start at 4 for
geometry and 3 for shading, subject to device limits. Animation/skinning data
preparation remains application-owned. Dictionary materials have higher-level
[vertex hooks](src/materials.md#time-and-vertex-displacement).

At 1x, `TransparencyTargets` requires these formats, each with one layer:

| Targets | Format and usage |
| --- | --- |
| `main_color`, `transparent_guide`, `transmittance` | `rgba16float`, render attachment + texture binding |
| `depth`, `transparent_depth`, `selection_depth` | `depth32float`, render attachment + texture binding |
| `winner` | `rgba32uint`, render attachment + texture binding |
| `winner_color` | `rgba16float`, storage binding + texture binding |

All targets must match `resize(size)` and the configured sample count, except the
always-single-sample `winner_color`. The next section describes 4x formats/layers.
Alternatively, supply only existing opaque inputs through `set_opaque_inputs()`
and let the pass allocate its other targets. Opaque inputs must be sampleable and
match size/sample count; drawing into them also requires render-attachment usage.
Set `target_format` to match the output `ColorAttachment` passed to `compose()`.

### Optional MSAA

`TransparencyPass(..., sample_count=4)` enables 4x MSAA; the default is 1.
`set_sample_count(1 or 4)` switches the default shaders and reallocates targets
while preserving registered DrawBatches geometry, camera and live settings.
The example exposes an **MSAA (M)** dropdown and the **M** shortcut.

Opaque, guide, transmittance and selection passes retain four samples.
Surface attributes interpolate at sample positions, and selection uses a static
per-sample noise seed, except `anchored_blue`, which reads a single field at the
native sample positions. Deferred shading processes each sample; composition filters
each sample separately and averages linear HDR before tone mapping. Integer winner
IDs are never resolved or averaged. There is no temporal accumulation.

Noise width follows sample density: `effective_noise_scale = noise_scale /
sqrt(sample_count)`. A base width of 1 targets approximately 1 px at 1x and
0.5 px at 4x. `noise_scale` keeps the user's chosen 1x width, so switching back
restores it, including edits made with MSAA enabled. The example displays both
the base setting and effective target. Each sample already has a different seed;
smaller cells change the spatial grain as well.

The backend cannot multisample `rgba32uint`. At 4x, selection instead splits the
same 128-bit payload into two `rgba16uint` attachments (`winner`, `winner_extra`).
Deferred `winner_color` is a single-sample `rgba16float` 2D array with four layers.
Caller-owned `TransparencyTargets` must supply those formats and layers; the other
attachments use sample count 4. At 1x the original formats and one-layer target
remain in use. External opaque color/depth must match the selected sample count.

The scene runs its single-sample Hi-Z prepass (`depth_shader`) only with MSAA
off. At 4x it skips the prepass, pyramid generation and Hi-Z culling, drawing
both lists from frustum visibility instead. Pixel-center depth cannot safely
occlude an object that remains visible at an uncovered MSAA sample.
Custom shader sets must implement the selected sample ABI at construction time;
runtime switching supports the default shader set, including `material_source`.

4x costs additional memory, raster/deferred shading and reconstruction work. It
smooths geometric silhouettes and adds stochastic samples, but does not eliminate
all movement in surface noise or octave transitions. Kernel size remains a separate
setting: **1x1** bypasses reconstruction while MSAA can still be enabled.

### Acceptance and winner data

For fragment opacity `alpha_rgb`, scalar acceptance probability is
`p = max(alpha_rgb)`. Multiplicative blending accumulates:

```text
T_rgb    *= 1 - alpha_rgb
T_scalar *= 1 - p
```

Selection compares `p` with a surface-anchored threshold. Fragments with
`u >= p` are discarded; hardware depth testing/writes retain the nearest accepted
fragment. No mode uses history, per-frame reseeding, cards or triangle-based noise.
Only selection changes: the guide and transmittance passes are independent of
sampling mode and grain size.

`noise_mode` / `set_noise_mode(name)` selects:

| Mode | Threshold construction |
| --- | --- |
| `object_hash` | Existing object-space 3D lattice, octave lerp and uniform CDF correction |
| `uv_blue_lerp` | Point-sampled blue tile in two UV octaves, lerp and uniform CDF correction |
| `uv_blue_pick` | Same UV octaves, selected by a separate surface-anchored hash channel |
| `uv_bayer` | Procedural 64x64 Bayer ranks, with the stochastic octave pick; diagnostic |
| `uv_blue_correlated` | GPU-generated parent/child rank hierarchy, shared spatial origin, static octave pick |
| `triplanar_blue` | Object-space XY/XZ/YZ tile projections, smooth normal weights and CDF-corrected blends |
| `triplanar_correlated` | Triplanar projection using the correlated hierarchy |
| `anchored_blue` | Pixel-lattice blue tile with first-order rest-cell anchors and finer rest-cell shifts; default |

UV sampling uses raw UV derivatives, before any discard or material-LOD clamp:

```text
fmax = max(length(dpdx(uv)), length(dpdy(uv)))
fmin = abs(det(dpdx(uv), dpdy(uv))) / max(fmax, epsilon)
f = fmax^(1 - noise_footprint_balance) * fmin^noise_footprint_balance
f *= 64 * effective_noise_scale
s = log2(f); k = floor(s); t = fract(s)
uA = tile[(floor(uv * 64 / 2^k)     + shift(stableID, k))     mod N]
uB = tile[(floor(uv * 64 / 2^(k+1)) + shift(stableID, k + 1)) mod N]
```

Octaves may be negative; numerical guards bound them to [-24,24]. Sampling uses
integer `textureLoad`, never bilinear noise filtering. Signed wrapping handles
negative UVs. Per-instance/per-octave translations leave a single octave's
threshold ranks and Fourier magnitudes unchanged; GPU tests verify both against
the original [blue-noise tile](src/data/blue_noise/README.md).
Mirroring and nonuniform UV density change the footprint, not the probability.
Grazing-angle anisotropy still distorts the spatial spectrum.

The size is a target, not a guarantee of square screen-space cells. A scalar
footprint cannot compensate separately for both axes. UV derivatives `(1/64, 0)`
and `(0, 4/64)`, for example, select approximately 4x1 screen-pixel cells at an
integer octave and a 1-pixel target. Octave blending and triplanar mixing further
complicate a single cell width. MSAA scaling does not remove this anisotropy.

#### UV footprint compromise experiment

`noise_footprint_balance` (constructor argument or
`set_noise_footprint_balance(a)`) interpolates in log space between the existing
maximum-column-length derivative estimate and `abs(det J)/fmax`. It is not a
singular-value decomposition for sheared Jacobians. The default **0** preserves
the prior sizing exactly, **0.5** uses the geometric mean/area footprint, and
**1** uses the minimum estimate. The example exposes **UV footprint balance**
beside the grain-size control; it is disabled for non-UV modes.

The arithmetic is now an optional shader feature, **disabled by default**.
Construct with `noise_footprint_balancing=True`, or use
`set_noise_footprint_balancing(True)` to compile/cache an enabled variant. The
example has an **Enable UV footprint balancing** checkbox. Disabled variants omit
the compromise calculations even if the stored balance is nonzero. Runtime
feature switching preserves targets and settings; custom shader sets must manage
their own variants. MaterialRenderer exposes the same setter to refresh its draw
registrations. Raw shader users enable `features=("NOISE_FOOTPRINT_BALANCE",)`.

The compromise applies to UV CDF lerp, stochastic pick, Bayer, and correlated
hierarchy modes. Object hash, triplanar sampling, and the degenerate-UV object
fallback retain their current footprints. Frame `noise_footprint_balance` holds
the value. Only selection reads it; opacity, guide data and filtering are unchanged.
The existing `surface_noise_threshold(...)` WGSL entry remains a zero-balance
wrapper; `surface_noise_threshold_balanced(..., balance)` exposes the experiment.

This changes scalar grain size, not cell anisotropy. For a 4:1 UV footprint,
targets move from approximately 4x1 px at 0 to 2x0.5 px at 0.5 and 1x0.25 px at 1.
The compressed axis becomes subpixel, so finer grain can alias or crawl more.
Use this to assess whether broad tilted grain is the dominant visible artifact.

`scenes/benchmarks/diagnose_noise_tilt.py` samples the production threshold function
on an analytic orthographic plane at 0, 60, 75 and 83 degrees. It outputs the full
width at half maximum of the mean x/y autocorrelation of the threshold image,
in screen pixels. This is a screen-space measurement, not a fixed-material-point
motion metric. Oversampling resolves some subpixel structure; `resolution_limited`
flags widths below two diagnostic samples. `--samples 4` applies the smaller cell
target, without simulating the four hardware sample locations or sample seeds.
No grazing fade, shading, reconstruction or temporal accumulation is involved.

Default UV CDF lerp measurements, 1 px target, 128x128 screen pixels, 8x diagnostic
oversampling, fixed surface ID 17 (values below are autocorrelation FWHM, not exact
lattice-cell dimensions):

| Tilt | Balance 0: x / y | Balance 0.5: x / y | Balance 1: x / y |
| --- | --- | --- | --- |
| 0 degrees | 0.83 / 0.84 | 0.83 / 0.84 | 0.83 / 0.84 |
| 60 degrees | 1.66 / 0.84 | 1.10 / 0.56 | 0.83 / 0.42 |
| 75 degrees | 3.36 / 0.88 | 1.67 / 0.43 | 0.83 / 0.22* |
| 83 degrees | 6.81 / 0.83 | 2.26 / 0.28 | 0.83 / 0.11* |

`*` Compressed-axis width is resolution-limited at this diagnostic sampling rate.
The pixel-anchored sibling below uses separate tilt and motion/layer diagnostics;
these UV measurements do not establish its temporal stability.

#### Pixel-lattice anchored blue

Select `noise_mode="anchored_blue"` or call `set_noise_mode("anchored_blue")`.
It requires no UVs, cards, triangle IDs, centroids, history or additional vertex
data. The existing interpolated rest/local position is the reference, including
when a material vertex hook deforms the surface.

With `D = [dpdx(local), dpdy(local)]`, its area footprint is
`sqrt(length(cross(D.x,D.y)))`. A power-of-two rest-space grid targets
`noise_anchor_pixels` (default 24). Both adjacent grid octaves are sampled and
CDF-blended. For each octave:

```text
center = (floor(local / cell_width) + 0.5) * cell_width
anchor = round(pixel + pinv(D) * (center - local))
shift_key = hash(floor(local * subdivisions / cell_width), stableID, octave, side)
u = tile[(floor((pixel - anchor) / effective_noise_scale) + shift(shift_key)) mod N]
```

The pseudoinverse projects through the measured rest-space derivatives of the
deformed surface, rather than projecting a rest center with an instance matrix.
Near-singular or numerically extreme footprints fall back to object hashing.
Anchor grids change with projected area; grain itself stays on a screen-space
lattice with equal x/y cell width. Both anchor octaves read this same grain size.
The complete field can still have seams and altered correlations at shift cells,
anchor rounding and octave transitions; it does not guarantee a blue spectrum.

`set_noise_anchor(pixels=24, subdivisions=4)` configures the coarse target and
number of finer shift cells per coarse-cell axis. The constructor accepts
`noise_anchor_pixels` and `noise_anchor_subdivisions`. The example displays both
controls when this mode is selected. More subdivisions reduce same-ID layer
correlation but add more shift seams. Dense self-overlap can still require
distinct stable IDs or object-hash sampling.

At 4x, the GPU supplies fractional fragment positions (measured on the test GPU:
`(.375,.125),(.875,.375),(.125,.625),(.625,.875)`). Selection uses these positions
and the existing half-pixel grain target. It does not add per-sample shifts, which
would replace the shared spatial field. GPU regressions compare actual production
winner masks against evaluation at the device's native positions for both 1x and
4x. Other modes retain their per-sample seed logic.

Tilt sweep, same settings as the UV diagnostic, autocorrelation FWHM in pixels:

| Tilt | Anchored blue x / y |
| --- | --- |
| 0 degrees | 0.77 / 0.79 |
| 60 degrees | 0.78 / 0.70 |
| 75 degrees | 0.81 / 0.72 |
| 83 degrees | 0.81 / 0.68 |

Run `python scenes/benchmarks/diagnose_noise_tilt.py --mode anchored_blue` to
repeat the sweep. Native cell width is isotropic; these correlation widths also
include the effect of fine-cell seams and octave blending.

`scenes/benchmarks/diagnose_anchored_noise.py` audits same-ID layers at separations
from 0.1 to 2 coarse cells, with a coincident control. At alpha 0.5, the rear win
rate `P(front rejected and rear accepted)` should be 25% for independent layers.
Its 32,768-probe results for shift subdivisions were:

| Separation | 1 | 4 (default) | 8 | 16 |
| --- | ---: | ---: | ---: | ---: |
| 0 cells | 0% | 0% | 0% | 0% |
| 0.1 cells | 2.40% | 10.08% | 19.79% | 25.31% |
| 0.25 cells | 6.15% | 25.18% | 25.23% | 25.10% |
| 0.5 cells | 12.40% | 25.30% | 24.83% | 25.00% |
| 1 cell | 24.88% | 24.76% | 25.02% | 25.11% |
| 2 cells | 25.09% | 25.19% | 25.20% | 25.10% |

This is not a guarantee for every mesh or cell phase. Surfaces in the same fine
cell can have identical thresholds; the coarser anchor alone cannot separate
them. The default is a moderate seam-density compromise, not full independence.

The same audit projects 4,096 fixed material points on a curved patch to a
512-pixel perspective view, quantizes to pixel centers, evaluates the first-order
rest position at each pixel, then invokes the production threshold function.
It uses 24 poses and therefore includes the pixel-lattice path omitted by the
older fixed-material-point diagnostic. Mean acceptance changes per pose:

| Motion | Object hash | UV CDF lerp | Anchored blue |
| --- | ---: | ---: | ---: |
| Pan, 0.6 units | 19.18% | 21.87% | 22.20% |
| Orbit, 0.6 radians | 25.95% | 29.08% | 33.68% |
| Dolly, distance 2.5 to 4 | 30.31% | 33.47% | 37.29% |
| Procedural wind | 21.53% | 23.84% | 24.86% |

These are nearest-pixel, first-order diagnostic samples, not full rasterized
visibility or perceptual scores, and are not comparable with the earlier
non-quantized table. Anchored blue's reduced tilt stretch comes with more
orbit/dolly reshuffling here. Anchored blue is now the default; UV CDF lerp
remains available for comparison. Neither diagnostic uses TAA or frame history.

#### Cross-object decorrelation audit

Every mode receives the stable renderable ID and face side in its seed; modes
other than anchored blue also include the sample index at 4x. World-space
separation is not part of that seed, so coincident
objects get different seeds too. IDs distinguish renderables, not every
self-overlapping surface within a mesh; repeated UVs within a primitive can
still share thresholds. The scene allocates IDs per loaded glTF primitive.

Distinct seeds do not guarantee independent threshold sets. Object-cell hashing
provides the strongest decorrelation here. UV blue and triplanar blue translate
a finite tile and retain some cross-correlation. Bayer shifts can strongly
correlate accept/reject masks. Both hierarchy modes can also strongly correlate
when a small coarse rank grid dominates; root jitter does not remove that
grid's threshold-set correlation.

A GPU audit used 32,768 coincident local-coordinate samples from
`default_rng(701).uniform(-30, 30, (32768, 3))`, UVs equal to local XY, IDs
`1,2,3,4,5,17,23,42`, four integer octaves `-3,0,2,3`, and a face-on projection.
Object and UV derivatives were axis-aligned with magnitude `2^octave/64`, the
normal was `(0,0,1)`, and the target was 1 pixel.
Independent selection of two opacity-0.5 layers would cover 75%. The most deviant
observed pair/octave for each mode produced:

| Mode | Observed union coverage |
| --- | ---: |
| Object hash | 74.33% |
| UV blue CDF lerp | 76.47% |
| UV blue pick | 76.47% |
| UV Bayer | 49.47% |
| UV correlated hierarchy | 83.99% |
| Triplanar blue | 73.08% |
| Triplanar correlated hierarchy | 64.51% |

These characterize the current implementation, not bounds for all IDs or
projections. Bayer and hierarchy modes cannot currently be described as
independently decorrelated layers. This is separate from distance-independent
ID seeding and the cell-footprint issue.

#### Octave transitions and UV fallback

The stochastic pick hashes the finer UV lattice with a salt independent of the
threshold phases. Its selector stays attached to the surface and selects B when
`selector < t`. A mixture of uniform thresholds remains uniform if selection is
independent; GPU marginal-distribution tests check this through fractional octaves.
It requires no remapping. The mixture is not guaranteed to retain the blue
spectrum, however, and introduces discrete threshold changes as the selector
crosses `t`. CDF lerp continuously blends the thresholds and remaps the resulting
nonuniform distribution. Both approaches gradually change the pattern during
zoom; neither eliminates all crawling.

Missing/constant and rank-one UVs (singular UV derivative Jacobian) fall back to
the existing object-space hash. The vertex shader passes local position before
instance transforms, so the fallback follows rigid object transforms. Custom
deformation shaders should pass a stable rest/object reference position in
`local`. The object field follows
[Wyman and McGuire, Hashed Alpha Testing, section 5.4](https://research.nvidia.com/sites/default/files/pubs/2017-02_Hashed-Alpha-Testing/Wyman2017Hashed.pdf).
It hashes 3D cells into the threshold table and therefore does not retain the
table's 2D blue-noise spectrum.

Bayer contains all 4096 midpoint ranks and is repeatable, but its periodic
structure can crosshatch and interact with the reconstruction kernel. Independent
octave shifts and stochastic picking disrupt coherent cross-octave nesting.
Translations of an ordered matrix can also correlate overlapping layers; it is
a comparison mode rather than a recommended transparency default.

A controlled diagnostic sampled 1600 fixed material points on a curved
parameterized surface over 24 camera poses, at alpha 0.5. Values below are the
percentage changing binary acceptance per consecutive pose; they isolate the
threshold field and exclude rasterization, shading, visibility and reconstruction.

| Camera motion | Object hash | UV CDF lerp | UV stochastic pick | UV Bayer |
| --- | ---: | ---: | ---: | ---: |
| Pan, 0.6 scene units | 0.071 | 0.046 | 0.071 | 0.073 |
| Orbit, 0.6 radians | 0.446 | 0.419 | 0.636 | 0.650 |
| Dolly, distance 2.5 to 4 | 1.147 | 1.231 | 1.476 | 1.419 |

This does not establish full-scene perceptual quality, but it does not support
promoting stochastic picking as the more stable default. Anchored blue is now the
default. These initial four modes and the three options below remain selectable
for visual comparison.

#### Correlated octave hierarchy

`uv_blue_correlated` builds a small noise hierarchy on first selection and reuses
it afterward. `build_correlated_noise_hierarchy()` in `TextureProcessing.py` and
`noise_hierarchy.shader` perform all generation on the GPU, with no CPU readback.
The source must be a square power-of-two noise tile up to 64x64.

Each parent averages its four child ranks, then receives unique ranks again.
Hash-based tie breaking avoids row-order stripes. Ordinary averaged noise mips
would lose threshold variance and bias coverage; re-ranking preserves the
marginal distribution. Each sample adds a uniform root-tile jitter within its
rank bin. That jitter is shared by parent and child samples and also gives the
1x1 root a varying threshold rather than a constant 0.5.

All levels use one origin translated in finest-level coordinates; there are no
independent per-octave shifts. The 64x64 pyramid spans requested octaves -4..2.
Outside that range the boundary mip is reused while the lattice continues to
scale; beyond the root, hashed root cells supply thresholds at the coarser scale.
Strong parent/child correlation applies inside the generated range. Re-ranking
does not guarantee that every downsampled level retains the blue-noise spectrum.

Adjacent hierarchy levels are correlated, so the independent-sample CDF formula
used by the old lerp mode would be incorrect. This option uses an independent,
surface-anchored octave selector instead. It avoids mismatched octave phases,
but individual acceptance changes can still occur while zooming; it is not a
guarantee of a perfectly static or pop-free raster image.

#### Triplanar projection

`triplanar_blue` and `triplanar_correlated` ignore material UVs. They sample XY,
XZ and YZ in the same object/reference space used by the object hash. Independent
axis seeds decorrelate the three fields. Weights are proportional to
`abs(local_normal)^4`; smooth local normals allow continuous changes between
projection axes. A two-stage CDF correction preserves the threshold distribution
while blending, rather than simply averaging uniforms. Zero normals fall back
to object hashing.

The local position and local normal stay attached to the object under instance
transforms. Custom deformation shaders should supply stable rest/reference
values for both. Triplanar projection is a whole-surface option selected manually;
it does not add UV-seam detection or anisotropic footprint filtering. Hard normal
discontinuities can still produce projection changes. No temporal selection or
TAA is involved, and the existing Bayer option is retained.

Repeating the fixed-material-point diagnostic above for the added modes gives:

| Mode | Pan (%) | Orbit (%) | Dolly (%) |
| --- | ---: | ---: | ---: |
| UV correlated hierarchy | 0.063 | 0.530 | 1.174 |
| Triplanar blue | 0.071 | 0.465 | 1.389 |
| Triplanar correlated hierarchy | 0.084 | 0.462 | 1.141 |

These are threshold-field measurements on one curved surface, not a guarantee
of perceptual improvement on every model. The current default is anchored blue;
select the other modes to compare them on the scene.

The default geometry shader's group 3 binding 2 is the sampled rank hierarchy.
Before generation it binds the base noise texture as an unused placeholder.
Custom geometry shaders using the shared helper must also supply the hierarchy
texture and the interpolated `local_normal` reference varying.

Use `TransparencyPass(..., noise_scale=1.0)` or `set_noise_scale(pixels)` to change
approximate cell width without changing opacity, guide data, filter settings or
resource allocation. Larger cells trade coarser grain for fewer threshold changes.
The value must be positive and representable as a normalized finite float32.
The setting is the 1x width; frame `clip.z` holds the effective MSAA-adjusted
width and `clip.w` the index in `NOISE_MODES`.

Overlapping instances must have distinct stable IDs in metadata, including
coincident geometry. Separate front/back seeds distinguish opposite physical
sides. Repeated/overlapping UV islands on same-side surfaces within a single
instance can share thresholds; per-instance shifts cannot decorrelate those
layers. Use distinct stable IDs or object-hash mode for such assets. The fallback
also has a limit: nearly coincident same-side surfaces can share a 3D cell.

The `rgba32uint` winner payload stores packed instance index plus one (zero means
empty), local triangle ID, two 16-bit barycentric coordinates, and half-float LOD
and face orientation. Deferred shading reconstructs local vertices and applies
the same instance translation, rotation, and scale as rasterization; normals use
inverse scale and rotation. Invalid instance/triangle/material addresses yield
zero samples. All draws sharing a selection target must use one metadata/vertex
address space for deferred shading.

Rasterization selects texture LOD from UV derivatives before any discard and
stores it in the winner payload. Deferred shading uses that LOD, so selection,
transmittance, and shading use matching texture filtering. Material mipmaps are
built by [TextureProcessing.py](src/TextureProcessing.py) in linear color with
alpha weighting; MASK coverage correction and transparent-edge RGB padding
limit cutout fringes without changing BLEND mean alpha.

Route MASK alongside BLEND through the three transparency raster passes to use
analytic cutout coverage. Each pass evaluates the same alpha and derivatives
before any discard, then uses
`coverage = clamp((alpha-cutoff)/max(fwidth(alpha), 1e-5)+0.5, 0, 1)`.
`alpha_rgb = coverage * opacity_rgb.rgb`; BLEND retains ordinary soft alpha.
Cutout interiors therefore accept deterministically, and only the boundary is
stochastic. The example explicitly loads `Simple_foliage.001` as MASK using
`load_transparency_scene(..., mask_materials=(...))`; this keeps generic BLEND
materials soft. OPAQUE geometry continues to use the opaque pass. Explicitly
sending MASK to the opaque pass retains the conventional hard-cutoff behavior.

### Grazing-angle opacity fade

`GrazingFade(enabled=True, start_angle=83, end_angle=86)` configures an optional
fade for transparent surfaces. Supply `grazing_fade=...` when constructing the
pass or call `set_grazing_fade(options)` to update it without reallocating targets.
The example exposes a checkbox, two degree fields and the G shortcut. These are
live controls; the example does not save settings to a profile file.

Angles are measured from the geometric normal to the view direction: 0 degrees
is face-on and 90 is edge-on. The absolute dot product makes front and back faces
behave symmetrically. Geometric normals come from world-position derivatives,
so smooth shading normals do not change the fade. Perspective views use the
fragment-to-camera vector; orthographic views use the constant camera direction.

```text
angle = degrees(acos(abs(dot(normalize(N_geom), normalize(to_eye)))))
fade = 1 - smoothstep(start_angle, end_angle, angle)
alpha_rgb *= fade
```

The interval must satisfy `0 <= start_angle < end_angle <= 90` and remain ordered
at float32 precision. The shader applies this factor after texture alpha
sharpening, identically in guide, transmittance and selection. Fully faded
fragments cannot contribute guide depth or win selection; deferred winner color
stays straight, so opacity is not applied twice. OPAQUE-pass geometry is unaffected.
`grazing_fade_factor` is reusable in `utils.shaderlib`.

The shared `Frame` ABI includes a `grazing: vec4f` after `clip`, storing start
degrees, end degrees, enabled (0/1), and a reserved component. Custom frame
declarations must include this member when using the default geometry pipeline.

### Reconstruction

In linear HDR space, define `A_rgb = 1 - T_rgb` and `A_s = 1 - T_scalar`.
For pixel `p` and its fixed 3x3, 5x5 or 9x9 neighborhood:

```text
C_trans(p) = A_rgb(p) * sum_q(w(p,q) * C_winner(q))
                     / max(sum_q(w(p,q) * hasWinner(q)), epsilon)

HDR(p) = opaque_color(p) * T_rgb(p) + C_trans(p)
```

The shaded target's alpha is a 0/1 winner flag, including for black winners.
No-winner pixels contribute neither color nor winner weight. This ratio removes
survivor-count brightness fluctuations for constant-color supported kernels;
coverage comes from the exact local `A_rgb`. It is a finite-sample ratio estimator,
not strictly unbiased: empty neighborhoods return zero via the epsilon floor,
and varying colors/weights can introduce spatial bias.

Certain nearest winners (`pNearest == 1`) skip neighborhood reads completely,
preserving noise-free leaf-interior texture and shading. Every fractional pixel
uses the full normalized reconstruction. Blending in its own sample near full
opacity would expose rare missing winners as dark holes during camera movement.
Before half-float storage, the guide reserves `1.0` for exact float32 certainty
and caps fractional probabilities at `0.99951171875` (the next lower half-float).
This prevents nearly opaque fragments from rounding up into the bypass. Selection
and transmittance continue using the original opacity, so coverage is unchanged.

The 1x1 option bypasses reconstruction with a direct point estimator:
`C_trans = A_rgb * C_winner / max(A_s, epsilon)`. With scalar opacity,
`A_rgb/A_s == 1`: the surviving sample retains its full brightness. Multiplying
by alpha again would darken the mean. There are no neighborhood or guide reads
in this path; composition and presentation still run normally. Empty samples
stay zero, making the raw stochastic noise visible for comparison. Callers can
omit the guide raster pass for 1x1, as the example does.

Cross-bilateral weights use pixel distance, opaque depth, nearest-transparent
depth, and RGBA transmittance. Always-on slope-aware transparent-depth weights predict
neighbor depths from the stored reciprocal-depth gradient (affine on a projected
planar card). Both forward and backward plane residuals must pass, keeping card
discontinuities while admitting oblique surfaces. Opaque depth remains a separate
constraint. Composition applies
`HDR / (1 + HDR)` and encodes sRGB only when the output format does not do so.

`BilateralFilter` settings update uniforms without rebuilding pipelines/targets:

| Setting | Meaning |
| --- | --- |
| `kernel_size` | 1 (reconstruction off), 3, 5 or 9 pixels per side; default 3 |
| `spatial_sigma` | Pixel-distance standard deviation; default 0.9 |
| `depth_sigma` | Relative view-depth tolerance; default 0.05 |
| `min_depth_sigma` | Minimum depth tolerance in scene units; default 0.01 |
| `transmittance_sigma` | RGBA transmittance tolerance; default 0.3 |

Sigmas must be finite and positive. The chosen kernel and sigmas remain fixed
at every distance; there is no adaptive search or distance interpolation. Both
depth constraints and transmittance-edge rejection remain active. Looser
tolerances increase support but can blur across overlapping surfaces.

Slope correction is retained as default behavior rather than a setting: the
oblique-card GPU regression verifies over twice the effective support of raw
depth differences under tight tolerances, while the depth-step regression checks
that card discontinuities remain protected. Its benefit is support on tilted
surfaces, not surface anchoring or direct temporal stabilization.

The caller-owned `TransparencyTargets` set includes the `transparent_guide`
rgba16float attachment, storing `(pNearest, d(1/z)/dx, d(1/z)/dy, 1/z)` and cleared
to zero. It shares the nearest-depth test with `transparent_depth`. Custom
geometry shaders must output this guide data as well as depth, reserving
`pNearest == 1` for exact certainty before half-float conversion. Raster
`VERTEX_DTYPE` contains position, normal, UV, barycentric coordinates and triangle
ID; `SHADING_VERTEX_DTYPE` stores the original vertices for deferred shading.

### Diagnostic views and limits

Call `set_debug_view(name)`:

- `final`: tone-mapped composition.
- `transmittance`: scalar transmittance as grayscale, without reconstruction or tone mapping; black cutouts on white are expected for nearly opaque geometry.
- `winner`: unfiltered, tone-mapped winner color.
- `reconstruction`: tone-mapped transparent contribution without the opaque background.
- `filter_support`: `(sum(w)**2 / sum(w*w)) / kernel_area`, including no-winner samples.

If transmittance is stable while reconstruction flickers, inspect winner variance
and filter support. Bright support means substantial spatial weight, not many
accepted winners. Smaller tolerances can leave too few contributing samples.

A single winner is noisy; spatial reconstruction introduces bias and can blur
edges. Using `max(alpha_rgb)` is an approximation for overlapping layers with
differing RGB opacities. Geometric silhouettes are point-sampled with MSAA off;
optional 4x MSAA smooths their coverage. Pure absorption has no separate
pass-skipping material path. The technique provides no temporal accumulation,
additive-material path, or multiple winners per sample. Custom material code or a
matched geometry/shading shader pair can be supplied under the existing payload
and shared-address-space contracts.
