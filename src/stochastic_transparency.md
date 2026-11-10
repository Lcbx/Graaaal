# Stochastic transparency

[TransparencyPass.py](TransparencyPass.py) implements single-winner stochastic
order-independent transparency. Applications provide geometry, instances,
materials, camera values, targets, and command submission. The
[engine guide](../README.md#composable-transparency) documents its resource API;
the [interactive scene](../scenes/transparency_example.py) shows complete recording.

## Useful APIs

Import from `numpyecs.TransparencyPass`. This overview is non-exhaustive; initialize
RenderContext before constructing GPU resources.

| Class or method | Purpose |
| --- | --- |
| `TransparencyPass(...)` | Owns persistent pipelines and size-dependent targets; accepts material, vertex, texture and optional opaque inputs. |
| `resize(size, targets=...)` | Allocates or adopts targets; call before updating camera data or recording. `targets` is keyword-only. |
| `update_frame(...)`, `update_camera(camera, ...)` | Upload explicit arrays or camera-object values and return `view @ projection`; accept `time` and `light_dir`. |
| `create_geometry_passes(metadata, transparent=..., instances=...)` | Returns `GeometryPasses`: `.render_shader` for DrawBatches, `.guide` for the guide override, and deferred bindings for `shade_winner`. |
| `opaque_pass`, `guide_pass`, `transmittance_pass`, `selection_pass` | Open raster passes on caller-owned commands; issue draws inside each scope. |
| `shade_winner(commands, geometry)`, `compose(commands, output)` | Record deferred shading and composition; `output` is a matching `ColorAttachment`. Neither submits work. |
| `set_opaque_inputs(color, depth)` | Uses caller-owned opaque textures; pass `None, None` to restore internal targets. |
| `set_sample_count`, `set_noise_mode`, `set_noise_scale`, `set_debug_view` | Update live sampling and diagnostic settings. |
| `BilateralFilter`, `set_bilateral_filter` | Configure reconstruction footprint and tolerances. |
| `GrazingFade`, `set_grazing_fade` | Configure angular opacity fade; default enabled from 83 to 86 degrees. |
| `TransparencyTargets` | Describes caller-owned color, depth, guide, transmittance and winner textures. |

## Pass order

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

## Geometry and caller-owned targets

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
[vertex hooks](materials.md#time-and-vertex-displacement).

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

## Optional MSAA

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

## Acceptance and winner data

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
| `uv_blue_lerp` | Point-sampled blue tile in two UV octaves, lerp and uniform CDF correction; default |
| `uv_blue_pick` | Same UV octaves, selected by a separate surface-anchored hash channel |
| `uv_bayer` | Procedural 64x64 Bayer ranks, with the stochastic octave pick; diagnostic |
| `uv_blue_correlated` | GPU-generated parent/child rank hierarchy, shared spatial origin, static octave pick |
| `triplanar_blue` | Object-space XY/XZ/YZ tile projections, smooth normal weights and CDF-corrected blends |
| `triplanar_correlated` | Triplanar projection using the correlated hierarchy |
| `anchored_blue` | Pixel-lattice blue tile with first-order rest-cell anchors and finer rest-cell shifts |

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
the original [blue-noise tile](data/blue_noise/README.md).
Mirroring and nonuniform UV density change the footprint, not the probability.
Grazing-angle anisotropy still distorts the spatial spectrum.

The size is a target, not a guarantee of square screen-space cells. A scalar
footprint cannot compensate separately for both axes. UV derivatives `(1/64, 0)`
and `(0, 4/64)`, for example, select approximately 4x1 screen-pixel cells at an
integer octave and a 1-pixel target. Octave blending and triplanar mixing further
complicate a single cell width. MSAA scaling does not remove this anisotropy.

### UV footprint compromise experiment

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

### Pixel-lattice anchored blue

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
orbit/dolly reshuffling here. It remains an optional comparison mode; UV CDF lerp
is still the default. Neither diagnostic uses TAA or frame history.

### Cross-object decorrelation audit

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

### Octave transitions and UV fallback

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
promoting stochastic picking as the more stable default. UV CDF lerp is the
default. These initial four modes and the three options below remain selectable
for visual comparison.

### Correlated octave hierarchy

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

### Triplanar projection

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
of perceptual improvement on every model. The existing UV CDF-lerp default is
unchanged; select the added modes to compare them on the scene.

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
built by [TextureProcessing.py](TextureProcessing.py) in linear color with
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

## Grazing-angle opacity fade

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

## Reconstruction

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

## Diagnostic views and limits

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
