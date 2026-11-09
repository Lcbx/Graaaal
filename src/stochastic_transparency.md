# Stochastic transparency

[TransparencyPass.py](TransparencyPass.py) implements single-winner stochastic
order-independent transparency. Applications provide geometry, instances,
materials, camera values, targets, and command submission. The
[engine guide](../README.md#composable-transparency) documents its resource API;
the [interactive scene](../scenes/transparency_example.py) shows complete recording.

## Pass order

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
transmittance, and selection. The technique requires no transparent sorting.

## Optional MSAA

`TransparencyPass(..., sample_count=4)` enables 4x MSAA; the default is 1.
`set_sample_count(1 or 4)` switches the default shaders and reallocates targets
while preserving registered DrawBatches geometry, camera and live settings.
The example exposes an **MSAA (M)** dropdown and the **M** shortcut.

Opaque, guide, transmittance and selection passes retain four independent samples.
Surface attributes interpolate at sample positions, and selection uses a static
per-sample noise seed. Deferred shading processes each sample; composition filters
each sample separately and averages linear HDR before tone mapping. Integer winner
IDs are never resolved or averaged. There is no temporal accumulation.

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

UV sampling uses raw UV derivatives, before any discard or material-LOD clamp:

```text
f = max(length(dpdx(uv)), length(dpdy(uv))) * 64 * noise_scale
s = log2(f); k = floor(s); t = fract(s)
uA = tile[(floor(uv * 64 / 2^k)     + shift(stableID, k))     mod N]
uB = tile[(floor(uv * 64 / 2^(k+1)) + shift(stableID, k + 1)) mod N]
```

Octaves may be negative; numerical guards bound them to [-24,24]. Sampling uses
integer `textureLoad`, never bilinear noise filtering. Signed wrapping handles
negative UVs. Per-instance/per-octave translations leave a single octave's
threshold ranks and Fourier magnitudes unchanged; GPU tests verify both against
the original [blue-noise tile](../scenes/resources/blue_noise/README.md).
Mirroring and nonuniform UV density change the footprint, not the probability.
Grazing-angle anisotropy still distorts the spatial spectrum.

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
Frame `clip.z` holds scale and `clip.w` the index in `NOISE_MODES`.

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

`GrazingFade(enabled=False, start_angle=83, end_angle=86)` configures an optional
fade for transparent surfaces. Supply `grazing_fade=...` when constructing the
pass or call `set_grazing_fade(options)` to update it without reallocating targets.
The example exposes a checkbox, two degree fields and the G shortcut. These are
live controls outside the filter-only JSON profile.

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

The caller-owned `TransparencyTargets` set includes the new `transparent_guide`
rgba16float attachment, storing `(pNearest, d(1/z)/dx, d(1/z)/dy, 1/z)` and cleared
to zero. It shares the nearest-depth test with `transparent_depth`. Custom
geometry shaders must output this guide data as well as depth, reserving
`pNearest == 1` for exact certainty before half-float conversion. The card-unit
attributes have been removed from `VERTEX_DTYPE`; raster geometry needs only
position, normal, UV, barycentric coordinates and triangle ID. Deferred shading's
vertex ABI is unchanged.

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
