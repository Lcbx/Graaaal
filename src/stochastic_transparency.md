# Stochastic transparency

[TransparencyPass.py](TransparencyPass.py) implements stochastic transparency with
one/two averaged draws or any positive number of conditional fill attempts.
Applications provide geometry, instances,
materials, camera values, targets, and command submission. The
[engine guide](../README.md#composable-transparency) documents its resource API;
the [interactive scene](../scenes/transparency_example.py) shows complete recording.

`TransparencyPass`, `MaterialRenderer` and the examples default to
`tail_allocation="banded", kawase_passes=1`: exact nearest capture, two
complementary banded winners and one Kawase down/up pair.
Tail estimation uses 1x; scene MSAA applies to opaque/MASK geometry, transmission
and the exact nearest layer. Subset shading is off. MASK materials use fixed
alpha-cutoff discard and opaque depth; only BLEND uses stochastic transparency.

Choose `tail_allocation="stochastic"` for the earlier winner/filter options.
That mode's defaults are UV CDF noise with footprint balancing at 1.0, two
complemented averaged winners and one 3x3 fragment bilateral pass.
Low-level custom shader sets require `tail_allocation="stochastic", winner_count=1,
reconstruction="fragment"`.

## Useful APIs

Import from `numpyecs.TransparencyPass`. This overview is non-exhaustive; initialize
RenderContext before constructing GPU resources.

| Class or method | Purpose |
| --- | --- |
| `TransparencyPass(...)` | Owns persistent pipelines and size-dependent targets; accepts material, vertex, texture and optional opaque inputs. |
| `resize(size, targets=...)` | Allocates or adopts targets; call before updating camera data or recording. `targets` is keyword-only. |
| `update_frame(...)`, `update_camera(camera, ...)` | Upload explicit arrays or camera-object values and return `view @ projection`; accept `frame` and `lighting` mappings. |
| `create_geometry_passes(metadata, transparent=..., instances=...)` | Returns `GeometryPasses`: `.render_shader` for DrawBatches, `.guide` and optional `.nearest` overrides, and deferred shading bindings. |
| `opaque_pass`, `guide_pass`, `transmittance_pass`, `selection_pass` | Open raster passes on caller-owned commands; issue draws inside each scope. |
| `shade_winner(commands, geometry)`, `compose(commands, output)` | Record deferred shading and composition; `output` is a matching `ColorAttachment`. Neither submits work. |
| `set_opaque_inputs(color, depth)` | Uses caller-owned opaque textures; pass `None, None` to restore internal targets. |
| `exact_nearest=True`, `set_exact_nearest(bool)` | Deterministic nearest transparent layer; enabled and required in default banded mode, optional in stochastic mode. |
| `set_sample_count`, `set_noise_mode`, `set_noise_scale`, `set_debug_view` | Update live sampling and diagnostic settings. |
| `set_winner_count(N)`, `set_winner_mix(mode)` | Positive uint32 draw count and `"average"`/`"fill"` mode; averaging supports 1 or 2. Default: 2 in average mode. |
| `set_winner_coupling("complement" or "independent")` | Pair two averaged selections with one complemented tile (default), or independent sources. Fill always uses independent retries. |
| `set_reconstruction("atrous", "push_pull", "fragment" or "compute")` | Select reconstruction after choosing stochastic allocation; banded uses Kawase. |
| `ConfidenceFilter`, `set_confidence_filter` | Set budget (default 4), cascade passes (1–4, default 4), and pyramid levels (2 or 3, default 3). |
| `set_reconstruction_passes(N)` | Stack a positive uint32 number of bilateral passes; default 1. A 1x1 kernel bypasses all passes. |
| `set_early_b_rejection(bool)` | Experimental per-sample stencil rejection for fill B; default off and requires `depth32float-stencil8`. |
| `BilateralFilter`, `set_bilateral_filter` | Configure reconstruction footprint and tolerances. |
| `GrazingFade`, `set_grazing_fade` | Configure angular opacity fade; default enabled from 83 to 86 degrees. |
| `TransparencyTargets` | Describes caller-owned color, depth, guide, transmittance and winner textures. |

## Exact nearest with banded tail and dual Kawase

Set `tail_allocation="banded", kawase_passes=1` on `TransparencyPass` or
`MaterialRenderer`. This selects exact nearest capture, a complementary pair of
banded tail winners, and dual Kawase reconstruction. These are the defaults.

```python
renderer = MaterialRenderer(items)
renderer.set_kawase_passes(3)
renderer.set_tail_allocation("stochastic")  # returns to fragment reconstruction
```

The banded mode defaults to `banded_proxy=True`, `banded_subset=False`,
`banded_tail_msaa=False`. The proxy uses the texture albedo already fetched for
opacity. Its full-resolution contribution is restored after blurring signed
residuals, reducing color halos where albedo approximates the shaded color.
Custom materials and lighting remain supported; the proxy need not equal them.

```python
renderer.set_banded_options(proxy=True, subset=True, tail_msaa=False)
renderer.set_kawase_passes(3)
```

`subset=True` selects one fixed pixel in every 2x2 block for allocation/lighting
and stores zero residual/confidence at the other pixels. It increases noise;
compare a wider blur. Alpha derivatives still execute before the allocation
discard. `tail_msaa=False` runs statistics, allocation, lighting and blur at 1x,
while physical transmission and exact nearest remain at the scene MSAA count.
At 4x this adds a separate `tail_statistics_pass()` before allocation. Direct
recorders must draw `"tail_statistics"` into it when `mixed_tail` is true.
`tail_sample_count` gives the estimator's sample count.

This mixed mode approximates subpixel tail coverage: opaque depth uses native
sample zero; fragments matching any native nearest ID are excluded to avoid
false contributions at internal triangle boundaries. Tail-only features missing
from the 1x raster can be lost. Subset sampling remains opt-in. The settings UI
has no tail-MSAA toggle; the programmatic `tail_msaa=True` variant remains
available for comparisons and workloads that need subpixel tail detail.

`kawase_passes` is an integer from 0 through 8: each level records one downsample
and one upsample, stopping when the smallest level reaches 1x1. Zero uses the
unblurred tail. All settings preserve the sharp exact nearest contribution.
The tail's numerator (signed residual when enabled) and allocation denominator are blurred together;
division and RGB opacity application occur after blurring, before composition
under the nearest layer. There is no depth/opacity edge stopping in this blur,
so increasing its radius can mix colors across tail boundaries.

`set_tail_allocation("banded")` enables exact nearest and resets incompatible
settings to two complemented averaged winners, no force acceptance or stencil,
and one reconstruction stage. Those settings cannot be changed while banding
is active. Switch back to `"stochastic"` to use the other techniques; exact
nearest stays enabled. `set_kawase_passes()` changes blur depth live without
re-registering geometry. `MaterialRenderer.record()` handles the passes.

For direct recording, after opaque rendering and camera/target setup:

```python
with transparency.nearest_pass(commands) as render:
	batches.draw(render, visibility, "main", pass_override=geometry.nearest)
transparency.shade_nearest(commands, geometry)
with transparency.transmittance_pass(commands) as render:
	batches.draw(render, visibility, "transmittance")
if transparency.mixed_tail:
	with transparency.tail_statistics_pass(commands) as render:
		batches.draw(render, visibility, "tail_statistics")
with transparency.banded_pass(commands) as render:
	batches.draw(render, visibility, "bands")
transparency.shade_winner(commands, geometry)  # shades both tail winners
transparency.compose(commands, output)
```

Record transmission exactly once before allocation. Do not record the guide or
per-stream selection passes in this mode. Counters and payloads clear as part
of `transmittance_pass()`. Caller-supplied `TransparencyTargets` can set `winner`
and `selection_depth` to `None`; band counters, array payloads and blur levels
are managed internally. Native 1x/4x, external opaque inputs and material hooks
including vertex animation are supported; custom low-level shader sets are not.

Banding shares one screen-space noise field across the tail. `noise_scale`
controls its size (halved when tail estimation itself uses 4x MSAA); surface noise mode, UV balance, bilateral
settings and confidence budgets do not affect this path. Diagnostics are
`final`, `transmittance`, `winner` (raw numerator, or signed residual when proxy
is enabled; zero residual displays as neutral grey),
`reconstruction`, `nearest`, `nearest_mask`, and `tail_transmittance`.

Banded allocation uses an unordered WebGPU atomic path. Its band assignment may
change between frames in overlap regions. Tail colors approach a p-weighted
mixture, which differs from depth-ordered over for thick tails. Probabilities
use 1/65536 fixed-point quantization; extremely faint tail fragments can round
to zero width. Totals must remain below 2^32 units (65536 full-opacity tail
fragments per sample), and target dimensions must fit the device's storage
buffer binding limit. Exact nearest capture retains every nonzero opacity.

For caller-owned targets in mixed mode, `winner_color` and
`reconstruction_color` have `tail_sample_count` layers; nearest-color storage
keeps `sample_count` layers. In mixed mode `reconstruction_color.rgb` holds tail
mean color, with physical per-sample opacity applied during composition. With
matched sample counts it holds the premultiplied tail contribution.

## Recording and settings

For classic stochastic allocation, choose `tail_allocation="stochastic"` first.
Call `resize(size)` before updating frame/camera data or recording passes.
All stages record into the caller's `CommandContext`; the application submits.
Render transparent geometry into `guide_pass`, `transmittance_pass` and
`selection_pass`, then call `shade_winner` and `compose`. The guide is optional
when the reconstruction kernel is 1x1.

| Stage | API | Output |
| --- | --- | --- |
| Opaque | `opaque_pass()` or an external renderer | Color and depth |
| Transparent guide | `guide_pass()` | Depth and reconstruction guide |
| Transmission | `transmittance_pass()` | Accumulated transmittance |
| Winner selection | `selection_pass()` | Accepted fragment payload |
| Shading | `shade_winner(commands, geometry)` | Shaded winner |
| Composition | `compose(commands, output)` | Output attachment |

With an existing `TransparencyPass` named `transparency`, for example:

```python
transparency.set_tail_allocation("stochastic")
transparency.set_noise_mode("anchored_blue")
transparency.set_winner_mix("fill")
transparency.set_winner_count(4)
transparency.set_confidence_filter(ConfidenceFilter(target_winners=8, cascade_passes=4))
```

`winner_mix="average"` supports one or two draws; `"fill"` supports any positive
uint32 draw count with `object_hash` or `anchored_blue` and the default noise
sources. Switch back to at most two draws before selecting average mode.
`set_early_b_rejection(True)` optionally requires the
`depth32float-stencil8` GPU feature for fill mode.

Enable `exact_nearest=True` to shade the nearest nonzero-opacity transparent
fragment exactly at each hardware sample, then composite it over the reconstructed
tail. All noise, winner-count, force/fill, filter and MSAA choices apply to the
tail. Every nonzero opacity is eligible; there is no minimum-alpha cutoff.
`MaterialRenderer.record()` handles these passes automatically. Direct recording
must call `nearest_pass()` and draw using `geometry.nearest`, followed by
`shade_nearest(commands, geometry)`, before the tail guide/transmittance/selection
passes. The nearest pass is required even at a 1x1 reconstruction kernel.
`set_exact_nearest()` is available on both pass and material renderer. Material
templates and frame animation are supported; low-level custom shader sets are not.

Direct DrawBatches recording after opaque rendering, with a shared `geometry`:

```python
with transparency.nearest_pass(commands) as render:
	batches.draw(render, visibility, "main", pass_override=geometry.nearest)
transparency.shade_nearest(commands, geometry)
# Record the tail guide, transmittance and winner passes next, then compose.
```

```python
renderer.set_exact_nearest(True)  # MaterialRenderer, or TransparencyPass
renderer.transparency.set_debug_view("nearest")  # exact nearest contribution
```

With this option, `transmittance` shows total scalar transmission; `winner` and
stream diagnostics describe the tail only. Additional views are `nearest`,
`nearest_mask` and `tail_transmittance`. Disabling the option resets an active
nearest-only diagnostic to `final`. Existing depth/transmittance filter controls
guide the tail; single-layer nearest contributions bypass reconstruction.
`set_reconstruction("atrous"|"push_pull"|"fragment"|"compute")` chooses the implementation.
The optional a-trous cascade expands the footprint where retained acceptance
support is low. `ConfidenceFilter.target_winners` sets its support budget D;
higher values give broader smoothing. D is not a measured effective sample count.
`cascade_passes=1..4` sets maximum radii of 1/3/7/15 pixels. The comparison
pyramid uses `levels=2` or `3`. Both use the depth/transmittance tolerances in
`BilateralFilter`; spatial sigma applies to the pyramid and bilateral filters.
Its 1x1 kernel disables reconstruction; other kernel sizes enable the confidence
filters' 3x3 gathers. `reconstruction_passes` applies only to fixed bilateral modes.

Complemented averaging uses the same final remapped threshold and phase for
both selections. Set `winner_coupling="independent"` to compare independent
sources. `blue_noise_second` is used by independent pairing and fill, and is not
sampled by complemented pairing. Winner validity diagnostics still show actual
accepted fractions; `filter_support` shows final retained D / budget for the
cascade, or local confidence for the pyramid.

## Geometry and targets

`create_geometry_passes(metadata, transparent=True, instances=...)` returns
`GeometryPasses`; use its `.render_shader` with DrawBatches, `.guide` for the
guide pass, and the same object in `shade_winner()`. All transparent draws sharing
a selection target must share the metadata, vertices, and material address space.
The default geometry uses `VERTEX_DTYPE` and `MESH_INSTANCE_DTYPE`. The metadata
fields hold triangle offset, material index, stable surface ID, and a reserved
slot (used by the dictionary-material adapter for instance parameters).

Supply opaque color and depth with `set_opaque_inputs(color, depth)`, or let the
pass allocate them. `TransparencyTargets` lets callers provide the remaining
attachments; each must match the configured size and sampling layout.
Set `target_format` to match the output color attachment.

| 1x target | Format |
| --- | --- |
| `main_color`, `transparent_guide`, `transmittance` | `rgba16float` |
| `depth`, `transparent_depth`, `selection_depth` | `depth32float` |
| `winner` | `rgba32uint` |
| `winner_color` | `rgba16float` storage/texture |

`set_sample_count(1 or 4)` selects 1x/4x MSAA with default shaders and
reallocates size-dependent targets. For 4x, winner payloads use two `rgba16uint`
attachments (`winner`, `winner_extra`); `winner_color` is a four-layer,
single-sample `rgba16float` texture. Externally owned opaque attachments must
match the selected sample count. In fill mode, caller-owned targets also require
`winner_source` (`r32uint`, one layer per sample). Multiple reconstruction
bilateral passes require `reconstruction_pair`, and more than two require
`reconstruction_pair_scratch` (`rgba16float`).
Complemented two-draw averaging additionally requires `winner_first_payload`
(`rgba32uint`, sampled/storage, one layer per sample, including an array layer
at 1x). Push-pull and compute require `reconstruction_color` (`rgba16float`,
sampled/storage, layered only at 4x); internal confidence-filter textures are pass-owned.

Caller-owned targets for `exact_nearest=True` additionally need `nearest_depth`
(`depth32float`), `nearest_alpha` (`rgba16float`), and `nearest_payload`
(`rgba32uint` at 1x; `rgba16uint` plus `nearest_extra` at 4x). These are raster
attachments matching native sample count. `nearest_color` is `rgba16float`
sampled/storage, a 2D texture at 1x or four single-sample array layers at 4x.

For custom material shading or geometry deformation, use `material_source`, a
matching `geometry_shader`/`shading_shader` pair, or the extra resource arguments
to `create_geometry_passes()`. Custom shader sets support only one winner unless
explicitly implemented for additional draws and the selected MSAA ABI.

## Noise and opacity controls

`set_noise_mode(name)` accepts:

| Name | Use |
| --- | --- |
| `anchored_blue` | Pixel-lattice anchored noise without material UVs |
| `object_hash` | Object-space hash; useful with overlapping or missing UVs |
| `uv_blue_lerp` (default), `uv_blue_pick` | UV blue-noise variants |
| `uv_bayer` | Diagnostic ordered-noise comparison |
| `uv_blue_correlated` | UV correlated hierarchy |
| `triplanar_blue`, `triplanar_correlated` | Object-space triplanar variants |

`set_noise_scale(pixels)` adjusts the positive, finite 1x noise-cell target;
4x MSAA uses half that target. `set_noise_anchor(pixels=24, subdivisions=4)`
configures anchored-blue grain. `set_noise_footprint_balancing(True)` enables
a UV-only shader feature (enabled by default), and `set_noise_footprint_balance(a)` sets
its 0–1 derivative balance (default 1). Other modes ignore this control.
Distinct overlapping instances should use distinct stable surface IDs.

`GrazingFade(enabled=True, start_angle=83, end_angle=86)` fades transparent
opacity near a silhouette. Pass it to the constructor as `grazing_fade=` or
call `set_grazing_fade(...)`. Require `0 <= start_angle < end_angle <= 90`.
BLEND uses transparent passes. OPAQUE and MASK use the opaque pass; MASK discards
when texture/base-color alpha is below `alpha_cutoff`, writes depth for accepted
samples, and ignores stochastic noise, grazing fade and tail reconstruction.
UVs are interpolated at native MSAA sample locations, so cutoff coverage and
opaque depth agree per sample. Pixel-center Hi-Z culling is bypassed at 4x.
For direct geometry registration, pass `transparent=False` for OPAQUE/MASK and
`True` for BLEND. No temporal accumulation is performed.

## Reconstruction and diagnostics

Pass `BilateralFilter(...)` to `set_bilateral_filter()`:

| Parameter | Effect | Default |
| --- | --- | --- |
| `kernel_size` | Square footprint, 1/3/5/9 | 3 |
| `spatial_sigma` | Pixel-distance weight | 0.9 |
| `depth_sigma` | Relative depth tolerance | 0.05 |
| `min_depth_sigma` | Minimum depth tolerance | 0.01 |
| `transmittance_sigma` | Transmittance-edge tolerance | 0.3 |

Sigmas must be finite and positive. `kernel_size=1` disables reconstruction.
`set_debug_view(name)` accepts `final`, `transmittance`, `winner`,
`reconstruction`, and `filter_support`. Stream-specific views are available
when their selected winner mode enables them.

This approach is spatial and can introduce noise or bias around insufficient
winner samples; MSAA improves silhouettes but does not guarantee motion stability.
For a runnable example and live controls, see the
[scene guide](../scenes/README.md#transparency-settings-and-controls).
