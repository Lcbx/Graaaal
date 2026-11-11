# Stochastic transparency

[TransparencyPass.py](TransparencyPass.py) implements stochastic transparency with
one/two averaged draws or any positive number of conditional fill attempts.
Applications provide geometry, instances,
materials, camera values, targets, and command submission. The
[engine guide](../README.md#composable-transparency) documents its resource API;
the [interactive scene](../scenes/transparency_example.py) shows complete recording.

`TransparencyPass` and the example default to `noise_mode="anchored_blue"`,
`winner_count=2` and `winner_mix="average"`. MaterialRenderer inherits these
defaults. Explicit settings override them; low-level custom shader sets require
`winner_count=1`.

## Useful APIs

Import from `numpyecs.TransparencyPass`. This overview is non-exhaustive; initialize
RenderContext before constructing GPU resources.

| Class or method | Purpose |
| --- | --- |
| `TransparencyPass(...)` | Owns persistent pipelines and size-dependent targets; accepts material, vertex, texture and optional opaque inputs. |
| `resize(size, targets=...)` | Allocates or adopts targets; call before updating camera data or recording. `targets` is keyword-only. |
| `update_frame(...)`, `update_camera(camera, ...)` | Upload explicit arrays or camera-object values and return `view @ projection`; accept `frame` and `lighting` mappings. |
| `create_geometry_passes(metadata, transparent=..., instances=...)` | Returns `GeometryPasses`: `.render_shader` for DrawBatches, `.guide` for the guide override, and deferred bindings for `shade_winner`. |
| `opaque_pass`, `guide_pass`, `transmittance_pass`, `selection_pass` | Open raster passes on caller-owned commands; issue draws inside each scope. |
| `shade_winner(commands, geometry)`, `compose(commands, output)` | Record deferred shading and composition; `output` is a matching `ColorAttachment`. Neither submits work. |
| `set_opaque_inputs(color, depth)` | Uses caller-owned opaque textures; pass `None, None` to restore internal targets. |
| `set_sample_count`, `set_noise_mode`, `set_noise_scale`, `set_debug_view` | Update live sampling and diagnostic settings. |
| `set_winner_count(N)`, `set_winner_mix(mode)` | Positive uint32 draw count and `"average"`/`"fill"` mode; averaging supports 1 or 2. Default: 2 in average mode. |
| `set_reconstruction("fragment" or "compute")` | Select original filtering or experimental shared-memory compute; fragment remains default. |
| `set_reconstruction_passes(N)` | Stack a positive uint32 number of bilateral passes; default 1. A 1x1 kernel bypasses all passes. |
| `set_early_b_rejection(bool)` | Experimental per-sample stencil rejection for fill B; default off and requires `depth32float-stencil8`. |
| `BilateralFilter`, `set_bilateral_filter` | Configure reconstruction footprint and tolerances. |
| `GrazingFade`, `set_grazing_fade` | Configure angular opacity fade; default enabled from 83 to 86 degrees. |
| `TransparencyTargets` | Describes caller-owned color, depth, guide, transmittance and winner textures. |

## Recording and settings

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
transparency.set_noise_mode("anchored_blue")
transparency.set_winner_mix("fill")
transparency.set_winner_count(4)
transparency.set_reconstruction_passes(2)
```

`winner_mix="average"` supports one or two draws; `"fill"` supports any positive
uint32 draw count with `object_hash` or `anchored_blue` and the default noise
sources. Switch back to at most two draws before selecting average mode.
`set_early_b_rejection(True)` optionally requires the
`depth32float-stencil8` GPU feature for fill mode.
`set_reconstruction("fragment"|"compute")` chooses the implementation; the
fragment path is the default. `set_reconstruction_passes(N)` stacks N filters;
a 1x1 kernel bypasses them.

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
passes require `reconstruction_pair`, and more than two require
`reconstruction_pair_scratch` (`rgba16float`).

For custom material shading or geometry deformation, use `material_source`, a
matching `geometry_shader`/`shading_shader` pair, or the extra resource arguments
to `create_geometry_passes()`. Custom shader sets support only one winner unless
explicitly implemented for additional draws and the selected MSAA ABI.

## Noise and opacity controls

`set_noise_mode(name)` accepts:

| Name | Use |
| --- | --- |
| `anchored_blue` (default) | Pixel-lattice anchored noise without material UVs |
| `object_hash` | Object-space hash; useful with overlapping or missing UVs |
| `uv_blue_lerp`, `uv_blue_pick` | UV blue-noise variants |
| `uv_bayer` | Diagnostic ordered-noise comparison |
| `uv_blue_correlated` | UV correlated hierarchy |
| `triplanar_blue`, `triplanar_correlated` | Object-space triplanar variants |

`set_noise_scale(pixels)` adjusts the positive, finite 1x noise-cell target;
4x MSAA uses half that target. `set_noise_anchor(pixels=24, subdivisions=4)`
configures anchored-blue grain. `set_noise_footprint_balancing(True)` enables
an optional UV-only shader feature, and `set_noise_footprint_balance(a)` sets
its 0–1 derivative balance (default 0). Other modes ignore this control.
Distinct overlapping instances should use distinct stable surface IDs.

`GrazingFade(enabled=True, start_angle=83, end_angle=86)` fades transparent
opacity near a silhouette. Pass it to the constructor as `grazing_fade=` or
call `set_grazing_fade(...)`. Require `0 <= start_angle < end_angle <= 90`.
MASK and BLEND geometry can share the transparent passes; OPAQUE geometry uses
the opaque pass. No temporal accumulation is performed.

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
For shader formulas, experiments, measured results, and design rationale see
[implementation work](../IMPLEMENTATION_WORK.md#stochastic-transparency-implementation-and-validation-record)
and the [dated multiple-winner report](transparency_multiple_winners.md).
