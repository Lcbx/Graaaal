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
| Transparent depth guide | `guide_pass()` | Nearest nonzero-opacity transparent depth, independent of stochastic acceptance |
| Transmittance | `transmittance_pass()` | RGB/scalar transmittance, initialized to one |
| Winner selection | `selection_pass()` | Nearest accepted fragment and its payload, using separate writable depth |
| Deferred shading | `shade_winner()` | Straight HDR winner color; empty pixels become zero |
| Reconstruction/composition | `compose()` | Filtered transparency over the opaque color, tone-mapped for output |

The guide pass rejects opaque occlusion and fully transparent texels while
retaining low nonzero opacity. Transmittance tests opaque depth without writing
it. Selection clears its own depth and explicitly rejects fragments behind
opaque geometry. Transparent geometry is rasterized three times: guide,
transmittance, and selection. The technique requires no transparent sorting.

## Acceptance and winner data

For fragment opacity `alpha_rgb`, scalar acceptance probability is
`p = max(alpha_rgb)`. Multiplicative blending accumulates:

```text
T_rgb    *= 1 - alpha_rgb
T_scalar *= 1 - p
```

Selection compares `p` with a blue-noise threshold at the pixel, offset by a hash
of stable surface ID and local triangle ID. Fragments with `u >= p` are discarded;
hardware depth testing/writes retain the nearest accepted fragment. The noise
tile is static, with no per-frame randomization or temporal accumulation. Its
[source and license](../scenes/resources/blue_noise/README.md) are bundled.
Pixel-space sampling can swim when the camera or geometry moves.

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

## Reconstruction

In linear HDR space, define `A_rgb = 1 - T_rgb` and `A_s = 1 - T_scalar`.
For pixel `p` and its 3x3 or 5x5 neighborhood:

```text
C_trans(p) = A_rgb(p) * sum_q(w(p,q) * C_winner(q))
                     / max(sum_q(w(p,q) * A_s(q)), epsilon)

HDR(p) = opaque_color(p) * T_rgb(p) + C_trans(p)
```

No-winner pixels contribute zero color and still contribute their aggregate
acceptance probability to the denominator. Skipping them would bias low-opacity
regions brighter. Cross-bilateral weights use pixel distance, opaque depth,
nearest-transparent depth, and RGBA transmittance. Composition applies
`HDR / (1 + HDR)` and encodes sRGB only when the output format does not do so.

`BilateralFilter` settings update uniforms without rebuilding pipelines/targets:

| Setting | Meaning |
| --- | --- |
| `kernel_size` | 3 or 5 pixels per side; default 5 |
| `spatial_sigma`, `far_spatial_sigma` | Pixel-distance standard deviation |
| `depth_sigma`, `far_depth_sigma` | Relative view-depth tolerance |
| `min_depth_sigma` | Minimum depth tolerance in scene units |
| `transmittance_sigma`, `far_transmittance_sigma` | RGBA transmittance tolerance |
| `adaptive`, `filter_near`, `filter_far` | Enable/interpolate near/far settings using view depth |
| `far_transmittance_weight` | Remaining transmittance edge weight at the far endpoint, in [0,1] |

Sigmas must be finite and positive, and distances satisfy
`0 <= filter_near < filter_far`. Adaptation uses a smoothstep of the nearer opaque
or transparent-guide depth. Both depth constraints remain active when the
transmittance edge weight fades out. Looser tolerances increase support but can
blur across overlapping surfaces; empty neighborhoods remain zero. View depth
is a proxy for feature size, not a measurement of individual leaves or particles.

## Diagnostic views and limits

Call `set_debug_view(name)`:

- `final`: tone-mapped composition.
- `transmittance`: scalar transmittance as grayscale, without reconstruction or tone mapping.
- `winner`: unfiltered, tone-mapped winner color.
- `reconstruction`: tone-mapped transparent contribution without the opaque background.
- `filter_support`: `(sum(w)**2 / sum(w*w)) / kernel_area`, including no-winner samples.

If transmittance is stable while reconstruction flickers, inspect winner variance
and filter support. Bright support means substantial spatial weight, not many
accepted winners. Smaller tolerances can leave too few contributing samples.

A single winner is noisy; spatial reconstruction introduces bias and can blur
edges. Using `max(alpha_rgb)` is an approximation for overlapping layers with
differing RGB opacities. The technique provides no temporal accumulation,
additive-material path, or multiple-winner storage. Custom material code or a
matched geometry/shading shader pair can be supplied under the existing payload
and shared-address-space contracts.
