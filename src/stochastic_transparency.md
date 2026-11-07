Stochastic order-independent transparency

# Stochastic Order-Independent Transparency — Single Winner

## 1. Opaque pass

Render opaque geometry normally:

* `main_color`: HDR opaque color

* `depth_opq`: opaque depth buffer

## 2. Transparent transmittance pass

Render transparent geometry unsorted, depth-tested against `depth_opq`, without depth writes.

For each fragment:

1. Compute RGB opacity `alpha_rgb`.

2. Define scalar acceptance probability `p = max(alpha_rgb)`.

3. Accumulate transmittance through multiplicative blending:

   ```
   T_rgb *= 1 - alpha_rgb
   T_scalar *= 1 - p
   ```

Store both in one `RGBA16Float` attachment initialized to 1, using `srcFactor = ZERO`, `dstFactor = SRC_COLOR` (and corresponding alpha factors).

Optionally accumulate nearest transparent depth for spatial filtering.

## 3. Stochastic winner selection

Re-rasterize transparent geometry without expensive shading.

1. Compute a deterministic threshold:

   `u = BlueNoise(pixel, hash(InstanceID, PrimitiveID))`

2. Discard fragments where `u >= p`.

3. Use hardware depth testing and writes to retain the nearest accepted fragment.

4. Write its `InstanceID`, `PrimitiveID`, and packed barycentrics into a G-buffer.

Use a separate selection depth buffer, initialized from opaque depth or with opaque occlusion tested explicitly.

Hardware depth testing ensures the winning payload and depth remain consistent, without atomic races.

## 4. Deferred compute shading

Dispatch compute over pixels with transparency:

1. Fetch the winning triangle and instance.

2. Reconstruct/interpolate vertex attributes using barycentrics.

3. Evaluate material and lighting only for the winner.

4. Output `C_winner`, or zero if no winner exists.

Use straight, non-premultiplied fragment color. No opacity division is required here.

## 5. Spatial reconstruction and composition

Define:

Argb=1−Trgb,As=1−TscalarA_{rgb}=1-T_{rgb},\qquad A_s=1-T_{scalar}Argb=1−Trgb,As=1−Tscalar

Apply a 3×3 or 5×5 cross-bilateral reconstruction:

Ctrans,p=Argb,p∑q∈N(p)wpqCwinner,qmax⁡(∑q∈N(p)wpqAs,q,ϵ)C_{\mathrm{trans},p} = A_{rgb,p} \frac{\sum_{q\in N(p)}w_{pq}C_{\mathrm{winner},q}} {\max(\sum_{q\in N(p)}w_{pq}A_{s,q},\epsilon)}Ctrans,p=Argb,pmax(∑q∈N(p)wpqAs,q,ϵ)∑q∈N(p)wpqCwinner,q

* Include pixels with no winner as zero-valued samples.

* Guide filtering using opaque depth, optional transparent depth, and transmittance.

* Filter in linear HDR space.

Final composition:

```
final_color = main_color * T_rgb + C_trans
```

## Pipeline

```
Opaque pass
    ├─ main_color
    └─ depth_opq
          ↓
Transparent transmittance pass
    ├─ T_rgb
    └─ T_scalar
          ↓
Stochastic winner selection
    ├─ selection depth
    └─ winner G-buffer
          ↓
Deferred compute shading
    └─ C_winner
          ↓
Spatial reconstruction + composition
    └─ final_color
```

## Implementation notes

* Correctness: Unbiased stochastic color estimator for scalar opacity before spatial filtering; approximate for overlapping layers with differing RGB opacities.

* Temporal stability: Static surface-decorrelated blue noise, without temporal accumulation. Camera movement may still produce stochastic swimming.

* Additive materials: Accumulate separately using additive blending.

* Low-alpha particles: Do not omit their stochastic contribution unless accepting additional color bias.

* Performance: Two inexpensive transparent rasterization passes, one deferred shading pass, one reconstruction pass. No sorting, atomic winner buffers, or linked lists.

Extension: The single winner can be generalized to a fixed number KKK of winners using bounded depth peeling or per-pixel top-KKK storage, with deferred shading of the retained layers.
