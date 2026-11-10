# Default blue-noise tile

`64x64.npy` contains the unchanged uint16 sample values from Christoph Peters'
`Data/64_64/HDR_L_0.png` in `FreeBlueNoiseTextures.zip`. The NumPy representation
avoids an image-decoding dependency for the renderer's default texture.

`64x64_second.npy` likewise retains the uint16 samples from
`Data/64_64/HDR_L_1.png` in the same archive. Two-winner rendering lazily loads
this separate noise realization for stream B; stream A remains unchanged.
Both use the same CC0 license. `get_default_blue_noise(stream=1)` accesses B.
Custom single-tile overrides use hashed phases of that tile unless a separate
`blue_noise_second` texture is supplied. Separate sources do not guarantee
independence for every projection, hierarchy level, or overlapping surface.

Source: [Free blue noise textures](https://momentsingraphics.de/BlueNoise.html).
The original [LICENSE.txt](LICENSE.txt) and [COPYING.txt](COPYING.txt) apply (CC0).

## Loading APIs

This directory contains data, not a Python module. Useful public helpers include:

| API | Result and requirements |
| --- | --- |
| `numpyecs.Assets.load_blue_noise(path=None)` | Returns a contiguous 2-D `float32` threshold array using `(sample + 0.5) / 65536`, without sRGB conversion. The packaged NPY needs only NumPy; custom uint16 PNG loading also needs Pillow. |
| `numpyecs.TextureProcessing.get_default_blue_noise()` | Uploads thresholds to an `r32float` texture and reuses it while active on the same RenderContext device/generation. Requires an initialized GPU backend. |
| `TransparencyPass(..., blue_noise=texture)` / `MaterialRenderer(..., blue_noise=texture)` | Accepts a caller-supplied texture; omitting it uses the shared default. |
