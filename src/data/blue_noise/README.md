# Default blue-noise tile

`64x64.npy` contains the unchanged uint16 sample values from Christoph Peters'
`Data/64_64/HDR_L_0.png` in `FreeBlueNoiseTextures.zip`. The NumPy representation avoids an
image-decoding dependency when loading the renderer's default texture.

Source: [Free blue noise textures](https://momentsingraphics.de/BlueNoise.html).
The accompanying original LICENSE.txt and COPYING.txt apply (CC0).

The loader converts each sample to its quantization-bin midpoint in (0, 1),
without sRGB conversion, and uploads an R32float texture.
