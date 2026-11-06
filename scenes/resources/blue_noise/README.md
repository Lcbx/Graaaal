# Tileable blue noise

`HDR_L_0.png` is Christoph Peters' original 64×64 single-channel, 16-bit blue
noise texture, copied unchanged from `Data/64_64/HDR_L_0.png` in
`FreeBlueNoiseTextures.zip`.

Source: [Free blue noise textures](https://momentsingraphics.de/BlueNoise.html).
The original `LICENSE.txt` and `COPYING.txt` accompany the texture (CC0).

The transparency scene loads the values as linear data, maps 16-bit bins to
midpoint thresholds in (0,1), and repeats the tile with integer pixel wrapping.
Each surface gets a deterministic offset; there is no per-frame randomization.
The lookup is in pixel space, so the pattern is not anchored to surface positions.
