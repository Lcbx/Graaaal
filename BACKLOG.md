# Engine backlog and explicit exclusions

* add proper monitoring and logging
* make SSAO respect normals
* add FXAA
* implement Forward+ / tiled shading
* add compute light binning
* implement ramp / toon shading
* implement shadow maps
* make wgpu-py Canvas use msaa for main buffer (?)
* implement grid-based ambient probes
* experiment with a kuwahara material
* add post-process depth/normal edge detection outlines
* ensure depth prepass only has close / big, not alpha-tested surfaces
* implement gpu animation using custom vertex attributes and animation texture atlas
* use openAL for binaural audio
* use bullet for physics (at first ; maybe migrate to ctypes & jolt or box3d later)
* add a particle system
* implement parallax texture mapping
* implement triplanar mapping
* networking : use udp hole punching (users must exchange IP) or steamPeer

not doing :
- Meshlets / task shaders
- Dynamic GI / radiance cascades
- Inverted-hull outlines
- TAA, DLSS/FSR upscaling