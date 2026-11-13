#if PARAMS.get("source_srgb_encoded", false) or not PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif
#include "ambient_occlusion.shaderlib"

<AOSettings> projection: mat4x4f;
<AOSettings> clip: vec4f;
uniform<0, 0> ao_settings: AOSettings;
texture<1, 0> scene_color: texture_2d<f32>;
sample_texture<1, 1> opaque_depth: depth;

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

@fragment
fn apply_ambient_occlusion(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	var occlusion = 0.0;
	for (var sample = 0u; sample < {{PARAMS.get("sample_count", 1)}}u; sample += 1u) {
		let depth = textureLoad(opaque_depth, pixel, i32(sample));
		occlusion += ambient_occlusion(pixel, sample, depth, ao_settings.projection, ao_settings.clip.xy);
	}
	occlusion /= {{PARAMS.get("sample_count", 1)}}.0;
	var color = textureLoad(scene_color, pixel, 0);
#if PARAMS.get("source_srgb_encoded", false)
	color = vec4f(srgb_to_linear(color.rgb), color.a);
#endif
	color = vec4f(color.rgb*occlusion, color.a);
#if not PARAMS.srgb_surface
	color = vec4f(linear_to_srgb(color.rgb), color.a);
#endif
	return color;
}
