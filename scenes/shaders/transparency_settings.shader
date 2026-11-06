#include "utils.shaderlib"

<OverlaySettings> viewport: vec4f;
@group(0) @binding(0) var<uniform> settings: OverlaySettings;
@group(0) @binding(1) var settings_image: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let coordinates = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
		vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
	let pixel = vec2f(12.0)+coordinates[index]*settings.viewport.zw;
	return vec4f(pixel/settings.viewport.xy*vec2f(2.0, -2.0)+vec2f(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let color = textureLoad(settings_image, vec2i(position.xy)-vec2i(12), 0);
#if PARAMS.srgb_surface
	return vec4f(srgb_to_linear(color.rgb), color.a);
#else
	return color;
#endif
}
