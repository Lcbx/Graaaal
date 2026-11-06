#include "stochastic_common.shaderlib"

<ShadingVertex> position: vec4f;
<ShadingVertex> normal: vec4f;
<ShadingVertex> uv: vec4f;
@group(2) @binding(4) var<storage, read> vertices: array<ShadingVertex>;
@group(1) @binding(0) var winners: texture_2d<u32>;
@group(1) @binding(1) var winner_color: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn shade_winner(@builtin(global_invocation_id) id: vec3u) {
	let dimensions = textureDimensions(winners);
	if any(id.xy >= dimensions) { return; }
	let pixel = vec2i(id.xy);
	let payload = textureLoad(winners, pixel, 0);
	// Empty pixels are zero samples, including pixels where all low-alpha
	// fragments lost the stochastic test. Never leave last frame's value here.
	if payload.x == 0u {
		textureStore(winner_color, pixel, vec4f(0.0));
		return;
	}
	let info = metadata[payload.x - 1u];
	let triangle = (info.x + payload.y) * 3u;
	let a = vertices[triangle];
	let b = vertices[triangle + 1u];
	let c = vertices[triangle + 2u];
	let xy = unpack2x16unorm(payload.z);
	var bary = max(vec3f(xy, 1.0-xy.x-xy.y), vec3f(0.0));
	bary /= max(bary.x+bary.y+bary.z, 1e-6);
	let position = a.position.xyz*bary.x + b.position.xyz*bary.y + c.position.xyz*bary.z;
	let normal = a.normal.xyz*bary.x + b.normal.xyz*bary.y + c.normal.xyz*bary.z;
	let uv = a.uv.xy*bary.x + b.uv.xy*bary.y + c.uv.xy*bary.z;
	let lod_face = unpack2x16float(payload.w);
	let color = shade(materials[info.y], uv, position, select(-normal, normal, lod_face.y > 0.0), lod_face.x);
	// Straight HDR color: no multiplication or division by opacity here.
	textureStore(winner_color, pixel, vec4f(color, 1.0));
}
