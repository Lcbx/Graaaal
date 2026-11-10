#include "stochastic_common.shaderlib"
#include "draw_common.shaderlib"
#include "utils.shaderlib"
#include "instance_transform.shaderlib"

<ShadingVertex> position: vec4f;
<ShadingVertex> normal: vec4f;
<ShadingVertex> uv: vec4f;
@group(2) @binding(4) var<storage, read> vertices: array<ShadingVertex>;
@group(2) @binding(5) var<storage, read> shading_instances: array<MeshInstance>;
#if PARAMS.get("sample_count", 1) > 1
@group(1) @binding(0) var winners: texture_multisampled_2d<u32>;
@group(1) @binding(1) var winner_color: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(2) var winners_extra: texture_multisampled_2d<u32>;
#else
@group(1) @binding(0) var winners: texture_2d<u32>;
@group(1) @binding(1) var winner_color: texture_storage_2d<rgba16float, write>;
#endif

fn shade_payload(payload: vec4u) -> vec4f {
	// Empty pixels are zero samples, including pixels where all low-alpha
	// fragments lost the stochastic test. Never leave last frame's value here.
	if payload.x == 0u {
		return vec4f(0.0);
	}
	let instance_id = payload.x - 1u;
	if instance_id >= arrayLength(&metadata) || instance_id >= arrayLength(&shading_instances) {
		return vec4f(0.0);
	}
	let info = metadata[instance_id];
	let triangle_count = arrayLength(&vertices) / 3u;
#if PARAMS.get("material_library") is not none
	let material_count = library_material_count();
#else
	let material_count = arrayLength(&materials);
#endif
	if info.x >= triangle_count || payload.y >= triangle_count - info.x || info.y >= material_count {
		return vec4f(0.0);
	}
	let triangle = (info.x + payload.y) * 3u;
	let raw_a = vertices[triangle];
	let raw_b = vertices[triangle + 1u];
	let raw_c = vertices[triangle + 2u];
	// Deform each original vertex before interpolating, exactly as rasterization.
	let a = deform_vertex(instance_id, raw_a.position.xyz, raw_a.normal.xyz, raw_a.uv.xy);
	let b = deform_vertex(instance_id, raw_b.position.xyz, raw_b.normal.xyz, raw_b.uv.xy);
	let c = deform_vertex(instance_id, raw_c.position.xyz, raw_c.normal.xyz, raw_c.uv.xy);
	let xy = unpack2x16unorm(payload.z);
	var bary = max(vec3f(xy, 1.0-xy.x-xy.y), vec3f(0.0));
	bary /= max(bary.x+bary.y+bary.z, 1e-6);
	let inst = shading_instances[instance_id];
	let position = instance_position(inst, a.position.xyz*bary.x + b.position.xyz*bary.y + c.position.xyz*bary.z);
	let normal = instance_normal(inst, a.normal.xyz*bary.x + b.normal.xyz*bary.y + c.normal.xyz*bary.z);
	let uv = a.uv.xy*bary.x + b.uv.xy*bary.y + c.uv.xy*bary.z;
	let lod_face = unpack2x16float(payload.w);
	let color = shade_material(instance_id, get_material(instance_id), uv, position, select(-normal, normal, lod_face.y > 0.0), lod_face.x);
	// Straight HDR color: no multiplication or division by opacity here.
	return vec4f(color, 1.0);
}

@compute @workgroup_size(8, 8)
fn shade_winner(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let pixel = vec2i(id.xy);
#if PARAMS.get("sample_count", 1) > 1
	let low = textureLoad(winners, pixel, i32(id.z));
	let high = textureLoad(winners_extra, pixel, i32(id.z));
	let payload = vec4u(low.x | (low.y << 16u), low.z | (low.w << 16u),
		high.x | (high.y << 16u), high.z | (high.w << 16u));
	textureStore(winner_color, pixel, i32(id.z), shade_payload(payload));
#else
	textureStore(winner_color, pixel, shade_payload(textureLoad(winners, pixel, 0)));
#endif
}
