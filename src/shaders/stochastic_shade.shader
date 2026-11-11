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

#if PARAMS.get("force_single_contributor", false) or PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
@group(1) @binding(6) var shade_transmittance: texture_multisampled_2d<f32>;
#else
@group(1) @binding(6) var shade_transmittance: texture_2d<f32>;
#endif
#endif
#if PARAMS.get("force_single_contributor", false)
#if PARAMS.get("sample_count", 1) > 1
@group(1) @binding(7) var contributor_count: texture_multisampled_2d<f32>;
#else
@group(1) @binding(7) var contributor_count: texture_2d<f32>;
#endif
#endif

#if PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
@group(1) @binding(8) var winner_source: texture_storage_2d_array<r32uint, write>;
#else
@group(1) @binding(8) var winner_source: texture_storage_2d<r32uint, write>;
#endif
#endif

#if PARAMS.get("winner_count", 1) > 1 or PARAMS.get("winner_mix", "average") == "fill"
<WinnerSettings> info: vec4u;
@group(1) @binding(5) var<uniform> winner_settings: WinnerSettings;
#endif
#if PARAMS.get("winner_count", 1) == 2 and PARAMS.get("winner_mix", "average") == "average"
#if PARAMS.get("sample_count", 1) > 1
@group(1) @binding(3) var first_color: texture_2d_array<f32>;
@group(1) @binding(4) var winner_guide: texture_multisampled_2d<f32>;
#else
@group(1) @binding(3) var first_color: texture_2d<f32>;
@group(1) @binding(4) var winner_guide: texture_2d<f32>;
#endif
#endif

fn shade_payload(payload: vec4u, pixel: vec2i, sample: u32) -> vec4f {
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
	let lod_face = unpack2x16float(payload.w & ~WINNER_FORCED_BIT);
	let color = shade_material(instance_id, get_material(instance_id), uv, position, select(-normal, normal, lod_face.y > 0.0), lod_face.x);
	var weight = 1.0;
#if PARAMS.get("winner_mix", "average") == "fill"
	// Each attempt uses the total-budget geometric sum, including attempt A.
	// This remains stable near T=1 and avoids cancellation in (1-T)/(1-T^N).
	let transmission = textureLoad(shade_transmittance, pixel, i32(sample)).a;
	var total = 0.0;
	var prefix = 1.0;
	var block_sum = 1.0;
	var block_power = transmission;
	var remaining = winner_settings.info.z;
	while remaining > 0u {
		if (remaining & 1u) != 0u {
			total += prefix*block_sum;
			prefix *= block_power;
		}
		block_sum *= 1.0+block_power;
		block_power *= block_power;
		remaining >>= 1u;
	}
	weight = 1.0/total;
#endif
#if PARAMS.get("force_single_contributor", false)
	if (payload.w & WINNER_FORCED_BIT) != 0u {
		weight = 1.0-textureLoad(shade_transmittance, pixel, i32(sample)).a;
	}
#endif
	return weight*vec4f(color, 1.0);
}

fn winner_payload(id: vec3u) -> vec4u {
	let pixel = vec2i(id.xy);
#if PARAMS.get("sample_count", 1) > 1
	let low = textureLoad(winners, pixel, i32(id.z));
	let high = textureLoad(winners_extra, pixel, i32(id.z));
	return vec4u(low.x | (low.y << 16u), low.z | (low.w << 16u),
		high.x | (high.y << 16u), high.z | (high.w << 16u));
#else
	return textureLoad(winners, pixel, 0);
#endif
}

fn store_winner(id: vec3u, color: vec4f) {
#if PARAMS.get("sample_count", 1) > 1
	textureStore(winner_color, vec2i(id.xy), i32(id.z), color);
#else
	textureStore(winner_color, vec2i(id.xy), color);
#endif
}

@compute @workgroup_size(8, 8)
fn shade_winner(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let payload = winner_payload(id);
	let shaded = shade_payload(payload, vec2i(id.xy), id.z);
	store_winner(id, shaded);
#if PARAMS.get("winner_mix", "average") == "fill"
	store_source(id, select(0u, 1u, shaded.a > 0.0));
#endif
}

#if PARAMS.get("winner_mix", "average") == "fill"
fn store_source(id: vec3u, source: u32) {
#if PARAMS.get("sample_count", 1) > 1
	textureStore(winner_source, vec2i(id.xy), i32(id.z), vec4u(source));
#else
	textureStore(winner_source, vec2i(id.xy), vec4u(source));
#endif
}

@compute @workgroup_size(8, 8)
fn shade_fill(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let payload = winner_payload(id);
	// A's full-screen dispatch clears every empty pixel at frame start. B must
	// retain A where its payload is empty; no storage texture read is required.
	if payload.x == 0u { return; }
	let shaded = shade_payload(payload, vec2i(id.xy), id.z);
	if shaded.a <= 0.0 { return; }
	store_winner(id, shaded);
	store_source(id, winner_settings.info.x+1u);
}
#endif

#if PARAMS.get("winner_count", 1) == 2 and PARAMS.get("winner_mix", "average") == "average"
@compute @workgroup_size(8, 8)
fn shade_mix(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let pixel = vec2i(id.xy);
#if PARAMS.get("sample_count", 1) > 1
	let first = textureLoad(first_color, pixel, i32(id.z), 0);
#else
	let first = textureLoad(first_color, pixel, 0);
#endif
#if PARAMS.get("force_single_contributor", false)
	if textureLoad(shade_transmittance, pixel, i32(id.z)).a < 1.0 &&
		textureLoad(contributor_count, pixel, i32(id.z)).r == 1.0 && first.a > 0.0 {
		store_winner(id, first);
		return;
	}
#endif
	if winner_settings.info.y != 0u {
		if textureLoad(winner_guide, pixel, i32(id.z)).x == 1.0 {
			store_winner(id, first);
			return;
		}
	}
	// Pool unnormalized colors and validity; do not normalize each stream.
	store_winner(id, 0.5*(first+shade_payload(winner_payload(id), pixel, id.z)));
}
#endif
