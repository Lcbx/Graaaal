#set MATERIAL_STAGE = "shading"
#include "stochastic_common.shaderlib"
#include "draw_common.shaderlib"
#include "utils_transform.shaderlib"
#include "instance_transform.shaderlib"

<ShadingVertex> position: vec4f;
<ShadingVertex> normal: vec4f;
<ShadingVertex> uv: vec4f;
storage<2, 4, read> vertices: array<ShadingVertex>;
storage<2, 5, read> shading_instances: array<MeshInstance>;
sample_texture<1, 0> winners: u32;
#if PARAMS.get("sample_count", 1) > 1
texture<1, 1> winner_color: texture_storage_2d_array<rgba16float, write>;
#else
texture<1, 1> winner_color: texture_storage_2d<rgba16float, write>;
#endif
#if PARAMS.get("sample_count", 1) > 1
texture<1, 2> winners_extra: texture_multisampled_2d<u32>;
#endif

#if PARAMS.get("force_single_contributor", false) or PARAMS.get("winner_mix", "average") == "fill"
sample_texture<1, 6> shade_transmittance: f32;
#endif
#if PARAMS.get("force_single_contributor", false)
sample_texture<1, 7> contributor_count: f32;
#endif

#if PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
texture<1, 8> winner_source: texture_storage_2d_array<r32uint, write>;
#else
texture<1, 8> winner_source: texture_storage_2d<r32uint, write>;
#endif
#endif

#if PARAMS.get("winner_count", 1) > 1 or PARAMS.get("winner_mix", "average") == "fill"
<WinnerSettings> info: vec4u;
uniform<1, 5> winner_settings: WinnerSettings;
#endif
#if PARAMS.get("winner_count", 1) == 2 and PARAMS.get("winner_mix", "average") == "average"
sample_texture<1, 3, layered> first_color: f32;
sample_texture<1, 4> winner_guide: f32;
#endif

#if PARAMS.get("complemented_winners", false)
#if PARAMS.get("mix_only", false)
texture<1, 9> first_payloads: texture_2d_array<u32>;
#else
texture<1, 9> first_payloads: texture_storage_2d_array<rgba32uint, write>;
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
	let transmission = sample_texture_load(shade_transmittance, pixel, sample).a;
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
		weight = 1.0-sample_texture_load(shade_transmittance, pixel, sample).a;
	}
#endif
	return weight*vec4f(color, 1.0);
}

fn winner_payload(id: vec3u) -> vec4u {
	let pixel = vec2i(id.xy);
#if PARAMS.get("sample_count", 1) > 1
	let low = sample_texture_load(winners, pixel, id.z);
	let high = textureLoad(winners_extra, pixel, i32(id.z));
	return vec4u(low.x | (low.y << 16u), low.z | (low.w << 16u),
		high.x | (high.y << 16u), high.z | (high.w << 16u));
#else
	return sample_texture_load(winners, pixel, id.z);
#endif
}

fn store_winner(id: vec3u, color: vec4f) {
#if PARAMS.get("sample_count", 1) > 1
	textureStore(winner_color, vec2i(id.xy), i32(id.z), color);
#else
	textureStore(winner_color, vec2i(id.xy), color);
#endif
}

#if PARAMS.get("keyed_tail", false)
texture<1, 10> key_payload_a: texture_2d<u32>;
texture<1, 11> key_payload_b: texture_2d<u32>;
texture<1, 12> tail_coverage: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn shade_keyed(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(key_payload_a)) { return; }
	let pixel = vec2i(id.xy);
	let a = textureLoad(key_payload_a, pixel, 0);
	let b = textureLoad(key_payload_b, pixel, 0);
	let ca = shade_payload(a, pixel, 0u);
	var cb = ca;
	if any(a != b) { cb = shade_payload(b, pixel, 0u); }
	let den = min(1.0, textureLoad(tail_coverage, pixel, 0).r);
	store_winner(id, vec4f(den*0.5*(ca.rgb+cb.rgb), den));
}
#endif

#if PARAMS.get("banded_tail", false)
#include "banded_sampling.shaderlib"
<BandCount> total: u32;
<BandCount> running: u32;
storage<1, 11, read> band_counts: array<BandCount>;
texture<1, 10> band_payloads: texture_2d_array<u32>;
#if PARAMS.get("banded_proxy", false)
sample_texture<1, 12> band_proxy: f32;
#endif

@compute @workgroup_size(8, 8)
fn shade_bands(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(winner_color);
#if PARAMS.get("banded_subset", false)
	let origin = id.xy*2u;
	if any(origin >= size) { return; }
	let selected = band_selected_pixel(id.xy);
	for (var y = 0u; y < 2u; y += 1u) {
		for (var x = 0u; x < 2u; x += 1u) {
			let q = origin+vec2u(x, y);
			if all(q < size) && any(q != selected) { store_winner(vec3u(q, id.z), vec4f(0.0)); }
		}
	}
	if any(selected >= size) { return; }
	let actual_id = vec3u(selected, id.z);
	let payload_pixel = vec2i(id.xy);
#else
	if any(id.xy >= size) { return; }
	let actual_id = id;
	let payload_pixel = vec2i(id.xy);
#endif
	let pixel = vec2i(actual_id.xy);
	let a = textureLoad(band_payloads, payload_pixel, i32(id.z*2u), 0);
	let b = textureLoad(band_payloads, payload_pixel, i32(id.z*2u+1u), 0);
	let ca = shade_payload(a, pixel, id.z);
	var cb = ca;
	if any(a != b) { cb = shade_payload(b, pixel, id.z); }
	let index = (actual_id.y*size.x+actual_id.x)*{{PARAMS.get("sample_count", 1)}}u+id.z;
	let s = f32(band_counts[index].total)/65536.0;
	let den = min(1.0, s);
	var residual = 0.5*(ca.rgb+cb.rgb);
#if PARAMS.get("banded_proxy", false)
	residual -= sample_texture_load(band_proxy, pixel, id.z).rgb/max(1.0, s);
#endif
	store_winner(actual_id, vec4f(select(vec3f(0.0), residual, den > 0.0), den));
}
#elif not PARAMS.get("keyed_tail", false)
#if not PARAMS.get("mix_only", false)
@compute @workgroup_size(8, 8)
fn shade_winner(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let payload = winner_payload(id);
#if PARAMS.get("complemented_winners", false)
	textureStore(first_payloads, vec2i(id.xy), i32(id.z), payload);
#endif
	let shaded = shade_payload(payload, vec2i(id.xy), id.z);
	store_winner(id, shaded);
#if PARAMS.get("winner_mix", "average") == "fill"
	store_source(id, select(0u, 1u, shaded.a > 0.0));
#endif
}
#endif

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

#if PARAMS.get("winner_count", 1) == 2 and PARAMS.get("winner_mix", "average") == "average" and (not PARAMS.get("complemented_winners", false) or PARAMS.get("mix_only", false))
@compute @workgroup_size(8, 8)
fn shade_mix(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	let pixel = vec2i(id.xy);
	let first = sample_texture_load(first_color, pixel, id.z, layered=true);
#if PARAMS.get("force_single_contributor", false)
	if sample_texture_load(shade_transmittance, pixel, id.z).a < 1.0 &&
		sample_texture_load(contributor_count, pixel, id.z).r == 1.0 && first.a > 0.0 {
		store_winner(id, first);
		return;
	}
#endif
	if winner_settings.info.y != 0u {
		if sample_texture_load(winner_guide, pixel, id.z).x == 1.0 {
			store_winner(id, first);
			return;
		}
	}
	// Pool unnormalized colors and validity; do not normalize each stream.
#if PARAMS.get("complemented_winners", false)
	let payload = winner_payload(id);
	let previous = textureLoad(first_payloads, pixel, i32(id.z), 0);
	if payload.x != 0u && all(payload == previous) {
		store_winner(id, first);
		return;
	}
#endif
	store_winner(id, 0.5*(first+shade_payload(winner_payload(id), pixel, id.z)));
}
#endif
#endif
