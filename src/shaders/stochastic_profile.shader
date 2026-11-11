#set RECONSTRUCT_MODE = "profile"
#include "stochastic_reconstruction.shaderlib"
storage<2, 0, read_write> counters: array<atomic<u32>>;
var<workgroup> tile_filter: atomic<u32>;

@compute @workgroup_size(8, 8)
fn classify(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32) {
	if local == 0u { atomicStore(&tile_filter, 0u); }
	workgroupBarrier();
	if all(id.xy < textureDimensions(transmittance)) {
		let pixel = vec2i(id.xy);
		let t = textureLoad(transmittance, pixel, i32(id.z));
		let winner = winner_at(pixel, id.z);
		let visible = 1.0-t.a > 1e-6;
		let forced = forced_at(pixel, id.z);
		var certain = false;
#if PARAMS.get("guide_available", true)
		certain = textureLoad(transparent_guide, pixel, i32(id.z)).x == 1.0 && !forced && visible;
#endif
		let stochastic = visible && !forced && !certain;
		let filtering = stochastic && bilateral.kernel.x != 0u;
		atomicAdd(&counters[0], 1u);
		atomicAdd(&counters[1], select(0u, 1u, !visible));
		atomicAdd(&counters[2], select(0u, 1u, forced));
		atomicAdd(&counters[3], select(0u, 1u, certain));
		atomicAdd(&counters[4], select(0u, 1u, stochastic));
		atomicAdd(&counters[5], select(0u, 1u, winner.a > 0.0));
		atomicAdd(&counters[6], select(0u, 1u, visible && winner.a <= 0.0));
		var first_empty = winner.a <= 0.0;
#if PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
		first_empty = textureLoad(winner_source, pixel, i32(id.z), 0).r != 1u;
#else
		first_empty = textureLoad(winner_source, pixel, 0).r != 1u;
#endif
#elif PARAMS.get("winner_count", 1) == 2
#if PARAMS.get("sample_count", 1) > 1
		first_empty = textureLoad(first_color, pixel, i32(id.z), 0).a <= 0.0;
#else
		first_empty = textureLoad(first_color, pixel, 0).a <= 0.0;
#endif
#endif
		atomicAdd(&counters[7], select(0u, 1u, visible && first_empty));
		atomicAdd(&counters[8], select(0u, 1u, filtering));
		if filtering { atomicOr(&tile_filter, 1u); }
	}
	workgroupBarrier();
	if local == 0u && atomicLoad(&tile_filter) != 0u { atomicAdd(&counters[9], 1u); }
}
