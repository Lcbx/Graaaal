#set RECONSTRUCT_MODE = "pair" if PARAMS.get("filter_pairs", false) else "color"
#include "stochastic_reconstruction.shaderlib"

#if PARAMS.get("sample_count", 1) > 1
texture<2, 0> filter_output: texture_storage_2d_array<rgba16float, write>;
#else
texture<2, 0> filter_output: texture_storage_2d<rgba16float, write>;
#endif

var<workgroup> needs_cache: atomic<u32>;
var<workgroup> cache_flag: u32;

fn store_filtered(pixel: vec2i, sample: u32, result: vec4f) {
#if PARAMS.get("sample_count", 1) > 1
	textureStore(filter_output, pixel, i32(sample), result);
#else
	textureStore(filter_output, pixel, result);
#endif
}

@compute @workgroup_size(8, 8)
fn reconstruct(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32,
	@builtin(workgroup_id) group: vec3u) {
	let dimensions = vec2i(textureDimensions(transmittance));
	let pixel = vec2i(id.xy);
	let valid = all(pixel < dimensions);
	if local == 0u { atomicStore(&needs_cache, 0u); }
	workgroupBarrier();
	if valid {
		if 1.0-sample_texture_load(transmittance, pixel, id.z).a > 1e-6 {
			atomicOr(&needs_cache, 1u);
			if needs_reconstruction(pixel, id.z) { atomicOr(&needs_cache, 2u); }
		}
	}
	workgroupBarrier();
	if local == 0u { cache_flag = atomicLoad(&needs_cache); }
	// Uniform load makes the barrier-containing branch uniform, including edge
	// workgroups with out-of-bounds invocations. No thread returns before it.
	let load_halo = workgroupUniformLoad(&cache_flag);
	// Presentation checks current transmittance before reading the intermediate,
	// so empty tiles need neither cache loads nor output writes/clears.
	if (load_halo & 1u) == 0u {
#if PARAMS.get("filter_pairs", false)
		if valid { store_filtered(pixel, id.z, vec4f(0.0)); }
#endif
		return;
	}
	let radius = bilateral.kernel.x;
	let side = 8u+2u*radius;
	let origin = vec2i(group.xy*8u)-vec2i(i32(radius));
	if (load_halo & 2u) != 0u {
		for (var index = local; index < side*side; index += 64u) {
			let q = origin+vec2i(i32(index%side), i32(index/side));
			if all(q >= vec2i(0)) && all(q < dimensions) {
				let data = load_filter_data(q, id.z);
				cached_transmission[index] = data.transmission;
				cached_winner[index] = data.winner;
				cached_guide[index] = data.guide;
				cached_depths[index] = data.depths;
			}
		}
		workgroupBarrier();
	}
	if !valid { return; }
	if 1.0-sample_texture_load(transmittance, pixel, id.z).a <= 1e-6 {
#if PARAMS.get("filter_pairs", false)
		store_filtered(pixel, id.z, vec4f(0.0));
#endif
		return;
	}
#if PARAMS.get("filter_pairs", false)
	let result = reconstruct_pair(pixel, id.z, origin, side);
#else
	let result = reconstruct_transparency(pixel, id.z, origin, side);
#endif
	store_filtered(pixel, id.z, result);
}
