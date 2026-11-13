#set RECONSTRUCT_MODE = "cascade"
#include "stochastic_reconstruction.shaderlib"

<PyramidSettings> info: vec4u;
<PyramidSettings> confidence: vec4f;
uniform<2, 0> pyramid_settings: PyramidSettings;
texture<2, 1> cascade_pair: texture_2d_array<f32>;
texture<2, 2> cascade_result: texture_2d_array<f32>;
texture<3, 0> output_pair: texture_storage_2d_array<rgba32float, write>;
texture<3, 1> output_result: texture_storage_2d_array<rgba32float, write>;

fn pair_at(pixel: vec2i, sample: u32) -> vec4f {
	if pyramid_settings.info.x == 0u {
		return vec4f(winner_at(pixel, sample).rgb, max(1.0-sample_texture_load(transmittance, pixel, sample).a, 0.0));
	}
	return textureLoad(cascade_pair, pixel, i32(sample), 0);
}

fn opaque_reciprocal(pixel: vec2i, sample: u32) -> f32 {
	let p = clamp(pixel, vec2i(0), vec2i(textureDimensions(opaque_depth))-vec2i(1));
	return 1.0/linear_depth(sample_texture_load(opaque_depth, p, sample), frame.clip.x, frame.clip.y);
}

fn opaque_slope(pixel: vec2i, sample: u32) -> vec2f {
	let z = opaque_reciprocal(pixel, sample);
	let a = vec2f(z-opaque_reciprocal(pixel-vec2i(1, 0), sample), z-opaque_reciprocal(pixel-vec2i(0, 1), sample));
	let b = vec2f(opaque_reciprocal(pixel+vec2i(1, 0), sample)-z, opaque_reciprocal(pixel+vec2i(0, 1), sample)-z);
	// A one-sided discontinuity must not become a plane across a silhouette.
	return select(vec2f(0.0), select(b, a, abs(a) < abs(b)), a*b > vec2f(0.0));
}

fn cascade_edge(pixel: vec2i, q: vec2i, sample: u32) -> f32 {
	let tp = sample_texture_load(transmittance, pixel, sample);
	let tq = sample_texture_load(transmittance, q, sample);
	let ap = max(1.0-tp.a, 0.0);
	let aq = max(1.0-tq.a, 0.0);
	if ap <= 0.0 || aq <= 0.0 { return 0.0; }
	let delta = vec2f(q-pixel);
	let zp = linear_depth(sample_texture_load(transparent_depth, pixel, sample), frame.clip.x, frame.clip.y);
	let zq = linear_depth(sample_texture_load(transparent_depth, q, sample), frame.clip.x, frame.clip.y);
	let gp = sample_texture_load(transparent_guide, pixel, sample).yz;
	let gq = sample_texture_load(transparent_guide, q, sample).yz;
	let predicted_q = 1.0/max(1.0/zp+dot(gp, delta), 1e-6);
	let predicted_p = 1.0/max(1.0/zq-dot(gq, delta), 1e-6);
	let dz = max(abs(zq-predicted_q), abs(zp-predicted_p))/max(min(zp, zq)*bilateral.sigmas.y, bilateral.sigmas.w);
	let op = opaque_reciprocal(pixel, sample);
	let oq = opaque_reciprocal(q, sample);
	let predicted_oq = 1.0/max(op+dot(opaque_slope(pixel, sample), delta), 1e-6);
	let predicted_op = 1.0/max(oq-dot(opaque_slope(q, sample), delta), 1e-6);
	let od = max(abs(1.0/oq-predicted_oq), abs(1.0/op-predicted_op))/max(min(1.0/op, 1.0/oq)*bilateral.sigmas.y, bilateral.sigmas.w);
	// A factor-of-three acceptance difference is one sigma by default.
	let da = log((aq+1e-6)/(ap+1e-6))/log(3.0);
	// Retain a separate chromatic transmission constraint for colored stacks.
	let tint = (tq.rgb-vec3f(tq.a))-(tp.rgb-vec3f(tp.a));
	return exp(-0.5*(dz*dz+od*od+da*da)-dot(tint, tint)/(2.0*bilateral.sigmas.z*bilateral.sigmas.z));
}

@compute @workgroup_size(8, 8)
fn cascade(@builtin(global_invocation_id) id: vec3u) {
	let dimensions = vec2i(textureDimensions(transmittance));
	let pixel = vec2i(id.xy);
	if any(pixel >= dimensions) { return; }
	let level = pyramid_settings.info.x;
	let step = i32(1u << level);
	var acc = vec4f(0.0);
	for (var j = -1; j <= 1; j += 1) {
		for (var i = -1; i <= 1; i += 1) {
			let q = pixel+vec2i(i, j)*step;
			if any(q < vec2i(0)) || any(q >= dimensions) { continue; }
			let weight = 0.25*select(0.5, 1.0, i == 0)*select(0.5, 1.0, j == 0)*cascade_edge(pixel, q, id.z);
			acc += weight*pair_at(q, id.z);
		}
	}
	let count = acc.a*exp2(2.0*f32(level+1u));
	let budget = pyramid_settings.confidence.x;
	let beta = select(smoothstep(0.5*budget, budget, count), 1.0, level+1u == pyramid_settings.info.y);
	var result = vec4f(0.0, 0.0, 0.0, 1.0);
	if level != 0u { result = textureLoad(cascade_result, pixel, i32(id.z), 0); }
	result = vec4f(result.rgb+result.a*beta*acc.rgb/max(acc.a, 1e-12), result.a*(1.0-beta));
	textureStore(output_pair, pixel, i32(id.z), acc);
	textureStore(output_result, pixel, i32(id.z), result);
}

// The final HDR output has a different format from the accumulation textures.
#if PARAMS.get("sample_count", 1) > 1
texture<3, 2> final_color: texture_storage_2d_array<rgba16float, write>;
#else
texture<3, 2> final_color: texture_storage_2d<rgba16float, write>;
#endif

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
	let pixel = vec2i(id.xy);
	if any(id.xy >= textureDimensions(transmittance)) { return; }
	let t = sample_texture_load(transmittance, pixel, id.z);
	var result = vec4f(0.0);
	if 1.0-t.a > 1e-6 {
		var mu = textureLoad(cascade_result, pixel, i32(id.z), 0).rgb;
		if sample_texture_load(transparent_guide, pixel, id.z).x == 1.0 || forced_at(pixel, id.z) {
			let own = winner_at(pixel, id.z);
			mu = own.rgb/max(own.a, 1e-12);
		}
		let count = textureLoad(cascade_pair, pixel, i32(id.z), 0).a*exp2(2.0*f32(pyramid_settings.info.y));
		result = vec4f((vec3f(1.0)-t.rgb)*mu, clamp(count/pyramid_settings.confidence.x, 0.0, 1.0));
	}
#if PARAMS.get("sample_count", 1) > 1
	textureStore(final_color, pixel, i32(id.z), result);
#else
	textureStore(final_color, pixel, result);
#endif
}
