#set RECONSTRUCT_MODE = "pyramid"
#include "stochastic_reconstruction.shaderlib"

<PyramidSettings> info: vec4u;
<PyramidSettings> sizes: vec4u;
<PyramidSettings> confidence: vec4f;
uniform<2, 0> pyramid_settings: PyramidSettings;
{{ binding('level_pair', 'texture_2d_array<f32>', 2, 1) }}
{{ binding('level_depth', 'texture_2d_array<f32>', 2, 2) }}
{{ binding('level_transmission', 'texture_2d_array<f32>', 2, 3) }}
{{ binding('parent_mu', 'texture_2d_array<f32>', 2, 4) }}
{{ binding('parent_depth', 'texture_2d_array<f32>', 2, 5) }}
{{ binding('parent_transmission', 'texture_2d_array<f32>', 2, 6) }}
{{ binding('output_pair', 'texture_storage_2d_array<rgba32float, write>', 3, 0) }}
{{ binding('output_depth', 'texture_storage_2d_array<rgba32float, write>', 3, 1) }}
{{ binding('output_transmission', 'texture_storage_2d_array<rgba16float, write>', 3, 2) }}
{{ binding('output_mu', 'texture_storage_2d_array<rgba32float, write>', 3, 3) }}
#if PARAMS.get("sample_count", 1) > 1
{{ binding('output_result', 'texture_storage_2d_array<rgba16float, write>', 3, 4) }}
#else
{{ binding('output_result', 'texture_storage_2d<rgba16float, write>', 3, 4) }}
#endif

<PyramidSample> pair: vec4f;
<PyramidSample> depth: vec4f;
<PyramidSample> transmission: vec4f;

fn source_at(pixel: vec2i, sample: u32) -> PyramidSample {
	if pyramid_settings.info.x == 0u {
		let t = textureLoad(transmittance, pixel, i32(sample));
		let opaque = textureLoad(opaque_depth, pixel, i32(sample));
		let transparent = textureLoad(transparent_depth, pixel, i32(sample));
		let guide = textureLoad(transparent_guide, pixel, i32(sample));
		return PyramidSample(vec4f(winner_at(pixel, sample).rgb, max(1.0-t.a, 0.0)),
			vec4f(linear_depth(min(opaque, transparent), frame.clip.x, frame.clip.y), linear_depth(opaque, frame.clip.x, frame.clip.y), guide.yz), t);
	}
	return PyramidSample(textureLoad(level_pair, pixel, i32(sample), 0), textureLoad(level_depth, pixel, i32(sample), 0), textureLoad(level_transmission, pixel, i32(sample), 0));
}

fn edge_weight(a: PyramidSample, b: PyramidSample, delta: vec2f) -> f32 {
	let sigmas = bilateral.sigmas;
	let predicted_b = 1.0/max(1.0/a.depth.x+dot(a.depth.zw, delta), 1e-6);
	let predicted_a = 1.0/max(1.0/b.depth.x-dot(b.depth.zw, delta), 1e-6);
	let residual = max(abs(b.depth.x-predicted_b), abs(a.depth.x-predicted_a));
	let depth_delta = residual/max(a.depth.x*sigmas.y, sigmas.w);
	let opaque_delta = (b.depth.y-a.depth.y)/max(a.depth.y*sigmas.y, sigmas.w);
	let dt = b.transmission-a.transmission;
	return exp(-0.5*(depth_delta*depth_delta+opaque_delta*opaque_delta)-dot(dt, dt)/(2.0*sigmas.z*sigmas.z));
}

@compute @workgroup_size(8, 8)
fn pull(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= pyramid_settings.sizes.zw) { return; }
	let first = vec2i(id.xy*2u);
	var reference = source_at(first, id.z);
	var reference_offset = vec2i(0);
	for (var y = 0; y < 2; y += 1) {
		for (var x = 0; x < 2; x += 1) {
			let q = first+vec2i(x, y);
			if any(q >= vec2i(pyramid_settings.sizes.xy)) { continue; }
			let candidate = source_at(q, id.z);
			if candidate.pair.a > 0.0 && (reference.pair.a <= 0.0 || candidate.depth.x < reference.depth.x) {
				reference = candidate;
				reference_offset = vec2i(x, y);
			}
		}
	}
	let scale = exp2(f32(pyramid_settings.info.x));
	var pair = vec4f(0.0);
	for (var y = 0; y < 2; y += 1) {
		for (var x = 0; x < 2; x += 1) {
			let q = first+vec2i(x, y);
			if any(q >= vec2i(pyramid_settings.sizes.xy)) { continue; }
			let neighbor = source_at(q, id.z);
			pair += edge_weight(reference, neighbor, vec2f(vec2i(x, y)-reference_offset)*scale)*neighbor.pair;
		}
	}
	// Move the guide to the next cell center; reciprocal depth is planar.
	let delta = (vec2f(0.5)-vec2f(reference_offset))*scale;
	let depth = 1.0/max(1.0/reference.depth.x+dot(reference.depth.zw, delta), 1e-6);
	textureStore(output_pair, vec2i(id.xy), i32(id.z), pair);
	textureStore(output_depth, vec2i(id.xy), i32(id.z), vec4f(depth, reference.depth.yzw));
	textureStore(output_transmission, vec2i(id.xy), i32(id.z), reference.transmission);
}

<PushedEstimate> value: vec4f;
<PushedEstimate> confidence: f32;

fn pushed_estimate(pixel: vec2i, sample: u32) -> PushedEstimate {
	let center = source_at(pixel, sample);
	let scale = exp2(f32(pyramid_settings.info.x));
	var pair = vec4f(0.0);
	for (var y = -1; y <= 1; y += 1) {
		for (var x = -1; x <= 1; x += 1) {
			let q = pixel+vec2i(x, y);
			if any(q < vec2i(0)) || any(q >= vec2i(pyramid_settings.sizes.xy)) { continue; }
			let offset = vec2f(f32(x), f32(y));
			let weight = edge_weight(center, source_at(q, sample), offset*scale)*exp(-dot(offset, offset)/(2.0*bilateral.sigmas.x*bilateral.sigmas.x));
			pair += weight*source_at(q, sample).pair;
		}
	}
	let local_mu = pair.rgb/max(pair.a, 1e-6);
	var confidence = clamp(pair.a/pyramid_settings.confidence.x, 0.0, 1.0);
	if pyramid_settings.info.y != 0u || confidence >= 1.0 { return PushedEstimate(vec4f(local_mu, pair.a), confidence); }
	let parent_position = (vec2f(pixel)+vec2f(0.5))*0.5-vec2f(0.5);
	let parent_center = vec2i(floor(parent_position+vec2f(0.5)));
	let dimensions = vec2i(textureDimensions(parent_mu));
	var inherited = vec3f(0.0);
	var total = 0.0;
	for (var y = -1; y <= 1; y += 1) {
		for (var x = -1; x <= 1; x += 1) {
			let q = parent_center+vec2i(x, y);
			if any(q < vec2i(0)) || any(q >= dimensions) { continue; }
			let mu = textureLoad(parent_mu, q, i32(sample), 0);
			if mu.a <= 0.0 { continue; }
			let guide = PyramidSample(vec4f(0.0), textureLoad(parent_depth, q, i32(sample), 0), textureLoad(parent_transmission, q, i32(sample), 0));
			let offset = vec2f(q)-parent_position;
			let weight = edge_weight(center, guide, offset*(2.0*scale))*exp(-dot(offset, offset)/(2.0*bilateral.sigmas.x*bilateral.sigmas.x));
			inherited += weight*mu.rgb;
			total += weight;
		}
	}
	if total <= 1e-6 { confidence = 1.0; }
	return PushedEstimate(vec4f(mix(inherited/max(total, 1e-6), local_mu, confidence), max(pair.a, total)), confidence);
}

@compute @workgroup_size(8, 8)
fn push(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= pyramid_settings.sizes.zw) { return; }
	textureStore(output_mu, vec2i(id.xy), i32(id.z), pushed_estimate(vec2i(id.xy), id.z).value);
}

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= pyramid_settings.sizes.zw) { return; }
	let pixel = vec2i(id.xy);
	let t = textureLoad(transmittance, pixel, i32(id.z));
	var result = vec4f(0.0);
	if 1.0-t.a > 1e-6 {
		if textureLoad(transparent_guide, pixel, i32(id.z)).x == 1.0 || forced_at(pixel, id.z) {
			let own = winner_at(pixel, id.z);
			result = vec4f((vec3f(1.0)-t.rgb)*own.rgb/max(own.a, 1e-6), 1.0);
		} else {
			let estimate = pushed_estimate(pixel, id.z);
			// The displayed confidence is the final local expected-winner fraction.
			result = vec4f((vec3f(1.0)-t.rgb)*estimate.value.rgb, estimate.confidence);
		}
	}
#if PARAMS.get("sample_count", 1) > 1
	textureStore(output_result, pixel, i32(id.z), result);
#else
	textureStore(output_result, pixel, result);
#endif
}
