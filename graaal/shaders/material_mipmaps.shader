#include "utils_color.shaderlib"

<MipParams> info: vec4u;
<MipParams> cutoff: vec4f;
uniform<0, 0> params: MipParams;
texture<0, 1> source: texture_2d<f32>;
texture<0, 2> linear_destination: texture_storage_2d<rgba32float, write>;
texture<0, 3> encoded_destination: texture_storage_2d_array<rgba8unorm, write>;

<CoverageState> target_hits: atomic<u32>;
<CoverageState> candidate_hits: atomic<u32>;
<CoverageState> min_alpha_bits: atomic<u32>;
<CoverageState> target_total: u32;
<CoverageState> candidate_total: u32;
<CoverageState> low: f32;
<CoverageState> high: f32;
<CoverageState> scale: f32;
<CoverageState> best_scale: f32;
<CoverageState> best_error: f32;
<CoverageState> phase: u32;
<CoverageState> done: u32;
storage<0, 4, read_write> coverage: CoverageState;
var<workgroup> hit_counts: array<u32, 64>;
var<workgroup> minimums: array<u32, 64>;

@compute @workgroup_size(8, 8)
fn decode(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(source)) { return; }
	// Reading the source's sRGB view performs linear color decoding on the GPU.
	let color = textureLoad(source, vec2i(id.xy), 0);
	let alpha = select(color.a, 1.0, params.info.y == 0u);
	textureStore(linear_destination, vec2i(id.xy), vec4f(color.rgb*alpha, alpha));
}

fn wrap(pixel: vec2i, dimensions: vec2i) -> vec2i {
	return ((pixel % dimensions) + dimensions) % dimensions;
}

@compute @workgroup_size(8, 8)
fn resample(@builtin(global_invocation_id) id: vec3u) {
	let dst_size = textureDimensions(linear_destination);
	if any(id.xy >= dst_size) { return; }
	let src_size = vec2f(textureDimensions(source));
	let ratio = src_size/vec2f(dst_size);
	let lo = vec2f(id.xy)*ratio;
	let hi = vec2f(id.xy+vec2u(1u))*ratio;
	let center = (lo+hi)*0.5-0.5;
	let upscale = ratio < vec2f(1.0);
	let first = vec2i(floor(select(lo, center, upscale)));
	let last = vec2i(select(ceil(hi)-1.0, floor(center)+1.0, upscale));
	var total = vec4f(0.0);
	var weight_sum = 0.0;
	for (var y = first.y; y <= last.y; y += 1) {
		for (var x = first.x; x <= last.x; x += 1) {
			let p = vec2f(f32(x), f32(y));
			let area = max(vec2f(0.0), min(p+1.0, hi)-max(p, lo));
			let bilinear = max(vec2f(0.0), 1.0-abs(p-center));
			let weights = select(area, bilinear, upscale);
			let weight = weights.x*weights.y;
			total += weight*textureLoad(source, wrap(vec2i(x, y), vec2i(src_size)), 0);
			weight_sum += weight;
		}
	}
	// Carry uncorrected, premultiplied linear data through every mip, including
	// NPOT area resampling and repeat-aware bilinear upsampling of small sources.
	textureStore(linear_destination, vec2i(id.xy), total/max(weight_sum, 1e-30));
}

fn quantized_alpha(pixel: vec2i, scale: f32) -> f32 {
	let alpha = textureLoad(source, wrap(pixel, vec2i(textureDimensions(source))), 0).a;
	return round(clamp(alpha*scale, 0.0, 1.0)*255.0)/255.0;
}

fn coverage_hits(pixel: vec2i, scale: f32) -> u32 {
	let a = quantized_alpha(pixel, scale);
	var hits = 0u;
	for (var y = -1; y <= 1; y += 2) {
		let b = quantized_alpha(pixel+vec2i(0, y), scale);
		for (var x = -1; x <= 1; x += 2) {
			let c = quantized_alpha(pixel+vec2i(x, 0), scale);
			let d = quantized_alpha(pixel+vec2i(x, y), scale);
			let alpha = a*0.5625 + (b+c)*0.1875 + d*0.0625;
			hits += select(0u, 1u, alpha >= params.cutoff.x);
		}
	}
	return hits;
}

@compute @workgroup_size(1)
fn init_source() {
	atomicStore(&coverage.target_hits, 0u);
	let size = textureDimensions(source);
	coverage.target_total = size.x*size.y*4u;
}

@compute @workgroup_size(8, 8)
fn measure_source(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
	var hits = 0u;
	if all(id.xy < textureDimensions(source)) { hits = coverage_hits(vec2i(id.xy), 1.0); }
	hit_counts[lane] = hits;
	workgroupBarrier();
	for (var stride = 32u; stride > 0u; stride /= 2u) {
		if lane < stride { hit_counts[lane] += hit_counts[lane+stride]; }
		workgroupBarrier();
	}
	if lane == 0u { atomicAdd(&coverage.target_hits, hit_counts[0]); }
}

@compute @workgroup_size(1)
fn begin_search() {
	atomicStore(&coverage.candidate_hits, 0u);
	atomicStore(&coverage.min_alpha_bits, 0x7f800000u);
	let size = textureDimensions(source);
	coverage.candidate_total = size.x*size.y*4u;
	coverage.low = 0.0;
	coverage.high = 1.0;
	coverage.scale = 1.0;
	coverage.best_scale = 1.0;
	coverage.best_error = 2.0;
	coverage.phase = 0u;
	coverage.done = 0u;
}

@compute @workgroup_size(8, 8)
fn measure_candidate(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
	var hits = 0u;
	var minimum = 0x7f800000u;
	if all(id.xy < textureDimensions(source)) && coverage.done == 0u {
		hits = coverage_hits(vec2i(id.xy), coverage.scale);
		let alpha = textureLoad(source, vec2i(id.xy), 0).a;
		if alpha > 0.0 { minimum = bitcast<u32>(alpha); }
	}
	hit_counts[lane] = hits;
	minimums[lane] = minimum;
	workgroupBarrier();
	for (var stride = 32u; stride > 0u; stride /= 2u) {
		if lane < stride {
			hit_counts[lane] += hit_counts[lane+stride];
			minimums[lane] = min(minimums[lane], minimums[lane+stride]);
		}
		workgroupBarrier();
	}
	if lane == 0u {
		atomicAdd(&coverage.candidate_hits, hit_counts[0]);
		atomicMin(&coverage.min_alpha_bits, minimums[0]);
	}
}

@compute @workgroup_size(1)
fn update_search() {
	if coverage.done != 0u { return; }
	let desired = f32(atomicLoad(&coverage.target_hits))/f32(coverage.target_total);
	let current = f32(atomicLoad(&coverage.candidate_hits))/f32(coverage.candidate_total);
	atomicStore(&coverage.candidate_hits, 0u);
	let error = abs(current-desired);
	if error < coverage.best_error {
		coverage.best_error = error;
		coverage.best_scale = coverage.scale;
	}
	if error == 0.0 { coverage.done = 1u; return; }
	let minimum = bitcast<f32>(atomicLoad(&coverage.min_alpha_bits));
	let limit = max(1.0, params.cutoff.x/minimum*1.01);
	if coverage.phase == 0u {
		coverage.phase = 2u;
		if current < desired { coverage.low = 1.0; coverage.phase = 1u; }
	}
	if coverage.phase == 1u {
		if current < desired && coverage.scale < limit {
			coverage.low = coverage.scale;
			coverage.high = min(coverage.scale*2.0, limit);
			coverage.scale = coverage.high;
			return;
		}
		coverage.high = coverage.scale;
		coverage.phase = 2u;
	} else {
		if current < desired { coverage.low = coverage.scale; }
		else { coverage.high = coverage.scale; }
	}
	let next = (coverage.low+coverage.high)*0.5;
	if next == coverage.scale { coverage.done = 1u; }
	coverage.scale = next;
}

@compute @workgroup_size(8, 8)
fn encode(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(source)) { return; }
	let linear = textureLoad(source, vec2i(id.xy), 0);
	var color = linear.rgb/max(linear.a, 1e-30);
	if linear.a == 0.0 && params.info.y != 0u {
		// Straight-color bilinear/trilinear sampling still reads RGB from zero-
		// alpha neighbors. Pad those texels from the nearest nonzero-alpha color
		// instead of encoding black. Keep this out of the premultiplied scratch
		// chain so neither coverage nor later mip averages change.
		var nearest = 9;
		var donor = vec4f(0.0);
		for (var y = -2; y <= 2; y += 1) {
			for (var x = -2; x <= 2; x += 1) {
				let distance = x*x+y*y;
				let sample = textureLoad(source, wrap(vec2i(id.xy)+vec2i(x, y), vec2i(textureDimensions(source))), 0);
				if sample.a > 0.0 && distance <= nearest {
					if distance < nearest { donor = vec4f(0.0); nearest = distance; }
					donor += sample;
				}
			}
		}
		if donor.a > 0.0 { color = donor.rgb/donor.a; }
	}
	var scale = 1.0;
	if params.info.z != 0u { scale = coverage.best_scale; }
	let alpha = round(clamp(linear.a*scale, 0.0, 1.0)*255.0)/255.0;
	textureStore(encoded_destination, vec2i(id.xy), i32(params.info.x), vec4f(linear_to_srgb(color), alpha));
}
