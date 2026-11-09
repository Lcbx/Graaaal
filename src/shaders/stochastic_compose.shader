#include "utils.shaderlib"

<Frame> view: mat4x4f;
<Frame> proj: mat4x4f;
<Frame> light_dir: vec4f;
<Frame> camera_position: vec4f;
<Frame> clip: vec4f;
<Frame> grazing: vec4f;
@group(0) @binding(0) var<uniform> frame: Frame;
#if PARAMS.get("sample_count", 1) > 1
@group(0) @binding(1) var main_color: texture_multisampled_2d<f32>;
@group(0) @binding(2) var transmittance: texture_multisampled_2d<f32>;
@group(0) @binding(3) var winner_color: texture_2d_array<f32>;
@group(0) @binding(4) var opaque_depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var main_color: texture_2d<f32>;
@group(0) @binding(2) var transmittance: texture_2d<f32>;
@group(0) @binding(3) var winner_color: texture_2d<f32>;
@group(0) @binding(4) var opaque_depth: texture_depth_2d;
#endif
<BilateralSettings> sigmas: vec4f;
<BilateralSettings> kernel: vec4u;
@group(0) @binding(5) var<uniform> bilateral: BilateralSettings;
#if PARAMS.get("sample_count", 1) > 1
@group(0) @binding(6) var transparent_depth: texture_depth_multisampled_2d;
@group(0) @binding(7) var transparent_guide: texture_multisampled_2d<f32>;
#else
@group(0) @binding(6) var transparent_depth: texture_depth_2d;
@group(0) @binding(7) var transparent_guide: texture_2d<f32>;
#endif

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

fn present_linear(color: vec3f) -> vec4f {
#if PARAMS.srgb_surface
	return vec4f(color, 1.0);
#else
	return vec4f(linear_to_srgb(color), 1.0);
#endif
}

fn winner_at(pixel: vec2i, sample: u32) -> vec4f {
#if PARAMS.get("sample_count", 1) > 1
	return textureLoad(winner_color, pixel, i32(sample), 0);
#else
	return textureLoad(winner_color, pixel, 0);
#endif
}

fn guide_depth(pixel: vec2i, sample: u32) -> f32 {
	let depth = min(textureLoad(opaque_depth, pixel, i32(sample)), textureLoad(transparent_depth, pixel, i32(sample)));
	return linear_depth(depth, frame.clip.x, frame.clip.y);
}

fn compose_sample(pixel: vec2i, sample: u32) -> vec3f {
	let dimensions = vec2i(textureDimensions(transmittance));
	let t = textureLoad(transmittance, pixel, i32(sample));
	// Display T_scalar directly, with no exposure curve or reconstruction.
	if bilateral.kernel.y == 1u {
		return vec3f(t.a);
	}
	if bilateral.kernel.y == 2u {
		let winner = max(winner_at(pixel, sample).rgb, vec3f(0.0));
		return winner;
	}
	let main = textureLoad(main_color, pixel, i32(sample)).rgb;
	var transparent = vec3f(0.0);
	var weight_sum = 0.0;
	var weight_squared_sum = 0.0;
	let acceptance = 1.0-t.a;
	if acceptance > 1e-6 {
		let own = winner_at(pixel, sample).rgb;
		if bilateral.kernel.x == 0u {
			// Unfiltered stochastic estimator, without neighborhood/guide reads.
			// Division by expected acceptance keeps single-sample brightness: for
			// scalar opacity A_rgb/A_s is exactly one, not a second alpha factor.
			transparent = (vec3f(1.0)-t.rgb) * own / max(acceptance, 1e-6);
			weight_sum = 1.0;
			weight_squared_sum = 1.0;
		} else {
			let guide = textureLoad(transparent_guide, pixel, i32(sample));
			if guide.x == 1.0 && bilateral.kernel.y != 4u {
				// Guide encoding reserves 1.0 for exact certainty before quantization.
				// A certain nearest winner has no stochastic noise to reconstruct.
				transparent = (vec3f(1.0)-t.rgb) * own;
			} else {
				let depth = guide_depth(pixel, sample);
				let opaque_z = linear_depth(textureLoad(opaque_depth, pixel, i32(sample)), frame.clip.x, frame.clip.y);
				let sigmas = bilateral.sigmas;
				var numerator = vec3f(0.0);
				var denominator = 0.0;
				let radius = i32(bilateral.kernel.x);
				for (var y = -radius; y <= radius; y += 1) {
					for (var x = -radius; x <= radius; x += 1) {
						let q = pixel + vec2i(x, y);
						if any(q < vec2i(0)) || any(q >= dimensions) { continue; }
						let tq = textureLoad(transmittance, q, i32(sample));
						let depth_q = guide_depth(q, sample);
						let guide_q = textureLoad(transparent_guide, q, i32(sample));
						let opaque_q = linear_depth(textureLoad(opaque_depth, q, i32(sample)), frame.clip.x, frame.clip.y);
						let dt = tq-t;
						let spatial = f32(x*x+y*y)/(2.0*sigmas.x*sigmas.x);
						var depth_difference = abs(depth_q-depth);
						if guide.w > 0.0 && guide_q.w > 0.0 {
							let offset = vec2f(f32(x), f32(y));
							let predicted_q = 1.0/max(1.0/depth+dot(guide.yz, offset), 1e-6);
							let predicted_p = 1.0/max(1.0/depth_q-dot(guide_q.yz, offset), 1e-6);
							// Reciprocal-depth planes admit tilted surfaces; checking both
							// directions retains rejection at surface depth breaks.
							depth_difference = max(abs(depth_q-predicted_q), abs(depth-predicted_p));
						}
						let depth_delta = depth_difference/max(depth*sigmas.y, sigmas.w);
						let opaque_delta = (opaque_q-opaque_z)/max(opaque_z*sigmas.y, sigmas.w);
						let transmittance_delta = dot(dt, dt)/(2.0*sigmas.z*sigmas.z);
						let weight = exp(-spatial - 0.5*(depth_delta*depth_delta + opaque_delta*opaque_delta) - transmittance_delta);
						let winner = winner_at(q, sample);
						numerator += weight * winner.rgb;
						denominator += weight * winner.a;
						weight_sum += weight;
						weight_squared_sum += weight*weight;
					}
				}
				// Every fractional pixel uses the full normalized reconstruction.
				// Mixing in its own sample would reintroduce dark missing winners.
				transparent = (vec3f(1.0)-t.rgb) * numerator / max(denominator, 1e-6);
			}
		}
	}
	if bilateral.kernel.y == 4u {
		// Effective sample count / maximum footprint size. This includes empty
		// winner samples; brightness measures filter support, not survivor count.
		let side = f32(2u*bilateral.kernel.x + 1u);
		let support = weight_sum*weight_sum/max(weight_squared_sum, 1e-6)/(side*side);
		return vec3f(support);
	}
	// Reconstruction and composition happen in linear HDR. Encode only when the
	// presentation surface does not already perform the sRGB conversion.
	let hdr = max(select(main*t.rgb + transparent, transparent, bilateral.kernel.y == 3u), vec3f(0.0));
	return hdr;
}

@fragment
fn compose(@builtin(position) position: vec4f) -> @location(0) vec4f {
	var color = vec3f(0.0);
	// Keep winner IDs, depth and transmittance separate until linear HDR resolve.
	for (var sample = 0u; sample < {{PARAMS.get("sample_count", 1)}}u; sample += 1u) {
		color += compose_sample(vec2i(position.xy), sample);
	}
	color /= {{PARAMS.get("sample_count", 1)}}.0;
	if bilateral.kernel.y != 1u && bilateral.kernel.y != 4u {
		color /= vec3f(1.0)+color;
	}
	return present_linear(color);
}
