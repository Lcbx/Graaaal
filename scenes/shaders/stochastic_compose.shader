#include "utils.shaderlib"

<Frame> view: mat4x4f;
<Frame> proj: mat4x4f;
<Frame> light_dir: vec4f;
<Frame> camera_position: vec4f;
<Frame> clip: vec4f;
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var main_color: texture_2d<f32>;
@group(0) @binding(2) var transmittance: texture_2d<f32>;
@group(0) @binding(3) var winner_color: texture_2d<f32>;
@group(0) @binding(4) var opaque_depth: texture_depth_2d;
<BilateralSettings> near_sigmas: vec4f;
<BilateralSettings> far_sigmas: vec4f;
<BilateralSettings> distance_range: vec4f;
<BilateralSettings> kernel: vec4u;
@group(0) @binding(5) var<uniform> bilateral: BilateralSettings;
@group(0) @binding(6) var transparent_depth: texture_depth_2d;

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

fn guide_depth(pixel: vec2i) -> f32 {
	let depth = min(textureLoad(opaque_depth, pixel, 0), textureLoad(transparent_depth, pixel, 0));
	return linear_depth(depth, frame.clip.x, frame.clip.y);
}

@fragment
fn compose(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	let dimensions = vec2i(textureDimensions(transmittance));
	let t = textureLoad(transmittance, pixel, 0);
	// Display T_scalar directly, with no exposure curve or reconstruction.
	if bilateral.kernel.y == 1u {
		return present_linear(vec3f(t.a));
	}
	if bilateral.kernel.y == 2u {
		let winner = max(textureLoad(winner_color, pixel, 0).rgb, vec3f(0.0));
		return present_linear(winner/(vec3f(1.0)+winner));
	}
	let main = textureLoad(main_color, pixel, 0).rgb;
	var transparent = vec3f(0.0);
	var weight_sum = 0.0;
	var weight_squared_sum = 0.0;
	if 1.0-t.a > 1e-6 {
		let depth = guide_depth(pixel);
		let opaque_z = linear_depth(textureLoad(opaque_depth, pixel, 0), frame.clip.x, frame.clip.y);
		let distance_blend = select(0.0, smoothstep(bilateral.distance_range.x,
			bilateral.distance_range.y, depth), bilateral.kernel.z != 0u);
		let sigmas = mix(bilateral.near_sigmas.xyz, bilateral.far_sigmas.xyz, distance_blend);
		let transmittance_weight = mix(1.0, bilateral.far_sigmas.w, distance_blend);
		var numerator = vec3f(0.0);
		var denominator = 0.0;
		let radius = i32(bilateral.kernel.x);
		for (var y = -radius; y <= radius; y += 1) {
			for (var x = -radius; x <= radius; x += 1) {
				let q = pixel + vec2i(x, y);
				if any(q < vec2i(0)) || any(q >= dimensions) { continue; }
				let tq = textureLoad(transmittance, q, 0);
				let depth_q = guide_depth(q);
				let opaque_q = linear_depth(textureLoad(opaque_depth, q, 0), frame.clip.x, frame.clip.y);
				let dt = tq-t;
				let spatial = f32(x*x+y*y)/(2.0*sigmas.x*sigmas.x);
				let depth_delta = (depth_q-depth)/max(depth*sigmas.y, bilateral.near_sigmas.w);
				let opaque_delta = (opaque_q-opaque_z)/max(opaque_z*sigmas.y, bilateral.near_sigmas.w);
				let transmittance_delta = transmittance_weight*dot(dt, dt)/(2.0*sigmas.z*sigmas.z);
				let weight = exp(-spatial - 0.5*(depth_delta*depth_delta + opaque_delta*opaque_delta) - transmittance_delta);
				// No-winner samples are zero in the numerator; their aggregate
				// acceptance probability still enters the denominator.
				numerator += weight * textureLoad(winner_color, q, 0).rgb;
				denominator += weight * (1.0-tq.a);
				weight_sum += weight;
				weight_squared_sum += weight*weight;
			}
		}
		transparent = (vec3f(1.0)-t.rgb) * numerator / max(denominator, 1e-6);
	}
	if bilateral.kernel.y == 4u {
		// Effective sample count / maximum footprint size. This includes empty
		// winner samples; brightness measures filter support, not survivor count.
		let side = f32(2u*bilateral.kernel.x + 1u);
		let support = weight_sum*weight_sum/max(weight_squared_sum, 1e-6)/(side*side);
		return present_linear(vec3f(support));
	}
	// Reconstruction and composition happen in linear HDR. Encode only when the
	// presentation surface does not already perform the sRGB conversion.
	let hdr = max(select(main*t.rgb + transparent, transparent, bilateral.kernel.y == 3u), vec3f(0.0));
	let mapped = hdr / (vec3f(1.0)+hdr);
	return present_linear(mapped);
}
