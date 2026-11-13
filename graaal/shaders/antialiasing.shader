// LXAA and FAAA adapted from shaders by Germán Rebord. See antialiasing_faaa_lxaa_LICENSE.txt
// DDAA2 adapted from shader by almarklein. See antialiasing_ddaa2_LICENSE.txt

#if PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif

<AASettings> values: vec4f;
<AASettings> flags: vec4u;
<AASettings> ddaa2: vec4f;
uniform<0, 0> aa_settings: AASettings;
texture<0, 1> scene_color: texture_2d<f32>;

fn texel(pixel: vec2i) -> vec4f {
	let dimensions = vec2i(textureDimensions(scene_color));
	var color = textureLoad(scene_color, clamp(pixel, vec2i(0), dimensions-vec2i(1)), 0);
#if PARAMS.srgb_surface
	color = vec4f(linear_to_srgb(color.rgb), color.a);
#endif
	return color;
}

fn sample_color(uv: vec2f) -> vec4f {
	let position = uv*vec2f(textureDimensions(scene_color))-vec2f(0.5);
	let base = vec2i(floor(position));
	let f = fract(position);
	return mix(mix(texel(base), texel(base+vec2i(1, 0)), f.x), mix(texel(base+vec2i(0, 1)), texel(base+vec2i(1, 1)), f.x), f.y);
}

fn output_color(color: vec4f) -> vec4f {
#if PARAMS.srgb_surface
	return vec4f(srgb_to_linear(color.rgb), color.a);
#else
	return color;
#endif
}

fn luma(color: vec3f) -> f32 { return dot(color, vec3f(0.299, 0.587, 0.114)); }
fn sample_luma(uv: vec2f) -> f32 { return luma(sample_color(uv).rgb); }

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

@fragment
fn lxaa(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	let uv = position.xy/vec2f(textureDimensions(scene_color));
	let step = 1.0/vec2f(textureDimensions(scene_color));
	var a = vec4f(luma(texel(pixel+vec2i(-1, 1)).rgb), luma(texel(pixel+vec2i(1, 1)).rgb), luma(texel(pixel+vec2i(1, -1)).rgb), luma(texel(pixel+vec2i(-1, -1)).rgb));
	let gradient = vec2f(a.x-a.z, a.y-a.w);
	let direction = vec2f(gradient.x+gradient.y, gradient.x-gradient.y);
	let magnitude = abs(direction);
	let maximum = max(max(a.x, a.y), max(a.z, a.w));
	let minimum_luma = min(min(a.x, a.y), min(a.z, a.w));
	let center = texel(pixel);
	if abs(magnitude.x-magnitude.y) < aa_settings.values.x*(maximum*0.5+0.5) { return output_color(center); }
	let minimum = min(magnitude.x, magnitude.y);
	let offsets = min(vec2f(1.0), 0.0625*magnitude/max(minimum, 1e-20));
	var multiplier = 2.0*step*sign(direction);
	if aa_settings.flags.x != 0u && max(offsets.x, offsets.y) == 1.0 {
		let horizontal = offsets.x == 1.0;
		let negative = select(multiplier.y < 0.0, multiplier.x < 0.0, horizontal);
		let southwest = horizontal == negative;
		var shifted = uv;
		if horizontal { shifted.x += 2.0*multiplier.x; } else { shifted.y += 2.0*multiplier.y; }
		if southwest { a.x = sample_luma(shifted+vec2f(-step.x, step.y)); }
		if !negative { a.y = sample_luma(shifted+step); }
		if !southwest { a.z = sample_luma(shifted+vec2f(step.x, -step.y)); }
		if negative { a.w = sample_luma(shifted-step); }
		let g = vec2f(a.x-a.z, a.y-a.w);
		var d = vec2f(g.x+g.y, g.x-g.y);
		if !horizontal { d = d.yx; }
		if abs(d.x) > 2.0*abs(d.y) { multiplier *= 2.0; }
	}
	let offset = offsets*multiplier;
	let result = (sample_color(uv-offset)+sample_color(uv+offset))*0.3+center*0.4;
	let result_luma = luma(result.rgb);
	if result_luma < minimum_luma || result_luma > maximum { return output_color(center); }
	return output_color(vec4f(result.rgb, center.a));
}

fn quality_step(i: u32, quality: u32) -> f32 {
	if i == 0u { return 1.0; }
	if quality == 9u {
		if i <= 4u { return 1.0; }
		if i == 5u { return 1.5; }
	}
	if i == quality+2u && quality <= 8u { return 8.0; }
	if quality >= 4u && quality <= 8u && i == quality+1u { return select(4.0, 3.0, quality == 4u); }
	if i == 1u { return 1.5; }
	if i == 10u { return 4.0; }
	if i == 11u { return 8.0; }
	return 2.0;
}

@fragment
fn faaa(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	let uv = position.xy/vec2f(textureDimensions(scene_color));
	let step = 1.0/vec2f(textureDimensions(scene_color));
	let center = texel(pixel);
	let sw = luma(texel(pixel+vec2i(-1, 1)).rgb);
	let se = luma(texel(pixel+vec2i(1, 1)).rgb);
	let ne = luma(texel(pixel+vec2i(1, -1)).rgb);
	let nw = luma(texel(pixel+vec2i(-1, -1)).rgb);
	let gradient = vec2f(sw-ne, se-nw);
	let direction = abs(vec2f(gradient.x+gradient.y, gradient.x-gradient.y));
	let threshold = (10.0-aa_settings.values.y)*0.0625*(max(max(sw, se), max(ne, nw))*0.5+0.5);
	if aa_settings.flags.z != 0u {
		if abs(direction.x-direction.y) < threshold { return output_color(vec4f(vec3f(luma(center.rgb)*0.9), center.a)); }
		return output_color(vec4f(0.0, 1.0, 0.0, center.a));
	}
	if abs(direction.x-direction.y) < threshold { return output_color(center); }
	let horizontal = direction.x > direction.y;
	let m = luma(center.rgb);
	let n = luma(texel(pixel+select(vec2i(-1, 0), vec2i(0, -1), horizontal)).rgb);
	let s = luma(texel(pixel+select(vec2i(1, 0), vec2i(0, 1), horizontal)).rgb);
	let pair_n = abs(n-m) > abs(s-m);
	let neighbor = select(s, n, pair_n);
	let gradient_scaled = abs(neighbor-m)*0.25;
	let middle = (neighbor+m)*0.5;
	var length_sign = select(step.x, step.y, horizontal);
	if pair_n { length_sign = -length_sign; }
	var base = uv;
	if horizontal { base.y += length_sign*0.5; } else { base.x += length_sign*0.5; }
	let along = select(vec2f(0.0, step.y), vec2f(step.x, 0.0), horizontal);
	var pos_n = base-along;
	var pos_p = base+along;
	var end_n = sample_luma(pos_n)-middle;
	var end_p = sample_luma(pos_p)-middle;
	var done_n = abs(end_n) > gradient_scaled;
	var done_p = abs(end_p) > gradient_scaled;
	for (var i = 1u; i < aa_settings.flags.y+3u; i += 1u) {
		let distance = quality_step(i, aa_settings.flags.y);
		if !done_n { pos_n -= along*distance; end_n = sample_luma(pos_n)-middle; done_n = abs(end_n) > gradient_scaled; }
		if !done_p { pos_p += along*distance; end_p = sample_luma(pos_p)-middle; done_p = abs(end_p) > gradient_scaled; }
		if done_n && done_p { break; }
	}
	let dst_n = select(uv.y-pos_n.y, uv.x-pos_n.x, horizontal);
	let dst_p = select(pos_p.y-uv.y, pos_p.x-uv.x, horizontal);
	let near_n = dst_n < dst_p;
	let good_span = (select(end_p, end_n, near_n) < 0.0) != (m-middle < 0.0);
	let offset = 0.5-min(dst_n, dst_p)/max(dst_n+dst_p, 1e-20);
	if offset < 0.0 || !good_span { return output_color(center); }
	var result_uv = uv;
	if horizontal { result_uv.y += offset*length_sign; } else { result_uv.x += offset*length_sign; }
	return output_color(vec4f(sample_color(result_uv).rgb, center.a));
}


// DDAA2 v2.4 by Almar Klein; see ddaa2_LICENSE.txt.
const MAX_EDGE_ITERS = 5;  // length(EDGE_STEP_LIST)
const MAX_EDGE_SAMPLES = 15; // sum(EDGE_STEP_LIST), how far the algirithm can look along an edge

// The strength of the diffusion. A value of 3 seems to work well.




// Trims the algorithm from processing darks.
// low: 0.0833, medium: 0.0625, high: 0.0312, ultra: 0.0156, extreme: 0.0078




// The minimum amount of local contrast required to apply algorithm.
// low: 0.250, medium: 0.166, high: 0.125, ultra: 0.063, extreme: 0.031





// ========== Constants and helper functions ==========

const sqrt2  = sqrt(2.0);

fn rgb2luma(rgb: vec3f) -> f32 {
	return sqrt(dot(rgb, vec3f(0.299, 0.587, 0.114)));  // trick for perceived lightness, used in Bevy
	// return dot(rgb, vec3f(0.299, 0.587, 0.114));  // real luma
}


@fragment
fn ddaa2(@builtin(position) position: vec4f) -> @location(0) vec4<f32> {

	let texCoord = position.xy/vec2f(textureDimensions(scene_color));

	let resolution = vec2f(textureDimensions(scene_color));
	let pixelStep = 1.0 / resolution.xy;

	// Sample the center pixel
	let centerSample = sample_color(texCoord);
	let lumaCenter = rgb2luma(centerSample.rgb);

	// Luma at the four direct neighbors of the current fragment.
	let lumaN = rgb2luma(sample_color(texCoord+vec2f(vec2i(0, 1))*pixelStep).rgb);
	let lumaE = rgb2luma(sample_color(texCoord+vec2f(vec2i(1, 0))*pixelStep).rgb);
	let lumaS = rgb2luma(sample_color(texCoord+vec2f(vec2i(0, -1))*pixelStep).rgb);
	let lumaW = rgb2luma(sample_color(texCoord+vec2f(vec2i(-1, 0))*pixelStep).rgb);

	// Query the 4 remaining corners lumas.
	let lumaNW = rgb2luma(sample_color(texCoord+vec2f(vec2i(-1, 1))*pixelStep).rgb);
	let lumaNE = rgb2luma(sample_color(texCoord+vec2f(vec2i(1, 1))*pixelStep).rgb);
	let lumaSW = rgb2luma(sample_color(texCoord+vec2f(vec2i(-1, -1))*pixelStep).rgb);
	let lumaSE = rgb2luma(sample_color(texCoord+vec2f(vec2i(1, -1))*pixelStep).rgb);

	// Compute the range
	let lumaMin = min(lumaCenter, min(min(lumaS, lumaN), min(lumaW, lumaE)));
	let lumaMax = max(lumaCenter, max(max(lumaS, lumaN), max(lumaW, lumaE)));
	let lumaRange = lumaMax - lumaMin;

	// If the luma variation is lower that a threshold (or if we are in a really dark area), we are not on an edge, don't perform any AA.
	if (lumaRange < max(aa_settings.ddaa2.y, lumaMax * aa_settings.ddaa2.z)) {
		return output_color(centerSample);
	}

	// Combine the four edges lumas (using intermediary variables for future computations with the same values).
	let lumaSUp = lumaS + lumaN;
	let lumaWRight = lumaW + lumaE;
	let lumaWCorners = lumaSW + lumaNW;

	// Same for corners
	let lumaSCorners = lumaSW + lumaSE;
	let lumaECorners = lumaSE + lumaNE;
	let lumaNCorners = lumaNE + lumaNW;

	// Calculate the image gradient using the Schar kernel, which is (relatively) rotationally invariant.
	const k1 = 162.0 / 256.0; // 61
	const k2 = 47.0 / 256.0; // 17
	let imDx = (lumaW * k1 + lumaSW * k2 + lumaNW * k2) - (lumaE * k1 + lumaSE * k2 + lumaNE * k2);
	let imDy = (lumaS * k1 + lumaSW * k2 + lumaSE * k2) - (lumaN * k1 + lumaNW * k2 + lumaNE * k2);

	// Get the edge vector (orthogonal to the gradient), and calculate strength and direction.
	let edgeVector = vec2f(-imDy, imDx);
	var diffuseStrength = sqrt(length(edgeVector)) * aa_settings.ddaa2.x;
	var diffuseDirection = normalize(edgeVector);
	if diffuseStrength < 1e-6 {
		diffuseDirection = vec2f(0.0, 0.0);
		diffuseStrength = 0.0;
	}
	diffuseStrength = min(1.0, diffuseStrength);

	// Is the local edge horizontal or vertical ?
	let edgeHorizontal = abs(-2.0 * lumaW + lumaWCorners) + abs(-2.0 * lumaCenter + lumaSUp) * 2.0 + abs(-2.0 * lumaE + lumaECorners);
	let edgeVertical = abs(-2.0 * lumaN + lumaNCorners) + abs(-2.0 * lumaCenter + lumaWRight) * 2.0 + abs(-2.0 * lumaS + lumaSCorners);
	let isHorizontal = (edgeHorizontal >= edgeVertical);
	//let isHorizontal = (abs(diffuseDirection.x) >= abs(diffuseDirection.y)); -> different, resulting in wrong ridge detection

	// Calculate gradient on both sides of the current pixel
	var luma1 = select(lumaW, lumaS, isHorizontal);
	var luma2 = select(lumaE, lumaN, isHorizontal);    // Compute gradients in this direction.
	let gradient1 = luma1 - lumaCenter;
	let gradient2 = luma2 - lumaCenter;

	// Maintain ridges and thin lines. This is inspired by AXAA's 2nd enhancement, except we also apply it to negative edges and do a smooth transition instead of a threshold.
	// Note that we can diminish quite hard, because the neighbouring pixels likely get diffused in the direction of the edge (this is one of our advantages over fxaa).
	if sign(gradient1) == sign(gradient2) {
		// This is a ridge or a valley, e.g. a thin line. We want to presereve these.
		let ridgeness = min(abs(gradient1), abs(gradient2));
		let diminish_factor = 1.0 - (min(1.0, 10 * ridgeness));
		diffuseStrength *= diminish_factor;
	}

	// For long edges, the diffusion has to be huge to remove the jaggies. What algorithms like FXAA do instead, is detect
	// the length of the edge segment (successive horizontal/vertical pixels), and the use that to calculate the subpixel
	// texture coordinate offset, perpendicular to the edge. So technically this is diffusion perpendicular to the edge,
	// but in a controlled manner to smooth the step/jaggy.
	var subpixelEdgeOffset = vec2f(0.0);
	if (aa_settings.flags.w > 0u) {

		// Choose the step size (one pixel) accordingly.
		var stepLength = select(pixelStep.x, pixelStep.y, isHorizontal);

		// Gradient in the corresponding direction, normalized.
		let gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

		// Average luma in the current direction.
		var lumaLocalAverage = 0.0;
		let gradient2IsHigher = abs(gradient2) > abs(gradient1);
		if  gradient2IsHigher {
			lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
		} else {
			stepLength = -stepLength;  // switch the direction
			lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
		}

		// Shift UV in the correct direction by half a pixel (orthogonal to the edge)
		var currentUv = texCoord;
		if isHorizontal {
			currentUv.y += stepLength * 0.5;
		} else {
			currentUv.x += stepLength * 0.5;
		}

		// Initialize edge search params

		var distance1 = 999.0;
		var distance2 = 999.0;

		var lumaEnd1: f32;
		var lumaEnd2: f32;


		var lumaEnd_0: f32;

		var lumaEnd_1: f32;

		var lumaEnd_2: f32;

		var lumaEnd_3: f32;

		var lumaEnd_4: f32;

		var lumaEnd_5: f32;

		//


		// Step 0 (3 steps)

		// Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
		// We take 3 steps in both directions, but we get one sample for free (for each direction), so we take 4 samples.
		if isHorizontal {
			lumaEnd_0 = 0.5 * (lumaW + select(lumaSW, lumaNW, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_3 = 0.5 * (lumaE + select(lumaSE, lumaNE, gradient2IsHigher)) - lumaLocalAverage;

			lumaEnd_1 = rgb2luma(sample_color(currentUv+vec2f(-vec2i(2, 0))*pixelStep).rgb) - lumaLocalAverage;

			lumaEnd_2 = rgb2luma(sample_color(currentUv+vec2f(-vec2i(3, 0))*pixelStep).rgb) - lumaLocalAverage;


			lumaEnd_4 = rgb2luma(sample_color(currentUv+vec2f(vec2i(2, 0))*pixelStep).rgb) - lumaLocalAverage;

			lumaEnd_5 = rgb2luma(sample_color(currentUv+vec2f(vec2i(3, 0))*pixelStep).rgb) - lumaLocalAverage;

		} else {
			lumaEnd_0 = 0.5 * (lumaS + select(lumaSW, lumaSE, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_3 = 0.5 * (lumaN + select(lumaNW, lumaNE, gradient2IsHigher)) - lumaLocalAverage;

			lumaEnd_1 = rgb2luma(sample_color(currentUv+vec2f(-vec2i(0, 2))*pixelStep).rgb) - lumaLocalAverage;

			lumaEnd_2 = rgb2luma(sample_color(currentUv+vec2f(-vec2i(0, 3))*pixelStep).rgb) - lumaLocalAverage;


			lumaEnd_4 = rgb2luma(sample_color(currentUv+vec2f(vec2i(0, 2))*pixelStep).rgb) - lumaLocalAverage;

			lumaEnd_5 = rgb2luma(sample_color(currentUv+vec2f(vec2i(0, 3))*pixelStep).rgb) - lumaLocalAverage;

		}

		// Search for left endpoint in the current 3 samples

		if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 3.0; lumaEnd1 = lumaEnd_2; }

		if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 2.0; lumaEnd1 = lumaEnd_1; }

		if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 1.0; lumaEnd1 = lumaEnd_0; }

		// Same for the right endpoint

		if (abs(lumaEnd_5) >= gradientScaled) { distance2 = 3.0; lumaEnd2 = lumaEnd_5; }

		if (abs(lumaEnd_4) >= gradientScaled) { distance2 = 2.0; lumaEnd2 = lumaEnd_4; }

		if (abs(lumaEnd_3) >= gradientScaled) { distance2 = 1.0; lumaEnd2 = lumaEnd_3; }

		//

		// Now search for endpoints in a series of rounds, using a templated (i.e. unrolled) loop.
		// This is much faster (in WGSL) than a normal loop, probably due to optimization related to the texture lookups.
		// We use textureSampleLevel with the offset parameter to help the gpu compiler batch the samples.

		let max_distance = f32(aa_settings.flags.w)*3.0;



		// Step 1 (3 steps, offset 3)
		if (aa_settings.flags.w >= 2u && distance1 > 900.0) {
			if isHorizontal {
				let currentUv1 = currentUv - vec2f(5.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv1 = currentUv - vec2f(0.0, 5.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd1 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 6.0; lumaEnd1 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 5.0; lumaEnd1 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 4.0; lumaEnd1 = lumaEnd_0; }

		}
		if (aa_settings.flags.w >= 2u && distance2 > 900.0) {
			if isHorizontal {
				let currentUv2 = currentUv + vec2f(5.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv2 = currentUv + vec2f(0.0, 5.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd2 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance2 = 6.0; lumaEnd2 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance2 = 5.0; lumaEnd2 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance2 = 4.0; lumaEnd2 = lumaEnd_0; }

		}



		// Step 2 (3 steps, offset 6)
		if (aa_settings.flags.w >= 3u && distance1 > 900.0) {
			if isHorizontal {
				let currentUv1 = currentUv - vec2f(8.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv1 = currentUv - vec2f(0.0, 8.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd1 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 9.0; lumaEnd1 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 8.0; lumaEnd1 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 7.0; lumaEnd1 = lumaEnd_0; }

		}
		if (aa_settings.flags.w >= 3u && distance2 > 900.0) {
			if isHorizontal {
				let currentUv2 = currentUv + vec2f(8.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv2 = currentUv + vec2f(0.0, 8.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd2 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance2 = 9.0; lumaEnd2 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance2 = 8.0; lumaEnd2 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance2 = 7.0; lumaEnd2 = lumaEnd_0; }

		}



		// Step 3 (3 steps, offset 9)
		if (aa_settings.flags.w >= 4u && distance1 > 900.0) {
			if isHorizontal {
				let currentUv1 = currentUv - vec2f(11.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv1 = currentUv - vec2f(0.0, 11.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd1 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 12.0; lumaEnd1 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 11.0; lumaEnd1 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 10.0; lumaEnd1 = lumaEnd_0; }

		}
		if (aa_settings.flags.w >= 4u && distance2 > 900.0) {
			if isHorizontal {
				let currentUv2 = currentUv + vec2f(11.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv2 = currentUv + vec2f(0.0, 11.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd2 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance2 = 12.0; lumaEnd2 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance2 = 11.0; lumaEnd2 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance2 = 10.0; lumaEnd2 = lumaEnd_0; }

		}



		// Step 4 (3 steps, offset 12)
		if (aa_settings.flags.w >= 5u && distance1 > 900.0) {
			if isHorizontal {
				let currentUv1 = currentUv - vec2f(14.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv1 = currentUv - vec2f(0.0, 14.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv1+vec2f(-vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd1 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 15.0; lumaEnd1 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 14.0; lumaEnd1 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 13.0; lumaEnd1 = lumaEnd_0; }

		}
		if (aa_settings.flags.w >= 5u && distance2 > 900.0) {
			if isHorizontal {
				let currentUv2 = currentUv + vec2f(14.0, 0.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( -1, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 0, 0))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i( 1, 0))*pixelStep).rgb) - lumaLocalAverage;

			} else {
				let currentUv2 = currentUv + vec2f(0.0, 14.0) * pixelStep;

				lumaEnd_0 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, -1 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_1 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 0 ))*pixelStep).rgb) - lumaLocalAverage;

				lumaEnd_2 = rgb2luma(sample_color(currentUv2+vec2f(vec2i(0, 1 ))*pixelStep).rgb) - lumaLocalAverage;

			}
			lumaEnd2 = lumaEnd_2;

			if (abs(lumaEnd_2) >= gradientScaled) { distance2 = 15.0; lumaEnd2 = lumaEnd_2; }

			if (abs(lumaEnd_1) >= gradientScaled) { distance2 = 14.0; lumaEnd2 = lumaEnd_1; }

			if (abs(lumaEnd_0) >= gradientScaled) { distance2 = 13.0; lumaEnd2 = lumaEnd_0; }

		}

		// Clip the distance (if we did not find the end, we assume it's one pixel further)
		distance1 = min(distance1, max_distance + 1.0);
		distance2 = min(distance2, max_distance + 1.0);

		// UV offset: read in the direction of the closest side of the edge.
		let pixelOffset = - min(distance1, distance2) / (distance1 + distance2) + 0.5;

		// If the luma at center is smaller than at its neighbor, the delta luma at each end should be positive (same variation).
		let isLumaCenterSmaller = lumaCenter < lumaLocalAverage;
		var correctVariation: bool;
		if (distance1 < distance2) {
			correctVariation = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
		} else {
			correctVariation = (lumaEnd2 < 0.0) != isLumaCenterSmaller;
		}

		// Set subpixel texCoord offset
		if (!correctVariation) {
			subpixelEdgeOffset = vec2f(0.0);
		} else if isHorizontal {
			subpixelEdgeOffset = vec2f(0.0, pixelOffset * stepLength);
		} else {
			subpixelEdgeOffset = vec2f(pixelOffset * stepLength, 0.0);
		}
	}

	// We mix the effects of the edge-search with the directional diffusion.
	// Basically, we allow more diffusion if the edge-effect is smaller.
	let edgeStrength = (min(1.0, length(2.0 * subpixelEdgeOffset / pixelStep)));
	diffuseStrength = diffuseStrength * (1.0 - edgeStrength);

	// The step to take for the diffusion effect (blur in the direction of the
	// edge). Note that for diagonal-ish lines, the most blur is obtained when
	// stepping halfway to the next pixel, i.e. 0.707, because then the neighbour
	// pixels are taken into account more. Let's use no more than 0.6 because then
	// for horizontal lines, the max diffusion kernel is effectively [0.6, 2 * 0.5,
	// 0.6] which is still somewhat bell-shaped. Actually, if we use 0.5, we trade a
	// bit more smoothness for perceived sharpness. Make its 0.51 so it does not
	// look like some sort of offset.
	let max_step_size = 0.51;
	let diffuseStep = diffuseDirection * pixelStep * (max_step_size * diffuseStrength);

	// Compose the three texture coordinates, combining the ede-offset with the diffusion componennt, depending on the steepness of the edge.
	let texCoord1 = texCoord - diffuseStep + subpixelEdgeOffset;
	let texCoord2 = texCoord + diffuseStep + subpixelEdgeOffset;

	// Sample the final color
	var finalColor = vec3f(0.0);
	finalColor += 0.5 * sample_color(texCoord1).rgb;
	finalColor += 0.5 * sample_color(texCoord2).rgb;

	return output_color(vec4f(finalColor, centerSample.a));

}
