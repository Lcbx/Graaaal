// LXAA and FAAA adapted from shaders by Germán Rebord. See antialiasing_LICENSE.txt.
// FAAA is based on NVIDIA FXAA 3.11, COPYRIGHT (C) 2010, 2011 NVIDIA CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL NVIDIA
// OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION,
// OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE
// THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGES.
#if PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif

<AASettings> values: vec4f;
<AASettings> flags: vec4u;
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
