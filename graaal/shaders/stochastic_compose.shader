#set RECONSTRUCT_MODE = "compose"
#include "stochastic_reconstruction.shaderlib"
#if not PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif

#if PARAMS.get("compute_reconstruction", false)
sample_texture<1, 11, layered> reconstructed: f32;
#endif

#if PARAMS.get("exact_nearest", false)
sample_texture<1, 13> nearest_alpha: f32;
sample_texture<1, 14, layered> nearest_color: f32;
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

fn compose_sample(pixel: vec2i, sample: u32) -> vec3f {
	let dimensions = vec2i(textureDimensions(transmittance));
	let t = sample_texture_load(transmittance, pixel, sample);
#if PARAMS.get("exact_nearest", false)
	if bilateral.kernel.y == 11u { return vec3f(t.a); }
	if bilateral.kernel.y == 10u { return vec3f(select(0.0, 1.0, sample_texture_load(nearest_alpha, pixel, sample).a > 0.0)); }
	if bilateral.kernel.y == 9u {
		let color = sample_texture_load(nearest_color, pixel, sample, layered=true).rgb;
		return sample_texture_load(nearest_alpha, pixel, sample).rgb*color;
	}
#endif
	// Display T_scalar directly, with no exposure curve or reconstruction.
	if bilateral.kernel.y == 1u {
#if PARAMS.get("exact_nearest", false)
		return vec3f(t.a*(1.0-sample_texture_load(nearest_alpha, pixel, sample).a));
#else
		return vec3f(t.a);
#endif
	}
	if bilateral.kernel.y == 2u {
#if PARAMS.get("banded_tail", false) and PARAMS.get("banded_proxy", false)
		return clamp(vec3f(0.5)+0.5*winner_at(pixel, sample).rgb, vec3f(0.0), vec3f(1.0));
#else
		return debug_color(winner_at(pixel, sample));
#endif
	}
	if bilateral.kernel.y == 8u { return vec3f(select(0.0, 1.0, forced_at(pixel, sample))); }
	if bilateral.kernel.y >= 5u {
		let mixed = winner_at(pixel, sample);
		if bilateral.kernel.y == 7u && forced_at(pixel, sample) { return vec3f(1.0); }
#if PARAMS.get("winner_mix", "average") == "fill"
		let source = sample_texture_load(winner_source, pixel, sample, layered=true).r;
		if bilateral.kernel.y == 7u { return vec3f(select(0.0, 1.0, source != 0u)); }
		let wanted = select(1u, 2u, bilateral.kernel.y == 6u);
		return select(vec3f(0.0), debug_color(mixed), source == wanted);
#else
		if bilateral.kernel.y == 7u { return vec3f(mixed.a); }
#if PARAMS.get("winner_count", 1) == 2
		let first = sample_texture_load(first_color, pixel, sample, layered=true);
		if forced_at(pixel, sample) { return debug_color(first); }
		return debug_color(select(first, 2.0*mixed-first, bilateral.kernel.y == 6u));
#else
		return debug_color(mixed);
#endif
#endif
	}
	let main = sample_texture_load(main_color, pixel, sample).rgb;
	var filtered: vec4f;
#if PARAMS.get("compute_reconstruction", false)
#if PARAMS.get("banded_tail", false)
#if PARAMS.get("banded_mixed", false)
	filtered = sample_texture_load(reconstructed, pixel, 0u, layered=true);
	filtered = vec4f((vec3f(1.0)-t.rgb)*filtered.rgb, filtered.a);
#else
	filtered = sample_texture_load(reconstructed, pixel, sample, layered=true);
#endif
#else
	if 1.0-t.a <= 1e-6 {
		filtered = vec4f(0.0);
	} else if bilateral.kernel.x != 0u {
		filtered = sample_texture_load(reconstructed, pixel, sample, layered=true);
	} else {
		filtered = point_reconstruction(pixel, sample);
	}
#endif
#else
	filtered = reconstruct_transparency(pixel, sample, vec2i(0), 0u);
#endif
	if bilateral.kernel.y == 4u { return vec3f(filtered.a); }
	let transparent = filtered.rgb;
	// Reconstruction and composition happen in linear HDR. Encode only when the
	// presentation surface does not already perform the sRGB conversion.
	var hdr = select(main*t.rgb + transparent, transparent, bilateral.kernel.y == 3u);
#if PARAMS.get("exact_nearest", false)
	let a1 = sample_texture_load(nearest_alpha, pixel, sample).rgb;
	let c1 = sample_texture_load(nearest_color, pixel, sample, layered=true).rgb;
	hdr = a1*c1+(vec3f(1.0)-a1)*hdr;
#endif
	hdr = max(hdr, vec3f(0.0));
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
	var tone_map = bilateral.kernel.y != 1u && bilateral.kernel.y != 4u && bilateral.kernel.y != 7u && bilateral.kernel.y != 8u && bilateral.kernel.y != 10u && bilateral.kernel.y != 11u;
#if PARAMS.get("banded_tail", false) and PARAMS.get("banded_proxy", false)
	tone_map = tone_map && bilateral.kernel.y != 2u;
#endif
	if tone_map {
		color /= vec3f(1.0)+color;
	}
	return present_linear(color);
}
