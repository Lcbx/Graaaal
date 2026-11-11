#set RECONSTRUCT_MODE = "compose"
#include "stochastic_reconstruction.shaderlib"
#if not PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif

#if PARAMS.get("compute_reconstruction", false)
#if PARAMS.get("sample_count", 1) > 1
{{ binding('reconstructed', 'texture_2d_array<f32>', 1, 11) }}
#else
{{ binding('reconstructed', 'texture_2d<f32>', 1, 11) }}
#endif
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
	let t = textureLoad(transmittance, pixel, i32(sample));
	// Display T_scalar directly, with no exposure curve or reconstruction.
	if bilateral.kernel.y == 1u {
		return vec3f(t.a);
	}
	if bilateral.kernel.y == 2u {
		return debug_color(winner_at(pixel, sample));
	}
	if bilateral.kernel.y == 8u { return vec3f(select(0.0, 1.0, forced_at(pixel, sample))); }
	if bilateral.kernel.y >= 5u {
		let mixed = winner_at(pixel, sample);
		if bilateral.kernel.y == 7u && forced_at(pixel, sample) { return vec3f(1.0); }
#if PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
		let source = textureLoad(winner_source, pixel, i32(sample), 0).r;
#else
		let source = textureLoad(winner_source, pixel, 0).r;
#endif
		if bilateral.kernel.y == 7u { return vec3f(select(0.0, 1.0, source != 0u)); }
		let wanted = select(1u, 2u, bilateral.kernel.y == 6u);
		return select(vec3f(0.0), debug_color(mixed), source == wanted);
#else
		if bilateral.kernel.y == 7u { return vec3f(mixed.a); }
#if PARAMS.get("winner_count", 1) == 2
#if PARAMS.get("sample_count", 1) > 1
		let first = textureLoad(first_color, pixel, i32(sample), 0);
#else
		let first = textureLoad(first_color, pixel, 0);
#endif
		if forced_at(pixel, sample) { return debug_color(first); }
		return debug_color(select(first, 2.0*mixed-first, bilateral.kernel.y == 6u));
#else
		return debug_color(mixed);
#endif
#endif
	}
	let main = textureLoad(main_color, pixel, i32(sample)).rgb;
	var filtered: vec4f;
#if PARAMS.get("compute_reconstruction", false)
	if 1.0-t.a <= 1e-6 {
		filtered = vec4f(0.0);
	} else if bilateral.kernel.x != 0u {
#if PARAMS.get("sample_count", 1) > 1
		filtered = textureLoad(reconstructed, pixel, i32(sample), 0);
#else
		filtered = textureLoad(reconstructed, pixel, 0);
#endif
	} else {
		filtered = point_reconstruction(pixel, sample);
	}
#else
	filtered = reconstruct_transparency(pixel, sample, vec2i(0), 0u);
#endif
	if bilateral.kernel.y == 4u { return vec3f(filtered.a); }
	let transparent = filtered.rgb;
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
	if bilateral.kernel.y != 1u && bilateral.kernel.y != 4u && bilateral.kernel.y != 7u && bilateral.kernel.y != 8u {
		color /= vec3f(1.0)+color;
	}
	return present_linear(color);
}
