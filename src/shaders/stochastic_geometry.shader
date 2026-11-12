#include "draw_common.shaderlib"
#set MATERIAL_STAGE = "geometry"
#include "stochastic_common.shaderlib"
#include "utils_transform.shaderlib"
#include "utils_texture.shaderlib"
#include "utils_hash.shaderlib"
#include "utils_alpha.shaderlib"
#include "instance_transform.shaderlib"
#include "stochastic_noise.shaderlib"

storage<1, 0, read> instances: array<MeshInstance>;
storage<1, 1, read> visible_instances: array<u32>;
#if PARAMS.get("sample_count", 1) > 1
{{ binding('opaque_depth', 'texture_depth_multisampled_2d', 3, 0) }}
#else
{{ binding('opaque_depth', 'texture_depth_2d', 3, 0) }}
#endif
{{ binding('blue_noise', 'texture_2d<f32>', 3, 1) }}
{{ binding('noise_hierarchy', 'texture_2d<f32>', 3, 2) }}
#if PARAMS.get("winner_count", 1) > 1 or PARAMS.get("winner_mix", "average") == "fill"
<WinnerSettings> info: vec4u;
uniform<3, 3> winner_settings: WinnerSettings;
#if PARAMS.get("sample_count", 1) > 1
{{ binding('winner_guide', 'texture_multisampled_2d<f32>', 3, 4) }}
#else
{{ binding('winner_guide', 'texture_2d<f32>', 3, 4) }}
#endif
#endif

#if PARAMS.get("winner_mix", "average") == "fill"
#if PARAMS.get("sample_count", 1) > 1
{{ binding('previous_winner', 'texture_2d_array<f32>', 3, 7) }}
#else
{{ binding('previous_winner', 'texture_2d<f32>', 3, 7) }}
#endif
#endif

#if PARAMS.get("force_single_contributor", false)
#if PARAMS.get("sample_count", 1) > 1
{{ binding('selection_transmittance', 'texture_multisampled_2d<f32>', 3, 5) }}
{{ binding('contributor_count', 'texture_multisampled_2d<f32>', 3, 6) }}
#else
{{ binding('selection_transmittance', 'texture_2d<f32>', 3, 5) }}
{{ binding('contributor_count', 'texture_2d<f32>', 3, 6) }}
#endif
<TransmittanceOutput> @location(0) transmission: vec4f;
<TransmittanceOutput> @location(1) count: f32;
#endif

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;
<VertexInput> barycentric: vec2f;
<VertexInput> triangle: u32;

<VertexOutput> @builtin(position) position: vec4f;
#if PARAMS.get("sample_count", 1) > 1
<VertexOutput> @interpolate(perspective, sample) world: vec3f;
<VertexOutput> @interpolate(perspective, sample) normal: vec3f;
<VertexOutput> @interpolate(perspective, sample) uv: vec2f;
<VertexOutput> @interpolate(perspective, sample) barycentric: vec2f;
<VertexOutput> @interpolate(perspective, sample) local: vec3f;
<VertexOutput> @interpolate(perspective, sample) local_normal: vec3f;
#else
<VertexOutput> world: vec3f;
<VertexOutput> normal: vec3f;
<VertexOutput> uv: vec2f;
<VertexOutput> barycentric: vec2f;
<VertexOutput> local: vec3f;
<VertexOutput> local_normal: vec3f;
#endif
<VertexOutput> @interpolate(flat) triangle: u32;
<VertexOutput> @interpolate(flat) instance: u32;

@vertex
fn vertex(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
	let id = visible_instances[instance_index];
	let deformed = deform_vertex(id, input.position, input.normal, input.uv);
	let world = instance_position(instances[id], deformed.position);
	var output: VertexOutput;
	output.position = frame.proj * frame.view * vec4f(world, 1.0);
	output.world = world;
	// Keep inverse-transformed normals unnormalized until after interpolation.
	output.normal = instance_normal_vector(instances[id], deformed.normal);
	output.uv = deformed.uv;
	output.barycentric = input.barycentric;
	output.triangle = input.triangle;
	output.instance = id;
	output.local = input.position;
	output.local_normal = input.normal;
	return output;
}

fn surface_grazing_fade(id: u32, world: vec3f) -> f32 {
	// Geometric derivatives reflect the projected surface, independent of smooth
	// shading normals and instance scale. Evaluate them before every discard.
	let geometric_normal = cross(dpdxFine(world), dpdyFine(world));
#if PARAMS.get("material_library") is not none
	let grazing = library_grazing(id);
#else
	let grazing = frame.grazing.xyz;
#endif
	if grazing.z == 0.0 { return 1.0; }
	let orthographic_to_eye = vec3f(frame.view[0].z, frame.view[1].z, frame.view[2].z);
	let to_eye = select(frame.camera_position.xyz-world, orthographic_to_eye, frame.proj[3].w == 1.0);
	let denominator = length(geometric_normal)*length(to_eye);
	if denominator <= 1e-20 { return 1.0; }
	return grazing_fade_factor(dot(geometric_normal, to_eye)/denominator, grazing.x, grazing.y);
}

fn raster_opacity(id: u32, material: Material, uv: vec2f, lod: f32, world: vec3f) -> vec3f {
	let alpha = base_color(material, uv, lod).a;
	let width = fwidth(alpha);
	let coverage = select(alpha, sharpen_alpha(alpha, material.surface.w, width), material.surface.z == 1.0);
	let fade = surface_grazing_fade(id, world);
	var result = coverage*material.opacity_rgb.rgb;
#if PARAMS.get("material_library") is not none
	result = library_opacity(id, material, result, uv, world, lod);
#endif
	return clamp(result, vec3f(0.0), vec3f(1.0))*fade;
}

@fragment
fn opaque_prepass(input: VertexOutput, @builtin(front_facing) front: bool) {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	if !front && material.flags.y == 0u { discard; }
	if material.surface.z == 1.0 && base_color(material, input.uv, lod).a < material.surface.w { discard; }
}

@fragment
fn opaque(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	if !front && material.flags.y == 0u { discard; }
	if material.surface.z == 1.0 && base_color(material, input.uv, lod).a < material.surface.w { discard; }
	let signed_normal = select(-input.normal, input.normal, front);
	let normal = signed_normal / max(length(signed_normal), 1e-8);
	return vec4f(shade_material(input.instance, material, input.uv, input.world, normal, lod), 1.0);
}

@fragment
fn transmittance(input: VertexOutput, @builtin(front_facing) front: bool)
#if PARAMS.get("force_single_contributor", false)
 -> TransmittanceOutput {
#else
 -> @location(0) vec4f {
#endif
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
	if !front && material.flags.y == 0u { discard; }
	let p = max(alpha.x, max(alpha.y, alpha.z));
	let transmission = vec4f(vec3f(1.0)-alpha, 1.0-p);
#if PARAMS.get("force_single_contributor", false)
	return TransmittanceOutput(transmission, select(0.0, 1.0, p > 0.0));
#else
	return transmission;
#endif
}

fn raster_lod(uv: vec2f) -> f32 {
	// Derivatives are evaluated before any divergent branch or discard. Match
	// the half-float LOD stored in the winner payload exactly across all passes.
	let lod = clamp(texture_lod_from_gradients(dpdxFine(uv), dpdyFine(uv),
		vec2f(textureDimensions(base_textures))), 0.0, f32(textureNumLevels(base_textures)-1u));
	return unpack2x16float(pack2x16float(vec2f(lod, 0.0))).x;
}

@fragment
fn transparent_depth(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
	// Reciprocal view depth is affine over a projected triangle. Storing its
	// gradient permits exact planar prediction, including perspective foreshortening.
	let reciprocal_depth = 1.0 / max(-(frame.view*vec4f(input.world, 1.0)).z, 1e-6);
	let slope = vec2f(dpdxFine(reciprocal_depth), dpdyFine(reciprocal_depth));
	if input.position.z >= textureLoad(opaque_depth, vec2i(input.position.xy), i32(sample)) { discard; }
	if !front && material.flags.y == 0u { discard; }
	// Ignore fully transparent texels; retain every nonzero opacity, however low.
	let p = max(alpha.x, max(alpha.y, alpha.z));
	if p <= 0.0 { discard; }
	// Reserve half-float 1.0 for exact certainty. Fractional values near one
	// would otherwise round up and incorrectly bypass missing-winner filtering.
	// 0.99951171875 is the largest representable half-float below one.
	let guide_probability = select(min(p, 0.99951171875), 1.0, p == 1.0);
	return vec4f(guide_probability, slope, reciprocal_depth);
}

#if PARAMS.get("sample_count", 1) > 1
<WinnerOutput> @location(0) low: vec4u;
<WinnerOutput> @location(1) high: vec4u;
#endif

@fragment
fn select_winner(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32)
#if PARAMS.get("sample_count", 1) > 1
 -> WinnerOutput {
#else
 -> @location(0) vec4u {
#endif
	let lod = raster_lod(input.uv);
	let pixel = vec2i(input.position.xy);
	let info = metadata[input.instance];
	let material = get_material(input.instance);
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
	// Derivatives precede every discard. The local field follows the mesh
	// through camera motion and instance transforms.
	// Opposite physical sides use separate seeds on two-sided/closed surfaces.
	var seed = hash_u32(info.z ^ select(0xa511e9b3u, 0u, front));
#if PARAMS.get("complemented_winners", false)
	seed = noise_primitive_seed(seed, input.triangle);
#endif
	let mode = u32(frame.clip.w);
#if PARAMS.get("sample_count", 1) > 1
	// Anchored blue reads one spatial field at the native sample positions.
	if mode != 7u { seed = hash_u32(seed ^ hash_u32(sample + 0x68bc21ebu)); }
#endif
	let local_dx = dpdxFine(input.local);
	let local_dy = dpdyFine(input.local);
	let uv_dx = dpdxFine(input.uv);
	let uv_dy = dpdyFine(input.uv);
	let p = max(alpha.x, max(alpha.y, alpha.z));
	// All derivative-dependent material/LOD/noise inputs are now evaluated.
	// Reject deterministic failures before count/guide reads and noise work.
	if p <= 0.0 { discard; }
	if !front && material.flags.y == 0u { discard; }
	if input.position.z >= textureLoad(opaque_depth, pixel, i32(sample)) { discard; }
	var forced = false;
#if PARAMS.get("force_single_contributor", false)
	let acceptance = 1.0-textureLoad(selection_transmittance, pixel, i32(sample)).a;
	forced = p > 0.0 && acceptance > 0.0 && textureLoad(contributor_count, pixel, i32(sample)).r == 1.0;
#endif
#if PARAMS.get("winner_count", 1) > 1 or PARAMS.get("winner_mix", "average") == "fill"
#if not PARAMS.get("complemented_winners", false)
	seed = noise_stream_seed(seed, winner_settings.info.x);
#endif
#if PARAMS.get("winner_mix", "average") == "fill"
	if winner_settings.info.x != 0u {
#if PARAMS.get("sample_count", 1) > 1
		if textureLoad(previous_winner, pixel, i32(sample), 0).a > 0.0 { discard; }
#else
		if textureLoad(previous_winner, pixel, 0).a > 0.0 { discard; }
#endif
	}
#endif
	if winner_settings.info.x != 0u && forced { discard; }
	// All material and noise derivatives have been evaluated before this discard.
	// The 1x1 path does not produce a guide and sets info.y to zero.
	if winner_settings.info.y != 0u {
		if textureLoad(winner_guide, pixel, i32(sample)).x == 1.0 { discard; }
	}
#endif
	var u = 0.0;
	if !forced {
		if mode == 7u {
			u = anchored_blue_threshold(blue_noise, input.position.xy, input.local, local_dx, local_dy,
				frame.clip.z, seed, frame.noise_anchor.x, frame.noise_anchor.y);
		} else {
#if FEATURES.NOISE_FOOTPRINT_BALANCE
			u = surface_noise_threshold_balanced(blue_noise, noise_hierarchy, input.local, local_dx, local_dy, input.local_normal,
				input.uv, uv_dx, uv_dy, frame.clip.z, seed, mode, frame.noise_footprint_balance);
#else
			u = surface_noise_threshold(blue_noise, noise_hierarchy, input.local, local_dx, local_dy, input.local_normal,
				input.uv, uv_dx, uv_dy, frame.clip.z, seed, mode);
#endif
		}
	}
#if PARAMS.get("complemented_winners", false)
	// Complement last, preserving the remapped marginal and the common phase.
	if winner_settings.info.x != 0u { u = 1.0-u; }
#endif
	if !forced && p < 1.0 && u >= p { discard; }
	// Zero instance is the empty sentinel. Hardware depth writes keep all four
	// payload components consistent with the nearest accepted triangle.
	let payload = vec4u(input.instance + 1u, input.triangle,
		pack2x16unorm(clamp(input.barycentric, vec2f(0.0), vec2f(1.0))),
		pack2x16float(vec2f(lod, select(-1.0, 1.0, front))) | select(0u, WINNER_FORCED_BIT, forced));
#if PARAMS.get("sample_count", 1) > 1
	// rgba32uint cannot be multisampled. Preserve every bit in two rgba16uint targets.
	return WinnerOutput(vec4u(payload.x & 65535u, payload.x >> 16u, payload.y & 65535u, payload.y >> 16u),
		vec4u(payload.z & 65535u, payload.z >> 16u, payload.w & 65535u, payload.w >> 16u));
#else
	return payload;
#endif
}
