#include "draw_common.shaderlib"
#include "stochastic_common.shaderlib"
#include "utils.shaderlib"
#include "instance_transform.shaderlib"

@group(1) @binding(0) var<storage, read> instances: array<MeshInstance>;
@group(1) @binding(1) var<storage, read> visible_instances: array<u32>;
@group(3) @binding(0) var opaque_depth: texture_depth_2d;
@group(3) @binding(1) var blue_noise: texture_2d<f32>;

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;
<VertexInput> barycentric: vec2f;
<VertexInput> triangle: u32;

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> world: vec3f;
<VertexOutput> normal: vec3f;
<VertexOutput> uv: vec2f;
<VertexOutput> barycentric: vec2f;
<VertexOutput> @interpolate(flat) triangle: u32;
<VertexOutput> @interpolate(flat) instance: u32;

@vertex
fn vertex(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
	let id = visible_instances[instance_index];
	let world = instance_position(instances[id], input.position);
	var output: VertexOutput;
	output.position = frame.proj * frame.view * vec4f(world, 1.0);
	output.world = world;
	// Keep inverse-transformed normals unnormalized until after interpolation.
	output.normal = instance_normal_vector(instances[id], input.normal);
	output.uv = input.uv;
	output.barycentric = input.barycentric;
	output.triangle = input.triangle;
	output.instance = id;
	return output;
}

@fragment
fn opaque_prepass(input: VertexOutput, @builtin(front_facing) front: bool) {
	let lod = raster_lod(input.uv);
	let material = materials[metadata[input.instance].y];
	if !front && material.flags.y == 0u { discard; }
	if material.surface.z == 1.0 && base_color(material, input.uv, lod).a < material.surface.w { discard; }
}

@fragment
fn opaque(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = materials[metadata[input.instance].y];
	if !front && material.flags.y == 0u { discard; }
	if material.surface.z == 1.0 && base_color(material, input.uv, lod).a < material.surface.w { discard; }
	let signed_normal = select(-input.normal, input.normal, front);
	let normal = signed_normal / max(length(signed_normal), 1e-8);
	return vec4f(shade(material, input.uv, input.world, normal, lod), 1.0);
}

@fragment
fn transmittance(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = materials[metadata[input.instance].y];
	if !front && material.flags.y == 0u { discard; }
	let alpha = opacity(material, input.uv, lod);
	let p = max(alpha.x, max(alpha.y, alpha.z));
	return vec4f(vec3f(1.0)-alpha, 1.0-p);
}

fn raster_lod(uv: vec2f) -> f32 {
	// Derivatives are evaluated before any divergent branch or discard. Match
	// the half-float LOD stored in the winner payload exactly across all passes.
	let lod = clamp(texture_lod_from_gradients(dpdxFine(uv), dpdyFine(uv),
		vec2f(textureDimensions(base_textures))), 0.0, f32(textureNumLevels(base_textures)-1u));
	return unpack2x16float(pack2x16float(vec2f(lod, 0.0))).x;
}

@fragment
fn transparent_depth(input: VertexOutput, @builtin(front_facing) front: bool) {
	let lod = raster_lod(input.uv);
	if input.position.z >= textureLoad(opaque_depth, vec2i(input.position.xy), 0) { discard; }
	let material = materials[metadata[input.instance].y];
	if !front && material.flags.y == 0u { discard; }
	let alpha = opacity(material, input.uv, lod);
	// Ignore fully transparent texels; retain every nonzero opacity, however low.
	if max(alpha.x, max(alpha.y, alpha.z)) <= 0.0 { discard; }
}

@fragment
fn select_winner(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4u {
	let lod = raster_lod(input.uv);
	let pixel = vec2i(input.position.xy);
	// Selection has its own writable depth; reject opaque occlusion explicitly.
	if input.position.z >= textureLoad(opaque_depth, pixel, 0) { discard; }
	let info = metadata[input.instance];
	let material = materials[info.y];
	if !front && material.flags.y == 0u { discard; }
	let alpha = opacity(material, input.uv, lod);
	let p = max(alpha.x, max(alpha.y, alpha.z));
	let seed = hash_u32(info.z ^ hash_u32(input.triangle + 1u));
	let dimensions = textureDimensions(blue_noise);
	let noise_pixel = (vec2u(pixel) + vec2u(seed & 65535u, seed >> 16u)) % dimensions;
	// Integer wrapping tiles the original rank map exactly. Surface-specific
	// translations preserve its blue-noise thresholds without modifying ranks.
	let u = textureLoad(blue_noise, vec2i(noise_pixel), 0).r;
	if u >= p { discard; }
	// Zero instance is the empty sentinel. Hardware depth writes keep all four
	// payload components consistent with the nearest accepted triangle.
	return vec4u(input.instance + 1u, input.triangle,
		pack2x16unorm(clamp(input.barycentric, vec2f(0.0), vec2f(1.0))),
		pack2x16float(vec2f(lod, select(-1.0, 1.0, front))));
}
