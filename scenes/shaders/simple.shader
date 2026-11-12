

<Uniforms> view: mat4x4f;
<Uniforms> proj: mat4x4f;
<Uniforms> light_dir: vec4f;
<Uniforms> light_view_proj: mat4x4f;

uniform<0, 0> uniforms: Uniforms;

texture<1, 0> shadow_map: texture_depth_2d;

{{ binding('shadow_sampler', 'sampler_comparison', 1, 1) }}

<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;
<VertexInput> iPosition: vec3f;
<VertexInput> iTint: u32;
<VertexInput> iRotation: vec2u;
<VertexInput> iScale: vec2u;

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> normal: vec3f;
<VertexOutput> @interpolate(flat) tint: u32;
<VertexOutput> shadow_pos: vec3f;


#include "utils.shaderlib"
#from "utils.shaderlib" import instance_transform_unpacking, unpack_srgb_color
{{ instance_transform_unpacking(InstanceType="VertexInput") }}
{{ unpack_srgb_color() }}


@vertex
fn vertex(input: VertexInput) -> VertexOutput {

	let trans = unpack_instance_transform(input.position, input);

	var normal = input.normal;
	if any(trans.scale != vec3f(1.0)) {
		normal /= trans.scale;
	}

	let normal_ws = normalize(quat_rotate(trans.rotation, normal));
	let world = vec4f(trans.world_pos, 1.0);
	let view_pos = uniforms.view * world;
	let shadow_clip = uniforms.light_view_proj * world;
	let shadow_ndc = shadow_clip.xyz / shadow_clip.w;

	var output: VertexOutput;
	output.position = uniforms.proj * view_pos;
	output.normal = normal_ws;
	output.tint = input.iTint;
	output.shadow_pos = vec3f(
		shadow_ndc.x * 0.5 + 0.5,
		0.5 - shadow_ndc.y * 0.5,
		shadow_ndc.z,
	);
	return output;
}

@vertex
fn shadow_vertex(input: VertexInput) -> @builtin(position) vec4f {

	let trans = unpack_instance_transform(input.position, input);
	return uniforms.light_view_proj * vec4f(trans.world_pos, 1.0);
}

fn sample_shadow(position: vec3f) -> f32 {
	let in_bounds = all(position.xy >= vec2f(0.0)) &&
		all(position.xy <= vec2f(1.0)) &&
		position.z >= 0.0 && position.z <= 1.0;

	if !in_bounds {
		return 1.0;
	}

	return textureSampleCompare(
		shadow_map,
		shadow_sampler,
		position.xy,
		position.z - 0.001,
	);
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
	let color = unpack_rgba8_srgb(input.tint);
	let light = max(dot(input.normal, uniforms.light_dir.xyz), 0.0);
	let shadow = sample_shadow(input.shadow_pos);
	let lighting = mix(0.35, 1.0, light) * mix(0.45, 1.0, shadow);
	return vec4f(color.rgb * lighting, color.a);
}