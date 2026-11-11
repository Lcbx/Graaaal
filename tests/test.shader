
<VertexInput> vertexPosition: vec3f;
<VertexInput> vertexNormal: vec3f;
<VertexInput> vertexTexCoord: vec2f;

<Uniforms> matModel: mat4x4f;
<Uniforms> mvp: mat4x4f;

// Kept to exercise struct/type parsing without requiring creation of
// a CPU uniform dtype for padded vec3 arrays yet.
<ArrayParsingTest> test: array<vec3f, 5>;
<ArrayParsingTest> test2: array<array<vec3f, 5>, 3>;

uniform<0, 0> uniforms: Uniforms;

#if FEATURES.BIAS

const bias: f32 = {{ PARAMS.bias | default(0.0001) }};

#include "test_include.shader"

#endif

<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> @location(0) fragColor: vec4f;

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;

	output.fragColor = vec4f(0.5, 0.1, 1.0, 1.0);
	var position = uniforms.mvp * vec4f(input.vertexPosition, 1.0);

#if FEATURES.BIAS
	position.z *= plus_one(bias);
#endif

	output.position = position;
	return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
	return input.fragColor;
}