#include "stochastic_reconstruction.shaderlib"

struct FilterVertex {
	@builtin(position) position: vec4f,
	@location(0) @interpolate(flat) sample: u32,
}

@vertex
fn fullscreen(@builtin(vertex_index) index: u32, @builtin(instance_index) sample: u32) -> FilterVertex {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return FilterVertex(vec4f(uv*2.0-1.0, 0.0, 1.0), sample);
}

@fragment
fn filter_pair(input: FilterVertex) -> @location(0) vec4f {
	return reconstruct_pair(vec2i(input.position.xy), input.sample, vec2i(0), 0u);
}
