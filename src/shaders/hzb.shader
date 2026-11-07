@group(0) @binding(0) var src_depth: texture_2d<f32>;
@group(0) @binding(1) var dst_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var src_prepass: texture_depth_2d;

@compute @workgroup_size(16, 16)
fn reduce_depth(@builtin(global_invocation_id) id: vec3u) {
	let dst_size = textureDimensions(dst_depth);
	if any(id.xy >= dst_size) { return; }
	let src_size = textureDimensions(src_prepass);
	// Reduce prepass depth directly, including the edges of odd-sized targets.
	let first = id.xy * src_size / dst_size;
	let end = ((id.xy + vec2u(1u)) * src_size + dst_size - vec2u(1u)) / dst_size;
	var max_d = 0.0;
	for (var y = first.y; y < end.y; y += 1u) {
		for (var x = first.x; x < end.x; x += 1u) {
			max_d = max(max_d, textureLoad(src_prepass, vec2i(i32(x), i32(y)), 0));
		}
	}
	textureStore(dst_depth, vec2i(id.xy), vec4f(max_d, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
	let dst_size = textureDimensions(dst_depth);
	if any(id.xy >= dst_size) { return; }
	let src_size = textureDimensions(src_depth, 0);
	// Overlap footprints for odd sizes so no edge texels are discarded.
	let first = id.xy * src_size / dst_size;
	let end = ((id.xy + vec2u(1u)) * src_size + dst_size - vec2u(1u)) / dst_size;
	var max_d = 0.0;
	for (var y = first.y; y < end.y; y += 1u) {
		for (var x = first.x; x < end.x; x += 1u) {
			max_d = max(max_d, textureLoad(src_depth, vec2i(i32(x), i32(y)), 0).r);
		}
	}
	textureStore(dst_depth, vec2i(id.xy), vec4f(max_d, 0.0, 0.0, 0.0));
}
