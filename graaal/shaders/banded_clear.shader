texture<1, 0> payloads: texture_storage_2d_array<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(payloads)) { return; }
	textureStore(payloads, vec2i(id.xy), i32(id.z), vec4u(0u));
}
