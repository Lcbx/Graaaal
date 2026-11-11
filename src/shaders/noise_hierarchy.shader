#include "utils_hash.shaderlib"
{{ binding('source', 'texture_2d<f32>', 0, 0) }}
{{ binding('destination', 'texture_storage_2d<r32float, write>', 0, 1) }}

@compute @workgroup_size(8, 8)
fn average_children(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(destination)) { return; }
	let p = vec2i(id.xy*2u);
	let value = (textureLoad(source, p, 0).r + textureLoad(source, p+vec2i(1, 0), 0).r
		+ textureLoad(source, p+vec2i(0, 1), 0).r + textureLoad(source, p+vec2i(1, 1), 0).r)*0.25;
	textureStore(destination, vec2i(id.xy), vec4f(value));
}

@compute @workgroup_size(8, 8)
fn rank_values(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(destination);
	if any(id.xy >= size) { return; }
	let index = id.y*size.x+id.x;
	let value = textureLoad(source, vec2i(id.xy), 0).r;
	let tie = hash_u32(index);
	var rank = 0u;
	for (var i = 0u; i < size.x*size.y; i += 1u) {
		let other = textureLoad(source, vec2i(vec2u(i % size.x, i / size.x)), 0).r;
		let other_tie = hash_u32(i);
		if other < value || (other == value && (other_tie < tie || (other_tie == tie && i < index))) {
			rank += 1u;
		}
	}
	textureStore(destination, vec2i(id.xy), vec4f(f32(rank)));
}
