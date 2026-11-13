<Optical> red: u32;
<Optical> green: u32;
<Optical> blue: u32;
<Optical> scalar: u32;
<Optical> coverage: u32;
storage<1, 0, read> optical: array<Optical>;
texture<1, 1> transmission: texture_storage_2d<rgba32float, write>;
texture<1, 2> coverage: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(transmission);
	if any(id.xy >= size) { return; }
	var t = vec4f(0.0);
	var s = 0.0;
	for (var sample = 0u; sample < {{PARAMS.get("sample_count", 1)}}u; sample += 1u) {
		let value = optical[(id.y*size.x+id.x)*{{PARAMS.get("sample_count", 1)}}u+sample];
		let tau = vec3f(vec3u(value.red, value.green, value.blue))*(1.0/1048576.0);
		t += vec4f(exp(-tau), exp(-f32(value.scalar)*(1.0/1048576.0)));
		s += f32(value.coverage)*(1.0/1048576.0);
	}
	textureStore(transmission, vec2i(id.xy), t/{{PARAMS.get("sample_count", 1)}}.0);
	textureStore(coverage, vec2i(id.xy), vec4f(s/{{PARAMS.get("sample_count", 1)}}.0, 0.0, 0.0, 0.0));
}
