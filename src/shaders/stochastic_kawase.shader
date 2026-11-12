texture<0, 0> source_pair: texture_2d_array<f32>;
#if not PARAMS.get("banded_mixed", false)
sample_texture<0, 1> transmission: f32;
#endif
#if PARAMS.get("banded_proxy", false)
sample_texture<0, 2> band_proxy: f32;
<BandCount> total: u32;
<BandCount> running: u32;
storage<0, 3, read> band_counts: array<BandCount>;
#endif
texture<1, 0> output_pair: texture_storage_2d_array<rgba16float, write>;
#if PARAMS.get("sample_count", 1) > 1
texture<1, 1> output_color: texture_storage_2d_array<rgba16float, write>;
#else
texture<1, 1> output_color: texture_storage_2d<rgba16float, write>;
#endif

fn read_pair(pixel: vec2i, layer: u32) -> vec4f {
	let size = vec2i(textureDimensions(source_pair));
	return textureLoad(source_pair, clamp(pixel, vec2i(0), size-1), i32(layer), 0);
}

fn linear_pair(position: vec2f, layer: u32) -> vec4f {
	let base = vec2i(floor(position));
	let t = fract(position);
	return mix(mix(read_pair(base, layer), read_pair(base+vec2i(1, 0), layer), t.x),
		mix(read_pair(base+vec2i(0, 1), layer), read_pair(base+vec2i(1, 1), layer), t.x), t.y);
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(output_pair);
	if any(id.xy >= size) { return; }
	let p = (vec2f(id.xy)+0.5)*vec2f(textureDimensions(source_pair))/vec2f(size)-0.5;
	var value = 4.0*linear_pair(p, id.z);
	for (var y = -1; y <= 1; y += 2) {
		for (var x = -1; x <= 1; x += 2) { value += linear_pair(p+vec2f(f32(x), f32(y)), id.z); }
	}
	textureStore(output_pair, vec2i(id.xy), i32(id.z), value/8.0);
}

@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(output_pair);
	if any(id.xy >= size) { return; }
	let p = (vec2f(id.xy)+0.5)*vec2f(textureDimensions(source_pair))/vec2f(size)-0.5;
	var value = linear_pair(p+vec2f(-2.0, 0.0), id.z)+linear_pair(p+vec2f(2.0, 0.0), id.z)
		+linear_pair(p+vec2f(0.0, -2.0), id.z)+linear_pair(p+vec2f(0.0, 2.0), id.z);
	for (var y = -1; y <= 1; y += 2) {
		for (var x = -1; x <= 1; x += 2) { value += 2.0*linear_pair(p+vec2f(f32(x), f32(y)), id.z); }
	}
	textureStore(output_pair, vec2i(id.xy), i32(id.z), value/12.0);
}

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(output_color)) { return; }
	let pixel = vec2i(id.xy);
	let pair = read_pair(pixel, id.z);
	var mu = pair.rgb/max(pair.a, 1e-8);
#if PARAMS.get("banded_proxy", false)
	let index = (id.y*textureDimensions(output_color).x+id.x)*{{PARAMS.get("sample_count", 1)}}u+id.z;
	let s = f32(band_counts[index].total)/65536.0;
	mu += sample_texture_load(band_proxy, pixel, id.z).rgb/max(s, 1e-8);
#endif
#if PARAMS.get("banded_mixed", false)
	let value = vec4f(mu, pair.a);
#else
	let t = sample_texture_load(transmission, pixel, id.z);
	let value = vec4f((vec3f(1.0)-t.rgb)*mu, pair.a);
#endif
#if PARAMS.get("sample_count", 1) > 1
	textureStore(output_color, pixel, i32(id.z), value);
#else
	textureStore(output_color, pixel, value);
#endif
}
