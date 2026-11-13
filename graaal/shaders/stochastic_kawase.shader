texture<0, 0> source_pair: texture_2d_array<f32>;
#if PARAMS.get("banded_proxy", false)
texture<0, 4> local_pair: texture_2d_array<f32>;
#endif
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

#if PARAMS.get("banded_proxy", false)
fn proxy_guide(position: vec2f, layer: u32) -> vec4f {
	let full_size = vec2i(textureDimensions(band_proxy));
	let source_size = vec2f(textureDimensions(source_pair));
	let full_position = (position+0.5)*vec2f(full_size)/source_size-0.5;
	let pixel = clamp(vec2i(round(full_position)), vec2i(0), full_size-1);
	let index = (u32(pixel.y)*u32(full_size.x)+u32(pixel.x))*{{PARAMS.get("sample_count", 1)}}u+layer;
	let total = f32(band_counts[index].total)/65536.0;
	let proxy = sample_texture_load(band_proxy, pixel, layer).rgb/max(total, 1e-6);
	return vec4f(proxy, select(0.0, 1.0, total > 1e-5));
}

fn edge_weight(center: vec4f, neighbor: vec4f) -> f32 {
	if center.a != neighbor.a { return 0.0; }
	if center.a == 0.0 { return 1.0; }
	let difference = length(center.rgb-neighbor.rgb);
	return exp(-difference*difference/(2.0*0.2*0.2));
}
fn guided_pair(position: vec2f, layer: u32, guide: vec4f) -> vec4f {
	let base = vec2i(floor(position));
	let fraction = fract(position);
	var value = vec4f(0.0);
	var mass = 0.0;
	for (var y = 0; y < 2; y += 1) {
		for (var x = 0; x < 2; x += 1) {
			let q = base+vec2i(x, y);
			let spatial = select(1.0-fraction.x, fraction.x, x == 1)*select(1.0-fraction.y, fraction.y, y == 1);
			let weight = spatial*edge_weight(guide, proxy_guide(vec2f(q), layer));
			value += weight*read_pair(q, layer);
			mass += weight;
		}
	}
	return value/max(mass, 1e-6);
}
#else
fn proxy_guide(position: vec2f, layer: u32) -> vec4f { return vec4f(0.0); }
fn edge_weight(center: vec4f, neighbor: vec4f) -> f32 { return 1.0; }
fn guided_pair(position: vec2f, layer: u32, guide: vec4f) -> vec4f { return linear_pair(position, layer); }
#endif

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(output_pair);
	if any(id.xy >= size) { return; }
	let p = (vec2f(id.xy)+0.5)*vec2f(textureDimensions(source_pair))/vec2f(size)-0.5;
	let center_guide = proxy_guide(p, id.z);
	var value = 4.0*guided_pair(p, id.z, center_guide);
	var mass = 0.0;
	for (var y = -1; y <= 1; y += 2) {
		for (var x = -1; x <= 1; x += 2) {
			let q = p+vec2f(f32(x), f32(y));
			let weight = edge_weight(center_guide, proxy_guide(q, id.z));
			value += weight*guided_pair(q, id.z, center_guide);
			mass += weight;
		}
	}
	textureStore(output_pair, vec2i(id.xy), i32(id.z), value/(mass+4.0));
}

@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(output_pair);
	if any(id.xy >= size) { return; }
	let p = (vec2f(id.xy)+0.5)*vec2f(textureDimensions(source_pair))/vec2f(size)-0.5;
	let center_guide = proxy_guide(p, id.z);
	var value = vec4f(0.0);
	var mass = 0.0;
	for (var axis = 0u; axis < 4u; axis += 1u) {
		let offset = select(select(vec2f(-2.0, 0.0), vec2f(2.0, 0.0), axis == 1u), select(vec2f(0.0, -2.0), vec2f(0.0, 2.0), axis == 3u), axis >= 2u);
		let q = p+offset;
		let weight = edge_weight(center_guide, proxy_guide(q, id.z));
		value += weight*guided_pair(q, id.z, center_guide);
		mass += weight;
	}
	for (var y = -1; y <= 1; y += 2) {
		for (var x = -1; x <= 1; x += 2) {
			let q = p+vec2f(f32(x), f32(y));
			let weight = 2.0*edge_weight(center_guide, proxy_guide(q, id.z));
			value += weight*guided_pair(q, id.z, center_guide);
			mass += weight;
		}
	}
#if PARAMS.get("banded_proxy", false)
	if mass <= 1e-6 && all(textureDimensions(output_pair) == textureDimensions(local_pair)) {
		value = textureLoad(local_pair, vec2i(id.xy), i32(id.z), 0);
		mass = 1.0;
	}
#endif
	textureStore(output_pair, vec2i(id.xy), i32(id.z), value/max(mass, 1e-6));
}

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(output_color)) { return; }
	let pixel = vec2i(id.xy);
	var pair = read_pair(pixel, id.z);
#if PARAMS.get("banded_proxy", false)
	if pair.a <= 1e-8 { pair = textureLoad(local_pair, pixel, i32(id.z), 0); }
#endif
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
