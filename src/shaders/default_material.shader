#extends "material_surface.shaderlib"

#block parameters
{{ super() }}
{{ parameter("base_color", "vec4f", (1, 1, 1, 1)) }}
{{ parameter("emission", "vec3f", (0, 0, 0)) }}
{{ parameter("roughness", "f32", 0.7) }}
{{ parameter("metallic", "f32", 0) }}
{{ parameter("opacity_rgb", "vec3f", (1, 1, 1)) }}
{{ parameter("alpha_cutoff", "f32", 0.5) }}
{{ parameter("alpha_mode", "u32", 2 if TRANSPARENT else 0) }}
{{ parameter("texture_layer", "u32", 0) }}
{{ parameter("double_sided", "bool", true) }}
{{ parameter("grazing_fade", "bool", false) }}
{{ parameter("grazing_start", "f32", 70) }}
{{ parameter("grazing_end", "f32", 85) }}
{{ parameter("ambient", "f32", 0.25) }}
{{ parameter("diffuse", "f32", 0.75) }}
{{ parameter("specular", "f32", 1) }}
{{ parameter("tint", "vec4f", (1, 1, 1, 1), "instance") }}
{{ parameter("opacity_multiplier", "f32", 1, "instance") }}
#endblock

#block helpers
fn default_lighting_{{PREFIX}}(material: Material, p: Params{{PREFIX}}, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
	let color = base_color(material, uv, lod).rgb;
	let n = normal/max(length(normal), 1e-6);
	let l = normalize(frame.light_dir.xyz);
	let v = normalize(frame.camera_position.xyz-position);
	let h = normalize(l+v);
	let diffuse = p.ambient+p.diffuse*max(dot(n, l), 0.0);
	let exponent = mix(128.0, 4.0, clamp(material.surface.x, 0.0, 1.0));
	let specular = p.specular*pow(max(dot(n, h), 0.0), exponent)*max(dot(n, l), 0.0);
	return color*diffuse*(1.0-0.6*material.surface.y)
		+mix(vec3f(0.04), color, material.surface.y)*specular+material.emission.rgb;
}
#endblock

#block surface
	var material: Material;
	material.base_color = p.base_color*p.tint;
	material.base_color.a *= p.opacity_multiplier;
	material.emission = vec4f(p.emission, 0.0);
	material.surface = vec4f(p.roughness, p.metallic, f32(p.alpha_mode), p.alpha_cutoff);
	material.opacity_rgb = vec4f(p.opacity_rgb, 0.0);
	material.flags = vec4u(p.texture_layer, select(0u, 1u, p.double_sided), 0u, 0u);
	return material;
#endblock

#block grazing
	return vec3f(p.grazing_start, p.grazing_end, select(0.0, 1.0, p.grazing_fade));
#endblock

#block lighting
	return default_lighting_{{PREFIX}}(material, p, uv, position, normal, lod);
#endblock
