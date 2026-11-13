#extends "material_surface.shaderlib"

#block parameters
{{ super() }}
<MaterialParameters> base_color: vec4f = (1, 1, 1, 1);
<MaterialParameters> emission: vec3f = (0, 0, 0);
<MaterialParameters> roughness: f32 = 0.7;
<MaterialParameters> metallic: f32 = 0;
<MaterialParameters> opacity_rgb: vec3f = (1, 1, 1);
<MaterialParameters> alpha_cutoff: f32 = 0.5;
<MaterialParameters> alpha_mode: u32 = 2 if TRANSPARENT else 0;
<MaterialParameters> texture_layer: u32 = 0;
<MaterialParameters> double_sided: bool = false;
<MaterialParameters> grazing_fade: bool = false;
<MaterialParameters> grazing_start: f32 = 70;
<MaterialParameters> grazing_end: f32 = 85;
<MaterialParameters> ambient: f32 = 0.25;
<MaterialParameters> diffuse: f32 = 0.75;
<MaterialParameters> specular: f32 = 1;
<InstanceParameters> tint: vec4f = (1, 1, 1, 1);
<InstanceParameters> opacity_multiplier: f32 = 1;
#endblock

#block helpers
fn default_lighting_{{PREFIX}}(material: Material, p: Params{{PREFIX}}, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
	return evaluate_lighting(material, uv, position, normal, lod, p.ambient, p.diffuse, p.specular);
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
