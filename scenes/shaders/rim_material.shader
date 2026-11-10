#extends "default_material.shader"

#block parameters
{{ super() }}
{{ parameter("rim_strength", "f32", 0.8) }}
{{ parameter("rim_color", "vec3f", (0.3, 0.7, 1)) }}
#endblock

#block lighting
	let view = normalize(frame.camera_position.xyz-position);
	let rim = pow(1.0-abs(dot(normalize(normal), view)), 3.0)*p.rim_strength;
	return default_lighting_{{PREFIX}}(material, p, uv, position, normal, lod)+p.rim_color*rim;
#endblock
