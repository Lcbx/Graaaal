"""Dictionary materials, shared values, instance overrides and inherited shaders.

Run: python main.py scenes/material_example.py [--frames 120]
M toggles MSAA; Space toggles orbit; D replaces the opaque shader's default.
The bottom-left pair shares a dictionary; the bottom-right transparent sphere
adds rim lighting through a Jinja child template. No transparency code is copied.
The top-middle sphere uses time-driven vertex displacement in the WGSL below.
"""
from pathlib import Path
from math import cos, sin

import numpy as np

from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.ECS import ECS
from numpyecs.EcsMaterials import EcsMaterialAdapter, MaterialInstance, MaterialTransform
from numpyecs.MaterialRenderer import MaterialMesh, DrawInstance, MaterialRenderer
from numpyecs.Math import Camera, Vec3
from numpyecs.RenderContext import RenderContext
from numpyecs.TransparencyData import VERTEX_DTYPE


STRIPED_SHADER = '''
// Extra shared parameters are reflected just like other shader structs.
<MaterialParameters> stripe_color: vec3f;
<MaterialParameters> stripe_frequency: f32;
<MaterialParameters> displacement: f32;
<MaterialParameters> speed: f32;
<FrameParameters> time: f32;

const default_stripe_color = vec3f(1.0, 0.15, 0.03);
const default_stripe_frequency = 8.0;
const default_displacement = 0.18;
const default_speed = 2.0;

fn vertex(p: Params, position: vec3f, normal: vec3f, uv: vec2f) -> MaterialVertex {
	// Ripple the sphere radially. Return the analytic normal of the radial graph.
	let radius = max(length(position), 1e-6);
	let direction = position/radius;
	let phase = direction.y*8.0-p.time*p.speed;
	let displaced_radius = radius+p.displacement*sin(phase);
	let gradient = p.displacement*8.0*cos(phase)*(vec3f(0.0, 1.0, 0.0)-direction.y*direction);
	let displaced_normal = normalize(direction-gradient/max(displaced_radius, 1e-6));
	return MaterialVertex(direction*displaced_radius, displaced_normal, uv);
}

fn shade(material: Material, p: Params, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
	// UV-based bands work in both fragment and compute shading without derivatives.
	let band = smoothstep(-0.2, 0.2, sin(uv.y*p.stripe_frequency*6.2831853));
	let color = mix(material.base_color.rgb, p.stripe_color, band);
	var striped = material;
	striped.base_color = vec4f(color, material.base_color.a);
	return default_lighting(striped, p, uv, position, normal, lod);
}
'''


def sphere_vertices():
	u, v = np.meshgrid(np.linspace(0, 2*np.pi, 33), np.linspace(0, np.pi, 17))
	normals = np.stack((np.sin(v)*np.cos(u), np.cos(v), np.sin(v)*np.sin(u)), axis=-1)
	grid = np.zeros((17, 33), dtype=VERTEX_DTYPE)
	grid["position"], grid["normal"] = normals, normals
	grid["uv"] = np.stack((u/(2*np.pi), v/np.pi), axis=-1)
	return np.stack((grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:],
		grid[:-1, :-1], grid[1:, 1:], grid[:-1, 1:]), axis=2).reshape(-1, 3)[:, (0, 2, 1)].reshape(-1)


def make_instances():
	"""CPU-only setup; materials and overrides are ordinary mutable dictionaries."""
	vertices = sphere_vertices()
	opaque = DefaultMaterialShader(default_material={"base_color": (0.8, 0.2, 0.08, 1)})
	glass = DefaultMaterialShader(transparent=True, template=Path(__file__).parent / "shaders/glass_material.shader")
	rim = DefaultMaterialShader(transparent=True, template=Path(__file__).parent / "shaders/rim_material.shader",
		default_material={"base_color": (0.25, 0.6, 0.8, 0.45)})
	striped = DefaultMaterialShader(source=STRIPED_SHADER,
		default_material={"base_color": (0.08, 0.2, 0.75, 1), "roughness": 0.3})
	shared_glass = {"base_color": (0.08, 0.35, 0.55, 0.65), "roughness": 0.2, "specular": 3.0,
		"metallic": 0.65, "ambient": 0.08, "diffuse": 0.35,
		"grazing_start": 55.0, "grazing_end": 85.0}
	glass_mesh = MaterialMesh(vertices, glass, material=shared_glass)
	opaque_mesh = MaterialMesh(vertices, opaque)
	instances = [
		DrawInstance(opaque_mesh, position=(-2.6, 2.6, 0), scale=(0.8, 0.8, 0.8)),
		DrawInstance(opaque_mesh, material={"base_color": (0.15, 0.65, 0.2, 1)}, position=(2.6, 2.6, 0), scale=(0.8, 0.8, 0.8)),
		DrawInstance(glass_mesh, overrides={"grazing_fade": True}, position=(-2.6, 0, 0)),
		DrawInstance(glass_mesh, overrides={"tint": (1, 0.7, 0.45, 1)}, position=(0, 0, 0)),
		DrawInstance(MaterialMesh(vertices, rim), position=(2.6, 0, 0)),
		DrawInstance(MaterialMesh(vertices, striped, bounds=((0, 0, 0), (1.3, 1.3, 1.3))),
			position=(0, 2.6, 0), scale=(0.8, 0.8, 0.8)),
	]
	return instances, shared_glass, opaque


def make_world():
	"""Entity-owned materials and separate numeric transform components."""
	items, shared, opaque = make_instances()
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	entities = world.create(len(items))
	for entity, item in zip(entities, items):
		world.add(entity, MaterialInstance(item.mesh, item.material, item.overrides, item.shader),
			MaterialTransform(Vec3(item.position), scale=Vec3(item.scale)))
	return world, entities, list(dict.fromkeys(item.mesh for item in items)), shared, opaque


def run_scene():
	from numpyecs.ImGui import ImGui
	from imgui_bundle import imgui

	RenderContext.init_window(1100, 760, "Materials | M: MSAA | D: default | Space: orbit",
		required_gpu_features=["indirect-first-instance"])
	world, entities, meshes, shared, opaque = make_world()
	renderer = MaterialRenderer(meshes=meshes)
	adapter = EcsMaterialAdapter(renderer, world)
	overrides = world.get(MaterialInstance)[entities].overrides
	striped = meshes[-1].shader.default_material
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50, near=0.05, far=100)
	orbit, angle, alternate = False, 0.0, False
	animation_time, animate = 0.0, True
	def set_alternate_color(enabled):
		nonlocal alternate
		alternate = enabled
		opaque.default_material = {"base_color": (0.6, 0.1, 0.8, 1) if alternate else (0.8, 0.2, 0.08, 1)}
		adapter.touch_materials(opaque.default_material)
	def on_char(code):
		nonlocal orbit
		if chr(code).lower() == "m": renderer.set_sample_count(5-renderer.transparency.sample_count)
		elif chr(code) == " ": orbit = not orbit
		elif chr(code).lower() == "d":
			set_alternate_color(not alternate)
		return True
	def draw_controls():
		nonlocal animate
		# Labels follow the objects during orbit and use ImGui's logical pixels.
		width, height = imgui.get_io().display_size
		vp = camera.view() @ camera.projection(width/height)
		labels = ("Default material (D)", "Explicit green override", "Shared glass A: instance fade",
			"Shared glass B: instance tint", "Inherited rim shader", "Animated WGSL shader")
		transforms = world.get(MaterialTransform)[entities]
		for position, scale, label in zip(transforms.position, transforms.scale, labels):
			anchor = np.array((*position, 1.0))
			anchor[1] -= scale[1]+0.15
			clip = anchor @ vp
			if clip[3] <= 0: continue
			xy = clip[:2]/clip[3]
			x, y = (xy[0]*0.5+0.5)*width, (0.5-xy[1]*0.5)*height
			imgui.get_background_draw_list().add_text((x-imgui.calc_text_size(label).x/2, y), 0xffffffff, label)
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((340, 510), imgui.Cond_.first_use_ever)
		imgui.begin("Dictionary materials")
		try:
			imgui.text_wrapped("Six spheres: three opaque above, three transparent below. The bottom-left and bottom-middle spheres share the sliders below.")
			changed, value = imgui.checkbox("Alternate opaque base color (D)", alternate)
			if changed: set_alternate_color(value)
			changed, value = imgui.slider_float("Shared roughness", shared["roughness"], 0.02, 1)
			if changed:
				shared["roughness"] = value
				adapter.touch_materials(shared)
			changed, value = imgui.slider_float("Shared specular", shared["specular"], 0, 8)
			if changed:
				shared["specular"] = value
				adapter.touch_materials(shared)
			changed, value = imgui.checkbox("Left glass grazing fade", overrides[2]["grazing_fade"])
			if changed:
				overrides[2]["grazing_fade"] = value
				adapter.touch_materials(overrides[2])
			changed, value = imgui.slider_float("Middle glass opacity multiplier", overrides[3].get("opacity_multiplier", 1), 0, 1)
			if changed:
				overrides[3]["opacity_multiplier"] = value
				adapter.touch_materials(overrides[3])
			imgui.separator()
			imgui.text("Top-middle: shader written in this script")
			changed, value = imgui.slider_float("Stripe frequency", striped.get("stripe_frequency", 8.0), 1, 20)
			if changed:
				striped["stripe_frequency"] = value
				adapter.touch_materials(striped)
			changed, value = imgui.color_edit3("Stripe color", striped.get("stripe_color", (1.0, 0.15, 0.03)))
			if changed:
				striped["stripe_color"] = tuple(value)
				adapter.touch_materials(striped)
			_, animate = imgui.checkbox("Animate displacement", animate)
			for name, label, default, low, high in (("displacement", "Displacement", 0.18, 0.0, 0.3),
				("speed", "Animation speed", 2.0, 0.0, 5.0)):
				changed, value = imgui.slider_float(label, striped.get(name, default), low, high)
				if changed:
					striped[name] = value
					adapter.touch_materials(striped)
			imgui.text_wrapped("D changes only the top-left default. Top-right has an explicit green material. Bottom-right adds inherited rim lighting. M: MSAA; Space: orbit.")
		finally: imgui.end()
	RenderContext.subscribe_event("char", on_char)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if min(RenderContext.window_dimensions) <= 0: continue
				if orbit: angle += RenderContext.frame_time*0.3
				if animate: animation_time += RenderContext.frame_time
				camera.position = Vec3(9*sin(angle), 1.3, 9*cos(angle))
				adapter.sync()
				commands = RenderContext.commands("material_example")
				renderer.record(commands, RenderContext.window_dimensions, camera, RenderContext.screen(), time=animation_time)
				RenderContext.submit(commands.finish())
				gui.render(draw_controls)
	finally:
		RenderContext.event_handlers["char"].remove(on_char)


if __name__ == "__main__":
	run_scene()
