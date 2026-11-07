"""Single-winner stochastic OIT, using DrawBatches for every geometry pass.

Run: python main.py scenes/transparency_example.py [--frames 120]
Space toggles orbit, scrolling zooms, R reverses transparent submission order.
3 and 5 select the bilateral reconstruction kernel size.
D cycles diagnostic views; A toggles adaptation; F shows/hides ImGui settings.
Drag settings or Ctrl+click to type; Up/Down and Left/Right also select/edit.
Imports only prepare CPU helpers; graphics are initialized by run_scene().
"""

from dataclasses import asdict, replace
import json
from math import cos, sin
from pathlib import Path

import numpy as np
import glfw

from ECS import ECS, component
from DrawBatches import DrawBatches
from RenderContext import (
	BufferUsage, GpuBuffer, Mesh, RenderContext, Texture, TextureUsage,
)
from Utils import (
	Camera, Vec3, Vec4, build_material_texture_array, load_blue_noise, load_scene,
	mesh_instance_dtype, pack_quaternion, pack_scale,
)
from tranparency_pass import (
	BilateralFilter, DEBUG_VIEWS, TransparencyPass,
)


PATH = Path(__file__).parent / "resources/small_scene.glb"
BLUE_NOISE_PATH = Path(__file__).parent / "resources/blue_noise/HDR_L_0.png"
TEXTURE_SIZE = 512


@component
class Transform:
	position: Vec3
	scale: Vec3
	rotation: Vec4


@component
class MeshRef:
	shader_id: np.uint32
	lod_id: np.uint32
	triangle_offset: np.uint32
	material_id: np.uint32
	stable_id: np.uint32


class GeometryBatches:
	def __init__(self, primitives, materials, transparency, transparent):
		self.world = ECS()
		transforms, refs = self.world.register(Transform, MeshRef)
		self.data = DrawBatches(mesh_instance_dtype)
		# This scene copies frustum results into the second pass's destinations.
		# Give only the two copy sources the additional usage before registration.
		for name in ("prepass_visible_instances", "prepass_draw_cmd"):
			buffer = self.data.buffers[name]
			self.data.buffers[name] = GpuBuffer(buffer.content,
				buffer.usage | BufferUsage.COPY_SRC, upload=False, label=name)
		self.metadata = GpuBuffer(np.zeros((1, 4), dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self.passes = transparency.create_geometry_passes(self.metadata, transparent=transparent)
		self.data.register_shader(0, self.passes.render_shader)
		for primitive in primitives:
			if (materials[primitive.material_id]["surface"][2] == 2) != transparent:
				continue
			mesh_id = primitive.stable_id
			mesh = Mesh(primitive.vertices, np.arange(len(primitive.vertices), dtype=np.uint32))
			self.data.register_mesh(mesh_id, mesh)
			self.data.register_lod_group(mesh_id, (mesh_id,))
			[entity] = self.world.create()
			self.world.add(entity, Transform(Vec3(0, 0, 0), Vec3(1, 1, 1), Vec4(0, 0, 0, 1)),
				MeshRef(0, mesh_id, primitive.triangle_offset, primitive.material_id, primitive.stable_id))
		self.data.sync_batches(self.world.query(Transform, MeshRef).entities(), transforms, refs)
		self.data.write_instance_fields(iPosition=(0, 0, 0), iTint=0,
			iRotation=pack_quaternion(np.array([0, 0, 0, 1], dtype=np.float32)),
			iScale=pack_scale(np.ones(3, dtype=np.float32)))
		# Metadata follows compact DrawBatches instance order, never entity order.
		metadata = np.zeros((max(1, len(self.data.entities)), 4), dtype=np.uint32)
		for column, name in enumerate(("triangle_offset", "material_id", "stable_id")):
			metadata[:len(self.data.entities), column] = self.data.mesh_field(name)
		self.metadata.write(metadata)

	def prepare(self, commands, camera, view_projection, dummy_hzb):
		self.data.refresh_bindings(dummy_hzb)
		self.data.update_cull_camera(camera.position, view_projection)
		self.data.reset_draw_counts(commands)
		self.data.cull_instances(commands, "frustum")
		# All batches participate in the first pass. Reuse its visibility and
		# indirect commands for selection; transparent culling never uses itself
		# as an occluder. No Hi-Z pass or sorting is needed for this small scene.
		for source, destination in (("prepass_visible_instances", "main_visible_instances"),
			("prepass_draw_cmd", "main_draw_cmd")):
			commands.copy_buffer(self.data.buffers[source], self.data.buffers[destination])


transparency: TransparencyPass
opaque: GeometryBatches
transparent: GeometryBatches
camera: Camera
center: np.ndarray
radius: float
angle = 0.65
orbit = False
distance = 0.0


def init_scene(path=PATH, *, bilateral_filter=None, blue_noise_path=BLUE_NOISE_PATH):
	"""Initialize the example's global render data after device setup."""
	global transparency, opaque, transparent, camera, center, radius, angle, orbit, distance
	primitives, materials, images, vertices = load_scene(path)
	# Ignore a large flat ground plane when framing the example.
	objects = [p.vertices["position"] for p in primitives if np.ptp(p.vertices["position"][:, 1]) > 0.01]
	positions = np.concatenate(objects or [p.vertices["position"] for p in primitives])
	center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
	radius = max(float(np.linalg.norm(np.ptp(positions, axis=0))) * 0.8, 1.0)
	textures = build_material_texture_array(images, materials, (TEXTURE_SIZE, TEXTURE_SIZE))
	noise = load_blue_noise(blue_noise_path)
	noise_size = (noise.shape[1], noise.shape[0])
	blue_noise = Texture(noise_size, format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": blue_noise.handle}, noise,
		{"bytes_per_row": noise.shape[1]*4}, (*noise_size, 1))
	transparency = TransparencyPass(
		materials=GpuBuffer(materials, BufferUsage.STORAGE | BufferUsage.COPY_DST),
		base_textures=textures.view(dimension="2d-array"),
		vertices=GpuBuffer(vertices, BufferUsage.STORAGE | BufferUsage.COPY_DST), blue_noise=blue_noise,
		bilateral_filter=bilateral_filter if bilateral_filter is not None else
			BilateralFilter(filter_near=radius, filter_far=radius*4))
	opaque = GeometryBatches(primitives, materials, transparency, False)
	transparent = GeometryBatches(primitives, materials, transparency, True)
	camera = Camera(tuple(center + (radius, radius*0.65, radius)), tuple(center), (0, 1, 0), 55,
		near=max(0.01, radius/1000), far=radius*100)
	angle, orbit, distance = 0.65, False, radius*1.4


# name, displayed label, increment, minimum, maximum
SETTINGS = (
	("adaptive", "Distance adaptation", None, None, None),
	("kernel_size", "Kernel size", None, None, None),
	("filter_near", "Near distance", 1.0, 0.0, None),
	("filter_far", "Far distance", 1.0, 0.0001, None),
	("spatial_sigma", "Near spatial sigma", 0.1, 0.01, None),
	("far_spatial_sigma", "Far spatial sigma", 0.1, 0.01, None),
	("depth_sigma", "Near relative depth sigma", 0.005, 0.0001, None),
	("far_depth_sigma", "Far relative depth sigma", 0.005, 0.0001, None),
	("transmittance_sigma", "Near transmittance sigma", 0.05, 0.001, None),
	("far_transmittance_sigma", "Far transmittance sigma", 0.05, 0.001, None),
	("min_depth_sigma", "Minimum depth tolerance", 0.001, 0.000001, None),
	("far_transmittance_weight", "Far transmittance weight", 0.05, 0.0, 1.0),
)


SETTING_HELP = {
	"adaptive": "Blend between the near and far filter settings as surfaces get farther from the camera. "
		"When off, only the near settings are used; the distance range and far settings have no effect.",
	"kernel_size": "Width of the reconstruction neighborhood: 3 x 3 uses up to 9 pixels; 5 x 5 uses up to 25. "
		"A larger neighborhood can smooth stochastic noise more, but costs more work and can soften fine detail.",
	"filter_near": "View depth, in scene units, where adaptation starts. Closer surfaces use the near settings. "
		"Depth is measured along the camera's viewing direction. Requires distance adaptation.",
	"filter_far": "View depth, in scene units, where adaptation reaches the far settings. "
		"Between the near and far distances, settings blend smoothly. Must exceed near distance. Requires distance adaptation.",
	"spatial_sigma": "Spatial spread in pixels for nearby surfaces. Larger values give neighboring pixels more influence, "
		"smoothing more noise; smaller values favor the center and preserve detail. The kernel size still limits the footprint.",
	"far_spatial_sigma": "Spatial spread in pixels at and beyond the far distance. Increase to smooth distant noise more; "
		"decrease to preserve finer detail. The kernel size still limits the footprint. Requires distance adaptation.",
	"depth_sigma": "Relative depth tolerance for nearby surfaces. For example, 0.02 is a 2% depth scale. "
		"Larger values blend more across depth changes; smaller values protect edges but leave fewer useful samples. "
		"Both opaque and transparent surface depths constrain the filter.",
	"far_depth_sigma": "Relative depth tolerance at and beyond the far distance. Larger values admit more neighbors "
		"across depth changes, reducing noise but risking color bleeding between surfaces. Requires distance adaptation.",
	"transmittance_sigma": "Tolerance for differences in RGB and scalar transmittance (the fraction of light passing through). "
		"Larger values blend more across opacity changes; smaller values protect opacity edges but can leave noisy gaps.",
	"far_transmittance_sigma": "Transmittance tolerance at and beyond the far distance. Larger values relax opacity edges. "
		"Has no effect at the far distance when far transmittance weight is zero. Requires distance adaptation.",
	"min_depth_sigma": "Minimum absolute depth tolerance, in scene units, used at every distance. "
		"The filter uses the larger of this value and depth times relative depth sigma. "
		"Increase to avoid overly strict filtering close to the camera; too much can blur nearby depth edges.",
	"far_transmittance_weight": "Strength of opacity-edge rejection at and beyond the far distance: "
		"0 ignores transmittance differences; 1 keeps their full influence. "
		"Depth edges still constrain the filter at zero. Requires distance adaptation.",
}

DEBUG_VIEW_HELP = {
	"final": "Final image: opaque color seen through the transparent surfaces, plus reconstructed transparent color.",
	"transmittance": "Scalar light transmission: white means fully transparent and black means fully opaque. "
		"Use this to check opacity independently of winner selection and reconstruction.",
	"winner": "Raw color of the selected transparent fragment at each pixel. "
		"Black pixels have no winner. This view reveals the stochastic pattern before filtering.",
	"reconstruction": "Filtered transparent contribution without the opaque background. "
		"Use this to compare smoothing and edge preservation while tuning.",
	"filter_support": "Effective filter support as a fraction of the kernel area. Brighter pixels have more useful "
		"filter weight spread across neighbors. This includes samples without a winner; brightness does not count surviving fragments.",
}


def _item_tooltip(text):
	from imgui_bundle import imgui

	if imgui.is_item_hovered():
		imgui.begin_tooltip()
		imgui.push_text_wrap_pos(imgui.get_font_size()*30)
		try:
			imgui.text_unformatted(text)
		finally:
			imgui.pop_text_wrap_pos()
			imgui.end_tooltip()


class FilterSettings:
	"""ImGui filter controls, keyboard shortcuts, and validated JSON profiles."""

	def __init__(self, renderer, profile_path=None):
		self.renderer = renderer
		self.defaults = renderer.bilateral_filter
		self.profile_path = Path(profile_path) if profile_path else Path(__file__).with_name("transparency_filter.json")
		self.selected = 0
		self.visible = True
		self.message = ""

	def move(self, direction):
		self.selected = (self.selected+direction) % len(SETTINGS)
		self.message = ""

	def adjust(self, direction, fine=False):
		name, _, step, _, _ = SETTINGS[self.selected]
		options = self.renderer.bilateral_filter
		current = getattr(options, name)
		if name == "adaptive":
			value = direction > 0
		elif name == "kernel_size":
			value = 5 if direction > 0 else 3
		else:
			value = round(current + direction*step*(0.1 if fine else 1), 9)
		self.set_value(name, value)

	def reset(self):
		self.renderer.set_bilateral_filter(self.defaults)
		self.message = "Restored startup settings"

	def save(self, path=None):
		path = Path(path) if path else self.profile_path
		path.write_text(json.dumps(asdict(self.renderer.bilateral_filter), indent=2)+"\n", encoding="utf-8")
		self.message = f"Saved {path.name}"

	def load(self, path=None):
		path = Path(path) if path else self.profile_path
		options = type(self.defaults)(**json.loads(path.read_text(encoding="utf-8")))
		self.renderer.set_bilateral_filter(options)
		self.message = f"Loaded {path.name}"

	def handle_key(self, key, scancode, action, mods):
		if action not in (glfw.PRESS, glfw.REPEAT):
			return False
		if key == glfw.KEY_F:
			if action == glfw.PRESS:
				self.visible = not self.visible
			return True
		if key in (glfw.KEY_F5, glfw.KEY_F9):
			if action == glfw.PRESS:
				try:
					self.save() if key == glfw.KEY_F5 else self.load()
				except (OSError, TypeError, ValueError) as error:
					self.message = str(error)
			return True
		if not self.visible:
			return False
		if key in (glfw.KEY_UP, glfw.KEY_DOWN):
			self.move(-1 if key == glfw.KEY_UP else 1)
		elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
			self.adjust(-1 if key == glfw.KEY_LEFT else 1, bool(mods & glfw.MOD_SHIFT))
		elif key == glfw.KEY_HOME:
			self.reset()
		else:
			return False
		return True

	def set_value(self, name, value):
		"""Apply the same valid ranges to mouse widgets and keyboard edits."""
		options = self.renderer.bilateral_filter
		_, _, _, minimum, maximum = next(row for row in SETTINGS if row[0] == name)
		if name not in ("adaptive", "kernel_size"):
			if not np.isfinite(value):
				return
			value = max(value, minimum)
			if maximum is not None:
				value = min(value, maximum)
			if name == "filter_near":
				value = min(value, options.filter_far-min(0.0001, options.filter_far*0.5))
			elif name == "filter_far":
				value = max(value, options.filter_near+0.0001)
		self.renderer.set_bilateral_filter(replace(options, **{name: value}))
		self.message = ""

	def draw(self):
		from imgui_bundle import imgui

		if not self.visible:
			return
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((460, 560), imgui.Cond_.first_use_ever)
		expanded, self.visible = imgui.begin("Transparency settings", self.visible)
		try:
			if not expanded:
				return
			imgui.text_disabled("Hover a setting or diagnostic view for an explanation.")
			if imgui.collapsing_header("Tuning guide"):
				imgui.text_wrapped("Start with the reconstruction view. Increase spatial spread or kernel size to smooth noise; "
					"reduce depth and transmittance tolerances if colors bleed across edges.")
				imgui.text_wrapped("Distance adaptation blends near and far settings using surface depth. "
					"With adaptation off, tune the near settings only. Filter support shows whether edge rejection leaves too few neighbors.")
				imgui.text_wrapped("These settings change reconstruction, not opacity or the stochastic winner pattern. "
					"There is no smoothing across frames, so some noise can remain during camera movement.")
			changed, index = imgui.combo("Diagnostic view", DEBUG_VIEWS.index(self.renderer.debug_view), DEBUG_VIEWS)
			_item_tooltip(DEBUG_VIEW_HELP[DEBUG_VIEWS[index]])
			if changed:
				self.renderer.set_debug_view(DEBUG_VIEWS[index])
			imgui.separator()
			for index, (name, label, step, minimum, maximum) in enumerate(SETTINGS):
				value = getattr(self.renderer.bilateral_filter, name)
				label = ("> " if index == self.selected else "") + label + "###" + name
				if name == "adaptive":
					changed, value = imgui.checkbox(label, value)
				elif name == "kernel_size":
					changed, selection = imgui.combo(label, 0 if value == 3 else 1, ("3 x 3", "5 x 5"))
					value = (3, 5)[selection]
				else:
					changed, value = imgui.drag_float(label, value, step, minimum, maximum or 0.0, "%.6g")
				_item_tooltip(SETTING_HELP[name])
				if changed:
					self.selected = index
					self.set_value(name, value)
			imgui.separator()
			if imgui.button("Reset (Home)"):
				self.reset()
			for label, action in (("Save (F5)", self.save), ("Load (F9)", self.load)):
				imgui.same_line()
				if imgui.button(label):
					try:
						action()
					except (OSError, TypeError, ValueError) as error:
						self.message = str(error)
			imgui.text_wrapped("Drag values or Ctrl+click to type. Shift gives fine keyboard steps. F hides/shows this window.")
			if self.message:
				imgui.text_wrapped(self.message)
		finally:
			imgui.end()


def scroll_callback(x, y):
	global distance
	distance = np.clip(distance * np.exp(-0.1*y), radius*0.15, radius*8)
	return True


def on_char(code):
	global orbit
	if chr(code) == " ":
		orbit = not orbit
	elif chr(code).lower() == "r":
		transparent.data.draw_batches.reverse()
	elif chr(code) in ("3", "5"):
		size = int(chr(code))
		transparency.set_bilateral_filter(replace(transparency.bilateral_filter, kernel_size=size))
		print(f"Bilateral filter: {size}x{size}")
	elif chr(code).lower() == "d":
		index = (DEBUG_VIEWS.index(transparency.debug_view)+1) % len(DEBUG_VIEWS)
		transparency.set_debug_view(DEBUG_VIEWS[index])
		print(f"View: {transparency.debug_view}")
	elif chr(code).lower() == "a":
		options = transparency.bilateral_filter
		transparency.set_bilateral_filter(replace(options, adaptive=not options.adaptive))
		print(f"Distance-adaptive filter: {transparency.bilateral_filter.adaptive}")
	return True


from Imgui import Imgui

RenderContext.init_window(1200, 900, "Stochastic transparency | F: settings | D: debug | Space: orbit",
	required_gpu_features=["indirect-first-instance"])
init_scene()
settings = FilterSettings(transparency)
RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)
RenderContext.event_handlers["char"].append(on_char)
RenderContext.event_handlers["key"].append(settings.handle_key)
try:
	with Imgui() as gui:
		while RenderContext.window_loop():
			if orbit:
				angle += RenderContext.frame_time * 0.2
			camera.position = Vec3(center + distance * np.array([cos(angle), 0.5, sin(angle)]))
			width, height = RenderContext.windowDimensions
			if width <= 0 or height <= 0:
				continue
			transparency.resize((width, height))
			view_projection = transparency.update_camera(camera)
			commands = RenderContext.commands("stochastic_renderer")
			for geometry in (opaque, transparent):
				geometry.prepare(commands, camera, view_projection, transparency.dummy_hzb)
			with transparency.opaque_pass(commands) as render:
				opaque.data.draw(render, "prepass")
			with transparency.guide_pass(commands) as render:
				transparent.data.draw(render, "main", pass_override=transparent.passes.guide)
			with transparency.transmittance_pass(commands) as render:
				transparent.data.draw(render, "prepass")
			with transparency.selection_pass(commands) as render:
				transparent.data.draw(render, "main")
			transparency.shade_winner(commands, transparent.passes)
			transparency.compose(commands, RenderContext.screen())
			RenderContext.submit(commands.finish())

			gui.render(settings.draw)
finally:
	RenderContext.event_handlers["mouse_scroll"].remove(scroll_callback)
	RenderContext.event_handlers["char"].remove(on_char)
	RenderContext.event_handlers["key"].remove(settings.handle_key)