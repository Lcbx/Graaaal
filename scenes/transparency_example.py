"""Single-winner stochastic OIT with frustum and Hi-Z culling via DrawBatches.

Run: python main.py scenes/transparency_example.py [--frames 120]
Space toggles orbit, scrolling zooms, R reverses transparent submission order.
1 disables reconstruction; 3, 5 and 9 select its kernel size.
D cycles diagnostic views; N cycles noise modes; [ / ] adjusts noise scale.
G toggles grazing-angle opacity fade (angles are adjustable in settings).
M toggles 4x MSAA (off by default).
F shows/hides ImGui settings.
Drag settings or Ctrl+click to type; Up/Down and Left/Right also select/edit.
Imports only prepare CPU helpers; graphics are initialized by run_scene().
"""

import json
from dataclasses import asdict, replace
from math import cos, sin
from pathlib import Path

import glfw
import numpy as np

from numpyecs.Assets import DEFAULT_BLUE_NOISE_PATH, load_blue_noise
from numpyecs.DrawBatches import DrawBatches, FrustumCuller, HiZCuller, ShaderPass
from numpyecs.HZB import HZB
from numpyecs.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale
from numpyecs.Math import Camera, Vec3
from numpyecs.RenderContext import (
    BufferUsage,
    GpuBuffer,
    Mesh,
    RenderContext,
    RenderPipeline,
    Texture,
    TextureUsage,
)
from numpyecs.TextureProcessing import build_material_texture_array
from numpyecs.TransparencyData import load_transparency_scene as load_transparency_data
from numpyecs.TransparencyPass import BilateralFilter, DEBUG_VIEWS, GrazingFade, KERNEL_SIZES, NOISE_MODES, TransparencyPass


PATH = Path(__file__).parent / "resources/small_scene.glb"
BLUE_NOISE_PATH = DEFAULT_BLUE_NOISE_PATH
TEXTURE_SIZE = 512


def load_transparency_scene(path):
	# This asset authors binary tree leaves as BLEND. Use MASK coverage/mips for
	# this named foliage material while keeping genuinely soft cards as BLEND.
	return load_transparency_data(path, mask_materials=("Simple_foliage.001",))


class GeometryBatches:
	"""Static glTF primitives submitted directly through DrawBatches."""
	def __init__(self, primitives, materials, transparency, transparent):
		primitives = [p for p in primitives if (materials[p.material_id]["surface"][2] != 0) == transparent]
		self.data = DrawBatches(MESH_INSTANCE_DTYPE)
		self.metadata = GpuBuffer(np.zeros((1, 4), dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self.passes = transparency.create_geometry_passes(self.metadata, transparent=transparent, instances=self.data.buffers["instances"])
		if not transparent:
			self.depth_pass = ShaderPass(RenderPipeline(transparency.depth_shader, vertex_entry="vertex",
				fragment_entry="opaque_prepass", cull_mode="none", label="opaque_depth"),
				self.passes.render_shader["prepass"].bindings)
		self.data.register_shader(0, self.passes.render_shader)
		for primitive in primitives:
			mesh_id = primitive.stable_id
			mesh = Mesh(primitive.vertices, np.arange(len(primitive.vertices), dtype=np.uint32))
			self.data.register_mesh(mesh_id, mesh)
			self.data.register_lod_group(mesh_id, (mesh_id,))
		mesh_ids = np.array([p.stable_id for p in primitives], dtype=np.uint32)
		self.data.sync(mesh_ids, np.zeros(len(mesh_ids), dtype=np.uint32))
		self.visibility = self.data.create_visibility()
		self.data.write_instance_fields(iPosition=(0, 0, 0), iTint=0,
			iRotation=pack_quaternion([0, 0, 0, 1]), iScale=pack_scale([1, 1, 1]))
		# Metadata follows packed instance order, which may differ from input order.
		metadata = np.zeros((max(1, len(primitives)), 4), dtype=np.uint32)
		for row, source in enumerate(self.data.instance_order):
			p = primitives[source]
			metadata[row] = (p.triangle_offset, p.material_id, p.stable_id, 0)
		self.metadata.write(metadata)

	def prepare(self, commands, camera, view_projection):
		FrustumCuller().record(commands, self.visibility, camera_position=camera.position,
			view_projection=view_projection)

	def draw(self, render, stage, **options):
		self.data.draw(render, self.visibility, stage, **options)


transparency: TransparencyPass
opaque: GeometryBatches
transparent: GeometryBatches
camera: Camera
hzb: HZB
center: np.ndarray
radius: float
angle = 0.65
orbit = False
distance = 0.0


def init_scene(path=PATH, *, bilateral_filter=None, blue_noise_path=BLUE_NOISE_PATH, grazing_fade=None, sample_count=1):
	"""Initialize the example's global render data after device setup."""
	global transparency, opaque, transparent, camera, hzb, center, radius, angle, orbit, distance
	primitives, materials, images, vertices = load_transparency_scene(path)
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
		bilateral_filter=bilateral_filter, grazing_fade=grazing_fade, sample_count=sample_count)
	hzb = HZB()
	opaque = GeometryBatches(primitives, materials, transparency, False)
	transparent = GeometryBatches(primitives, materials, transparency, True)
	camera = Camera(tuple(center + (radius, radius*0.65, radius)), tuple(center), (0, 1, 0), 55,
		near=max(0.01, radius/1000), far=radius*100)
	angle, orbit, distance = 0.65, False, radius*1.4


# name, displayed label, increment, minimum, maximum
SETTINGS = (
	("kernel_size", "Kernel size", None, None, None),
	("spatial_sigma", "Spatial sigma", 0.1, 0.01, None),
	("depth_sigma", "Relative depth sigma", 0.005, 0.0001, None),
	("transmittance_sigma", "Transmittance sigma", 0.05, 0.001, None),
	("min_depth_sigma", "Minimum depth tolerance", 0.001, 0.000001, None),
)


SETTING_HELP = {
	"kernel_size": "1 x 1 bypasses reconstruction and shows the unfiltered stochastic result. "
		"3 x 3, 5 x 5 and 9 x 9 filter progressively larger neighborhoods. Default: 3 x 3.",
	"spatial_sigma": "Gaussian spread in pixels. Increase for a softer result; the kernel size caps its reach. "
		"Default: 0.9 pixels. Certain nearest winners bypass filtering; fractional coverage is fully reconstructed.",
	"depth_sigma": "Relative depth tolerance, e.g. 0.02 is 2% of view depth. "
		"Larger values admit more neighbors but can mix overlapping surfaces. "
		"Slope correction is always enabled for transparent surfaces; opaque depth remains a separate constraint.",
	"transmittance_sigma": "Tolerance for differences in RGB and scalar transmittance. "
		"Increase to smooth more across opacity changes; smaller values protect edges but can leave grain. "
		"This changes color reconstruction, not the local accumulated opacity.",
	"min_depth_sigma": "Minimum depth tolerance in scene units. "
		"The filter uses the larger of this value and depth times relative depth sigma. "
		"Increase if nearby surfaces have too little filter support.",
}

NOISE_LABELS = ("Object hash", "UV blue: CDF lerp", "UV blue: stochastic pick", "UV Bayer: stochastic pick",
	"UV blue: correlated hierarchy", "Triplanar blue: CDF lerp", "Triplanar: correlated hierarchy")


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
	"""Live ImGui filter controls and keyboard shortcuts."""

	def __init__(self, renderer):
		self.renderer = renderer
		self.defaults = renderer.bilateral_filter
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
		if name == "kernel_size":
			index = min(max(KERNEL_SIZES.index(current)+direction, 0), len(KERNEL_SIZES)-1)
			value = KERNEL_SIZES[index]
		else:
			value = round(current + direction*step*(0.1 if fine else 1), 9)
		self.set_value(name, value)

	def reset(self):
		self.renderer.set_bilateral_filter(self.defaults)
		self.message = "Restored startup settings"

	def handle_key(self, key, scancode, action, mods):
		if action not in (glfw.PRESS, glfw.REPEAT):
			return False
		if key == glfw.KEY_F:
			if action == glfw.PRESS:
				self.visible = not self.visible
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
		if name != "kernel_size":
			if not np.isfinite(value):
				return
			value = max(value, minimum)
			if maximum is not None:
				value = min(value, maximum)
		self.renderer.set_bilateral_filter(replace(options, **{name: value}))
		self.message = ""

	def set_grazing_angle(self, name, value):
		"""Keep the editable fade interval ordered, including typed input."""
		if name not in ("start_angle", "end_angle"):
			raise ValueError("Expected start_angle or end_angle")
		if not np.isfinite(value):
			return
		options = self.renderer.grazing_fade
		if name == "start_angle":
			value = max(0, min(value, options.end_angle-0.01))
		else:
			value = min(90, max(value, options.start_angle+0.01))
		self.renderer.set_grazing_fade(replace(options, **{name: value}))

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
				imgui.text_wrapped("Use 1 x 1 to inspect the raw stochastic result. Defaults: 3 x 3 with spatial sigma 0.9. "
					"Slope correction is always enabled. Certain opaque interiors bypass blur. "
					"Filter support shows whether edge rejection leaves too few neighbors.")
				imgui.text_wrapped("Filter controls smooth reconstruction; noise controls choose winners; grazing fade reduces opacity at steep view angles. "
					"There is no smoothing across frames, so some noise can remain during camera movement.")
			changed, index = imgui.combo("Diagnostic view", DEBUG_VIEWS.index(self.renderer.debug_view), DEBUG_VIEWS)
			_item_tooltip(DEBUG_VIEW_HELP[DEBUG_VIEWS[index]])
			if changed:
				self.renderer.set_debug_view(DEBUG_VIEWS[index])
			changed, index = imgui.combo("MSAA (M)", int(self.renderer.sample_count == 4), ("Off (1x)", "4x"))
			_item_tooltip("4x samples geometry, opacity and winner selection at four positions per pixel. "
				"Reconstruction resolves linear HDR before tone mapping. Costs additional GPU time and memory; no temporal accumulation.")
			if changed:
				self.renderer.set_sample_count((1, 4)[index])
			changed, index = imgui.combo("Noise sampling (N)", NOISE_MODES.index(self.renderer.noise_mode), NOISE_LABELS)
			_item_tooltip("UV modes point-sample two surface-anchored octaves and use object hashing when UVs are degenerate. "
				"Stochastic pick uses an independent surface hash as selector; CDF lerp is the smoother alternative. "
				"Correlated hierarchy uses re-ranked parent levels with one shared origin. Triplanar uses object-space projections "
				"and smooth normal weights, ignoring UVs. Bayer is an ordered comparison with possible crosshatching. ")
			if changed:
				self.renderer.set_noise_mode(NOISE_MODES[index])
			changed, value = imgui.drag_float("Noise cell size (px)", self.renderer.noise_scale, 0.05, 0.25, 4.0, "%.3g")
			_item_tooltip("Approximate noise-cell width in pixels. Larger values give coarser, more stable grain. [ / ] also adjust it.")
			if changed and np.isfinite(value):
				self.renderer.set_noise_scale(max(0.25, min(4.0, value)))
			fade = self.renderer.grazing_fade
			changed, enabled = imgui.checkbox("Fade grazing surfaces (G)", fade.enabled)
			_item_tooltip("Smoothly reduce transparent-surface opacity between the two angles below. "
				"0 degrees is face-on; 90 is edge-on. Uses geometric normals on both sides. "
				"Fade and noise controls update immediately.")
			if changed:
				self.renderer.set_grazing_fade(replace(fade, enabled=enabled))
			for name, label in (("start_angle", "Fade start (degrees)"), ("end_angle", "Fully faded (degrees)")):
				changed, value = imgui.drag_float(label, getattr(self.renderer.grazing_fade, name), 0.5, 0.0, 90.0, "%.2f")
				_item_tooltip("Angle from the geometric surface normal. Opacity is unchanged at start and zero at the end, "
					"with a smooth fade between them. The start must be below the end.")
				if changed:
					self.set_grazing_angle(name, value)
			imgui.separator()
			for index, (name, label, step, minimum, maximum) in enumerate(SETTINGS):
				value = getattr(self.renderer.bilateral_filter, name)
				label = ("> " if index == self.selected else "") + label + "###" + name
				if name == "kernel_size":
					changed, selection = imgui.combo(label, KERNEL_SIZES.index(value), ("1 x 1 (off)", "3 x 3", "5 x 5", "9 x 9"))
					value = KERNEL_SIZES[selection]
				else:
					changed, value = imgui.drag_float(label, value, step, minimum, maximum or 0.0, "%.6g")
				_item_tooltip(SETTING_HELP[name])
				if changed:
					self.selected = index
					self.set_value(name, value)
			imgui.separator()
			if imgui.button("Reset filter (Home)"):
				self.reset()
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
	elif chr(code) in ("1", "3", "5", "9"):
		size = int(chr(code))
		transparency.set_bilateral_filter(replace(transparency.bilateral_filter, kernel_size=size))
		print(f"Bilateral filter: {size}x{size}")
	elif chr(code).lower() == "d":
		index = (DEBUG_VIEWS.index(transparency.debug_view)+1) % len(DEBUG_VIEWS)
		transparency.set_debug_view(DEBUG_VIEWS[index])
		print(f"View: {transparency.debug_view}")
	elif chr(code).lower() == "n":
		index = (NOISE_MODES.index(transparency.noise_mode)+1) % len(NOISE_MODES)
		transparency.set_noise_mode(NOISE_MODES[index])
		print(f"Noise: {NOISE_LABELS[index]}")
	elif chr(code) in ("[", "]"):
		factor = 2**(-0.25 if chr(code) == "[" else 0.25)
		transparency.set_noise_scale(max(0.25, min(4.0, transparency.noise_scale*factor)))
		print(f"Noise cell size: {transparency.noise_scale:.3g} px")
	elif chr(code).lower() == "g":
		options = transparency.grazing_fade
		transparency.set_grazing_fade(replace(options, enabled=not options.enabled))
		print(f"Grazing fade: {not options.enabled}, {options.start_angle:g} to {options.end_angle:g} degrees")
	elif chr(code).lower() == "m":
		transparency.set_sample_count(4 if transparency.sample_count == 1 else 1)
		print(f"MSAA: {transparency.sample_count}x")
	return True


def render_frame():
	"""Record frustum culling, opaque depth, Hi-Z culling, and transparency."""
	width, height = RenderContext.window_dimensions
	if width <= 0 or height <= 0:
		return False
	transparency.resize((width, height))
	view_projection = transparency.update_camera(camera)
	commands = RenderContext.commands("stochastic_renderer")
	for geometry in (opaque, transparent):
		geometry.prepare(commands, camera, view_projection)
	if transparency.sample_count == 1:
		hzb.resize((width, height))
		# Stochastic cutout surfaces cannot contribute conservative occluder depth.
		with commands.render_pass(color=(), depth=hzb.depth_texture.depth_attachment(clear=1.0),
			label="depth_prepass") as render:
			opaque.draw(render, "prepass", pass_override=opaque.depth_pass)
		hzb.build(commands)
		for geometry in (opaque, transparent):
			HiZCuller().record(commands, geometry.visibility, hzb_view=hzb.view)
	# Pixel-center depth cannot conservatively occlude all MSAA samples. At 4x,
	# draw both lists directly from the frustum visibility prepared above.
	with transparency.opaque_pass(commands) as render:
		opaque.draw(render, "main")
	# The point estimator never samples a reconstruction guide.
	if transparency.bilateral_filter.kernel_size != 1:
		with transparency.guide_pass(commands) as render:
			transparent.draw(render, "main", pass_override=transparent.passes.guide)
	with transparency.transmittance_pass(commands) as render:
		transparent.draw(render, "transmittance")
	with transparency.selection_pass(commands) as render:
		transparent.draw(render, "main")
	transparency.shade_winner(commands, transparent.passes)
	transparency.compose(commands, RenderContext.screen())
	RenderContext.submit(commands.finish())
	return True


def run_scene():
	global settings, angle
	from numpyecs.ImGui import ImGui

	RenderContext.init_window(1200, 900, "Stochastic transparency | M: MSAA | F: settings | D: debug | Space: orbit",
		required_gpu_features=["indirect-first-instance"])
	init_scene()
	settings = FilterSettings(transparency)
	RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)
	RenderContext.event_handlers["char"].append(on_char)
	RenderContext.event_handlers["key"].append(settings.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if orbit:
					angle += RenderContext.frame_time * 0.2
				camera.position = Vec3(center + distance * np.array([cos(angle), 0.5, sin(angle)]))
				if render_frame():
					gui.render(settings.draw)
	finally:
		RenderContext.event_handlers["mouse_scroll"].remove(scroll_callback)
		RenderContext.event_handlers["char"].remove(on_char)
		RenderContext.event_handlers["key"].remove(settings.handle_key)

if __name__ == "__main__":
	run_scene()
