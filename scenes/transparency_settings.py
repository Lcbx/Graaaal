"""Shared transparency settings for the interactive scenes.

Pass a TransparencyPass or MaterialRenderer to FilterSettings, register
handle_key with the key event, and call draw inside the ImGui frame.
F toggles visibility; visible=False starts with the settings hidden.
"""
from dataclasses import replace

import glfw
import numpy as np

from numpyecs.TransparencyPass import DEBUG_VIEWS, FILL_NOISE_MODES, KERNEL_SIZES, NOISE_MODES


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
	"UV blue: correlated hierarchy", "Triplanar blue: CDF lerp", "Triplanar: correlated hierarchy", "Anchored blue: pixel lattice")


DEBUG_VIEW_HELP = {
	"final": "Final image: opaque color seen through the transparent surfaces, plus reconstructed transparent color.",
	"transmittance": "Scalar light transmission: white means fully transparent and black means fully opaque. "
		"Use this to check opacity independently of winner selection and reconstruction.",
	"winner": "Raw selected color: averaged in Average mode, or the first successful attempt in Fill mode. "
		"Missing winners contribute zero. This view reveals the stochastic pattern before filtering.",
	"reconstruction": "transparent contribution without the opaque background: filtered when enabled, or the point estimate at 1 x 1. "
		"Use this to compare smoothing and edge preservation while tuning.",
	"filter_support": "Effective filter support as a fraction of the kernel area. Brighter pixels have more useful "
		"filter weight spread across neighbors. This includes samples without a winner; brightness does not count surviving fragments.",
	"winner_a": "Raw first-stream color, before filtering. Missing winners are black.",
	"winner_b": "Raw second-stream color in two-draw Average mode, before filtering. Equals the first stream at exact certainty.",
	"winner_count": "Average mode: valid-stream fraction (black/gray/white for zero/one/two). Fill mode: empty or filled, regardless of the successful attempt. MSAA averages hardware samples.",
	"forced": "White marks an exactly counted single contributor accepted deterministically at its accumulated opacity.",
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
	"""Live controls for a TransparencyPass or its owning MaterialRenderer."""

	def __init__(self, renderer, *, visible=True, position=(12, 12)):
		self.controls = renderer
		self.renderer = getattr(renderer, "transparency", renderer)
		self.defaults = self.renderer.bilateral_filter
		self.selected = 0
		self.visible = visible
		self.position = position
		self.message = ""

	@staticmethod
	def available_debug_views(renderer):
		"""UI choices for the current technique; shader diagnostic IDs stay fixed."""
		averaged_pair = renderer.winner_count == 2 and renderer.winner_mix == "average"
		return tuple(view for view in DEBUG_VIEWS
			if (view not in ("winner_a", "winner_b") or averaged_pair)
			and (view != "winner_count" or renderer.winner_count > 1)
			and (view != "forced" or renderer.force_single_contributor)
			and (view != "filter_support" or renderer.bilateral_filter.kernel_size != 1))

	@classmethod
	def sync_debug_view(cls, renderer):
		views = cls.available_debug_views(renderer)
		if renderer.debug_view not in views:
			renderer.set_debug_view("final")
		return views

	@staticmethod
	def debug_view_label(renderer, view):
		if view == "filter_support" and renderer.reconstruction == "atrous": return "retained_support"
		if view == "filter_support" and renderer.reconstruction in ("push_pull", "atrous"): return "confidence"
		return "winner_validity" if view == "winner_count" and renderer.winner_mix == "fill" else view

	def move(self, direction):
		indices = [index for index, row in enumerate(SETTINGS) if row[0] != "spatial_sigma" or self.renderer.reconstruction != "atrous"]
		if self.selected not in indices: self.selected = indices[0]
		self.selected = indices[(indices.index(self.selected)+direction) % len(indices)]
		self.message = ""

	def adjust(self, direction, fine=False):
		if SETTINGS[self.selected][0] == "spatial_sigma" and self.renderer.reconstruction == "atrous": self.move(direction)
		name, _, step, _, _ = SETTINGS[self.selected]
		options = self.renderer.bilateral_filter
		current = getattr(options, name)
		if name == "kernel_size":
			sizes = (1, 3) if self.renderer.reconstruction in ("push_pull", "atrous") else KERNEL_SIZES
			if self.renderer.reconstruction in ("push_pull", "atrous"): current = 1 if current == 1 else 3
			index = min(max(sizes.index(current)+direction, 0), len(sizes)-1)
			value = sizes[index]
		else:
			value = round(current + direction*step*(0.1 if fine else 1), 9)
		self.set_value(name, value)

	def reset(self):
		self.renderer.set_bilateral_filter(self.defaults)
		self.sync_debug_view(self.renderer)
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
		if name == "kernel_size": self.sync_debug_view(self.renderer)
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

		views = self.sync_debug_view(self.renderer)
		io = imgui.get_io()
		imgui.set_next_window_pos((io.display_size.x-12, 12), imgui.Cond_.always, (1, 0))
		imgui.set_next_window_bg_alpha(0.65)
		flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize
			| imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_focus_on_appearing
			| imgui.WindowFlags_.no_inputs)
		imgui.begin("Frame rate", flags=flags)
		try:
			fps = io.framerate  # ImGui's rolling average includes the whole frame.
			imgui.text(f"{fps:.0f} FPS | {1000/fps:.2f} ms" if fps > 0 else "FPS: --")
		finally:
			imgui.end()
		if not self.visible:
			return
		imgui.set_next_window_pos(self.position, imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((460, 560), imgui.Cond_.first_use_ever)
		expanded, self.visible = imgui.begin("Transparency settings", self.visible)
		try:
			if not expanded:
				return
			imgui.text_disabled("Hover a setting or diagnostic view for an explanation.")
			if imgui.collapsing_header("Tuning guide"):
				imgui.text_wrapped("Start with the reconstruction view. Increase the confidence budget, or bilateral spread/kernel size, to smooth noise; "
					"reduce depth and transmittance tolerances if colors bleed across edges.")
				imgui.text_wrapped("Use 1 x 1 to inspect the raw stochastic result. Bilateral filters use fixed kernels; confidence filters expose a support budget. "
					"Slope correction is always enabled. Certain opaque interiors bypass blur. "
					"Filter support shows whether edge rejection leaves too few neighbors.")
				imgui.text_wrapped("Filter controls smooth reconstruction; noise controls choose winners; grazing fade reduces opacity at steep view angles. "
					"There is no smoothing across frames, so some noise can remain during camera movement.")
			imgui.separator_text("Rendering")
			labels = tuple(self.debug_view_label(self.renderer, view) for view in views)
			changed, index = imgui.combo("Diagnostic view", views.index(self.renderer.debug_view), labels)
			help_text = DEBUG_VIEW_HELP[views[index]]
			if views[index] == "filter_support" and self.renderer.reconstruction in ("push_pull", "atrous"):
				help_text = "Local expected-winner weight divided by the target budget. Dark pixels prefer the broader pyramid estimate; white pixels trust the local estimate."
				if self.renderer.reconstruction == "atrous": help_text = "Final retained unit-peak acceptance D divided by the budget, clamped to one. Includes rejection from every cascade pass."
			_item_tooltip(help_text)
			if changed:
				self.renderer.set_debug_view(views[index])
			changed, index = imgui.combo("MSAA (M)", int(self.renderer.sample_count == 4), ("Off (1x)", "4x"))
			_item_tooltip("4x samples geometry, opacity and winner selection at four positions per pixel. "
				"Reconstruction resolves linear HDR before tone mapping. Costs additional GPU time and memory; no temporal accumulation.")
			if changed:
				self.controls.set_sample_count((1, 4)[index])
			changed, count = imgui.input_int("Winner draws", self.renderer.winner_count)
			_item_tooltip("Any positive draw count in Fill mode; Average supports one or two. Choose Object hash or Anchored blue and Fill before increasing beyond two. Each attempt adds selection and shading work; silhouettes are unchanged.")
			if changed:
				try: self.controls.set_winner_count(max(1, count))
				except ValueError as error: self.message = str(error)
			if self.renderer.winner_count > 1:
				if self.renderer.winner_mix == "average":
					changed, index = imgui.combo("Winner pairing", ("complement", "independent").index(self.renderer.winner_coupling), ("Complemented (one tile)", "Independent"))
					_item_tooltip("Complement the same fully remapped threshold for B. Fill always uses independent retries.")
					if changed: self.controls.set_winner_coupling(("complement", "independent")[index])
				if self.renderer.noise_mode not in FILL_NOISE_MODES:
					imgui.text_wrapped("Fill requires Object hash or Anchored blue noise.")
				changed, index = imgui.combo("Winner mix", ("average", "fill").index(self.renderer.winner_mix), ("Average", "Fill holes (experimental)"))
				_item_tooltip("Fill retries only empty samples, compensating neighbour weights. Audited for default-source object hash and anchored blue only; other modes retain averaging.")
				if changed:
					try: self.controls.set_winner_mix(("average", "fill")[index])
					except ValueError as error: self.message = str(error)
			changed, enabled = imgui.checkbox("Force single contributor", self.renderer.force_single_contributor)
			_item_tooltip("Exactly count nonzero visible contributors. A sole contributor is accepted deterministically and weighted by its opacity. Adds a small count attachment; disabled by default.")
			if changed: self.controls.set_force_single_contributor(enabled)
			imgui.separator_text("Noise sampling")
			changed, index = imgui.combo("Noise sampling (N)", NOISE_MODES.index(self.renderer.noise_mode), NOISE_LABELS)
			_item_tooltip("UV modes point-sample two surface-anchored octaves and use object hashing when UVs are degenerate. "
				"Stochastic pick uses an independent surface hash as selector; CDF lerp is the smoother alternative. "
				"Correlated hierarchy uses re-ranked parent levels with one shared origin. Triplanar uses object-space projections "
				"and smooth normal weights, ignoring UVs. Bayer is an ordered comparison with possible crosshatching. ")
			if changed:
				if self.renderer.winner_mix == "fill" and NOISE_MODES[index] not in FILL_NOISE_MODES:
					if self.renderer.winner_count > 2: self.controls.set_winner_count(2)
					self.controls.set_winner_mix("average")
				self.renderer.set_noise_mode(NOISE_MODES[index])
			changed, value = imgui.drag_float("Noise cell size at 1x (px)", self.renderer.noise_scale, 0.05, 0.25, 4.0, "%.3g")
			_item_tooltip("Base width at 1x; 4x MSAA halves the effective width. "
				"Size is approximate and cells can stretch at grazing angles. Larger values give coarser, more stable grain. [ / ] also adjust it.")
			if changed and np.isfinite(value):
				self.renderer.set_noise_scale(max(0.25, min(4.0, value)))
			imgui.text(f"Target cell size: {self.renderer.effective_noise_scale:.3g} px ({self.renderer.sample_count}x MSAA)")
			imgui.separator_text("UV footprint balancing")
			changed, enabled = imgui.checkbox("Enable UV footprint balancing", self.renderer.noise_footprint_balancing)
			_item_tooltip("UV shader feature, enabled by default at strength one. Shader variants are cached.")
			if changed: self.controls.set_noise_footprint_balancing(enabled)
			if self.renderer.noise_footprint_balancing:
				changed, value = imgui.slider_float("UV footprint balance", self.renderer.noise_footprint_balance, 0.0, 1.0, "%.2f")
				_item_tooltip("0: maximum derivative (current sizing). 0.5: geometric mean/area. 1: minimum derivative. "
					"Finer grain reduces the long cells at tilt but makes the compressed axis subpixel. UV modes only.")
				if changed: self.renderer.set_noise_footprint_balance(value)
				if not self.renderer.noise_mode.startswith("uv_"):
					imgui.text_wrapped("Select a UV noise mode to see the effect of footprint balancing.")
			if self.renderer.noise_mode == "anchored_blue":
				imgui.separator_text("Surface anchor")
				changed, value = imgui.drag_float("Anchor target (px)", self.renderer.noise_anchor_pixels, 0.5, 8.0, 64.0, "%.1f")
				_item_tooltip("Coarse rest-space anchor grid, sized by projected area. Grain itself remains pixel-sized.")
				if changed: self.renderer.set_noise_anchor(value, self.renderer.noise_anchor_subdivisions)
				changed, value = imgui.slider_int("Shift subdivisions", self.renderer.noise_anchor_subdivisions, 1, 32)
				_item_tooltip("Finer rest cells select independent tile shifts without changing the anchor. More subdivisions reduce layer correlation but add seams. Layers within one fine cell can still share thresholds.")
				if changed: self.renderer.set_noise_anchor(self.renderer.noise_anchor_pixels, value)
			imgui.separator_text("Grazing fade")
			fade = self.renderer.grazing_fade
			changed, enabled = imgui.checkbox("Fade grazing surfaces (G)", fade.enabled)
			_item_tooltip("Smoothly reduce transparent-surface opacity between the two angles below. "
				"0 degrees is face-on; 90 is edge-on. Uses geometric normals on both sides. "
				"Fade and noise controls update immediately.")
			if changed:
				self.renderer.set_grazing_fade(replace(fade, enabled=enabled))
			if self.renderer.grazing_fade.enabled:
				for name, label in (("start_angle", "Fade start (degrees)"), ("end_angle", "Fully faded (degrees)")):
					changed, value = imgui.drag_float(label, getattr(self.renderer.grazing_fade, name), 0.5, 0.0, 90.0, "%.2f")
					_item_tooltip("Angle from the geometric surface normal. Opacity is unchanged at start and zero at the end, "
						"with a smooth fade between them. The start must be below the end.")
					if changed:
						self.set_grazing_angle(name, value)
			imgui.separator_text("Reconstruction")
			implementations = ("atrous", "push_pull", "fragment", "compute")
			changed, index = imgui.combo("Filter implementation", implementations.index(self.renderer.reconstruction), ("A-trous cascade", "Confidence pyramid", "Bilateral fragment", "Bilateral compute"))
			_item_tooltip("A-trous uses full-resolution dilated tents and retained support. Push-pull keeps the downsampled comparison. Bilateral modes retain fixed kernels and optional stacked passes.")
			if changed: self.controls.set_reconstruction(implementations[index])
			if self.renderer.reconstruction in ("push_pull", "atrous"):
				options = self.renderer.confidence_filter
				cascade = self.renderer.reconstruction == "atrous"
				changed, target = imgui.drag_float("Support budget D" if cascade else "Target expected winners", options.target_winners, 0.25, 0.25, 32.0, "%.2f")
				_item_tooltip("Unit-peak retained acceptance budget. Four corresponds to about nine nominal effective samples for a broad unrestricted tent; edge rejection changes that relationship." if cascade else "Expected acceptance budget. Higher budgets prefer broader estimates.")
				if changed and np.isfinite(target): self.controls.set_confidence_filter(replace(options, target_winners=max(0.25, target)))
				if cascade:
					changed, index = imgui.combo("Cascade passes", self.renderer.confidence_filter.cascade_passes-1, ("1", "2", "3", "4"))
					if changed: self.controls.set_confidence_filter(replace(self.renderer.confidence_filter, cascade_passes=index+1))
				else:
					changed, index = imgui.combo("Pyramid levels", self.renderer.confidence_filter.levels-2, ("2", "3"))
					if changed: self.controls.set_confidence_filter(replace(self.renderer.confidence_filter, levels=index+2))
			else:
				changed, count = imgui.input_int("Reconstruction passes", self.renderer.reconstruction_passes)
				_item_tooltip("Stack bilateral passes on weighted color/validity. Opacity is applied once; 1 x 1 bypasses all passes.")
				if changed: self.controls.set_reconstruction_passes(max(1, count))
			for index, (name, label, step, minimum, maximum) in enumerate(SETTINGS):
				if name == "spatial_sigma" and self.renderer.reconstruction == "atrous": continue
				if name != "kernel_size" and self.renderer.bilateral_filter.kernel_size == 1:
					continue
				value = getattr(self.renderer.bilateral_filter, name)
				label = ("> " if index == self.selected else "") + label + "###" + name
				if name == "kernel_size":
					if self.renderer.reconstruction in ("push_pull", "atrous"):
						changed, enabled = imgui.checkbox("Enable reconstruction", value != 1)
						value = 3 if enabled else 1
					else:
						changed, selection = imgui.combo(label, KERNEL_SIZES.index(value), ("1 x 1 (off)", "3 x 3", "5 x 5", "9 x 9"))
						value = KERNEL_SIZES[selection]
				else:
					changed, value = imgui.drag_float(label, value, step, minimum, maximum or 0.0, "%.6g")
				_item_tooltip("Disable to inspect the raw stochastic result. Confidence reconstruction uses 3 x 3 gathers at each scale." if name == "kernel_size" and self.renderer.reconstruction in ("push_pull", "atrous") else SETTING_HELP[name])
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
			self.sync_debug_view(self.renderer)


