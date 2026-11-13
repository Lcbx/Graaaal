"""Display-space antialiasing as an independent postprocessing pass."""
from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np

from .RenderContext import RenderContext, RenderPipeline, Shader, Texture, TextureUsage


@dataclass(frozen=True)
class AntialiasingSettings:
	lxaa_edge_threshold: float = 0.375
	lxaa_line_check: bool = True
	faaa_edge_threshold: float = 5.0
	faaa_quality: int = 5
	faaa_show_edges: bool = False

	def __post_init__(self):
		if not np.isfinite(self.lxaa_edge_threshold) or not 0 <= self.lxaa_edge_threshold <= 1: raise ValueError("LXAA threshold must be between 0 and 1")
		if not np.isfinite(self.faaa_edge_threshold) or not 1 <= self.faaa_edge_threshold <= 9: raise ValueError("FAAA threshold must be between 1 and 9")
		if type(self.faaa_quality) is not int or not 1 <= self.faaa_quality <= 9: raise ValueError("FAAA quality must be an integer between 1 and 9")
		if type(self.lxaa_line_check) is not bool or type(self.faaa_show_edges) is not bool: raise ValueError("AA switches must be boolean")


class AntialiasingPass:
	"""Apply LXAA or FAAA to a single-sample color target."""
	def __init__(self, *, target_format=None, settings=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "antialiasing.shader"), params={"srgb_surface": self.target_format.endswith("-srgb")})
		self.pipelines = {mode: RenderPipeline(self.shader, vertex_entry="fullscreen", fragment_entry=mode, depth_test=None, depth_write=False, cull_mode="none", label=mode) for mode in ("lxaa", "faaa")}
		self.uniforms = self.shader.uniform_buffer("aa_settings")
		self.settings = None
		self.set_settings(settings if settings is not None else AntialiasingSettings())
		self.mode = "off"
		self.size = self.source = None

	def set_settings(self, settings):
		"""Set the LXAA and FAAA controls."""
		if not isinstance(settings, AntialiasingSettings): raise TypeError("Expected AntialiasingSettings")
		if settings == self.settings: return False
		self.settings = settings
		self.uniforms.content["values"] = (settings.lxaa_edge_threshold, settings.faaa_edge_threshold, 0, 0)
		self.uniforms.content["flags"] = (int(settings.lxaa_line_check), settings.faaa_quality, int(settings.faaa_show_edges), 0)
		self.uniforms.upload_changed()
		return True

	def resize(self, size):
		"""Allocate the input target for a width and height."""
		if len(size) != 2: raise ValueError("Expected antialiasing width and height")
		try: size = tuple(index(value) for value in size)
		except TypeError: raise ValueError("Expected integer antialiasing dimensions") from None
		if min(size) <= 0: return False
		if size == self.size: return False
		self.source = Texture(size, format=self.target_format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING, label="antialiasing_source")
		self.size = size
		return True

	def color_attachment(self, *, clear=None):
		"""Return the input attachment for the scene color."""
		if self.source is None: raise RuntimeError("Resize antialiasing targets first")
		return self.source.color_attachment(clear=clear)

	def record(self, commands, output, mode):
		"""Record LXAA or FAAA to a single-sample output attachment."""
		if mode not in self.pipelines: raise ValueError("Antialiasing mode must be lxaa or faaa")
		if self.source is None: raise RuntimeError("Resize antialiasing targets first")
		if output.format != self.target_format or output.sample_count != 1: raise ValueError("Antialiasing output must match the target format and be single-sample")
		bindings = self.shader.bind_group(0, aa_settings=self.uniforms, scene_color=self.source)
		with commands.render_pass(color=output, label=mode) as render:
			render.set_pipeline(self.pipelines[mode])
			render.set_bind_groups(bindings)
			render.draw(3)
