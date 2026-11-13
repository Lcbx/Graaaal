"""Screen-space ambient occlusion as an independent depth-aware pass."""
from operator import index
from pathlib import Path

import numpy as np

from .RenderContext import RenderContext, RenderPipeline, Shader, Texture, TextureUsage


SHADERS = Path(__file__).with_name("shaders")


class AmbientOcclusionPass:
	"""Composite ambient occlusion over a color target using the opaque depth buffer."""
	def __init__(self, *, target_format=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.srgb_surface = self.target_format.endswith("-srgb")
		self.size = None
		self.source = None
		self._camera_ready = False
		self.sample_count = None
		self._activate_sample_count(1)

	def _activate_sample_count(self, count):
		if type(count) is not int or count not in (1, 4):
			raise ValueError("Ambient occlusion sample_count must be 1 or 4")
		if count == self.sample_count:
			return False
		self.shader = Shader(filepath=str(SHADERS / "ambient_occlusion.shader"), params={
			"sample_count": count,
			"srgb_surface": self.srgb_surface,
			"source_srgb_encoded": not self.srgb_surface,
		})
		self.camera = self.shader.uniform_buffer("ao_settings")
		self.pipeline = RenderPipeline(self.shader, vertex_entry="fullscreen",
			fragment_entry="apply_ambient_occlusion", depth_test=None, depth_write=False,
			cull_mode="none", label="ambient_occlusion")
		self.sample_count = count
		self._camera_ready = False
		return True

	def resize(self, size):
		"""Allocate the sampled color target for the requested width and height."""
		if len(size) != 2:
			raise ValueError("Expected ambient occlusion target width and height")
		try:
			size = tuple(index(value) for value in size)
		except TypeError:
			raise ValueError("Expected integer ambient occlusion target dimensions") from None
		if size[0] <= 0 or size[1] <= 0:
			return False
		if size == self.size:
			return False
		self.source = Texture(size, format=self.target_format,
			usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING, label="ambient_occlusion_source")
		self.size = size
		self._camera_ready = False
		return True

	def color_attachment(self, *, clear=None):
		"""Return the color attachment for rendering the scene before AO."""
		if self.source is None:
			raise RuntimeError("Resize ambient occlusion targets before requesting the color attachment")
		return self.source.color_attachment(clear=clear)

	def update_camera(self, camera, aspect=None, *, sample_count=1):
		"""Upload the camera projection and depth sample count for the AO pass."""
		if self.size is None:
			raise RuntimeError("Resize ambient occlusion targets before updating camera uniforms")
		self._activate_sample_count(sample_count)
		if aspect is None:
			aspect = self.size[0]/self.size[1]
		projection = np.asarray(camera.projection(aspect), dtype=np.float32)
		if projection.shape != (4, 4) or not np.all(np.isfinite(projection)):
			raise ValueError("Camera projection must be a finite 4x4 matrix")
		if not (np.isfinite(camera.near) and np.isfinite(camera.far) and 0 < camera.near < camera.far):
			raise ValueError("Camera clip distances must satisfy 0 < near < far")
		self.camera.content["projection"] = projection
		self.camera.content["clip"] = (camera.near, camera.far, 0, 0)
		self.camera.upload_changed()
		self._camera_ready = True

	def record(self, commands, depth, output):
		"""Record the AO composite using the color target and opaque depth texture."""
		if self.source is None:
			raise RuntimeError("Resize ambient occlusion targets before recording the pass")
		if not self._camera_ready:
			raise RuntimeError("Update ambient occlusion camera uniforms before recording the pass")
		if depth.size[:2] != self.size or not depth.format.startswith("depth"):
			raise ValueError("Ambient occlusion requires matching opaque depth dimensions")
		if depth.sample_count != self.sample_count or not depth.usage & TextureUsage.TEXTURE_BINDING:
			raise ValueError("Ambient occlusion depth must match the configured sample_count and have TEXTURE_BINDING")
		if output.format != self.target_format:
			raise ValueError("Output format must match the ambient occlusion target_format")
		if output.sample_count != 1:
			raise ValueError("Ambient occlusion output must be single-sample")
		inputs = self.shader.bind_group(1, scene_color=self.source, opaque_depth=depth)
		with commands.render_pass(color=output, label="6_ambient_occlusion") as render:
			render.set_pipeline(self.pipeline)
			render.set_bind_groups(self.shader.bind_group(0, ao_settings=self.camera), inputs)
			render.draw(3)
