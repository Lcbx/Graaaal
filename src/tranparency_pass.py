"""Reusable stochastic transparency resources and explicit GPU passes.

The caller owns geometry, camera, command recording, output and submission.
No window, asset file, ECS world, or ImGui context is created by this module.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from DrawBatches import RenderShader, ShaderPass
from RenderContext import (
	BindGroup, ComputePipeline, RenderContext,
	RenderPipeline, Sampler, Shader, Texture, TextureUsage,
)

SHADERS = Path(__file__).with_name("shaders")
DEBUG_VIEWS = ("final", "transmittance", "winner", "reconstruction", "filter_support")
VERTEX_DTYPE = np.dtype([
	("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2),
	("barycentric", "<f4", 2), ("triangle", "<u4"),
])
SHADING_VERTEX_DTYPE = np.dtype([
	("position", "<f4", 4), ("normal", "<f4", 4), ("uv", "<f4", 4),
])
MATERIAL_DTYPE = np.dtype([
	("base_color", "<f4", 4), ("emission", "<f4", 4),
	("surface", "<f4", 4), ("opacity_rgb", "<f4", 4), ("flags", "<u4", 4),
])


@dataclass(frozen=True)
class BilateralFilter:
	"""Reconstruction footprint and spatial/depth/transmittance edge tolerances.

	Spatial sigma is in pixels; depth sigma is relative to view-space depth,
	with a world-space minimum. Near/far sigma sets interpolate with view depth.
	Transmittance sigma operates on linear RGBA; its far weight can fade to zero.
	"""
	kernel_size: int = 5
	# Retain the original reconstruction support near cutout-shaped BLEND edges.
	spatial_sigma: float = 2**0.5
	depth_sigma: float = 0.02
	transmittance_sigma: float = (1/24)**0.5
	min_depth_sigma: float = 0.01
	adaptive: bool = True
	filter_near: float = 10.0
	filter_far: float = 50.0
	far_spatial_sigma: float = 2.0
	far_depth_sigma: float = 0.05
	far_transmittance_sigma: float = 0.5
	far_transmittance_weight: float = 0.0

	def __post_init__(self):
		if not isinstance(self.kernel_size, int) or self.kernel_size not in (3, 5):
			raise ValueError("Bilateral kernel_size must be 3 or 5")
		for name in ("spatial_sigma", "depth_sigma", "transmittance_sigma", "min_depth_sigma",
			"far_spatial_sigma", "far_depth_sigma", "far_transmittance_sigma"):
			value = getattr(self, name)
			if not np.isfinite(value) or value <= 0:
				raise ValueError(f"Bilateral {name} must be finite and positive")
		if not isinstance(self.adaptive, bool):
			raise ValueError("Bilateral adaptive must be a boolean")
		if not (np.isfinite(self.filter_near) and np.isfinite(self.filter_far)
			and 0 <= self.filter_near < self.filter_far):
			raise ValueError("Bilateral distances must satisfy 0 <= filter_near < filter_far")
		if not np.isfinite(self.far_transmittance_weight) or not 0 <= self.far_transmittance_weight <= 1:
			raise ValueError("Bilateral far_transmittance_weight must be between zero and one")


@dataclass
class GeometryPasses:
	"""Register render_shader with DrawBatches; use guide as a pass override.

	metadata is a vec4u buffer in compact instance order: triangle offset,
	material index, stable surface ID, reserved. All transparent draws in one
	selection pass must share this metadata/vertex address space.
	"""
	render_shader: RenderShader
	guide: ShaderPass | None
	shading: BindGroup


class TransparencyPass:
	"""Persistent pipelines, bindings and size-dependent OIT targets.

	Inputs are GPU resources, independent of their source (glTF or procedural).
	Geometry uses VERTEX_DTYPE and mesh_instance_dtype. Supply baked world-space
	positions/normals in raster and SHADING_VERTEX_DTYPE deferred vertex buffers.
	The raster shader adds iPosition; deferred positions must include that same
	translation. Packed rotation/scale are not applied by these material shaders.
	"""
	def __init__(self, *, materials, base_textures, vertices, blue_noise,
		material_sampler=None, bilateral_filter=None, target_format=None):
		self.shader = Shader(filepath=str(SHADERS / "stochastic_geometry.shader"))
		self.shading_shader = Shader(filepath=str(SHADERS / "stochastic_shade.shader"))
		target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.compose_shader = Shader(filepath=str(SHADERS / "stochastic_compose.shader"),
			params={"srgb_surface": target_format.endswith("-srgb")})
		self.uniforms = self.shader.UniformBuffer("frame")
		self.bilateral_uniforms = self.compose_shader.UniformBuffer("bilateral")
		self.debug_view = "final"
		self.set_bilateral_filter(bilateral_filter if bilateral_filter is not None else BilateralFilter())
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry="shade_winner")
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none")
		self.vertices = vertices
		self.noise = blue_noise
		self.shared = dict(materials=materials, base_textures=base_textures,
			material_sampler=material_sampler or Sampler(address_mode_u="repeat", address_mode_v="repeat",
				mag_filter="linear", min_filter="linear", mipmap_filter="linear"))
		self.shade_frame = self.shading_shader.bind_group(0, frame=self.uniforms)
		self._transparent_geometries = []
		self.size = None
		# DrawBatches requires a float texture even when only frustum culling runs.
		self.dummy_hzb = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING).view()

	def create_geometry_passes(self, metadata, *, transparent):
		"""Build DrawBatches passes bound to caller-owned per-instance metadata."""
		bindings = (self.shader.bind_group(0, frame=self.uniforms),
			self.shader.bind_group(2, **self.shared, metadata=metadata))
		if transparent:
			multiply = {"color": {"src_factor": "zero", "dst_factor": "src", "operation": "add"},
				"alpha": {"src_factor": "zero", "dst_factor": "src-alpha", "operation": "add"}}
			first = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="transmittance",
				cull_mode="none", depth_write=False, blend=multiply, label="transmittance")
			last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="select_winner",
				cull_mode="none", label="select_winner")
			guide = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex",
				fragment_entry="transparent_depth", cull_mode="none", label="transparent_depth"), bindings)
		else:
			first = last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="opaque",
				cull_mode="none", label="opaque")
			guide = None
		geometry = GeometryPasses(RenderShader(ShaderPass(last, bindings), ShaderPass(first, bindings)),
			guide, self.shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices))
		if transparent:
			self._transparent_geometries.append(geometry)
			if self.size is not None:
				self._bind_targets(geometry)
		return geometry

	def _bind_targets(self, geometry):
		bindings = geometry.render_shader.prepass.bindings
		geometry.render_shader.main.bindings = bindings + (
			self.shader.bind_group(3, opaque_depth=self.depth, blue_noise=self.noise),)
		geometry.guide.bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth),)

	def set_bilateral_filter(self, options):
		"""Change reconstruction settings without recreating pipelines or targets."""
		self.bilateral_uniforms.content["near_sigmas"] = (
			options.spatial_sigma, options.depth_sigma, options.transmittance_sigma, options.min_depth_sigma,
		)
		self.bilateral_uniforms.content["far_sigmas"] = (
			options.far_spatial_sigma, options.far_depth_sigma, options.far_transmittance_sigma, options.far_transmittance_weight,
		)
		self.bilateral_uniforms.content["distance_range"] = (options.filter_near, options.filter_far, 0, 0)
		self.bilateral_uniforms.content["kernel"] = (options.kernel_size//2, DEBUG_VIEWS.index(self.debug_view), options.adaptive, 0)
		self.bilateral_uniforms.upload()
		self.bilateral_filter = options

	def set_debug_view(self, view):
		"""Inspect scalar transmittance, raw winners, or reconstructed contribution."""
		if view not in DEBUG_VIEWS:
			raise ValueError(f"Debug view must be one of {DEBUG_VIEWS}")
		self.debug_view = view
		self.set_bilateral_filter(self.bilateral_filter)

	def resize(self, size):
		"""Allocate targets and refresh geometry bindings; ignore minimized sizes."""
		if len(size) != 2:
			raise ValueError("Expected transparency target width and height")
		size = tuple(size)
		if size[0] <= 0 or size[1] <= 0:
			return
		if size == self.size:
			return
		attachment = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
		self.main_color = Texture(size, format="rgba16float", usage=attachment, label="opaque_hdr")
		self.depth = Texture(size, format="depth32float", usage=attachment, label="opaque_depth")
		self.transparent_depth = Texture(size, format="depth32float", usage=attachment, label="transparent_guide_depth")
		self.transmittance = Texture(size, format="rgba16float", usage=attachment, label="transmittance")
		self.selection_depth = Texture(size, format="depth32float", usage=attachment, label="selection_depth")
		self.winner = Texture(size, format="rgba32uint", usage=attachment, label="winner_payload")
		self.winner_color = Texture(size, format="rgba16float",
			usage=TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING, label="winner_hdr")
		for geometry in self._transparent_geometries:
			self._bind_targets(geometry)
		self.shade_targets = self.shading_shader.bind_group(1, winners=self.winner, winner_color=self.winner_color)
		self.compose_resources = self.compose_shader.bind_group(0, main_color=self.main_color,
			transmittance=self.transmittance, winner_color=self.winner_color, opaque_depth=self.depth,
			transparent_depth=self.transparent_depth, frame=self.uniforms, bilateral=self.bilateral_uniforms)
		self.size = size

	def update_camera(self, camera, *, light_dir=(0.4, 0.8, 0.3)):
		"""Upload frame uniforms after resize(); return the culling view-projection."""
		if self.size is None:
			raise RuntimeError("Resize transparency targets before updating the camera")
		width, height = self.size
		view, projection = camera.view(), camera.projection(width/height)
		for name, value in dict(view=view, proj=projection, light_dir=(*light_dir, 0),
			camera_position=(*camera.position, 1), clip=(camera.near, camera.far, 0, 0)).items():
			self.uniforms.content[name] = value
		self.uniforms.upload()
		return view @ projection

	def opaque_pass(self, commands, *, clear=(0.035, 0.05, 0.08, 1)):
		return commands.render_pass(color=self.main_color.color_attachment(clear=clear),
			depth=self.depth.depth_attachment(clear=1), label="1_opaque")

	def guide_pass(self, commands):
		# Includes low-alpha fragments and pixels with no stochastic winner.
		return commands.render_pass(depth=self.transparent_depth.depth_attachment(clear=1),
			label="2a_transparent_depth_guide")

	def transmittance_pass(self, commands):
		return commands.render_pass(color=self.transmittance.color_attachment(clear=(1, 1, 1, 1)),
			depth=self.depth.depth_attachment(), label="2_transmittance")

	def selection_pass(self, commands):
		return commands.render_pass(color=self.winner.color_attachment(clear=(0, 0, 0, 0)),
			depth=self.selection_depth.depth_attachment(clear=1), label="3_select_winner")

	def shade_winner(self, commands, geometry):
		"""Record deferred shading for the metadata used during winner selection."""
		width, height = self.size
		with commands.compute_pass(label="4_shade_winner") as compute:
			compute.set_pipeline(self.shade_pipeline)
			for bindings in (self.shade_frame, self.shade_targets, geometry.shading):
				compute.set_bind_group(bindings)
			compute.dispatch((width+7)//8, (height+7)//8)

	def compose(self, commands, output):
		"""Reconstruct into a caller-provided ColorAttachment; does not submit."""
		with commands.render_pass(color=output, label="5_reconstruct_compose") as render:
			render.set_pipeline(self.compose_pipeline)
			render.set_bind_groups(self.compose_resources)
			render.draw(3)
