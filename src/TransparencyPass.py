"""Stochastic transparency resources and explicit GPU passes.

The caller owns geometry, camera, command recording, output and submission.
No window, asset file, ECS world, or ImGui context is created by this module.
"""

from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .DrawBatches import RenderShader, ShaderPass
from .RenderContext import (
    BindGroup,
    ComputePipeline,
    RenderContext,
    RenderPipeline,
    Sampler,
    Shader,
    Texture,
    TextureUsage,
)
from .TransparencyData import MATERIAL_DTYPE, SHADING_VERTEX_DTYPE, VERTEX_DTYPE


SHADERS = Path(__file__).with_name("shaders")
DEBUG_VIEWS = ("final", "transmittance", "winner", "reconstruction", "filter_support")
KERNEL_SIZES = (1, 3, 5, 9)
MSAA_SAMPLE_COUNTS = (1, 4)
NOISE_MODES = ("object_hash", "uv_blue_lerp", "uv_blue_pick", "uv_bayer",
	"uv_blue_correlated", "triplanar_blue", "triplanar_correlated")


@dataclass(frozen=True)
class GrazingFade:
	"""Opacity fade by angle from the geometric normal: 0 face-on, 90 edge-on."""
	enabled: bool = True
	start_angle: float = 83.0
	end_angle: float = 86.0

	def __post_init__(self):
		if not isinstance(self.enabled, bool):
			raise ValueError("Grazing fade enabled must be a boolean")
		if not (np.isfinite(self.start_angle) and np.isfinite(self.end_angle)
			and 0 <= self.start_angle < self.end_angle <= 90
			and np.float32(self.start_angle) < np.float32(self.end_angle)):
			raise ValueError("Grazing fade angles must satisfy 0 <= start_angle < end_angle <= 90 degrees")


@dataclass(frozen=True)
class BilateralFilter:
	"""Reconstruction footprint and spatial/depth/transmittance edge tolerances.

	Spatial sigma is in pixels; depth sigma is relative to view-space depth,
	with a world-space minimum. Sloped surfaces use symmetric plane residuals.
	Transmittance sigma is linear RGBA. A 1x1 kernel bypasses reconstruction.
	"""
	kernel_size: int = 3
	spatial_sigma: float = 0.9
	depth_sigma: float = 0.05
	transmittance_sigma: float = 0.3
	min_depth_sigma: float = 0.01

	def __post_init__(self):
		if type(self.kernel_size) is not int or self.kernel_size not in KERNEL_SIZES:
			raise ValueError("Bilateral kernel_size must be 1, 3, 5 or 9")
		for name in ("spatial_sigma", "depth_sigma", "transmittance_sigma", "min_depth_sigma"):
			value = getattr(self, name)
			if not np.isfinite(value) or value <= 0:
				raise ValueError(f"Bilateral {name} must be finite and positive")


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
	owner: object
	shading_bindings: tuple[BindGroup, ...] = ()
	creation_args: dict | None = None


@dataclass(frozen=True)
class TransparencyTargets:
	main_color: Texture
	depth: Texture
	transparent_depth: Texture
	transparent_guide: Texture
	transmittance: Texture
	selection_depth: Texture
	winner: Texture
	winner_color: Texture
	winner_extra: Texture | None = None


class TransparencyPass:
	"""Persistent pipelines, bindings and size-dependent OIT targets.

	Inputs are GPU resources, independent of their source (glTF or procedural).
	Geometry uses VERTEX_DTYPE and MESH_INSTANCE_DTYPE. Raster and deferred vertex
	buffers share local coordinates; both use the same packed instance transforms.
	Custom deformation uses a matched geometry_shader/shading_shader pair with
	the documented winner/metadata ABI. Animation data preparation is caller-owned.
	"""
	def __init__(self, *, materials, base_textures, vertices, blue_noise,
		material_sampler=None, bilateral_filter=None, target_format=None, noise_scale=1.0, noise_mode="uv_blue_lerp",
		grazing_fade=None, sample_count=1,
		opaque_color=None, opaque_depth=None, material_source=None,
		geometry_shader=None, shading_shader=None, compose_shader=None):
		if (geometry_shader is None) != (shading_shader is None):
			raise ValueError("Custom deformation requires matched geometry and shading shaders")
		if type(sample_count) is not int or sample_count not in MSAA_SAMPLE_COUNTS:
			raise ValueError("Transparency sample_count must be 1 or 4")
		target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.target_format = target_format
		self.sample_count = sample_count
		self._custom_shaders = (geometry_shader, shading_shader, compose_shader)
		self._material_source = material_source
		self._shader_sets = {}
		self._activate_shaders(sample_count)
		self.depth_shader = geometry_shader or self._shader_set(1)[0]
		self.uniforms = self.shader.uniform_buffer("frame")
		self.noise = blue_noise
		self._noise_hierarchy = None
		self._transparent_geometries = []
		self._geometries = []
		self.size = None
		self.set_noise_scale(noise_scale)
		self.set_noise_mode(noise_mode)
		self.set_grazing_fade(grazing_fade if grazing_fade is not None else GrazingFade())
		self.bilateral_uniforms = self.compose_shader.uniform_buffer("bilateral")
		self.debug_view = "final"
		self.set_bilateral_filter(bilateral_filter if bilateral_filter is not None else BilateralFilter())
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry="shade_winner")
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none")
		self.vertices = vertices
		self.shared = dict(materials=materials, base_textures=base_textures,
			material_sampler=material_sampler or Sampler(address_mode_u="repeat", address_mode_v="repeat",
				mag_filter="linear", min_filter="linear", mipmap_filter="linear"))
		self.shade_frame = self.shading_shader.bind_group(0, frame=self.uniforms)
		self._opaque_inputs = None
		self._targets = None
		if opaque_color is not None or opaque_depth is not None:
			self.set_opaque_inputs(opaque_color, opaque_depth)

	def _shader_set(self, count):
		if count not in self._shader_sets:
			geometry, shading, compose = self._custom_shaders
			params = {"material_source": self._material_source, "sample_count": count}
			self._shader_sets[count] = (
				geometry or Shader(filepath=str(SHADERS / "stochastic_geometry.shader"), params=params),
				shading or Shader(filepath=str(SHADERS / "stochastic_shade.shader"), params=params),
				compose or Shader(filepath=str(SHADERS / "stochastic_compose.shader"),
					params={"srgb_surface": self.target_format.endswith("-srgb"), "sample_count": count}),
			)
		return self._shader_sets[count]

	def _activate_shaders(self, count):
		self.shader, self.shading_shader, self.compose_shader = self._shader_set(count)

	def set_sample_count(self, count):
		"""Switch 1x/4x MSAA, preserving geometry registration and live settings."""
		if type(count) is not int or count not in MSAA_SAMPLE_COUNTS:
			raise ValueError("Transparency sample_count must be 1 or 4")
		if count == self.sample_count:
			return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Runtime MSAA switching requires the default shader set")
		if self._opaque_inputs and self._opaque_inputs[0].sample_count != count:
			raise ValueError("External opaque inputs must match the selected sample count")
		self._activate_shaders(count)
		self.sample_count = count
		size, geometries = self.size, self._geometries
		self.size = None
		self._geometries, self._transparent_geometries = [], []
		for geometry in geometries:
			rebuilt = self.create_geometry_passes(**geometry.creation_args)
			geometry.render_shader.passes.clear()
			geometry.render_shader.passes.update(rebuilt.render_shader.passes)
			geometry.guide, geometry.shading = rebuilt.guide, rebuilt.shading
			self._geometries[-1] = geometry
			if geometry.creation_args["transparent"]:
				self._transparent_geometries[-1] = geometry
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry="shade_winner")
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none")
		self.shade_frame = self.shading_shader.bind_group(0, frame=self.uniforms)
		if size is not None:
			self.resize(size)
		return True

	def create_geometry_passes(self, metadata, *, transparent, instances,
		geometry_resources=None, shading_resources=None, geometry_bindings=(), shading_bindings=()):
		"""Build DrawBatches passes bound to caller-owned per-instance metadata."""
		if metadata.binding_size % 16 or not metadata.binding_size:
			raise ValueError("Metadata must contain nonempty vec4u records")
		if instances.binding_size % 32 or instances.binding_size // 32 < metadata.binding_size // 16:
			raise ValueError("Supply an instance transform for every metadata record")
		geometry_resources, shading_resources = dict(geometry_resources or {}), dict(shading_resources or {})
		if set(geometry_resources) & (set(self.shared) | {"metadata"}) or set(shading_resources) & (set(self.shared) | {"metadata", "vertices", "shading_instances"}):
			raise ValueError("Additional resources must not replace the geometry addressing resources")
		if any(group.group < 4 for group in geometry_bindings) or any(group.group < 3 for group in shading_bindings):
			raise ValueError("Extra geometry groups start at 4; extra shading groups start at 3")
		bindings = (self.shader.bind_group(0, frame=self.uniforms),
			self.shader.bind_group(2, **self.shared, metadata=metadata, **geometry_resources), *geometry_bindings)
		if transparent:
			multiply = {"color": {"src_factor": "zero", "dst_factor": "src", "operation": "add"},
				"alpha": {"src_factor": "zero", "dst_factor": "src-alpha", "operation": "add"}}
			first = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="transmittance",
				cull_mode="none", depth_write=False, blend=multiply, label="transmittance", sample_count=self.sample_count)
			last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="select_winner",
				cull_mode="none", label="select_winner", sample_count=self.sample_count)
			guide = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex",
				fragment_entry="transparent_depth", cull_mode="none", label="transparent_depth", sample_count=self.sample_count), bindings)
		else:
			first = last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="opaque",
				cull_mode="none", label="opaque", sample_count=self.sample_count)
			guide = None
		geometry = GeometryPasses(RenderShader({"main": ShaderPass(last, bindings), "prepass": ShaderPass(first, bindings)}),
			guide, self.shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices,
				shading_instances=instances, **shading_resources), self, tuple(shading_bindings))
		geometry.creation_args = dict(metadata=metadata, transparent=transparent, instances=instances,
			geometry_resources=geometry_resources, shading_resources=shading_resources,
			geometry_bindings=tuple(geometry_bindings), shading_bindings=tuple(shading_bindings))
		self._geometries.append(geometry)
		if transparent:
			self._transparent_geometries.append(geometry)
			if self.size is not None:
				self._bind_targets(geometry)
		return geometry

	def _bind_targets(self, geometry):
		bindings = geometry.render_shader['prepass'].bindings
		geometry.render_shader['main'].bindings = bindings + (
			self.shader.bind_group(3, opaque_depth=self.depth, blue_noise=self.noise,
				noise_hierarchy=self._noise_hierarchy if self._noise_hierarchy is not None else self.noise),)
		geometry.guide.bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth),)

	def set_bilateral_filter(self, options):
		"""Change reconstruction settings without recreating pipelines or targets."""
		self.bilateral_uniforms.content["sigmas"] = (
			options.spatial_sigma, options.depth_sigma, options.transmittance_sigma, options.min_depth_sigma,
		)
		self.bilateral_uniforms.content["kernel"] = (options.kernel_size//2, DEBUG_VIEWS.index(self.debug_view), 0, 0)
		self.bilateral_uniforms.upload()
		self.bilateral_filter = options

	def set_noise_scale(self, pixels):
		"""Set approximate surface noise-cell width in pixels (default 1).

		Larger cells trade coarser grain for fewer threshold changes under motion.
		No history or geometry preprocessing is required.
		"""
		limits = np.finfo(np.float32)
		if not np.isfinite(pixels) or not float(limits.tiny) <= pixels <= float(limits.max):
			raise ValueError("Noise scale must be a finite, positive normalized float32")
		self.noise_scale = float(pixels)
		self.uniforms.content["clip"][..., 2] = pixels
		self.uniforms.upload()

	def set_noise_mode(self, mode):
		"""Select object/UV/triplanar sampling, including correlated hierarchies.

		UV modes fall back to object hashing for missing/degenerate UVs. Only
		winner selection changes; opacity and reconstruction settings do not.
		"""
		if mode not in NOISE_MODES:
			raise ValueError(f"Noise mode must be one of {NOISE_MODES}")
		if mode in ("uv_blue_correlated", "triplanar_correlated") and self._noise_hierarchy is None:
			from .TextureProcessing import build_correlated_noise_hierarchy
			self._noise_hierarchy = build_correlated_noise_hierarchy(self.noise)
			if self.size is not None:
				for geometry in self._transparent_geometries:
					self._bind_targets(geometry)
		self.noise_mode = mode
		self.uniforms.content["clip"][..., 3] = NOISE_MODES.index(mode)
		self.uniforms.upload()

	def set_grazing_fade(self, options):
		"""Change transparent-surface opacity fade without rebuilding GPU resources."""
		self.uniforms.content["grazing"] = (options.start_angle, options.end_angle, options.enabled, 0)
		self.uniforms.upload()
		self.grazing_fade = options

	def set_debug_view(self, view):
		"""Inspect scalar transmittance, raw winners, or reconstructed contribution."""
		if view not in DEBUG_VIEWS:
			raise ValueError(f"Debug view must be one of {DEBUG_VIEWS}")
		self.debug_view = view
		self.set_bilateral_filter(self.bilateral_filter)

	def set_opaque_inputs(self, color, depth):
		"""Use another renderer's color/depth; pass both as None to restore defaults."""
		if (color is None) != (depth is None): raise ValueError("Supply both opaque color and depth")
		if color is not None:
			if color.size != depth.size or not depth.format.startswith("depth"):
				raise ValueError("Opaque inputs require matching dimensions and a depth format")
			for texture in (color, depth):
				if texture.sample_count != self.sample_count or not texture.usage & TextureUsage.TEXTURE_BINDING:
					raise ValueError("Opaque inputs must match sample_count and have TEXTURE_BINDING")
		self._opaque_inputs = None if color is None else (color, depth)
		self.size = None

	def resize(self, size: tuple[int, int], *, targets: TransparencyTargets | None = None) -> bool:
		"""Return whether targets changed; retain targets for minimized sizes."""
		if len(size) != 2:
			raise ValueError("Expected transparency target width and height")
		try:
			size = tuple(index(value) for value in size)
		except TypeError:
			raise ValueError("Expected integer transparency target dimensions") from None
		if size[0] <= 0 or size[1] <= 0:
			return False
		if size == self.size and (targets is None or targets is getattr(self, "_targets", None)):
			return False
		attachment = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
		inputs = getattr(self, "_opaque_inputs", None)
		if inputs and inputs[0].size[:2] != size: raise ValueError("Opaque input dimensions do not match transparency targets")
		formats = {"main_color": "rgba16float", "depth": "depth32float", "transparent_depth": "depth32float",
			"transparent_guide": "rgba16float",
			"transmittance": "rgba16float", "selection_depth": "depth32float", "winner": "rgba32uint", "winner_color": "rgba16float"}
		if self.sample_count == 4:
			formats["winner"] = "rgba16uint"
			formats["winner_extra"] = "rgba16uint"
		if targets is not None:
			for name, format in formats.items():
				texture = getattr(targets, name)
				usage = TextureUsage.TEXTURE_BINDING | (TextureUsage.STORAGE_BINDING if name == "winner_color" else TextureUsage.RENDER_ATTACHMENT)
				count = 1 if name == "winner_color" else self.sample_count
				layers = self.sample_count if name == "winner_color" else 1
				if texture is None or texture.size != (*size, layers) or texture.format != format or texture.sample_count != count or texture.usage & usage != usage:
					raise ValueError(f"Invalid transparency target {name}: expected {size}, {format}, {count} samples, {layers} layers, usage {usage}")
		else:
			values = {}
			for name, format in formats.items():
				if inputs and name in ("main_color", "depth"):
					values[name] = inputs[0 if name == "main_color" else 1]
				else:
					usage = TextureUsage.TEXTURE_BINDING | (TextureUsage.STORAGE_BINDING if name == "winner_color" else TextureUsage.RENDER_ATTACHMENT)
					count = 1 if name == "winner_color" else self.sample_count
					texture_size = (*size, self.sample_count) if name == "winner_color" else size
					values[name] = Texture(texture_size, format=format, usage=usage, label=name, sample_count=count)
			targets = TransparencyTargets(**values)
		for name in formats: setattr(self, name, getattr(targets, name))
		self.winner_extra = targets.winner_extra
		self._targets = targets
		for geometry in self._transparent_geometries:
			self._bind_targets(geometry)
		shade_resources = dict(winners=self.winner,
			winner_color=self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color)
		if self.sample_count > 1: shade_resources["winners_extra"] = self.winner_extra
		self.shade_targets = self.shading_shader.bind_group(1, **shade_resources)
		self.compose_resources = self.compose_shader.bind_group(0, main_color=self.main_color,
			transmittance=self.transmittance,
			winner_color=self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color, opaque_depth=self.depth,
			transparent_depth=self.transparent_depth, transparent_guide=self.transparent_guide,
			frame=self.uniforms, bilateral=self.bilateral_uniforms)
		self.size = size
		return True

	def _require_targets(self) -> tuple[int, int]:
		if self.size is None:
			raise RuntimeError("Resize transparency targets before recording passes or updating frame uniforms")
		return self.size

	def update_frame(self, *, view: ArrayLike, projection: ArrayLike,
		camera_position: ArrayLike, near: float, far: float,
		light_dir: ArrayLike = (0.4, 0.8, 0.3)) -> NDArray[np.float32]:
		"""Upload explicit camera values and return view @ projection for culling.

		Matrices follow the existing row-vector convention. No camera class is
		required; resize targets first. Invalid inputs leave uniforms unchanged.
		"""
		self._require_targets()
		view, projection = np.asarray(view, dtype=np.float32), np.asarray(projection, dtype=np.float32)
		position, light = np.asarray(camera_position, dtype=np.float32), np.asarray(light_dir, dtype=np.float32)
		if (view.shape != (4, 4) or projection.shape != (4, 4)
			or position.shape != (3,) or light.shape != (3,)
			or not all(np.all(np.isfinite(value)) for value in (view, projection, position, light))):
			raise ValueError("Expected finite 4x4 view/projection matrices and vec3 camera/light values")
		if not (np.isfinite(near) and np.isfinite(far) and 0 < near < far):
			raise ValueError("Camera clip distances must satisfy 0 < near < far")
		for name, value in dict(view=view, proj=projection, light_dir=(*light, 0),
			camera_position=(*position, 1), clip=(near, far, self.noise_scale, NOISE_MODES.index(self.noise_mode))).items():
			self.uniforms.content[name] = value
		self.uniforms.upload()
		return view @ projection

	def update_camera(self, camera, *, light_dir=(0.4, 0.8, 0.3)):
		"""Convenience adapter for camera objects; update_frame accepts plain arrays."""
		width, height = self._require_targets()
		return self.update_frame(view=camera.view(), projection=camera.projection(width/height),
			camera_position=camera.position, near=camera.near, far=camera.far, light_dir=light_dir)

	def opaque_pass(self, commands, *, clear=(0.035, 0.05, 0.08, 1)):
		self._require_targets()
		return commands.render_pass(color=self.main_color.color_attachment(clear=clear),
			depth=self.depth.depth_attachment(clear=1), label="1_opaque")

	def guide_pass(self, commands):
		self._require_targets()
		# Includes low-alpha fragments and pixels with no stochastic winner.
		return commands.render_pass(color=self.transparent_guide.color_attachment(clear=(0, 0, 0, 0)),
			depth=self.transparent_depth.depth_attachment(clear=1),
			label="2a_transparent_depth_guide")

	def transmittance_pass(self, commands):
		self._require_targets()
		return commands.render_pass(color=self.transmittance.color_attachment(clear=(1, 1, 1, 1)),
			depth=self.depth.depth_attachment(), label="2_transmittance")

	def selection_pass(self, commands):
		self._require_targets()
		colors = [self.winner.color_attachment(clear=(0, 0, 0, 0))]
		if self.sample_count > 1: colors.append(self.winner_extra.color_attachment(clear=(0, 0, 0, 0)))
		return commands.render_pass(color=colors,
			depth=self.selection_depth.depth_attachment(clear=1), label="3_select_winner")

	def shade_winner(self, commands, geometry):
		"""Record deferred shading for the metadata used during winner selection."""
		width, height = self._require_targets()
		if geometry.owner is not self: raise ValueError("Geometry passes belong to another transparency pass")
		with commands.compute_pass(label="4_shade_winner") as compute:
			compute.set_pipeline(self.shade_pipeline)
			for bindings in (self.shade_frame, self.shade_targets, geometry.shading, *geometry.shading_bindings):
				compute.set_bind_group(bindings)
			compute.dispatch((width+7)//8, (height+7)//8, self.sample_count)

	def compose(self, commands, output):
		"""Reconstruct into a caller-provided ColorAttachment; does not submit."""
		self._require_targets()
		if output.format != self.target_format: raise ValueError("Output format must match the configured transparency target_format")
		with commands.render_pass(color=output, label="5_reconstruct_compose") as render:
			render.set_pipeline(self.compose_pipeline)
			render.set_bind_groups(self.compose_resources)
			render.draw(3)
