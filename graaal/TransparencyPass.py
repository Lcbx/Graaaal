"""Stochastic transparency resources and explicit GPU passes.

The caller owns geometry, camera, command recording, output and submission.
No window, asset file, ECS world, or ImGui context is created by this module.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from operator import index
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .DrawBatches import RenderShader, ShaderPass
from .RenderContext import (
    BindGroup,
	BufferUsage,
	GpuBuffer,
    ColorAttachment,
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
DEBUG_VIEWS = ("final", "transmittance", "winner", "reconstruction", "filter_support",
	"winner_a", "winner_b", "winner_count", "forced", "nearest", "nearest_mask", "tail_transmittance")
KERNEL_SIZES = (1, 3, 5, 9)
MSAA_SAMPLE_COUNTS = (1, 4)
NOISE_MODES = ("object_hash", "uv_blue_lerp", "uv_blue_pick", "uv_bayer",
	"uv_blue_correlated", "triplanar_blue", "triplanar_correlated", "anchored_blue")
FILL_NOISE_MODES = ("object_hash", "anchored_blue")


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


@dataclass(frozen=True)
class ConfidenceFilter:
	"""Retained-support budget and cascade/pyramid scale counts."""
	target_winners: float = 4.0
	levels: int = 3
	cascade_passes: int = 4

	def __post_init__(self):
		if not np.isfinite(self.target_winners) or not 0 < self.target_winners <= np.finfo(np.float32).max:
			raise ValueError("target_winners must be finite and positive")
		if type(self.levels) is not int or self.levels not in (2, 3):
			raise ValueError("Confidence pyramid levels must be 2 or 3")
		if type(self.cascade_passes) is not int or not 1 <= self.cascade_passes <= 4:
			raise ValueError("Confidence cascade_passes must be 1 through 4")


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
	nearest: ShaderPass | None = None
	nearest_shading: BindGroup | None = None


@dataclass(frozen=True)
class TransparencyTargets:
	main_color: Texture
	depth: Texture
	transparent_depth: Texture
	transparent_guide: Texture
	transmittance: Texture
	selection_depth: Texture | None
	winner: Texture | None
	winner_color: Texture
	winner_extra: Texture | None = None
	winner_first: Texture | None = None
	contributor_count: Texture | None = None
	winner_source: Texture | None = None
	reconstruction_color: Texture | None = None
	reconstruction_pair: Texture | None = None
	reconstruction_pair_scratch: Texture | None = None
	winner_first_payload: Texture | None = None
	nearest_payload: Texture | None = None
	nearest_extra: Texture | None = None
	nearest_alpha: Texture | None = None
	nearest_depth: Texture | None = None
	nearest_color: Texture | None = None


class TransparencyPass:
	"""Persistent pipelines, bindings and size-dependent OIT targets.

	Inputs are GPU resources, independent of their source (glTF or procedural).
	Geometry uses VERTEX_DTYPE and MESH_INSTANCE_DTYPE. Raster and deferred vertex
	buffers share local coordinates; both use the same packed instance transforms.
	Custom deformation uses a matched geometry_shader/shading_shader pair with
	the documented winner/metadata ABI. Animation data preparation is caller-owned.
	"""
	def __init__(self, *, materials, base_textures, vertices, blue_noise=None, blue_noise_second=None,
		material_sampler=None, bilateral_filter=None, target_format=None, noise_scale=1.0, noise_mode="uv_blue_lerp",
		grazing_fade=None, sample_count=1, winner_count=2, noise_footprint_balance=1.0, noise_footprint_balancing=True,
		force_single_contributor=False, winner_mix="average", winner_coupling="complement", reconstruction="fragment", reconstruction_passes=1, confidence_filter=None, early_b_rejection=False,
		noise_anchor_pixels=24.0, noise_anchor_subdivisions=4,
		opaque_color=None, opaque_depth=None, material_source=None, material_library=None, frame_fields=None, exact_nearest=False,
		geometry_shader=None, shading_shader=None, compose_shader=None, tail_allocation="banded", kawase_passes=1,
		banded_proxy=False, banded_subset=False, banded_tail_msaa=False):
		if any(type(value) is not bool for value in (banded_proxy, banded_subset, banded_tail_msaa)): raise ValueError("Banded options must be boolean")
		self.banded_proxy, self.banded_subset, self.banded_tail_msaa = banded_proxy, banded_subset, banded_tail_msaa
		if tail_allocation not in ("stochastic", "banded"): raise ValueError("tail_allocation must be stochastic or banded")
		if type(kawase_passes) is not int or not 0 <= kawase_passes <= 8: raise ValueError("kawase_passes must be an integer from 0 through 8")
		self.tail_allocation, self.kawase_passes = tail_allocation, kawase_passes
		self.band_counts = self.band_payloads = None
		self.kawase_stages = []
		if self.banded_tail:
			if force_single_contributor or early_b_rejection or winner_mix != "average" or winner_count != 2 or winner_coupling != "complement":
				raise ValueError("Banded tail requires two complemented averaged winners without force acceptance or stencil rejection")
			exact_nearest, reconstruction, reconstruction_passes = True, "kawase", 1
		if (geometry_shader is None) != (shading_shader is None):
			raise ValueError("Custom deformation requires matched geometry and shading shaders")
		if type(exact_nearest) is not bool: raise ValueError("exact_nearest must be boolean")
		if exact_nearest and any(s is not None for s in (geometry_shader, shading_shader, compose_shader)):
			raise ValueError("Exact nearest requires the default shader set; material hooks remain supported")
		self.exact_nearest = exact_nearest
		if type(sample_count) is not int or sample_count not in MSAA_SAMPLE_COUNTS:
			raise ValueError("Transparency sample_count must be 1 or 4")
		self._validate_count(winner_count, "winner_count")
		self._validate_count(reconstruction_passes, "reconstruction_passes")
		if winner_coupling not in ("complement", "independent"): raise ValueError("winner_coupling must be complement or independent")
		if confidence_filter is not None and not isinstance(confidence_filter, ConfidenceFilter): raise TypeError("Expected ConfidenceFilter")
		if winner_count > 2 and winner_mix != "fill": raise ValueError("winner_count above two requires fill mixing")
		if type(early_b_rejection) is not bool: raise ValueError("early_b_rejection must be boolean")
		if early_b_rejection:
			if winner_mix != "fill" or winner_count < 2: raise ValueError("Early B rejection requires multiple-winner fill")
			RenderContext.require_features(("depth32float-stencil8",))
		if reconstruction not in ("fragment", "compute", "push_pull", "atrous", "kawase") or (reconstruction == "kawase" and not self.banded_tail):
			raise ValueError("reconstruction must be fragment, compute, push_pull or atrous")
		if (reconstruction != "fragment" or reconstruction_passes > 1) and any(s is not None for s in (geometry_shader, shading_shader, compose_shader)):
			raise ValueError("Compute reconstruction requires the default shader set; material hooks remain supported")
		if winner_mix not in ("average", "fill"): raise ValueError("winner_mix must be average or fill")
		if winner_mix == "fill" and noise_mode not in FILL_NOISE_MODES:
			raise ValueError(f"Fill requires a validated noise mode: {FILL_NOISE_MODES}")
		if winner_mix == "fill" and (blue_noise is not None or blue_noise_second is not None):
			raise ValueError("Fill requires the audited default noise sources")
		if type(force_single_contributor) is not bool:
			raise ValueError("force_single_contributor must be boolean")
		if force_single_contributor and any(s is not None for s in (geometry_shader, shading_shader, compose_shader)):
			raise ValueError("Force acceptance requires the default shader set; material hooks remain supported")
		if (winner_count > 1 or winner_mix == "fill") and any(s is not None for s in (geometry_shader, shading_shader, compose_shader)):
			raise ValueError("Multiple winners and fill require the default shader set; material hooks remain supported")
		if type(noise_footprint_balancing) is not bool:
			raise ValueError("Noise footprint balancing must be boolean")
		target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.target_format = target_format
		self.sample_count = sample_count
		self.winner_count = winner_count
		self.winner_mix = winner_mix
		self.winner_coupling = winner_coupling
		self.reconstruction = reconstruction
		self.confidence_filter = confidence_filter or ConfidenceFilter()
		self.reconstruction_passes = reconstruction_passes
		self.early_b_rejection = early_b_rejection
		self.force_single_contributor = force_single_contributor
		self.noise_footprint_balancing = noise_footprint_balancing
		self._custom_shaders = (geometry_shader, shading_shader, compose_shader)
		self._material_source = material_source
		self._material_library = material_library
		if frame_fields is None: frame_fields = {}
		if not isinstance(frame_fields, Mapping):
			raise TypeError("frame_fields must map names to (WGSL type, default) pairs")
		reserved = {"view", "proj", "camera_position", "clip", "grazing", "noise_footprint_balance", "noise_anchor"}
		self.frame_fields = {}
		for name, spec in frame_fields.items():
			if not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name in reserved:
				raise ValueError(f"Invalid or reserved frame field {name!r}")
			if not isinstance(spec, (tuple, list)) or len(spec) != 2 or not isinstance(spec[0], str) or not spec[0].strip():
				raise ValueError(f"Frame field {name!r} requires a WGSL type and default")
			self.frame_fields[name] = (spec[0], spec[1])
		self._frame_types = {name: spec[0] for name, spec in self.frame_fields.items()}
		self._shader_sets = {}
		self._geometry_shaders = {}
		self._reconstruction_shaders = {}
		self._pair_shaders = {}
		self._mix_shaders = {}
		self.pyramid = []
		self._layout_cache, self._pipeline_layouts = {}, {}
		self._winner_settings = None
		self._activate_shaders(sample_count)
		self.uniforms = self.shader.uniform_buffer("frame")
		for name, (_, default) in self.frame_fields.items():
			self.uniforms.content[name] = self._frame_value(name, default)
		lighting_shader = next((shader for shader in (self.shader, self.compose_shader)
			if "lighting" in shader.info.uniforms), None)
		self.lighting_uniforms = lighting_shader.uniform_buffer("lighting") if lighting_shader else None
		if self.lighting_uniforms is not None and "light_dir" in self.lighting_uniforms.dtype.names:
			self.lighting_uniforms.content["light_dir"] = (0.4, 0.8, 0.3, 0)
			self.lighting_uniforms.upload_changed()
		self._default_noise = blue_noise is None
		self._default_noise_pair = blue_noise is None and blue_noise_second is None
		if blue_noise is None:
			from .TextureProcessing import get_default_blue_noise
			blue_noise = get_default_blue_noise()
		self.noise = blue_noise
		self.noise_second = blue_noise_second
		self._noise_hierarchy = None
		self._noise_hierarchy_second = None
		self._transparent_geometries = []
		self._geometries = []
		self.size = None
		self.set_noise_scale(noise_scale)
		self.set_noise_footprint_balance(noise_footprint_balance)
		self.set_noise_anchor(noise_anchor_pixels, noise_anchor_subdivisions)
		self._ensure_second_noise()
		self.set_noise_mode(noise_mode)
		self.set_grazing_fade(grazing_fade if grazing_fade is not None else GrazingFade())
		self.bilateral_uniforms = self.compose_shader.uniform_buffer("bilateral")
		self.debug_view = "final"
		self.set_bilateral_filter(bilateral_filter if bilateral_filter is not None else BilateralFilter())
		entry = "shade_bands" if self.banded_tail else "shade_winner"
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry=entry, layout=self._pipeline_layout(self.shading_shader, entry))
		self.mix_pipeline = self._create_mix_pipeline()
		self._create_nearest_pipeline()
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none",
			layout=self._pipeline_layout(self.compose_shader, "fullscreen", "compose"))
		self._create_reconstruction_pipeline()
		self.vertices = vertices
		self.shared = dict(materials=materials, base_textures=base_textures,
			material_sampler=material_sampler or Sampler(address_mode_u="repeat", address_mode_v="repeat",
				mag_filter="linear", min_filter="linear", mipmap_filter="linear"))
		self.shade_frame = self._frame_group(self.shading_shader)
		self._opaque_inputs = None
		self._targets = None
		if opaque_color is not None or opaque_depth is not None:
			self.set_opaque_inputs(opaque_color, opaque_depth)

	def _geometry_shader(self, count):
		key = (count, self.noise_footprint_balancing, min(self.winner_count, 2), self.force_single_contributor, self.winner_mix, self.complemented_winners, self.exact_nearest, self.banded_tail, self._banded_flags)
		if key not in self._geometry_shaders:
			self._geometry_shaders[key] = self._custom_shaders[0] or Shader(
				filepath=str(SHADERS / "stochastic_geometry.shader"),
				features=("NOISE_FOOTPRINT_BALANCE",) if self.noise_footprint_balancing else (),
				params={**self._banded_params(count), "material_source": self._material_source, "sample_count": count,
					"winner_count": min(self.winner_count, 2), "force_single_contributor": self.force_single_contributor, "winner_mix": self.winner_mix,
					"material_library": self._material_library, "frame_fields": self._frame_types, "exact_nearest": self.exact_nearest, "complemented_winners": self.complemented_winners, "banded_tail": self.banded_tail})
		return self._geometry_shaders[key]

	@property
	def depth_shader(self):
		"""The auxiliary 1x geometry module is created only when requested."""
		return self._geometry_shader(1)

	def _frame_group(self, shader):
		if "frame" in shader.info.uniforms and shader.info.uniforms["frame"].dtype != self.uniforms.dtype:
			raise ValueError("Shader Frame uniform layout differs from the configured frame_fields")
		resources = dict(frame=self.uniforms)
		if "lighting" in shader.info.bindings: resources["lighting"] = self.lighting_uniforms
		return shader.bind_group(0, **resources)

	@staticmethod
	def _validate_count(count, name):
		if type(count) is not int or not 1 <= count <= 0xffffffff:
			raise ValueError(f"Transparency {name} must be a positive uint32 integer")

	def _pipeline_layout(self, shader, *entries):
		if any(custom is not None for custom in self._custom_shaders): return "auto"
		key = (shader, entries)
		if key not in self._pipeline_layouts:
			from .material_layout import material_pipeline_layout
			unfilterable = ("level_pair", "level_depth", "level_transmission", "parent_mu", "parent_depth", "parent_transmission", "cascade_pair", "cascade_result") if shader is getattr(self, "pyramid_shader", None) else ()
			self._pipeline_layouts[key] = material_pipeline_layout(shader, entries, self._layout_cache, unfilterable)
		return self._pipeline_layouts[key]

	def _shader_set(self, count):
		key = (count, self.noise_footprint_balancing, min(self.winner_count, 2), self.force_single_contributor, self.winner_mix, self.reconstruction, self.reconstruction_passes > 1, self.complemented_winners, self.exact_nearest, self.banded_tail, self._banded_flags)
		if key not in self._shader_sets:
			geometry, shading, compose = self._custom_shaders
			params = {**self._banded_params(count), "material_source": self._material_source, "sample_count": count,
				"material_library": self._material_library, "frame_fields": self._frame_types, "exact_nearest": self.exact_nearest, "banded_tail": self.banded_tail,
				"winner_count": min(self.winner_count, 2),
				"force_single_contributor": self.force_single_contributor, "winner_mix": self.winner_mix, "complemented_winners": self.complemented_winners}
			self._shader_sets[key] = (
				self._geometry_shader(count),
				shading or Shader(filepath=str(SHADERS / "stochastic_shade.shader"), params=dict(params, sample_count=(count if self.banded_tail_msaa else 1) if self.banded_tail else count)),
				compose or Shader(filepath=str(SHADERS / "stochastic_compose.shader"),
					params={**self._banded_params(count), "srgb_surface": self.target_format.endswith("-srgb"), "sample_count": count,
						"winner_count": min(self.winner_count, 2), "force_single_contributor": self.force_single_contributor, "winner_mix": self.winner_mix,
						"compute_reconstruction": self.reconstruction != "fragment", "filter_input": self.reconstruction_passes > 1 and self.reconstruction not in ("push_pull", "atrous"),
						"frame_fields": self._frame_types, "exact_nearest": self.exact_nearest, "banded_tail": self.banded_tail}),
			)
		return self._shader_sets[key]

	def _activate_shaders(self, count):
		self.shader, self.shading_shader, self.compose_shader = self._shader_set(count)
		self.tail_shader = self.shader
		if self.banded_tail and count > 1 and not self.banded_tail_msaa:
			key = ("tail", count, self._banded_flags, self.noise_footprint_balancing)
			if key not in self._geometry_shaders:
				params = dict(self._banded_params(count), sample_count=1, guide_sample_count=count, banded_statistics=True,
					material_source=self._material_source, material_library=self._material_library, frame_fields=self._frame_types,
					exact_nearest=True, banded_tail=True, winner_count=2)
				self._geometry_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_geometry.shader"), params=params,
					features=("NOISE_FOOTPRINT_BALANCE",) if self.noise_footprint_balancing else ())
			self.tail_shader = self._geometry_shaders[key]
		if self.winner_count > 1 or self.winner_mix == "fill":
			self._winner_settings = [self.shader.uniform_buffer("winner_settings") for _ in range(self.winner_count)]
			for stream, settings in enumerate(self._winner_settings):
				settings.content["info"] = (stream, int(stream > 0 and hasattr(self, "bilateral_filter") and self.bilateral_filter.kernel_size != 1), self.winner_count, 0)
				settings.upload()
		else: self._winner_settings = None

	def _create_mix_pipeline(self):
		self.mixing_shader = self.shading_shader
		if self.winner_count == 1 or self.banded_tail: return None
		entry = "shade_fill" if self.winner_mix == "fill" else "shade_mix"
		if self.complemented_winners:
			if self.shading_shader not in self._mix_shaders:
				self._mix_shaders[self.shading_shader] = Shader(filepath=str(SHADERS / "stochastic_shade.shader"), params={"material_source": self._material_source, "sample_count": self.sample_count, "material_library": self._material_library, "frame_fields": self._frame_types, "exact_nearest": self.exact_nearest, "winner_count": 2, "force_single_contributor": self.force_single_contributor, "winner_mix": "average", "complemented_winners": True, "mix_only": True})
			self.mixing_shader = self._mix_shaders[self.shading_shader]
		return ComputePipeline(self.mixing_shader, entry=entry, layout=self._pipeline_layout(self.mixing_shader, entry))

	def _create_nearest_pipeline(self):
		self.nearest_shading_shader = self.nearest_shade_pipeline = None
		if not self.exact_nearest: return
		key = ("nearest", self.sample_count)
		if key not in self._mix_shaders:
			self._mix_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_shade.shader"), params={"material_source": self._material_source, "sample_count": self.sample_count, "material_library": self._material_library, "frame_fields": self._frame_types, "winner_count": 1})
		self.nearest_shading_shader = self._mix_shaders[key]
		self.nearest_shade_pipeline = ComputePipeline(self.nearest_shading_shader, entry="shade_winner", layout=self._pipeline_layout(self.nearest_shading_shader, "shade_winner"))

	def set_exact_nearest(self, enabled):
		if type(enabled) is not bool: raise ValueError("exact_nearest must be boolean")
		if self.banded_tail and not enabled: raise ValueError("Banded tail requires exact_nearest")
		if enabled == self.exact_nearest: return False
		if any(s is not None for s in self._custom_shaders): raise ValueError("Exact nearest requires the default shader set")
		self.exact_nearest = enabled
		if not enabled and self.debug_view in ("nearest", "nearest_mask", "tail_transmittance"): self.set_debug_view("final")
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

	def _create_reconstruction_pipeline(self):
		self.reconstruct_shader = self.reconstruct_pipeline = self.reconstruct_frame = None
		self.pair_shader = self.pair_pipeline = self.pair_frame = None
		self.initial_pair_shader = self.initial_pair_pipeline = None
		self.pyramid_shader = None
		self.pyramid_frame = self.pyramid_inputs = self.pyramid_dummy = None
		self.pyramid_pipelines = {}
		self.pyramid, self.pyramid_stages = [], []
		if self.banded_tail: return
		if self.reconstruction in ("push_pull", "atrous"):
			key = (self.sample_count, self.force_single_contributor, self.reconstruction, self.exact_nearest)
			if key not in self._reconstruction_shaders:
				filename = "stochastic_atrous.shader" if self.reconstruction == "atrous" else "stochastic_push_pull.shader"
				self._reconstruction_shaders[key] = Shader(filepath=str(SHADERS / filename), params={"sample_count": self.sample_count, "force_single_contributor": self.force_single_contributor, "frame_fields": self._frame_types, "exact_nearest": self.exact_nearest})
			self.pyramid_shader = self._reconstruction_shaders[key]
			entries = ("cascade", "finish") if self.reconstruction == "atrous" else ("pull", "push", "finish")
			self.pyramid_pipelines = {entry: ComputePipeline(self.pyramid_shader, entry=entry, layout=self._pipeline_layout(self.pyramid_shader, entry)) for entry in entries}
			self.pyramid_frame = self._frame_group(self.pyramid_shader)
			return
		if self.reconstruction_passes > 1:
			key = (self.sample_count, self.force_single_contributor, self.reconstruction, self.exact_nearest)
			if key not in self._pair_shaders:
				self._pair_shaders[key] = Shader(filepath=str(SHADERS / ("stochastic_reconstruct.shader" if self.reconstruction == "compute" else "stochastic_filter.shader")),
					params={"sample_count": self.sample_count, "force_single_contributor": self.force_single_contributor,
						"filter_input": True, "filter_pairs": True, "tiled_reconstruction": self.reconstruction == "compute",
						"frame_fields": self._frame_types, "exact_nearest": self.exact_nearest})
			self.pair_shader = self._pair_shaders[key]
			if self.reconstruction == "compute":
				self.pair_pipeline = ComputePipeline(self.pair_shader, entry="reconstruct", layout=self._pipeline_layout(self.pair_shader, "reconstruct"))
			else:
				self.pair_pipeline = RenderPipeline(self.pair_shader, vertex_entry="fullscreen", fragment_entry="filter_pair",
					depth_test=None, depth_write=False, cull_mode="none", layout=self._pipeline_layout(self.pair_shader, "fullscreen", "filter_pair"))
			self.pair_frame = self._frame_group(self.pair_shader)
			if self.exact_nearest:
				key = (self.sample_count, self.force_single_contributor, self.reconstruction, "initial")
				if key not in self._pair_shaders:
					self._pair_shaders[key] = Shader(filepath=str(SHADERS / ("stochastic_reconstruct.shader" if self.reconstruction == "compute" else "stochastic_filter.shader")), params={"sample_count": self.sample_count, "force_single_contributor": self.force_single_contributor, "filter_input": True, "filter_pairs": True, "initial_pairs": True, "exact_nearest": True, "tiled_reconstruction": self.reconstruction == "compute", "frame_fields": self._frame_types})
				self.initial_pair_shader = self._pair_shaders[key]
				if self.reconstruction == "compute":
					self.initial_pair_pipeline = ComputePipeline(self.initial_pair_shader, entry="reconstruct", layout=self._pipeline_layout(self.initial_pair_shader, "reconstruct"))
				else:
					self.initial_pair_pipeline = RenderPipeline(self.initial_pair_shader, vertex_entry="fullscreen", fragment_entry="filter_pair", depth_test=None, depth_write=False, cull_mode="none", layout=self._pipeline_layout(self.initial_pair_shader, "fullscreen", "filter_pair"))
		if self.reconstruction == "fragment": return
		key = (self.sample_count, self.force_single_contributor, self.reconstruction_passes > 1, self.exact_nearest)
		if key not in self._reconstruction_shaders:
			self._reconstruction_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_reconstruct.shader"),
				params={"sample_count": self.sample_count, "force_single_contributor": self.force_single_contributor,
					"tiled_reconstruction": True, "compute_reconstruction": True, "filter_input": self.reconstruction_passes > 1,
					"frame_fields": self._frame_types, "exact_nearest": self.exact_nearest})
		self.reconstruct_shader = self._reconstruction_shaders[key]
		self.reconstruct_pipeline = ComputePipeline(self.reconstruct_shader, entry="reconstruct",
			layout=self._pipeline_layout(self.reconstruct_shader, "reconstruct"))
		self.reconstruct_frame = self._frame_group(self.reconstruct_shader)

	def set_reconstruction_passes(self, count):
		"""Stack positive-count bilateral passes without repeatedly applying opacity."""
		self._validate_count(count, "reconstruction_passes")
		if self.banded_tail and count != 1: raise ValueError("Use set_kawase_passes for banded tail")
		if count == self.reconstruction_passes: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Repeated reconstruction requires the default shader set")
		self.reconstruction_passes = count
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

	def set_reconstruction(self, mode):
		"""Choose a-trous, push-pull or fixed fragment/compute bilateral filtering."""
		if self.banded_tail:
			if mode == "kawase": return False
			raise ValueError("Banded tail uses dual Kawase reconstruction")
		if mode not in ("fragment", "compute", "push_pull", "atrous"): raise ValueError("reconstruction must be fragment, compute, push_pull or atrous")
		if mode == self.reconstruction: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Runtime reconstruction switching requires the default shader set")
		self.reconstruction = mode
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

	def _ensure_second_noise(self):
		if self.banded_tail: return
		if self.winner_count == 1 or self.complemented_winners or self.noise_second is not None: return
		if self._default_noise:
			from .TextureProcessing import get_default_blue_noise
			self.noise_second = get_default_blue_noise(stream=1)
		else:
			# Explicit single-channel overrides keep their source; stream hashing
			# selects a different phase. Supply blue_noise_second for a separate tile.
			self.noise_second = self.noise

	@property
	def complemented_winners(self):
		return not self.banded_tail and self.winner_count == 2 and self.winner_mix == "average" and self.winner_coupling == "complement"

	@property
	def banded_tail(self):
		return self.tail_allocation == "banded"

	@property
	def _banded_flags(self):
		return self.banded_proxy, self.banded_subset, self.banded_tail_msaa

	def _banded_params(self, samples):
		mixed = self.banded_tail and samples > 1 and not self.banded_tail_msaa
		return dict(banded_proxy=self.banded_tail and self.banded_proxy, banded_subset=self.banded_tail and self.banded_subset,
			banded_mixed=mixed, banded_statistics=not mixed, guide_sample_count=samples)

	@property
	def tail_sample_count(self):
		return 1 if self.banded_tail and not self.banded_tail_msaa else self.sample_count

	@property
	def mixed_tail(self):
		return self.banded_tail and self.tail_sample_count != self.sample_count

	def set_banded_options(self, *, proxy=None, subset=None, tail_msaa=None):
		"""Configure proxy residuals, one selected pixel per 2x2 block, and tail MSAA."""
		values = tuple(current if value is None else value for current, value in zip(self._banded_flags, (proxy, subset, tail_msaa)))
		if any(type(value) is not bool for value in values): raise ValueError("Banded options must be boolean")
		if values == self._banded_flags: return False
		if any(s is not None for s in self._custom_shaders): raise ValueError("Banded options require the default shader set")
		self.banded_proxy, self.banded_subset, self.banded_tail_msaa = values
		if self.banded_tail:
			self._activate_shaders(self.sample_count)
			self._rebuild_geometry()
		return True

	def set_tail_allocation(self, mode):
		"""Select stochastic winners or exact-nearest/banded-tail/Kawase rendering."""
		if mode not in ("stochastic", "banded"): raise ValueError("tail_allocation must be stochastic or banded")
		if mode == self.tail_allocation: return False
		if any(s is not None for s in self._custom_shaders): raise ValueError("Banded tail requires the default shader set; material hooks remain supported")
		self.tail_allocation = mode
		if self.banded_tail:
			self.exact_nearest, self.reconstruction, self.reconstruction_passes = True, "kawase", 1
			self.winner_count, self.winner_mix, self.winner_coupling = 2, "average", "complement"
			self.force_single_contributor = self.early_b_rejection = False
		else:
			self.reconstruction = "fragment"
		if self.debug_view not in ("final", "transmittance", "winner", "reconstruction", "nearest", "nearest_mask", "tail_transmittance"): self.set_debug_view("final")
		self._activate_shaders(self.sample_count)
		self._ensure_second_noise()
		self._rebuild_geometry()
		self.set_noise_mode(self.noise_mode)
		return True

	def set_kawase_passes(self, count):
		"""Set 0-8 down/up pairs; zero divides the unblurred banded pair."""
		if type(count) is not int or not 0 <= count <= 8: raise ValueError("kawase_passes must be an integer from 0 through 8")
		if count == self.kawase_passes: return False
		self.kawase_passes = count
		if self.banded_tail and self.size is not None:
			self._setup_kawase()
			for layout in self._layout_cache.values(): layout.clear_cache()
		return True

	def set_winner_coupling(self, mode):
		if mode not in ("complement", "independent"): raise ValueError("winner_coupling must be complement or independent")
		if self.banded_tail and mode != "complement": raise ValueError("Banded tail requires complemented winners")
		if mode == self.winner_coupling: return False
		if any(shader is not None for shader in self._custom_shaders): raise ValueError("Winner coupling requires the default shaders")
		self.winner_coupling = mode
		self._ensure_second_noise()
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		self.set_noise_mode(self.noise_mode)
		return True

	def set_confidence_filter(self, options):
		if not isinstance(options, ConfidenceFilter): raise TypeError("Expected ConfidenceFilter")
		if options == self.confidence_filter: return False
		previous = self.confidence_filter
		self.confidence_filter = options
		if self.reconstruction in ("push_pull", "atrous") and self.size is not None:
			if options.levels != previous.levels or options.cascade_passes != previous.cascade_passes:
				self._setup_pyramid(self.compose_resources.resources)
			else:
				for _, inputs, _, _ in self.pyramid_stages:
					settings = inputs.resources["pyramid_settings"]
					settings.content["confidence"] = (options.target_winners, 0, 0, 0)
					settings.upload_changed()
		return True

	def set_winner_count(self, count):
		"""Any positive fill draw count, or one/two averaged draws."""
		self._validate_count(count, "winner_count")
		if self.banded_tail and count != 2: raise ValueError("Banded tail produces two winners in one allocation pass")
		if count > 2 and self.winner_mix != "fill": raise ValueError("winner_count above two requires fill mixing")
		if count == self.winner_count: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Runtime winner switching requires the default shader set")
		self.winner_count = count
		if count == 1:
			self.winner_mix = "average"
			self.early_b_rejection = False
		self._ensure_second_noise()
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		self.set_noise_mode(self.noise_mode)
		self.set_bilateral_filter(self.bilateral_filter)
		return True

	def set_winner_mix(self, mode):
		"""Choose averaging or experimental fill, limited to audited sources/modes."""
		if mode not in ("average", "fill"): raise ValueError("winner_mix must be average or fill")
		if self.banded_tail and mode != "average": raise ValueError("Banded tail uses average mixing")
		if mode == self.winner_mix: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Winner mixing requires the default shader set")
		if mode == "average" and self.winner_count > 2: raise ValueError("Average mixing supports at most two winners; reduce winner_count first")
		if mode == "fill" and self.noise_mode not in FILL_NOISE_MODES:
			raise ValueError(f"Fill requires a validated noise mode: {FILL_NOISE_MODES}")
		if mode == "fill" and not self._default_noise_pair:
			raise ValueError("Fill requires the audited default noise sources")
		self.winner_mix = mode
		self._ensure_second_noise()
		if mode == "average": self.early_b_rejection = False
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

	def set_early_b_rejection(self, enabled):
		"""Experimental per-sample stencil gating of fill B; no depth precision loss."""
		if type(enabled) is not bool: raise ValueError("early_b_rejection must be boolean")
		if self.banded_tail and enabled: raise ValueError("Banded tail does not use stencil rejection")
		if enabled == self.early_b_rejection: return False
		if enabled:
			if self.winner_count < 2 or self.winner_mix != "fill": raise ValueError("Early B rejection requires multiple-winner fill")
			RenderContext.require_features(("depth32float-stencil8",))
		self.early_b_rejection = enabled
		self._rebuild_geometry()
		return True

	def set_force_single_contributor(self, enabled):
		"""Force only an exactly counted single nonzero visible contributor."""
		if type(enabled) is not bool: raise ValueError("force_single_contributor must be boolean")
		if self.banded_tail and enabled: raise ValueError("Banded tail does not use force acceptance")
		if enabled == self.force_single_contributor: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Force acceptance requires the default shader set")
		self.force_single_contributor = enabled
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

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
		self.set_noise_scale(self.noise_scale)
		self._rebuild_geometry()
		return True

	def _rebuild_geometry(self, *, preserve_targets=False):
		size, geometries = self.size, self._geometries
		self._create_nearest_pipeline()
		targets = self._targets if preserve_targets and size is not None else None
		self.size = None
		self._geometries, self._transparent_geometries = [], []
		for geometry in geometries:
			rebuilt = self.create_geometry_passes(**geometry.creation_args)
			geometry.render_shader.passes.clear()
			geometry.render_shader.passes.update(rebuilt.render_shader.passes)
			geometry.guide, geometry.shading = rebuilt.guide, rebuilt.shading
			geometry.nearest, geometry.nearest_shading = rebuilt.nearest, rebuilt.nearest_shading
			self._geometries[-1] = geometry
			if geometry.creation_args["transparent"]:
				self._transparent_geometries[-1] = geometry
		entry = "shade_bands" if self.banded_tail else "shade_winner"
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry=entry, layout=self._pipeline_layout(self.shading_shader, entry))
		self.mix_pipeline = self._create_mix_pipeline()
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none",
			layout=self._pipeline_layout(self.compose_shader, "fullscreen", "compose"))
		self._create_reconstruction_pipeline()
		self.shade_frame = self._frame_group(self.shading_shader)
		if size is not None:
			self.resize(size, targets=targets)

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
		bindings = (self._frame_group(self.shader),
			self.shader.bind_group(2, **self.shared, metadata=metadata, **geometry_resources), *geometry_bindings)
		if transparent:
			multiply = {"color": {"src_factor": "zero", "dst_factor": "src", "operation": "add"},
				"alpha": {"src_factor": "zero", "dst_factor": "src-alpha", "operation": "add"}}
			add = {"color": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
				"alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"}}
			first = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="transmittance",
				cull_mode="none", depth_write=False, blend=multiply, label="transmittance", sample_count=self.sample_count,
				targets=({"blend": multiply}, {"blend": add}) if self.force_single_contributor or (self.banded_tail and self.banded_proxy and not self.mixed_tail) else None,
				layout=self._pipeline_layout(self.shader, "vertex", "transmittance"))
			last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="select_winner",
				cull_mode="none", label="select_winner", sample_count=self.sample_count,
				depth_stencil=self._selection_stencil(False),
				layout=self._pipeline_layout(self.shader, "vertex", "select_winner"))
			guide = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex",
				fragment_entry="transparent_depth", cull_mode="none", label="transparent_depth", sample_count=self.sample_count,
				layout=self._pipeline_layout(self.shader, "vertex", "transparent_depth")), bindings)
		else:
			first = last = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="opaque",
				cull_mode="none", label="opaque", sample_count=self.sample_count,
				layout=self._pipeline_layout(self.shader, "vertex", "opaque"))
			guide = None
		# Transparent transmittance is a color pass, never an opaque depth prepass.
		first_pass = "transmittance" if transparent else "prepass"
		geometry = GeometryPasses(RenderShader({"main": ShaderPass(last, bindings), first_pass: ShaderPass(first, bindings)}),
			guide, self.shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices,
				shading_instances=instances, **shading_resources), self, tuple(shading_bindings))
		if transparent and self.exact_nearest:
			geometry.nearest = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="nearest_layer", cull_mode="none", sample_count=self.sample_count, label="nearest_layer", layout=self._pipeline_layout(self.shader, "vertex", "nearest_layer")), bindings)
			geometry.nearest_shading = self.nearest_shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices, shading_instances=instances, **shading_resources)
		if transparent and self.winner_count > 1:
			second = last if not self.early_b_rejection else RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="select_winner",
				cull_mode="none", label="select_fill_b_stencil", sample_count=self.sample_count,
				depth_stencil=self._selection_stencil(True), layout=self._pipeline_layout(self.shader, "vertex", "select_winner"))
			for stream in range(1, self.winner_count):
				geometry.render_shader.passes[self.selection_name(stream)] = ShaderPass(second, bindings)
		if transparent and self.banded_tail:
			tail_bindings = (self._frame_group(self.tail_shader), self.tail_shader.bind_group(2, **self.shared, metadata=metadata, **geometry_resources), *geometry_bindings)
			geometry.render_shader.passes["bands"] = ShaderPass(RenderPipeline(self.tail_shader, vertex_entry="vertex", fragment_entry="select_bands", cull_mode="none", depth_write=False, depth_test=None if self.mixed_tail else "always", sample_count=self.tail_sample_count, targets=({"write_mask": 0},) if self.mixed_tail else None, label="banded_tail", layout=self._pipeline_layout(self.tail_shader, "vertex", "select_bands")), tail_bindings)
			if self.mixed_tail:
				geometry.render_shader.passes["tail_statistics"] = ShaderPass(RenderPipeline(self.tail_shader, vertex_entry="vertex", fragment_entry="tail_statistics", cull_mode="none", depth_write=False, depth_test=None, sample_count=1, blend=add, label="tail_statistics", layout=self._pipeline_layout(self.tail_shader, "vertex", "tail_statistics")), tail_bindings)
		geometry.creation_args = dict(metadata=metadata, transparent=transparent, instances=instances,
			geometry_resources=geometry_resources, shading_resources=shading_resources,
			geometry_bindings=tuple(geometry_bindings), shading_bindings=tuple(shading_bindings))
		self._geometries.append(geometry)
		if transparent:
			self._transparent_geometries.append(geometry)
			if self.size is not None:
				self._bind_targets(geometry)
		return geometry

	@staticmethod
	def selection_name(stream):
		return "main" if stream == 0 else "second" if stream == 1 else f"winner_{stream}"

	def _bind_targets(self, geometry):
		bindings = tuple(b for b in geometry.render_shader['transmittance'].bindings if b.group != 3)
		nearest = {}
		if self.exact_nearest:
			nearest = dict(nearest_ids=self.nearest_payload)
			if self.sample_count > 1: nearest["nearest_ids_extra"] = self.nearest_extra
			geometry.render_shader['transmittance'].bindings = bindings + (self.shader.bind_group(3, **nearest),)
			geometry.nearest.bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth),)
		if self.banded_tail:
			geometry.render_shader['transmittance'].bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth, band_counts=self.band_counts, **nearest),)
			tail_bindings = tuple(b for b in geometry.render_shader['bands'].bindings if b.group != 3)
			geometry.render_shader['bands'].bindings = tail_bindings + (self.tail_shader.bind_group(3, opaque_depth=self.depth, blue_noise=self.noise, band_counts=self.band_counts, band_payloads=self.band_payloads.view(dimension="2d-array"), **nearest),)
			if self.mixed_tail: geometry.render_shader['tail_statistics'].bindings = tail_bindings + (self.tail_shader.bind_group(3, opaque_depth=self.depth, band_counts=self.band_counts, **nearest),)
			return
		for stream in range(self.winner_count):
			name = self.selection_name(stream)
			noise = self.noise if stream == 0 or self.complemented_winners else self.noise_second
			hierarchy = self._noise_hierarchy if stream == 0 or self.complemented_winners else self._noise_hierarchy_second
			resources = dict(opaque_depth=self.depth, blue_noise=noise,
				noise_hierarchy=hierarchy if hierarchy is not None else noise, **nearest)
			if self._winner_settings is not None:
				resources.update(winner_settings=self._winner_settings[stream], winner_guide=self.transparent_guide)
			if self.force_single_contributor:
				resources.update(selection_transmittance=self.transmittance, contributor_count=self.contributor_count)
			if self.winner_mix == "fill":
				resources["previous_winner"] = self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color
			geometry.render_shader[name].bindings = bindings + (self.shader.bind_group(3, **resources),)
		geometry.guide.bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth, **nearest),)

	def _selection_stencil(self, second):
		if not self.early_b_rejection: return None
		face = dict(compare="not-equal" if second else "always", fail_op="keep", depth_fail_op="keep",
			pass_op="keep" if second else "replace")
		return dict(depth_write_enabled=True, depth_compare="less", stencil_front=face, stencil_back=face,
			stencil_read_mask=255, stencil_write_mask=0 if second else 255)

	def set_bilateral_filter(self, options):
		"""Change reconstruction settings without recreating pipelines or targets."""
		self.bilateral_uniforms.content["sigmas"] = (
			options.spatial_sigma, options.depth_sigma, options.transmittance_sigma, options.min_depth_sigma,
		)
		self.bilateral_uniforms.content["kernel"] = (options.kernel_size//2, DEBUG_VIEWS.index(self.debug_view), 0, 0)
		self.bilateral_uniforms.upload_changed()
		self.bilateral_filter = options
		if self._winner_settings is not None:
			for stream, settings in enumerate(self._winner_settings):
				settings.content["info"] = (stream, int(stream > 0 and options.kernel_size != 1), self.winner_count, 0)
				settings.upload_changed()

	@property
	def effective_noise_scale(self):
		"""Approximate cell width after MSAA sample-density adjustment."""
		return self.noise_scale / self.sample_count**0.5

	def set_noise_scale(self, pixels):
		"""Set approximate 1x surface noise-cell width in pixels (default 1).

		Larger cells trade coarser grain for fewer threshold changes under motion.
		4x MSAA halves the effective width; the chosen base value is preserved.
		"""
		limits = np.finfo(np.float32)
		if not np.isfinite(pixels) or not float(limits.tiny) <= pixels <= float(limits.max):
			raise ValueError("Noise scale must be a finite, positive normalized float32")
		self.noise_scale = float(pixels)
		self.uniforms.content["clip"][..., 2] = self.effective_noise_scale
		self.uniforms.upload_changed()

	def set_noise_footprint_balance(self, value):
		"""UV footprint blend: 0=max derivative, .5=geometric mean, 1=min."""
		if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
			or not np.isfinite(value) or not 0 <= value <= 1):
			raise ValueError("Noise footprint balance must be finite and between 0 and 1")
		self.noise_footprint_balance = float(value)
		self.uniforms.content["noise_footprint_balance"] = self.noise_footprint_balance
		self.uniforms.upload_changed()

	def set_noise_footprint_balancing(self, enabled):
		"""Compile/cache the UV footprint feature; enabled by default."""
		if type(enabled) is not bool: raise ValueError("Noise footprint balancing must be boolean")
		if enabled == self.noise_footprint_balancing: return False
		if any(shader is not None for shader in self._custom_shaders):
			raise ValueError("Runtime shader-feature switching requires the default shader set")
		self.noise_footprint_balancing = enabled
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry(preserve_targets=True)
		return True

	def set_noise_anchor(self, pixels=24.0, subdivisions=4):
		"""Coarse anchor target and independent shift cells per coarse-cell axis."""
		if (not isinstance(pixels, (int, float, np.integer, np.floating)) or isinstance(pixels, (bool, np.bool_))
			or not np.isfinite(pixels) or not 1 <= pixels <= 256):
			raise ValueError("Noise anchor size must be finite and between 1 and 256 pixels")
		if type(subdivisions) is not int or not 1 <= subdivisions <= 32:
			raise ValueError("Noise anchor subdivisions must be an integer between 1 and 32")
		self.noise_anchor_pixels, self.noise_anchor_subdivisions = float(pixels), subdivisions
		self.uniforms.content["noise_anchor"] = (pixels, subdivisions, 0, 0)
		self.uniforms.upload_changed()

	def set_noise_mode(self, mode):
		"""Select object/UV/triplanar sampling, including correlated hierarchies.

		UV modes fall back to object hashing for missing/degenerate UVs. Only
		winner selection changes; opacity and reconstruction settings do not.
		"""
		if mode not in NOISE_MODES:
			raise ValueError(f"Noise mode must be one of {NOISE_MODES}")
		if self.winner_mix == "fill" and mode not in FILL_NOISE_MODES:
			raise ValueError(f"Fill is only validated for {FILL_NOISE_MODES}; select average mixing first")
		if not self.banded_tail and mode in ("uv_blue_correlated", "triplanar_correlated") and self._noise_hierarchy is None:
			from .TextureProcessing import build_correlated_noise_hierarchy
			self._noise_hierarchy = build_correlated_noise_hierarchy(self.noise)
			if self.size is not None:
				for geometry in self._transparent_geometries:
					self._bind_targets(geometry)
		if not self.banded_tail and mode in ("uv_blue_correlated", "triplanar_correlated") and self.winner_count > 1 and not self.complemented_winners and self._noise_hierarchy_second is None:
			from .TextureProcessing import build_correlated_noise_hierarchy
			self._noise_hierarchy_second = build_correlated_noise_hierarchy(self.noise_second)
			if self.size is not None:
				for geometry in self._transparent_geometries: self._bind_targets(geometry)
		self.noise_mode = mode
		self.uniforms.content["clip"][..., 3] = NOISE_MODES.index(mode)
		self.uniforms.upload_changed()

	def set_grazing_fade(self, options):
		"""Change transparent-surface opacity fade without rebuilding GPU resources."""
		self.uniforms.content["grazing"] = (options.start_angle, options.end_angle, options.enabled, 0)
		self.uniforms.upload_changed()
		self.grazing_fade = options

	def set_debug_view(self, view):
		"""Inspect scalar transmittance, raw winners, or reconstructed contribution."""
		if self.banded_tail and view in ("winner_a", "winner_b", "winner_count", "forced", "filter_support"):
			raise ValueError("This diagnostic is not recorded by banded tail")
		if view not in DEBUG_VIEWS:
			raise ValueError(f"Debug view must be one of {DEBUG_VIEWS}")
		if view in ("nearest", "nearest_mask", "tail_transmittance") and not self.exact_nearest:
			raise ValueError("Nearest-layer diagnostics require exact_nearest")
		self.debug_view = view
		self.set_bilateral_filter(self.bilateral_filter)

	@property
	def has_external_opaque_inputs(self):
		return self._opaque_inputs is not None

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
		inputs = getattr(self, "_opaque_inputs", None)
		if inputs and inputs[0].size[:2] != size: raise ValueError("Opaque input dimensions do not match transparency targets")
		formats = {"main_color": "rgba16float", "depth": "depth32float", "transparent_depth": "depth32float",
			"transparent_guide": "rgba16float",
			"transmittance": "rgba16float", "selection_depth": "depth32float", "winner": "rgba32uint", "winner_color": "rgba16float"}
		if self.banded_tail:
			del formats["winner"], formats["selection_depth"]
		if self.sample_count == 4 and not self.banded_tail:
			formats["winner"] = "rgba16uint"
			formats["winner_extra"] = "rgba16uint"
		if not self.banded_tail and self.winner_count == 2 and self.winner_mix == "average": formats["winner_first"] = "rgba16float"
		if self.winner_mix == "fill": formats["winner_source"] = "r32uint"
		if self.force_single_contributor: formats["contributor_count"] = "r16float"
		if self.reconstruction != "fragment": formats["reconstruction_color"] = "rgba16float"
		if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 1: formats["reconstruction_pair"] = "rgba16float"
		if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 2: formats["reconstruction_pair_scratch"] = "rgba16float"
		if self.early_b_rejection: formats["selection_depth"] = "depth32float-stencil8"
		if self.complemented_winners: formats["winner_first_payload"] = "rgba32uint"
		if self.exact_nearest:
			formats.update(nearest_depth="depth32float", nearest_alpha="rgba16float", nearest_payload="rgba16uint" if self.sample_count > 1 else "rgba32uint", nearest_color="rgba16float")
			if self.sample_count > 1: formats["nearest_extra"] = "rgba16uint"
		if targets is not None:
			for name, format in formats.items():
				texture = getattr(targets, name)
				shaded = name in ("winner_color", "nearest_color", "winner_first", "winner_source", "winner_first_payload", "reconstruction_color", "reconstruction_pair", "reconstruction_pair_scratch")
				usage = TextureUsage.TEXTURE_BINDING | (TextureUsage.STORAGE_BINDING if shaded else TextureUsage.RENDER_ATTACHMENT)
				if name.startswith("reconstruction_pair"): usage |= TextureUsage.RENDER_ATTACHMENT
				count = 1 if shaded else self.sample_count
				layers = (self.tail_sample_count if self.banded_tail and name in ("winner_color", "reconstruction_color") else self.sample_count) if shaded else 1
				if texture is None or texture.size != (*size, layers) or texture.format != format or texture.sample_count != count or texture.usage & usage != usage:
					raise ValueError(f"Invalid transparency target {name}: expected {size}, {format}, {count} samples, {layers} layers, usage {usage}")
		else:
			values = dict(winner=None, selection_depth=None) if self.banded_tail else {}
			for name, format in formats.items():
				if inputs and name in ("main_color", "depth"):
					values[name] = inputs[0 if name == "main_color" else 1]
				else:
					shaded = name in ("winner_color", "nearest_color", "winner_first", "winner_source", "winner_first_payload", "reconstruction_color", "reconstruction_pair", "reconstruction_pair_scratch")
					usage = TextureUsage.TEXTURE_BINDING | (TextureUsage.STORAGE_BINDING if shaded else TextureUsage.RENDER_ATTACHMENT)
					if name.startswith("reconstruction_pair"): usage |= TextureUsage.RENDER_ATTACHMENT
					count = 1 if shaded else self.sample_count
					layers = self.tail_sample_count if self.banded_tail and name in ("winner_color", "reconstruction_color") else self.sample_count
					texture_size = (*size, layers) if shaded else size
					values[name] = Texture(texture_size, format=format, usage=usage, label=name, sample_count=count)
			targets = TransparencyTargets(**values)
		for name in formats: setattr(self, name, getattr(targets, name))
		self.winner, self.selection_depth = targets.winner, targets.selection_depth
		self.winner_extra = targets.winner_extra
		self.winner_first = targets.winner_first if self.winner_count == 2 and self.winner_mix == "average" else None
		self.contributor_count = targets.contributor_count if self.force_single_contributor else None
		self.winner_source = targets.winner_source if self.winner_mix == "fill" else None
		self.reconstruction_color = targets.reconstruction_color if self.reconstruction != "fragment" else None
		self.reconstruction_pair = targets.reconstruction_pair if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 1 else None
		self.reconstruction_pair_scratch = targets.reconstruction_pair_scratch if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 2 else None
		self.winner_first_payload = targets.winner_first_payload if self.complemented_winners else None
		self._targets = targets
		for name in ("nearest_payload", "nearest_extra", "nearest_alpha", "nearest_depth", "nearest_color"):
			setattr(self, name, getattr(targets, name) if self.exact_nearest else None)
		self.nearest_shade_targets = None
		if self.exact_nearest:
			resources = dict(winners=self.nearest_payload, winner_color=self.nearest_color.view(dimension="2d-array") if self.sample_count > 1 else self.nearest_color)
			if self.sample_count > 1: resources["winners_extra"] = self.nearest_extra
			self.nearest_shade_targets = self.nearest_shading_shader.bind_group(1, **resources)
		if self.banded_tail:
			self._setup_banded(size)
			self.size = size
			for layout in self._layout_cache.values(): layout.clear_cache()
			return True
		self.band_counts = self.band_payloads = None
		self.band_proxy = None
		self.kawase_stages = []
		self.kawase_textures, self.band_clear_targets, self.kawase_dummy = [], None, None
		for geometry in self._transparent_geometries:
			self._bind_targets(geometry)
		def shaded_view(texture):
			return texture.view(dimension="2d-array") if self.sample_count > 1 else texture
		shade_resources = dict(winners=self.winner,
			winner_color=shaded_view(self.winner_first if self.winner_first is not None else self.winner_color))
		if self.sample_count > 1: shade_resources["winners_extra"] = self.winner_extra
		if self.complemented_winners: shade_resources["first_payloads"] = self.winner_first_payload.view(dimension="2d-array")
		if self.winner_count == 2 and self.winner_mix == "average":
			shade_resources.update(first_color=shaded_view(self.winner_color), winner_guide=self.transparent_guide,
				winner_settings=self._winner_settings[1])
		if self.force_single_contributor or self.winner_mix == "fill": shade_resources["shade_transmittance"] = self.transmittance
		if self.force_single_contributor: shade_resources["contributor_count"] = self.contributor_count
		if self.winner_mix == "fill": shade_resources["winner_source"] = shaded_view(self.winner_source)
		if self.winner_mix == "fill": shade_resources["winner_settings"] = self._winner_settings[0]
		self.shade_targets = self.shading_shader.bind_group(1, **shade_resources)
		self.mix_targets = None
		if self.winner_count == 2 and self.winner_mix == "average":
			self.mix_targets = self.mixing_shader.bind_group(1, **dict(shade_resources,
				winner_color=shaded_view(self.winner_color), first_color=shaded_view(self.winner_first)))
		self.fill_targets = []
		if self.winner_mix == "fill":
			self.fill_targets = [self.shading_shader.bind_group(1, **dict(shade_resources, winner_settings=settings)) for settings in self._winner_settings]
			self.shade_targets = self.fill_targets[0]
			self.mix_targets = self.fill_targets[1] if self.winner_count > 1 else None
		self.compose_frame = self._frame_group(self.compose_shader)
		compose_resources = dict(main_color=self.main_color,
			transmittance=self.transmittance,
			winner_color=self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color, opaque_depth=self.depth,
			transparent_depth=self.transparent_depth, transparent_guide=self.transparent_guide,
			bilateral=self.bilateral_uniforms)
		if self.exact_nearest:
			compose_resources.update(nearest_alpha=self.nearest_alpha, nearest_color=shaded_view(self.nearest_color))
		if self.winner_first is not None: compose_resources["first_color"] = shaded_view(self.winner_first)
		if self.winner_mix == "fill": compose_resources["winner_source"] = shaded_view(self.winner_source)
		if self.force_single_contributor: compose_resources["contributor_count"] = self.contributor_count
		self.pair_inputs, self.pair_outputs, self.pair_attachments = [], [], []
		if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 1:
			pairs = [self.reconstruction_pair] + ([self.reconstruction_pair_scratch] if self.reconstruction_passes > 2 else [])
			for texture in [self.winner_color, *pairs]:
				resources = dict(compose_resources, filter_input=shaded_view(texture))
				shader = self.initial_pair_shader if texture is self.winner_color and self.initial_pair_shader is not None else self.pair_shader
				self.pair_inputs.append(shader.bind_group(1, **{name: value for name, value in resources.items() if name in shader.info.bindings}))
			for texture in pairs:
				if self.reconstruction == "compute": self.pair_outputs.append(self.pair_shader.bind_group(2, filter_output=shaded_view(texture)))
				self.pair_attachments.append([ColorAttachment(texture.view(dimension="2d", base_array_layer=sample, array_layer_count=1).handle,
					texture.format, device_generation=RenderContext.device_generation)
					for sample in range(self.sample_count)])
			compose_resources["filter_input"] = shaded_view(pairs[(self.reconstruction_passes-2)%len(pairs)])
		self.reconstruct_inputs = self.reconstruct_output = None
		if self.reconstruction == "compute":
			self.reconstruct_inputs = self.reconstruct_shader.bind_group(1, **{name: resource for name, resource in compose_resources.items()
				if name in self.reconstruct_shader.info.bindings})
			self.reconstruct_output = self.reconstruct_shader.bind_group(2, filter_output=shaded_view(self.reconstruction_color))
			compose_resources["reconstructed"] = shaded_view(self.reconstruction_color)
		if self.reconstruction in ("push_pull", "atrous"):
			compose_resources["reconstructed"] = shaded_view(self.reconstruction_color)
			self._setup_pyramid(compose_resources)
		else: self.pyramid, self.pyramid_stages = [], []
		compose_group = self.compose_shader.info.bindings["main_color"].group
		if compose_group == 0:
			compose_resources.update(self.compose_frame.resources)
			self.compose_frame = None
		self.compose_resources = self.compose_shader.bind_group(compose_group, **compose_resources)
		self.size = size
		# Native bind groups retain their textures. Drop old target bindings on
		# resize/switch rather than keeping many full-resolution target sets alive.
		for layout in self._layout_cache.values(): layout.clear_cache()
		return True

	def _setup_banded(self, size):
		count = size[0]*size[1]*self.tail_sample_count*2
		if count*4 > RenderContext.device.limits["max-storage-buffer-binding-size"]:
			raise ValueError("Banded tail counters exceed max-storage-buffer-binding-size at this resolution/sample count")
		self.band_counts = GpuBuffer(None, BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC, dtype=np.uint32, count=count, label="band_counts")
		payload_size = tuple((value+1)//2 for value in size) if self.banded_subset else size
		self.band_payloads = Texture((*payload_size, self.tail_sample_count*2), format="rgba32uint", usage=TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC, label="band_payloads")
		self.band_proxy = Texture(size, format="rgba16float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING, sample_count=self.tail_sample_count, label="band_proxy") if self.banded_proxy or self.mixed_tail else None
		self.band_clear_shader = Shader(filepath=str(SHADERS / "banded_clear.shader"))
		self.band_clear_pipeline = ComputePipeline(self.band_clear_shader, entry="clear", layout=self._pipeline_layout(self.band_clear_shader, "clear"))
		self.band_clear_targets = self.band_clear_shader.bind_group(1, payloads=self.band_payloads.view(dimension="2d-array"))
		for geometry in self._transparent_geometries: self._bind_targets(geometry)
		color = self.winner_color.view(dimension="2d-array") if self.tail_sample_count > 1 else self.winner_color
		resources = dict(band_counts=self.band_counts, band_payloads=self.band_payloads.view(dimension="2d-array"), winner_color=color)
		if self.banded_proxy: resources["band_proxy"] = self.band_proxy
		self.shade_targets = self.shading_shader.bind_group(1, **resources)
		color = self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color
		self.compose_frame = self._frame_group(self.compose_shader)
		self.compose_resources = self.compose_shader.bind_group(1, main_color=self.main_color, transmittance=self.transmittance,
			winner_color=color, first_color=color, opaque_depth=self.depth, transparent_depth=self.transparent_depth,
			transparent_guide=self.transparent_guide, bilateral=self.bilateral_uniforms, nearest_alpha=self.nearest_alpha,
			nearest_color=self.nearest_color.view(dimension="2d-array") if self.sample_count > 1 else self.nearest_color,
			reconstructed=self.reconstruction_color.view(dimension="2d-array") if self.sample_count > 1 else self.reconstruction_color)
		self._setup_kawase()

	def _setup_kawase(self):
		self.kawase_shader = Shader(filepath=str(SHADERS / "stochastic_kawase.shader"), params={"sample_count": self.tail_sample_count, "banded_proxy": self.banded_proxy, "banded_mixed": self.mixed_tail})
		self.kawase_pipelines = {entry: ComputePipeline(self.kawase_shader, entry=entry, layout=self._pipeline_layout(self.kawase_shader, entry)) for entry in ("down", "up", "finish")}
		self.kawase_stages, self.kawase_textures = [], []
		self.kawase_dummy = Texture((1, 1, self.tail_sample_count), format="rgba16float", usage=TextureUsage.STORAGE_BINDING)
		def stage(entry, source, destination):
			resources = dict(source_pair=source.view(dimension="2d-array"))
			if not self.mixed_tail: resources["transmission"] = self.transmittance
			if self.banded_proxy: resources.update(band_proxy=self.band_proxy, band_counts=self.band_counts)
			inputs = self.kawase_shader.bind_group(0, **resources)
			pair = self.kawase_dummy if entry == "finish" else destination
			color = destination if entry == "finish" else self.kawase_dummy
			resources = dict(output_pair=pair.view(dimension="2d-array"), output_color=color.view(dimension="2d-array") if self.tail_sample_count > 1 else color.view(dimension="2d"))
			outputs = self.kawase_shader.bind_group(1, **resources)
			self.kawase_stages.append((entry, inputs, outputs, destination.size[:2]))
		def texture(size):
			result = Texture((*size, self.tail_sample_count), format="rgba16float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING, label="kawase_pair")
			self.kawase_textures.append(result)
			return result
		current = self.winner_color
		sizes = [current.size[:2]]
		for _ in range(self.kawase_passes):
			if sizes[-1] == (1, 1): break
			size = tuple((value+1)//2 for value in sizes[-1])
			destination = texture(size)
			stage("down", current, destination)
			current = destination
			sizes.append(size)
		for size in reversed(sizes[:-1]):
			destination = texture(size)
			stage("up", current, destination)
			current = destination
		stage("finish", current, self.reconstruction_color)

	def _setup_pyramid(self, resources):
		"""Create per-level storage and immutable dispatch bindings."""
		if self.reconstruction == "atrous":
			return self._setup_atrous(resources)
		width, height = self.transmittance.size[:2]
		shader = self.pyramid_shader
		self.pyramid_inputs = shader.bind_group(1, **{name: value for name, value in resources.items() if name in shader.info.bindings})
		usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
		self.pyramid_dummy = Texture((1, 1, self.sample_count), format="rgba32float", usage=usage)
		self.pyramid = []
		for level in range(1, self.confidence_filter.levels+1):
			size = ((width+(1<<level)-1)>>level, (height+(1<<level)-1)>>level, self.sample_count)
			self.pyramid.append({name: Texture(size, format="rgba16float" if name == "transmission" else "rgba32float", usage=usage) for name in ("pair", "depth", "transmission", "mu")})
		self.pyramid_stages = []
		def stage(entry, level, source, parent, output, dimensions):
			settings = shader.uniform_buffer("pyramid_settings")
			source_size = (width, height) if level == 0 else self.pyramid[level-1]["pair"].size[:2]
			settings.content["info"] = (level, int(parent is None), 0, 0)
			settings.content["sizes"] = (*source_size, *dimensions)
			settings.content["confidence"] = (self.confidence_filter.target_winners, 0, 0, 0)
			settings.upload()
			sampled = dict(level_pair=source["pair"], level_depth=source["depth"], level_transmission=source["transmission"], parent_mu=parent["mu"] if parent else self.pyramid_dummy, parent_depth=parent["depth"] if parent else self.pyramid_dummy, parent_transmission=parent["transmission"] if parent else self.pyramid_dummy)
			inputs = shader.bind_group(2, pyramid_settings=settings, **{name: texture.view(dimension="2d-array") for name, texture in sampled.items()})
			outputs = {"pull": ("pair", "depth", "transmission"), "push": ("mu",), "finish": ("result",)}[entry]
			output_group = shader.bind_group(3, **{f"output_{name}": output[name].view(dimension="2d-array") if name != "result" or self.sample_count > 1 else output[name] for name in outputs})
			self.pyramid_stages.append((entry, inputs, output_group, dimensions))
		dummy = dict.fromkeys(("pair", "depth", "transmission"), self.pyramid_dummy)
		for level, output in enumerate(self.pyramid):
			stage("pull", level, self.pyramid[level-1] if level else dummy, None, output, output["pair"].size[:2])
		for level in range(len(self.pyramid), 0, -1):
			output = self.pyramid[level-1]
			stage("push", level, output, self.pyramid[level] if level < len(self.pyramid) else None, output, output["pair"].size[:2])
		stage("finish", 0, dummy, self.pyramid[0], dict(result=self.reconstruction_color), (width, height))
		for layout in self._layout_cache.values(): layout.clear_cache()

	def _setup_atrous(self, resources):
		width, height = self.transmittance.size[:2]
		shader = self.pyramid_shader
		self.pyramid_inputs = shader.bind_group(1, **{name: value for name, value in resources.items() if name in shader.info.bindings})
		usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
		self.pyramid = [{name: Texture((width, height, self.sample_count), format="rgba32float", usage=usage) for name in ("pair", "result")} for _ in range(2)]
		self.pyramid_stages = []
		for level in range(self.confidence_filter.cascade_passes+1):
			final = level == self.confidence_filter.cascade_passes
			settings = shader.uniform_buffer("pyramid_settings")
			settings.content["info"] = (level, self.confidence_filter.cascade_passes, 0, 0)
			settings.content["confidence"] = (self.confidence_filter.target_winners, 0, 0, 0)
			settings.upload()
			source = self.pyramid[(level-1)%2]
			inputs = shader.bind_group(2, pyramid_settings=settings, cascade_pair=source["pair"].view(dimension="2d-array"), cascade_result=source["result"].view(dimension="2d-array"))
			entry = "finish" if final else "cascade"
			outputs = dict(final_color=self.reconstruction_color.view(dimension="2d-array") if self.sample_count > 1 else self.reconstruction_color) if final else {f"output_{name}": texture.view(dimension="2d-array") for name, texture in self.pyramid[level%2].items()}
			self.pyramid_stages.append((entry, inputs, shader.bind_group(3, **outputs), (width, height)))
		for layout in self._layout_cache.values(): layout.clear_cache()

	def _require_targets(self) -> tuple[int, int]:
		if self.size is None:
			raise RuntimeError("Resize transparency targets before recording passes or updating frame uniforms")
		return self.size

	@staticmethod
	def _uniform_value(buffer, name, value, kind):
		"""Check one uniform field before mutating its buffer."""
		field_dtype = buffer.dtype.fields[name][0]
		dtype, shape = field_dtype.subdtype or (field_dtype, ())
		try: supplied = np.asarray(value)
		except (TypeError, ValueError) as exc: raise ValueError(f"Invalid {kind} field {name!r}") from exc
		if supplied.shape != shape or supplied.dtype.kind not in "fiu" or (dtype.kind in "iu" and supplied.dtype.kind not in "iu"):
			raise ValueError(f"{kind.capitalize()} field {name!r} must have shape {shape} and numeric type {dtype}")
		try:
			with np.errstate(over="ignore", invalid="ignore"):
				converted = supplied.astype(dtype)
			if not np.all(np.isfinite(supplied)) or not np.all(np.isfinite(converted)):
				raise ValueError()
			if dtype.kind in "iu" and not np.all(supplied == converted):
				raise ValueError()
		except (TypeError, OverflowError, ValueError) as exc:
			raise ValueError(f"{kind.capitalize()} field {name!r} is out of range or non-finite") from exc
		return converted

	def _frame_value(self, name, value):
		return self._uniform_value(self.uniforms, name, value, "frame")

	def update_frame(self, *, view: ArrayLike, projection: ArrayLike,
		camera_position: ArrayLike, near: float, far: float,
		frame=None, lighting=None) -> NDArray[np.float32]:
		"""Upload explicit camera values and return view @ projection for culling.

		Matrices follow the existing row-vector convention. No camera class is
		required; resize targets first. Invalid inputs leave uniforms unchanged.
		"""
		self._require_targets()
		view, projection = np.asarray(view, dtype=np.float32), np.asarray(projection, dtype=np.float32)
		position = np.asarray(camera_position, dtype=np.float32)
		if (view.shape != (4, 4) or projection.shape != (4, 4)
			or position.shape != (3,)
			or not all(np.all(np.isfinite(value)) for value in (view, projection, position))):
			raise ValueError("Expected finite 4x4 view/projection matrices and vec3 camera position")
		if not (np.isfinite(near) and np.isfinite(far) and 0 < near < far):
			raise ValueError("Camera clip distances must satisfy 0 < near < far")
		if frame is None: frame = {}
		if not isinstance(frame, Mapping): raise ValueError("frame must be a mapping of field values")
		unknown = set(frame) - self.frame_fields.keys()
		if unknown: raise ValueError(f"Unknown frame fields: {sorted(map(str, unknown))}")
		custom = {name: self._frame_value(name, value) for name, value in frame.items()}
		if lighting is None: lighting = {}
		if not isinstance(lighting, Mapping): raise ValueError("lighting must be a mapping of field values")
		fields = set(self.lighting_uniforms.dtype.names) if self.lighting_uniforms is not None else set()
		unknown = set(lighting) - fields
		if unknown: raise ValueError(f"Unknown lighting fields: {sorted(map(str, unknown))}")
		light_values = {}
		for name, value in lighting.items():
			if name == "light_dir":
				try: value = np.asarray(value)
				except (TypeError, ValueError) as exc: raise ValueError("Invalid lighting field 'light_dir'") from exc
				if value.shape != (3,) or value.dtype.kind not in "fiu":
					raise ValueError("Lighting field 'light_dir' must be a numeric vec3")
				value = np.append(value, 0)
			light_values[name] = self._uniform_value(self.lighting_uniforms, name, value, "lighting")
		values = dict(view=view, proj=projection,
			camera_position=(*position, 1), clip=(near, far, self.effective_noise_scale, NOISE_MODES.index(self.noise_mode)))
		values.update(custom)
		for name, value in values.items(): self.uniforms.content[name] = value
		if light_values:
			for name, value in light_values.items(): self.lighting_uniforms.content[name] = value
			self.lighting_uniforms.upload_changed()
		self.uniforms.upload_changed()
		return view @ projection

	def update_camera(self, camera, *, frame=None, lighting=None):
		"""Convenience adapter for camera objects; update_frame accepts plain arrays."""
		width, height = self._require_targets()
		return self.update_frame(view=camera.view(), projection=camera.projection(width/height),
			camera_position=camera.position, near=camera.near, far=camera.far, frame=frame, lighting=lighting)

	def opaque_pass(self, commands, *, clear=(0.035, 0.05, 0.08, 1), load=False):
		"""Draw opaque geometry; load=True preserves existing color and depth."""
		self._require_targets()
		return commands.render_pass(color=self.main_color.color_attachment(clear=None if load else clear),
			depth=self.depth.depth_attachment(clear=None if load else 1), label="1_opaque")

	def guide_pass(self, commands):
		self._require_targets()
		if self.banded_tail: raise RuntimeError("Banded tail with Kawase does not use a guide pass")
		# Includes low-alpha fragments and pixels with no stochastic winner.
		return commands.render_pass(color=self.transparent_guide.color_attachment(clear=(0, 0, 0, 0)),
			depth=self.transparent_depth.depth_attachment(clear=1),
			label="2a_transparent_depth_guide")

	def nearest_pass(self, commands):
		self._require_targets()
		if not self.exact_nearest: raise RuntimeError("Enable exact_nearest before recording its pass")
		colors = [self.nearest_payload.color_attachment(clear=(0, 0, 0, 0)), self.nearest_alpha.color_attachment(clear=(0, 0, 0, 0))]
		if self.sample_count > 1: colors.append(self.nearest_extra.color_attachment(clear=(0, 0, 0, 0)))
		return commands.render_pass(color=colors, depth=self.nearest_depth.depth_attachment(clear=1), label="2_nearest_layer")

	def shade_nearest(self, commands, geometry):
		width, height = self._require_targets()
		if not self.exact_nearest: raise RuntimeError("Enable exact_nearest before shading its pass")
		if geometry.owner is not self: raise ValueError("Geometry passes belong to another transparency pass")
		with commands.compute_pass(label="2_shade_nearest") as compute:
			compute.set_pipeline(self.nearest_shade_pipeline)
			for bindings in (self._frame_group(self.nearest_shading_shader), self.nearest_shade_targets, geometry.nearest_shading, *geometry.shading_bindings): compute.set_bind_group(bindings)
			compute.dispatch((width+7)//8, (height+7)//8, self.sample_count)

	def transmittance_pass(self, commands):
		self._require_targets()
		if self.banded_tail:
			commands.clear_buffer(self.band_counts)
			width, height = self.band_payloads.size[:2]
			with commands.compute_pass(label="2_clear_bands") as compute:
				compute.set_pipeline(self.band_clear_pipeline)
				compute.set_bind_group(self.band_clear_targets)
				compute.dispatch((width+7)//8, (height+7)//8, self.tail_sample_count*2)
		colors = [self.transmittance.color_attachment(clear=(1, 1, 1, 1))]
		if self.banded_tail and self.banded_proxy and not self.mixed_tail: colors.append(self.band_proxy.color_attachment(clear=(0, 0, 0, 0)))
		if self.force_single_contributor: colors.append(self.contributor_count.color_attachment(clear=(0, 0, 0, 0)))
		return commands.render_pass(color=colors,
			depth=self.depth.depth_attachment(read_only=self.banded_tail), label="2_transmittance")

	def selection_pass(self, commands, *, stream=0):
		self._require_targets()
		if self.banded_tail: raise RuntimeError("Use banded_pass once instead of selection_pass")
		if type(stream) is not int or not 0 <= stream < self.winner_count: raise ValueError("Invalid selection stream")
		colors = [self.winner.color_attachment(clear=(0, 0, 0, 0))]
		if self.sample_count > 1: colors.append(self.winner_extra.color_attachment(clear=(0, 0, 0, 0)))
		depth = self.selection_depth.depth_attachment(clear=1,
			**({"stencil_clear": 0 if stream == 0 else None, "stencil_read_only": stream != 0} if self.early_b_rejection else {}))
		render_pass = commands.render_pass(color=colors, depth=depth, label=f"3_select_winner_{stream}")
		if not self.early_b_rejection: return render_pass
		@contextmanager
		def stencil_scope():
			with render_pass as render:
				render.handle.set_stencil_reference(1)
				yield render
		return stencil_scope()

	def banded_pass(self, commands):
		"""Allocate both banded payloads after recording transmittance once."""
		self._require_targets()
		if not self.banded_tail: raise RuntimeError("Enable banded tail before recording its pass")
		if self.mixed_tail: return commands.render_pass(color=self.band_proxy.color_attachment(), label="3_banded_tail")
		return commands.render_pass(depth=self.depth.depth_attachment(read_only=True), label="3_banded_tail")

	def tail_statistics_pass(self, commands):
		"""Accumulate full-resolution single-sample S/Q for a mixed-sample tail."""
		self._require_targets()
		if not self.mixed_tail: raise RuntimeError("Separate tail statistics require mixed tail sampling")
		return commands.render_pass(color=self.band_proxy.color_attachment(clear=(0, 0, 0, 0)), label="2_tail_statistics")

	def shade_winner(self, commands, geometry, *, stream=0):
		"""Record deferred shading for the metadata used during winner selection."""
		width, height = self._require_targets()
		if geometry.owner is not self: raise ValueError("Geometry passes belong to another transparency pass")
		if type(stream) is not int or not 0 <= stream < self.winner_count:
			raise ValueError("Winner stream must be less than winner_count")
		if self.banded_tail and stream != 0: raise ValueError("Banded tail shades both winners in one dispatch")
		step = 16 if self.banded_tail and self.banded_subset else 8
		with commands.compute_pass(label=f"4_shade_winner_{stream}") as compute:
			compute.set_pipeline(self.shade_pipeline if stream == 0 else self.mix_pipeline)
			targets = self.shade_targets if stream == 0 else self.mix_targets
			if self.winner_mix == "fill": targets = self.fill_targets[stream]
			for bindings in (self.shade_frame, targets, geometry.shading, *geometry.shading_bindings):
				compute.set_bind_group(bindings)
			compute.dispatch((width+step-1)//step, (height+step-1)//step, self.tail_sample_count if self.banded_tail else self.sample_count)

	def compose(self, commands, output):
		"""Reconstruct into a caller-provided ColorAttachment; does not submit."""
		self._require_targets()
		if output.format != self.target_format: raise ValueError("Output format must match the configured transparency target_format")
		if self.banded_tail and self.debug_view in ("final", "reconstruction"):
			for entry, inputs, outputs, size in self.kawase_stages:
				with commands.compute_pass(label=f"5_kawase_{entry}") as compute:
					compute.set_pipeline(self.kawase_pipelines[entry])
					compute.set_bind_group(inputs)
					compute.set_bind_group(outputs)
					compute.dispatch((size[0]+7)//8, (size[1]+7)//8, self.tail_sample_count)
		if self.reconstruction in ("push_pull", "atrous") and self.bilateral_filter.kernel_size != 1 and self.debug_view in ("final", "reconstruction", "filter_support"):
			for entry, inputs, output_group, dimensions in self.pyramid_stages:
				with commands.compute_pass(label=f"5a_pyramid_{entry}") as compute:
					compute.set_pipeline(self.pyramid_pipelines[entry])
					for bindings in (self.pyramid_frame, self.pyramid_inputs, inputs, output_group): compute.set_bind_group(bindings)
					compute.dispatch((dimensions[0]+7)//8, (dimensions[1]+7)//8, self.sample_count)
		if self.reconstruction not in ("push_pull", "atrous") and self.reconstruction_passes > 1 and self.bilateral_filter.kernel_size != 1 and self.debug_view in ("final", "reconstruction", "filter_support"):
			width, height = self.size
			for stage in range(self.reconstruction_passes-1):
				pipeline = self.initial_pair_pipeline if stage == 0 and self.initial_pair_pipeline is not None else self.pair_pipeline
				source = 0 if stage == 0 else 1+(stage-1)%len(self.pair_attachments)
				destination = stage%len(self.pair_attachments)
				if self.reconstruction == "compute":
					with commands.compute_pass(label=f"5a_reconstruct_pair_{stage}") as compute:
						compute.set_pipeline(pipeline)
						for bindings in (self.pair_frame, self.pair_inputs[source], self.pair_outputs[destination]): compute.set_bind_group(bindings)
						compute.dispatch((width+7)//8, (height+7)//8, self.sample_count)
				else:
					for sample, attachment in enumerate(self.pair_attachments[destination]):
						with commands.render_pass(color=attachment, label=f"5a_reconstruct_pair_{stage}_{sample}") as render:
							render.set_pipeline(pipeline)
							render.set_bind_groups(self.pair_frame, self.pair_inputs[source])
							render.draw(3, first_instance=sample)
		if self.reconstruction == "compute" and self.bilateral_filter.kernel_size != 1 and self.debug_view in ("final", "reconstruction", "filter_support"):
			width, height = self.size
			with commands.compute_pass(label="5a_reconstruct_compute") as compute:
				compute.set_pipeline(self.reconstruct_pipeline)
				for bindings in (self.reconstruct_frame, self.reconstruct_inputs, self.reconstruct_output): compute.set_bind_group(bindings)
				compute.dispatch((width+7)//8, (height+7)//8, self.sample_count)
		with commands.render_pass(color=output, label="5_reconstruct_compose") as render:
			render.set_pipeline(self.compose_pipeline)
			render.set_bind_groups(*((self.compose_frame,) if self.compose_frame is not None else ()), self.compose_resources)
			render.draw(3)
