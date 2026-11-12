"""DrawBatches integration for dictionary materials and inherited shading code."""
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from .DrawBatches import DrawBatches, FrustumCuller, RenderShader
from .InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale
from .Materials import MaterialBinding, MaterialLibrary, MaterialShader
from .EcsMaterials import MaterialInstance, MaterialTransform, EcsMaterialAdapter
from .RenderContext import BufferUsage, GpuBuffer, Mesh, RenderContext, Texture, TextureUsage
from .TransparencyData import VERTEX_DTYPE, SHADING_VERTEX_DTYPE, MATERIAL_DTYPE

_DRAW_IDS = count(1)
from .TransparencyPass import TransparencyPass


@dataclass(eq=False)
class MaterialMesh:
	vertices: np.ndarray
	shader: MaterialShader
	indices: np.ndarray | None = None
	material: dict | None = None
	bounds: tuple | None = None


@dataclass(eq=False)
class DrawInstance:
	"""Optional direct-draw input; ECS applications use MaterialInstance components."""
	mesh: MaterialMesh
	material: dict | None = None
	overrides: dict = field(default_factory=dict)
	position: tuple = (0, 0, 0)
	rotation: tuple = (0, 0, 0, 1)
	scale: tuple = (1, 1, 1)
	shader: MaterialShader | None = None
	surface_id: int = field(default_factory=lambda: next(_DRAW_IDS))


class MaterialRenderer:
	"""Registered mesh resources rendered from explicit material/transform columns.

	All shader families share winner, metadata and vertex addressing. Per-instance
	values do not split draw batches. EcsMaterialAdapter supplies versioned extraction;
	DrawInstance lists are an optional direct-draw convenience. record() uses frustum
	culling and caller-owned commands/output; presentation/submission remain external.
	"""
	def __init__(self, instances=None, *, meshes=None, shaders=(), blue_noise=None, base_textures=None, frame_fields=None, **transparency_options):
		self.instances = None if instances is None else list(instances)
		self.meshes = list(dict.fromkeys(meshes if meshes is not None else (i.mesh for i in self.instances or ())))
		if not self.meshes: raise ValueError("Register at least one mesh")
		self.mesh_ids = {mesh: index for index, mesh in enumerate(self.meshes)}
		self.library = MaterialLibrary([mesh.shader for mesh in self.meshes] + list(shaders)
			+ [i.shader for i in self.instances or () if i.shader is not None])
		self.draws = DrawBatches(MESH_INSTANCE_DTYPE)
		vertices, self.offsets = [], []
		for index, mesh in enumerate(self.meshes):
			source = mesh.vertices if mesh.indices is None else mesh.vertices[mesh.indices]
			if not len(source) or len(source) % 3: raise ValueError("Mesh must contain complete triangles")
			self.offsets.append(sum(len(v) for v in vertices)//3)
			raster = np.zeros(len(source), dtype=VERTEX_DTYPE)
			stored = np.zeros(len(source), dtype=SHADING_VERTEX_DTYPE)
			for name, width in (("position", 3), ("normal", 3), ("uv", 2)):
				raster[name] = source[name]
				stored[name][:, :width] = source[name]
			raster["triangle"] = np.repeat(np.arange(len(source)//3), 3)
			raster["barycentric"] = np.tile(((1, 0), (0, 1), (0, 0)), (len(source)//3, 1))
			vertices.append(stored)
			self.draws.register_mesh(index, Mesh(raster, np.arange(len(source), dtype=np.uint32)), bounds=mesh.bounds)
			self.draws.register_lod_group(index, (index,))
		usage = BufferUsage.STORAGE | BufferUsage.COPY_DST
		headers, shared, per_instance, _, _ = self.library.pack([])
		self.headers = GpuBuffer(headers, usage)
		self.parameters = dict(material_parameters=GpuBuffer(shared, usage), instance_parameters=GpuBuffer(per_instance, usage),
			material_headers=self.headers)
		self.metadata = GpuBuffer(np.zeros((1, 4), dtype=np.uint32), usage)
		if base_textures is None:
			white = Texture((1, 1, 1), format="rgba8unorm-srgb", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
			RenderContext.device.queue.write_texture({"texture": white.handle}, np.full(4, 255, dtype=np.uint8), {"bytes_per_row": 4}, (1, 1, 1))
			base_textures = white.view(dimension="2d-array")
		self.transparency = TransparencyPass(materials=GpuBuffer(np.zeros(1, dtype=MATERIAL_DTYPE), usage), base_textures=base_textures,
			vertices=GpuBuffer(np.concatenate(vertices), usage), blue_noise=blue_noise,
			material_library=self.library, frame_fields=frame_fields,
			**transparency_options)
		self.geometry = [self.transparency.create_geometry_passes(self.metadata, transparent=transparent,
			instances=self.draws.buffers["instances"], geometry_resources=self.parameters,
			shading_resources=self.parameters) for transparent in (False, True)]
		self._register_passes()
		self.visibility = self.draws.create_visibility()
		if self.instances is not None: self.sync()

	def _bindings(self, instances):
		return [MaterialBinding(i.shader or i.mesh.shader,
			i.material if i.material is not None else i.mesh.material, i.overrides) for i in instances]

	def _register_passes(self):
		opaque, transparent = self.geometry
		self.draws.register_shader(0, RenderShader({"opaque": opaque.render_shader["main"]}))
		passes = dict(transparent.render_shader.passes, guide=transparent.guide)
		if transparent.nearest is not None: passes["nearest"] = transparent.nearest
		self.draws.register_shader(1, RenderShader(passes))

	@staticmethod
	def _write(buffer, content):
		buffer.write_changed(content)

	def sync(self):
		"""Convenience for direct DrawInstance lists; ECS extraction calls update()."""
		if self.instances is None: return
		mesh_ids = np.array([self.mesh_ids[i.mesh] for i in self.instances], dtype=np.uint32)
		transforms = np.zeros(len(self.instances), dtype=MESH_INSTANCE_DTYPE)
		if self.instances:
			transforms["iPosition"] = [i.position for i in self.instances]
			transforms["iRotation"] = pack_quaternion([i.rotation for i in self.instances])
			transforms["iScale"] = pack_scale([i.scale for i in self.instances])
		ids = np.array([i.surface_id for i in self.instances], dtype=np.uint64)
		self.update(mesh_ids, self._bindings(self.instances), transforms, owners=ids, surface_ids=ids)

	def update(self, mesh_ids, bindings, transforms, *, owners, surface_ids, dirty_materials=None):
		"""Upload explicit columns. IDs are persistent renderable identities, not rows."""
		mesh_ids, owners, surface_ids = np.asarray(mesh_ids), np.asarray(owners), np.asarray(surface_ids)
		n = len(bindings)
		if (mesh_ids.shape != (n,) or owners.shape != (n,) or surface_ids.shape != (n,)
			or transforms.shape != (n,) or transforms.dtype != MESH_INSTANCE_DTYPE):
			raise ValueError("Supply one mesh, binding, transform, owner and surface ID per draw")
		if mesh_ids.dtype.kind not in "iu" or np.any(mesh_ids < 0) or np.any(mesh_ids >= len(self.meshes)):
			raise ValueError("Unknown mesh ID")
		if surface_ids.dtype.kind not in "iu" or np.any(surface_ids <= 0) or np.any(surface_ids > 0xffffffff) or len(np.unique(surface_ids)) != n:
			raise ValueError("Surface IDs must be distinct positive uint32 values")
		# Pack once, before mutations. Instance offsets remain valid after regrouping.
		headers, shared, instance, ids, offsets = self.library.pack(bindings, dirty_materials=dirty_materials)
		routes = np.array([int(binding.shader.is_blended(self.library.material_values(binding), binding.overrides)) for binding in bindings], dtype=np.uint32)
		self.draws.sync(mesh_ids, routes, labels=owners.astype(np.uint64))
		order = self.draws.instance_order
		self._write(self.headers, headers)
		self._write(self.parameters["material_parameters"], shared)
		self._write(self.parameters["instance_parameters"], instance)
		metadata = np.zeros((max(1, n), 4), dtype=np.uint32)
		metadata[:n, 0] = np.asarray(self.offsets, dtype=np.uint32)[mesh_ids[order]]
		metadata[:n, 1], metadata[:n, 2], metadata[:n, 3] = ids[order], surface_ids[order], offsets[order]
		self._write(self.metadata, metadata)
		self.update_transforms(transforms)

	def update_transforms(self, transforms):
		"""Transform-only ECS edits do not repack or validate material dictionaries."""
		ordered = transforms[self.draws.instance_order]
		if not np.array_equal(self.draws.buffers["instances"].content[:len(ordered)], ordered):
			self.draws.write_instances(ordered)

	def set_sample_count(self, count):
		if self.transparency.set_sample_count(count):
			self._register_passes()

	def set_exact_nearest(self, enabled):
		if self.transparency.set_exact_nearest(enabled): self._register_passes()

	def set_tail_allocation(self, mode):
		if self.transparency.set_tail_allocation(mode): self._register_passes()

	def set_kawase_passes(self, count):
		return self.transparency.set_kawase_passes(count)

	def set_banded_options(self, **options):
		if self.transparency.set_banded_options(**options): self._register_passes()

	def set_noise_footprint_balancing(self, enabled):
		if self.transparency.set_noise_footprint_balancing(enabled):
			self._register_passes()

	def set_winner_count(self, count):
		if self.transparency.set_winner_count(count):
			self._register_passes()

	def set_force_single_contributor(self, enabled):
		if self.transparency.set_force_single_contributor(enabled): self._register_passes()

	def set_winner_mix(self, mode):
		if self.transparency.set_winner_mix(mode): self._register_passes()

	def set_winner_coupling(self, mode):
		if self.transparency.set_winner_coupling(mode): self._register_passes()

	def set_confidence_filter(self, options):
		return self.transparency.set_confidence_filter(options)

	def set_reconstruction(self, mode):
		if self.transparency.set_reconstruction(mode): self._register_passes()

	def set_reconstruction_passes(self, count):
		if self.transparency.set_reconstruction_passes(count): self._register_passes()

	def set_early_b_rejection(self, enabled):
		if self.transparency.set_early_b_rejection(enabled): self._register_passes()

	def record(self, commands, size, camera, output, *, frame=None, lighting=None):
		"""Record a frame, updating any supplied frame and lighting fields."""
		self.sync()
		oit = self.transparency
		if size[0] <= 0 or size[1] <= 0: return
		oit.resize(size)
		vp = oit.update_camera(camera, frame=frame, lighting=lighting)
		FrustumCuller().record(commands, self.visibility, camera_position=camera.position, view_projection=vp)
		external = oit.has_external_opaque_inputs
		if not external or any(batch.shader_id == 0 and batch.mesh_id is not None for batch in self.draws.draw_batches):
			# Existing scene color and occlusion depth belong to the caller. Local
			# opaque geometry adds to those targets using their current contents.
			with oit.opaque_pass(commands, load=external) as render:
				self.draws.draw(render, self.visibility, "opaque")
		if oit.exact_nearest:
			with oit.nearest_pass(commands) as render:
				self.draws.draw(render, self.visibility, "nearest")
			oit.shade_nearest(commands, self.geometry[1])
		for begin, name in ((oit.guide_pass, "guide"),
			(oit.transmittance_pass, "transmittance")):
			if name == "guide" and oit.banded_tail: continue
			if name == "guide" and oit.bilateral_filter.kernel_size == 1: continue
			with begin(commands) as render:
				self.draws.draw(render, self.visibility, name)
		if oit.banded_tail:
			if oit.mixed_tail:
				with oit.tail_statistics_pass(commands) as render: self.draws.draw(render, self.visibility, "tail_statistics")
			with oit.banded_pass(commands) as render:
				self.draws.draw(render, self.visibility, "bands")
			oit.shade_winner(commands, self.geometry[1])
		for stream in range(0 if oit.banded_tail else oit.winner_count):
			with oit.selection_pass(commands, stream=stream) as render:
				self.draws.draw(render, self.visibility, oit.selection_name(stream))
			oit.shade_winner(commands, self.geometry[1], stream=stream)
		oit.compose(commands, output)
