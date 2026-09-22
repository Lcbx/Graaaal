from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from numpy.typing import ArrayLike, NDArray

import numpy as np

from ECS import ComponentAccessor, Entity, EntityArray, EntityLike, FieldArray, as_entities
from RenderContext import BindGroup, GpuBuffer, Texture, Shader, Mesh, ComputePipeline, RenderPipeline, BufferUsage, TextureUsage, higher_pow2
from Utils import extract_frustum_planes


mesh_metadata_dtype = np.dtype([
	("box_center", "<f4", 4), ("box_extents", "<f4", 4),
	("lod_offset", "<u4"), ("lod_count", "<u4"), ("padding", "<u4", 2),
])
frustum_candidate_dtype = np.dtype([("instance_id", "<u4"), ("command_id", "<u4")])
batch_dtype = np.dtype([
	("group_id", "<u4"), ("instance_offset", "<u4"), ("instance_count", "<u4"),
	("command_offset", "<u4"), ("prepass", "<u4"), ("frustum_count", "<u4"),
])
workgroup_dtype = np.dtype([("batch_id", "<u4"), ("instance_offset", "<u4")])
indirect_dtype = np.dtype([
	("index_count", "<u4"), ("instance_count", "<u4"), ("first_index", "<u4"),
	("base_vertex", "<i4"), ("first_instance", "<u4"),
])

MeshId = int | None

@dataclass
class SourceBatch:
	lod_group_id: int
	shader_id: int
	offset: int
	count: int

@dataclass(frozen=True)
class LodGroup:
	lod_ids: tuple[MeshId, ...]
	distances: tuple[float, ...]

@dataclass
class DrawBatch:
	"""Main-pass destination; offset/count describe its reserved index region."""
	mesh_id: MeshId
	shader_id: int
	offset: int
	count: int
	command_index: int


@dataclass
class ShaderPass:
	"""Group 1 binds instances and visible_instances and is owned by DrawBatches.
	Supply rendering uniforms and extra resources through bindings in other groups.

	Prepass and main must agree on positions, coverage, and rasterization.
	Additional instance attributes can use storage in an extra group, indexed
	by visible_instances[instance_idx] in the current DrawBatches.entities order.
	Owners may repeat; use mesh_field() for per-component attributes.
	"""
	pipeline: RenderPipeline | ComputePipeline
	bindings: tuple[BindGroup, ...] = ()

@dataclass
class RenderShader:
	main: ShaderPass
	prepass: ShaderPass | None

def standard_RenderShader(shader: Shader, uniform_buffer: GpuBuffer | None = None, vertex_entry: str = "vertex", fragment_entry: str | None = "fragment") -> RenderShader:
	prepass_pipeline = RenderPipeline(
		shader,
		vertex_entry=vertex_entry,
		fragment_entry=None,
		label="prepass",
	)
	main_pipeline = RenderPipeline(
		shader,
		vertex_entry=vertex_entry,
		fragment_entry=fragment_entry,
		depth_test="less-equal",
		label="main",
	)
	uniforms_bg = shader.bind_group(0, uniforms= uniform_buffer or shader.UniformBuffer())
	bindings_tup = (uniforms_bg,)
	return RenderShader(
		ShaderPass(main_pipeline, bindings_tup),
		ShaderPass(prepass_pipeline, bindings_tup)
	)


@dataclass
class MeshInfo:
	mesh: Mesh
	box_center: FieldArray
	box_extents: FieldArray

class _MeshSelection(Protocol):
	def __getitem__(self, name: str) -> FieldArray: ...

class DrawBatches:
	"""Persistent draw resources; record passes explicitly using the caller's commands."""

	def __init__(self, instance_dtype: np.dtype[Any], cull_shader: Shader | None = None) -> None:
		"""The caller supplies the instance layout; cull and draw shaders must match it.

		The default cull shader uses Utils.mesh_instance_dtype. Supply a compatible
		cull_shader when using another ABI; dtype selection does not generate WGSL.
		"""
		self.instance_dtype: np.dtype[Any] = np.dtype(instance_dtype)
		if self.instance_dtype.hasobject or self.instance_dtype.subdtype is not None or not self.instance_dtype.itemsize or self.instance_dtype.itemsize % 4:
			raise ValueError("Expected fixed-size instance records with a stride divisible by four")
		self._cull_shader: Shader = cull_shader or Shader(filepath='scenes/shaders/cull.shader', label="cull")
		self._cull_pipelines: dict[str, ComputePipeline] = {}
		self.meshes: dict[MeshId, MeshInfo] = {}
		self.group_rows: dict[int, int] = {}
		self.lod_groups: dict[int, LodGroup] = {}
		self.source_batches: list[SourceBatch] = []
		self._mesh_commands: dict[int, list[int]] = {}
		self._dirty_meshes: set[int] = set()
		self.shaders: dict[int, RenderShader] = {}
		self.buffers: dict[str, GpuBuffer] = {}
		self.draw_batches: list[DrawBatch] = []
		self.entities: EntityArray = np.empty(0, dtype=Entity)
		self._batch_version: tuple[Any, ...] | None = None
		self._selection: EntityArray = np.empty(0, dtype=Entity)
		self._mesh_selection: _MeshSelection | None = None
		self._mesh_order: FieldArray = np.empty(0, dtype=np.intp)
		self.component_rows: FieldArray = np.empty(0, dtype=np.uint32)
		self._workgroup_signature: tuple[int, ...] = ()
		self._destinations_dirty: bool = True
		self._dirty_lod_distances: set[int] = set()
		self.instance_order_version: int = 0
		self.bindings: dict[str | tuple[int, str], BindGroup] = {}
		self.workgroup_count: int = 0
		storage = BufferUsage.STORAGE | BufferUsage.COPY_DST
		for name, dtype in (
			("instances", self.instance_dtype), ("frustum_candidates", frustum_candidate_dtype),
			("prepass_visible_instances", np.dtype("<u4")), ("lod_distances", np.dtype("<f4")),
			("main_visible_instances", np.dtype("<u4")), ("mesh_metadata", mesh_metadata_dtype),
			("batches", batch_dtype), ("workgroups", workgroup_dtype),
			("prepass_draw_cmd", indirect_dtype), ("main_draw_cmd", indirect_dtype),
		):
			usage = storage | (BufferUsage.INDIRECT if name in ("prepass_draw_cmd", "main_draw_cmd") else 0)
			self.buffers[name] = GpuBuffer(np.zeros(1, dtype=dtype), usage, upload=False, label=name)
		self.camera_params_buffer: GpuBuffer = self._cull_shader.UniformBuffer("camera_params")
		self._camera_params_dirty: bool = False

	def register_lod_group(self, group_id: int, lod_ids: Sequence[MeshId], distances: Sequence[float] = ()) -> None:
		"""MeshRef.lod_ids names a group. Distances are increasing world-space switch distances.

		Supply one fewer distance than meshes; equality selects the coarser LoD.
		None meshId renders nothing in that LoD range; at least one mesh must be non-None.
		Distances are measured from the culling camera to the transformed group bounds center.
		All variants must use the same local coordinate system and compatible shaders.
		Example: register_lod_group(10, (100, 101, None), (30.0, 100.0)).
		"""
		lod_ids = tuple(lod_ids)
		distances = tuple(float(value) for value in distances)
		if not lod_ids or len(distances) != len(lod_ids) - 1:
			raise ValueError("Expected at least one mesh and one fewer LoD distance")
		values = np.asarray(distances, dtype=np.float32)
		if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(np.diff(values) <= 0):
			raise ValueError("LoD distances must be finite, positive and strictly increasing in float32")
		if all(mesh_id is None for mesh_id in lod_ids):
			raise ValueError("Expected at least one non-None mesh for group bounds")
		for mesh_id in lod_ids:
			if mesh_id is None: continue
			if mesh_id not in self.meshes: raise KeyError(f"Unregistered mesh_id {mesh_id}")
		group = LodGroup(lod_ids, distances)
		previous = self.lod_groups.get(group_id)
		if previous != group:
			self.lod_groups[group_id] = group
			if previous is None or previous.lod_ids != lod_ids:
				self._destinations_dirty = True
			else:
				self._dirty_lod_distances.add(group_id)

	def get_cull_pipeline(self, pipeline_name : str) -> ComputePipeline:
		if (pipeline := self._cull_pipelines.get(pipeline_name)) is None:
			pipeline = self._cull_pipelines[pipeline_name] = ComputePipeline(self._cull_shader, entry=pipeline_name, label=pipeline_name)
		return pipeline

	def _reserve_buffer(self, name: str, count: int) -> FieldArray:
		"""Keep CPU capacity stable between growths; GpuBuffer owns GPU growth.

		Callers replace active inputs after growth. GPU outputs are regenerated.
		"""
		buffer = self.buffers[name]
		if count > buffer.content.size:
			capacity = higher_pow2(count - 1)
			buffer.content = np.zeros(capacity, dtype=buffer.content.dtype)
			buffer.resize(capacity)
		return buffer.content[:count]

	def _upload_array(self, name: str, values: FieldArray | Sequence[Any]) -> None:
		self._reserve_buffer(name, len(values))[:] = values
		if len(values): self.buffers[name].upload_range(0, len(values))

	def register_mesh(self, mesh_id: int, mesh: Mesh, bounds: tuple[ArrayLike, ArrayLike] | None = None) -> None:
		"""Use explicit conservative local bounds for displaced geometry.

		Re-register after changing mesh geometry, bounds, or pooled draw ranges.
		Only changed metadata/commands are uploaded during sync_batches.
		Registration does not invalidate entity grouping.
		Registry IDs need not be dense.
		"""
		if bounds is None:
			pos = mesh.vertices["position"]
			box_min, box_max = np.min(pos, axis=0), np.max(pos, axis=0)
			bounds = (box_min + box_max) * 0.5, (box_max - box_min) * 0.5
		center, extents = (np.asarray(value, dtype=np.float32).copy() for value in bounds)
		if center.shape != (3,) or extents.shape != (3,) or not np.all(np.isfinite([center, extents])) or np.any(extents < 0):
			raise ValueError("Mesh bounds must be finite vec3 center and nonnegative extents")
		self.meshes[mesh_id] = MeshInfo(mesh, center, extents)
		self._dirty_meshes.add(mesh_id)

	def register_shader(self, shader_id: int, shader: RenderShader) -> None:
		"""Replace bindings; rebuild destinations only when prepass participation changes."""

		for spec in (shader.prepass, shader.main):
			if spec is not None and any(bindings.group == 1 for bindings in spec.bindings):
				raise ValueError("Group 1 is reserved for instance data")

		had_prepass = (shader_id, "prepass") in self.bindings
		if had_prepass != (shader.prepass is not None):
			self._destinations_dirty = True

		self.shaders[shader_id] = shader
		for name, spec, visible in (
			("prepass", shader.prepass, "prepass_visible_instances"),
			("main", shader.main, "main_visible_instances"),
		):
			if spec is None:
				self.bindings.pop((shader_id, name), None)
				continue
			self.bindings[shader_id, name] = spec.pipeline.shader.bind_group(1, instances=self.buffers["instances"], visible_instances=self.buffers[visible])


	def _group_bounds(self, group_id: int) -> tuple[FieldArray, FieldArray]:
		infos = [self.meshes[mesh_id] for mesh_id in self.lod_groups[group_id].lod_ids if mesh_id is not None]
		box_min = np.min([info.box_center - info.box_extents for info in infos], axis=0)
		box_max = np.max([info.box_center + info.box_extents for info in infos], axis=0)
		return (box_min + box_max) * 0.5, (box_max - box_min) * 0.5

	def _sync_meshes(self) -> None:
		"""Refresh group bounds and every destination using changed geometry."""
		if not self._dirty_meshes: return
		for group_id, row in self.group_rows.items():
			if self._dirty_meshes.isdisjoint(self.lod_groups[group_id].lod_ids): continue
			center, extents = self._group_bounds(group_id)
			buffer = self.buffers["mesh_metadata"]
			metadata = buffer.content[row]
			if not (np.array_equal(metadata["box_center"][:3], center) and np.array_equal(metadata["box_extents"][:3], extents)):
				metadata["box_center"][:3], metadata["box_extents"][:3] = center, extents
				buffer.upload_range(row, 1)
		for mesh_id in self._dirty_meshes:
			mesh = self.meshes[mesh_id].mesh
			fields = ("index_count", "first_index", "base_vertex")
			values = (mesh.index_count, mesh.index_range[0], mesh.vertex_range[0])
			for command_index in self._mesh_commands.get(mesh_id, ()):
				for name in ("prepass_draw_cmd", "main_draw_cmd"):
					buffer = self.buffers[name]
					command = buffer.content[command_index]
					if all(command[field] == value for field, value in zip(fields, values)): continue
					for field, value in zip(fields, values): command[field] = value
					buffer.upload_range(command_index, 1)
		self._dirty_meshes.clear()

	def sync_batches(self, entities: EntityLike, transforms: ComponentAccessor, mesh_refs: ComponentAccessor) -> bool:
		"""Group mesh components into contiguous LoD group/shader ranges.

		Owners may repeat. Selection, membership or grouping-key changes invalidate
		instance inputs; no component identity is retained across regrouping.
		When this returns True (or instance_order_version changes), rewrite all
		instance fields and custom attributes in the current entities/mesh_field order.
		Synchronize before recording any reset/culling/draw passes for the frame.
		"""
		ents = as_entities(entities)
		if ents.ndim != 1: raise ValueError("Expected a 1D entity selection")
		version = (transforms, mesh_refs, transforms.membership_version, mesh_refs.version("lod_id", "shader_id"))
		regroup = version != self._batch_version or not np.array_equal(ents, self._selection)
		if regroup:
			refs = mesh_refs[ents]
			rows = refs.get_rows()
			order, batches = build_batches(np.arange(len(rows), dtype=Entity), refs["lod_id"], refs["shader_id"])
			component_rows = rows[order]
			owners = mesh_refs.owner_ids(component_rows)
			if batches != self.source_batches or self._destinations_dirty:
				self._rebuild_destinations(owners, batches)
			self._mesh_selection, self._mesh_order = refs, order
			self.component_rows, self.entities = component_rows, owners
			self.instance_order_version += 1
			self._selection = ents
			self._batch_version = version
		self.sync_resources()
		return regroup

	def mesh_field(self, name: str) -> FieldArray:
		"""Read a MeshRef field in the current compact instance order.

		Do not index mesh_refs with the repeated owners in entities: that would
		expand components again. Synchronize after ECS membership edits first.
		"""
		if self._mesh_selection is None: raise RuntimeError("Call sync_batches before mesh_field")
		return self._mesh_selection[name][self._mesh_order]

	def sync_resources(self) -> None:
		"""Apply registry changes without rereading or reordering mesh components."""
		if self._destinations_dirty:
			self._rebuild_destinations(self.entities, self.source_batches)
		self._sync_lod_distances()
		self._sync_meshes()

	def write_instances(self, values: FieldArray) -> None:
		"""Upload packed records in the current entities/mesh_field order."""
		count = len(self.entities)
		if values.ndim != 1 or len(values) != count:
			raise ValueError("Expected one packed record per current mesh component")
		buffer = self.buffers["instances"]
		values = values.copy() if np.shares_memory(values, buffer.content) else values
		buffer.content[:len(values)] = values
		if count: buffer.upload_range(0, count)

	def write_instance_fields(self, **fields: Any) -> None:
		"""Write packed dtype fields; omitted fields retain their current values.

		Values broadcast to (live count, *field shape). No field names or packing
		conventions are assumed. The caller tracks changes; supplied fields cause
		one upload of the live records. Initialize all fields after regrouping.
		"""
		if not fields: return
		buffer = self.buffers["instances"]
		count = len(self.entities)
		# Validate before modifying storage; preserve aliased inputs across writes.
		values: dict[str, NDArray[Any]] = {}
		for name, value in fields.items():
			target = buffer.content[name][:count]
			array = np.broadcast_to(np.asarray(value, dtype=target.dtype), target.shape)
			values[name] = array.copy() if np.shares_memory(array, buffer.content) else array
		for name, array in values.items(): buffer.content[name][:count] = array
		if count: buffer.upload_range(0, count)

	def _sync_lod_distances(self) -> None:
		buffer = self.buffers["lod_distances"]
		for group_id in self._dirty_lod_distances:
			row = self.group_rows.get(group_id)
			if row is None: continue
			metadata = self.buffers["mesh_metadata"].content[row]
			offset = int(metadata["lod_offset"]) + 1
			distances = self.lod_groups[group_id].distances
			if distances:
				buffer.content[offset:offset + len(distances)] = distances
				buffer.upload_range(offset, len(distances))
		self._dirty_lod_distances.clear()

	def _rebuild_destinations(self, ordered_entities: EntityArray, batches: list[SourceBatch]) -> None:
		used_groups = sorted({batch.lod_group_id for batch in batches})
		group_rows = {group_id: i for i, group_id in enumerate(used_groups)}
		for batch in batches:
			if batch.lod_group_id not in self.lod_groups: raise KeyError(f"Unregistered LoD group {batch.lod_group_id}")
			if batch.shader_id not in self.shaders: raise KeyError(f"Unregistered shader_id {batch.shader_id}")
		metadata = np.zeros(len(used_groups), dtype=mesh_metadata_dtype)
		distances : list[float] = []
		for group_id, row in group_rows.items():
			group = self.lod_groups[group_id]
			center, extents = self._group_bounds(group_id)
			metadata[row]["box_center"][:3], metadata[row]["box_extents"][:3] = center, extents
			metadata[row]["lod_offset"] = len(distances)
			metadata[row]["lod_count"] = len(group.lod_ids)
			distances.extend((0.0, *group.distances))
		params = np.zeros(len(batches), dtype=batch_dtype)
		draw_batches     : list[DrawBatch]      = []
		commands         : list[tuple[int,...]] = []
		prepass_commands : list[tuple[int,...]] = []
		main_count = prepass_count = 0
		mesh_commands : dict[int, list[int]]= {}
		for batch_id, batch in enumerate(batches):
			prepass = self.shaders[batch.shader_id].prepass is not None
			params[batch_id] = (group_rows[batch.lod_group_id], batch.offset, batch.count, len(commands), prepass, 0)
			for mesh_id in self.lod_groups[batch.lod_group_id].lod_ids:
				command_index = len(commands)
				draw_batches.append(DrawBatch(mesh_id, batch.shader_id, main_count, batch.count, command_index))
				if mesh_id is None:
					command = (0, 0, 0, 0)
					commands.append((*command, main_count))
				else:
					mesh = self.meshes[mesh_id].mesh
					command = (mesh.index_count, 0, mesh.index_range[0], mesh.vertex_range[0])
					commands.append((*command, main_count))
					mesh_commands.setdefault(mesh_id, []).append(command_index)
				prepass_commands.append((*command, prepass_count if prepass else 0))
				main_count += batch.count
				if prepass: prepass_count += batch.count
		if max(
				main_count,
				prepass_count,
				len(commands),
				entities_len := len(ordered_entities)
			) > np.iinfo(np.uint32).max:
			raise ValueError("Draw destinations exceed uint32 addressing")
		self._reserve_buffer("instances", entities_len)
		self._reserve_buffer("frustum_candidates", entities_len)
		self._reserve_buffer("main_visible_instances", main_count)
		self._reserve_buffer("prepass_visible_instances", prepass_count)
		for name, values in (
			("mesh_metadata", metadata), ("lod_distances", np.asarray(distances, dtype="<f4")),
			("batches", params),
			("prepass_draw_cmd", np.asarray(prepass_commands, dtype=indirect_dtype)),
			("main_draw_cmd", np.asarray(commands, dtype=indirect_dtype)),
		):
			self._upload_array(name, values)
		self.source_batches, self.draw_batches = batches, draw_batches
		self._mesh_commands = mesh_commands
		self.group_rows = group_rows
		self._dirty_meshes.clear()
		self._sync_workgroups()
		counts = (self.workgroup_count, len(self.source_batches), len(self.draw_batches))
		for field, value in zip(("workgroup_count", "batch_count", "command_count"), counts):
			if np.any(self.camera_params_buffer.content[field] != value):
				self.camera_params_buffer.content[field] = value
				self._camera_params_dirty = True
		self._destinations_dirty = False
		self._dirty_lod_distances.clear()

	def _sync_workgroups(self) -> None:
		# Source offsets are read from batches; only per-batch dispatch counts matter.
		signature = tuple((batch.count + 63) // 64 for batch in self.source_batches)
		if signature == self._workgroup_signature: return
		group_counts = np.asarray(signature, dtype=np.int64)
		workgroups = np.zeros(int(group_counts.sum()), dtype=workgroup_dtype)
		if len(group_counts):
			workgroups["batch_id"] = np.repeat(np.arange(len(group_counts)), group_counts)
			starts = np.cumsum(group_counts) - group_counts
			workgroups["instance_offset"] = (np.arange(len(workgroups)) - np.repeat(starts, group_counts)) * 64
		self._upload_array("workgroups", workgroups)
		self.workgroup_count = len(workgroups)
		self._workgroup_signature = signature

	def refresh_bindings(self, hzb_view: Any) -> None:
		cull_shader = self._cull_shader
		buffers = self.buffers
		if "cull_frustum" not in self.bindings:
			common = {name: buffers[name] for name in ("instances", "prepass_draw_cmd", "frustum_candidates", "mesh_metadata", "batches", "workgroups", "prepass_visible_instances", "lod_distances")}
			self.bindings["cull_frustum"] = cull_shader.bind_group(0, **common)
			common = {name: buffers[name] for name in ("instances", "frustum_candidates", "mesh_metadata", "batches", "workgroups")}
			self.bindings["cull_hiz"] = cull_shader.bind_group(0, **common)
			self.bindings["frustum"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer)
			self.bindings["reset_prepass"] = cull_shader.bind_group(0, prepass_draw_cmd=buffers["prepass_draw_cmd"], batches=buffers["batches"])
			self.bindings["reset_main"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer, main_draw_cmd=buffers["main_draw_cmd"])
		bindings = self.bindings.get("hiz")
		if bindings is None or bindings.resources["hzb_texture"] is not hzb_view:
			self.bindings["hiz"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer, hzb_texture=hzb_view, main_draw_cmd=buffers["main_draw_cmd"], main_visible_instances=buffers["main_visible_instances"])

	def reset_draw_counts(self, cmd: Any, pipeline_name: str = "reset_draw_counts") -> None:
		# Reset instance counters for indirect draw targets
		count = max(len(self.draw_batches), len(self.source_batches))
		if not count: return
		groups = (count + 63) // 64
		x = min(groups, 65535)
		y = (groups + x - 1) // x
		if y > 65535: raise ValueError("Draw counter reset exceeds dispatch limits")

		pipeline = self.get_cull_pipeline(pipeline_name)

		self._upload_camera_params()
		with cmd.compute_pass(label=pipeline_name) as cp:
			cp.set_pipeline(pipeline)
			cp.set_bind_group(self.bindings["reset_prepass"])
			cp.set_bind_group(self.bindings["reset_main"])
			cp.dispatch(x, y)

	def cull_instances(self, cmd: Any, stage: str = "frustum") -> None:
		if not self.workgroup_count: return
		# Flatten a 2D dispatch so large scenes do not exceed the portable X limit.
		x = min(self.workgroup_count, 65535)
		y = (self.workgroup_count + x - 1) // x
		if y > 65535: raise ValueError("Culling workgroup table exceeds dispatch limits")

		pipeline_name = f"cull_{stage}"
		pipeline = self.get_cull_pipeline(pipeline_name) 

		self._upload_camera_params()
		with cmd.compute_pass(label=pipeline_name) as cp:
			cp.set_pipeline(pipeline)
			cp.set_bind_group(self.bindings[f"cull_{stage}"])
			cp.set_bind_group(self.bindings[stage])
			cp.dispatch(x, y)

	def draw(self, rp: Any, stage: str) -> None:
		commands = self.buffers[f"{stage}_draw_cmd"]
		for batch in self.draw_batches:
			if batch.mesh_id is None: continue
			spec: ShaderPass | None = getattr(self.shaders[batch.shader_id], stage)
			if spec is None: continue
			rp.set_pipeline(spec.pipeline)
			rp.set_bind_group(self.bindings[batch.shader_id, stage])
			for bindings in spec.bindings:
				rp.set_bind_group(bindings)
			mesh = self.meshes[batch.mesh_id].mesh
			rp.set_vertex_buffer(0, mesh.vertex_buffer)
			rp.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			rp.draw_indexed_indirect(commands, batch.command_index * indirect_dtype.itemsize)

	def invalidate_batches(self) -> None:
		"""Force regrouping and instance input invalidation on the next sync_batches call."""
		self._batch_version = None

	def update_cull_camera(self, cameraPosition: Sequence[float], viewProjectionMatrix: Sequence[Sequence[float]] | FieldArray) -> None:
		"""Update when the camera changes; upload before the next reset/culling operation."""
		vp = np.asarray(viewProjectionMatrix)
		buffer = self.camera_params_buffer
		buffer.content["view_proj"] = vp
		buffer.content["planes"] = extract_frustum_planes(vp)
		buffer.content["camera_position"] = [*cameraPosition, 0.0]
		self._camera_params_dirty = True

	def _upload_camera_params(self) -> None:
		if not self._camera_params_dirty: return
		self.camera_params_buffer.upload()
		self._camera_params_dirty = False



class HZB:
	def __init__(self,
		shader: Shader | None = None,
		pipelines: tuple[ComputePipeline, ComputePipeline] | None = None
	) -> None:
		self._shader = shader or Shader(filepath='scenes/shaders/hzb.shader', label="hzb")
		self._pipelines = pipelines or (
			ComputePipeline(self._shader, entry="reduce_depth", label="hzb_depth"),
			ComputePipeline(self._shader, entry="main", label="hzb")
		)
		self.size: tuple[int, int] | tuple[()] = ()
		self.depth_texture: Texture | None = None
		self.texture: Texture | None = None
		self.view: Any = None
		self.passes: list[tuple[ComputePipeline, BindGroup, int, int]] = []

	def resize(self, size: tuple[int, int]) -> bool:
		shader, pipelines = self._shader, self._pipelines
		width, height = size
		if width <= 0 or height <= 0 or self.size == size: return False
		usage = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
		self.depth_texture = Texture(size, format="depth32float", usage=usage, label="prepass_depth")
		width, height = max(1, width // 2), max(1, height // 2)
		num_mips = max(width, height).bit_length()
		usage = TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING
		self.texture = Texture((width, height), format="r32float", usage=usage, mip_level_count=num_mips, label="hzb_pyramid")
		self.view = self.texture.view()
		views = [self.texture.view(base_mip_level=i, mip_level_count=1) for i in range(num_mips)]
		bindings = shader.bind_group(0, src_prepass=self.depth_texture.view(), dst_depth=views[0])
		self.passes = [(pipelines[0], bindings, width, height)]
		for mip in range(1, num_mips):
			width, height = max(1, width // 2), max(1, height // 2)
			bindings = shader.bind_group(0, src_depth=views[mip - 1], dst_depth=views[mip])
			self.passes.append((pipelines[1], bindings, width, height))
		self.size = size
		return True

	def build(self, cmd: Any) -> None:
		for mip, (pipeline, bindings, width, height) in enumerate(self.passes):
			with cmd.compute_pass(label=f"hzb_mip_{mip}") as cp:
				cp.set_pipeline(pipeline)
				cp.set_bind_group(bindings)
				cp.dispatch((width + 15) // 16, (height + 15) // 16)


def build_batches(entities: EntityLike, lod_ids: FieldArray, shader_ids: FieldArray) -> tuple[EntityArray, list[SourceBatch]]:
	"""Pure CPU grouping; lod_ids are LoD group IDs. Preserve order inside each pair."""
	ents = as_entities(entities)
	if not (ents.ndim == lod_ids.ndim == shader_ids.ndim == 1 and len(ents) == len(lod_ids) == len(shader_ids)):
		raise ValueError("Expected equally sized 1D entity, mesh ID, and shader ID arrays")
	order = np.lexsort((lod_ids, shader_ids))
	lod_ids, shader_ids = lod_ids[order], shader_ids[order]
	changes = (lod_ids[1:] != lod_ids[:-1]) | (shader_ids[1:] != shader_ids[:-1])
	starts = np.r_[0, np.flatnonzero(changes) + 1] if len(order) else np.empty(0, dtype=int)
	ends = np.r_[starts[1:], len(order)] if len(order) else starts
	batches = [SourceBatch(int(lod_ids[start]), int(shader_ids[start]), int(start), int(end - start)) for start, end in zip(starts, ends)]
	return ents[order].copy(), batches
