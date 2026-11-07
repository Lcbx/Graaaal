"""Internal retained draw registries, resource synchronization, and instance uploads."""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import higher_pow2
from .DrawList import DrawList, SourceBatch
from .draw_compiler import compile_destinations, compile_workgroups, group_bounds
from .draw_layout import (
    BATCH_DTYPE,
    FRUSTUM_CANDIDATE_DTYPE,
    INDIRECT_DTYPE,
    MESH_METADATA_DTYPE,
    WORKGROUP_DTYPE,
    DrawBatch,
    LodGroup,
    MeshId,
)
from .RenderContext import BufferUsage, GpuBuffer, Mesh


Entity = np.uint64
EntityArray = FieldArray = NDArray[Any]
EntityLike = ComponentAccessor = Any


@dataclass
class MeshInfo:
	mesh: Mesh
	box_center: FieldArray
	box_extents: FieldArray

class _MeshSelection(Protocol):
	def __getitem__(self, name: str) -> FieldArray: ...


class _ArrayTemplate:
	"""CPU compiler output; per-view GPU buffers are allocated by Visibility."""
	def __init__(self, content, *args, **kwargs): self.content = content
	def resize(self, count): pass
	def upload(self): pass
	def upload_range(self, start, count): pass


class _DrawResources:
	"""Retain draw registries, grouping caches, CPU templates, and shared GPU buffers."""

	def __init__(self, instance_dtype: np.dtype[Any]) -> None:
		"""The caller supplies the instance layout; matching shader code is caller-owned."""
		self.instance_dtype: np.dtype[Any] = np.dtype(instance_dtype)
		if self.instance_dtype.hasobject or self.instance_dtype.subdtype is not None or not self.instance_dtype.itemsize or self.instance_dtype.itemsize % 4:
			raise ValueError("Expected fixed-size instance records with a stride divisible by four")
		self.meshes: dict[MeshId, MeshInfo] = {}
		self._mesh_versions = {}
		self.resource_version = 0
		self.group_rows: dict[int, int] = {}
		self.lod_groups: dict[int, LodGroup] = {}
		self.source_batches: list[SourceBatch] = []
		self._mesh_commands: dict[int, list[int]] = {}
		self._dirty_meshes: set[int] = set()
		self._shader_prepass: dict[int, bool] = {}
		self.buffers: dict[str, GpuBuffer | _ArrayTemplate] = {}
		self.draw_batches: list[DrawBatch] = []
		self.entities: EntityArray = np.empty(0, dtype=Entity)
		self._batch_version: tuple[Any, ...] | None = None
		self._selection: EntityArray = np.empty(0, dtype=Entity)
		self._array_keys: tuple[FieldArray, FieldArray] | None = None
		self.instance_order: FieldArray = np.empty(0, dtype=np.intp)
		self._mesh_selection: _MeshSelection | None = None
		self._mesh_order: FieldArray = np.empty(0, dtype=np.intp)
		self.component_rows: FieldArray = np.empty(0, dtype=np.uint32)
		self._workgroup_signature: tuple[int, ...] = ()
		self._destinations_dirty: bool = True
		self._dirty_lod_distances: set[int] = set()
		self.instance_order_version: int = 0
		self.workgroup_count: int = 0
		storage = BufferUsage.STORAGE | BufferUsage.COPY_DST
		for name, dtype in (
			("instances", self.instance_dtype), ("frustum_candidates", FRUSTUM_CANDIDATE_DTYPE),
			("prepass_visible_instances", np.dtype("<u4")), ("lod_distances", np.dtype("<f4")),
			("main_visible_instances", np.dtype("<u4")), ("mesh_metadata", MESH_METADATA_DTYPE),
			("batches", BATCH_DTYPE), ("workgroups", WORKGROUP_DTYPE),
			("prepass_draw_cmd", INDIRECT_DTYPE), ("main_draw_cmd", INDIRECT_DTYPE),
		):
			usage = storage | (BufferUsage.INDIRECT if name in ("prepass_draw_cmd", "main_draw_cmd") else 0)
			buffer_type = _ArrayTemplate if name not in ("instances", "mesh_metadata", "lod_distances", "workgroups") else GpuBuffer
			self.buffers[name] = buffer_type(np.zeros(1, dtype=dtype), usage, upload=False, label=name)

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

	def register_shader(self, shader_id: int, *, prepass: bool = True) -> None:
		"""Register culling participation without depending on render pipelines."""
		if self._shader_prepass.get(shader_id) != prepass:
			self._shader_prepass[shader_id] = prepass
			self._destinations_dirty = True


	def unregister_lod_group(self, group_id: int) -> None:
		"""Remove an unused group after synchronizing away its instances."""
		if group_id not in self.lod_groups: raise KeyError(group_id)
		if any(batch.lod_group_id == group_id for batch in self.source_batches):
			raise ValueError("Synchronize away the group's instances before unregistering it")
		del self.lod_groups[group_id]
		self._dirty_lod_distances.discard(group_id)

	def unregister_mesh(self, mesh_id: int) -> None:
		"""Remove registry ownership; releasing the mesh remains caller-owned."""
		if mesh_id not in self.meshes: raise KeyError(mesh_id)
		if any(mesh_id in group.lod_ids for group in self.lod_groups.values()):
			raise ValueError("Remove LoD groups referencing this mesh before unregistering it")
		del self.meshes[mesh_id]
		self._mesh_versions.pop(mesh_id, None)
		self._dirty_meshes.discard(mesh_id)
		self._mesh_commands.pop(mesh_id, None)

	def _sync_meshes(self) -> None:
		"""Refresh group bounds and every destination using changed geometry."""
		if not self._dirty_meshes: return
		for group_id, row in self.group_rows.items():
			if self._dirty_meshes.isdisjoint(self.lod_groups[group_id].lod_ids): continue
			center, extents = group_bounds(self.lod_groups[group_id], self.meshes)
			buffer = self.buffers["mesh_metadata"]
			metadata = buffer.content[row]
			if not (np.array_equal(metadata["box_center"][:3], center) and np.array_equal(metadata["box_extents"][:3], extents)):
				metadata["box_center"][:3], metadata["box_extents"][:3] = center, extents
				buffer.upload_range(row, 1)
		for mesh_id in self._dirty_meshes:
			command_indices = self._mesh_commands.get(mesh_id, ())
			if not command_indices: continue
			mesh = self.meshes[mesh_id].mesh
			fields = ("index_count", "first_index", "base_vertex")
			values = (mesh.index_count, mesh.index_range[0], mesh.vertex_range[0])
			for command_index in command_indices:
				for name in ("prepass_draw_cmd", "main_draw_cmd"):
					buffer = self.buffers[name]
					command = buffer.content[command_index]
					if all(command[field] == value for field, value in zip(fields, values)): continue
					for field, value in zip(fields, values): command[field] = value
					buffer.upload_range(command_index, 1)
		self._dirty_meshes.clear()

	def _group_instances(self, labels, lod_ids, shader_ids):
		"""Common grouping, destination rebuild, and order invalidation for all inputs."""
		grouping = DrawList.build(lod_ids, shader_ids)
		order, batches = grouping.order, list(grouping.batches)
		owners = labels[order]
		if batches != self.source_batches or self._destinations_dirty:
			self._rebuild_destinations(owners, batches)
		self.entities, self.instance_order = owners, order.astype(np.intp)
		self.instance_order_version += 1
		return order

	def sync_batches(self, entities: EntityLike, transforms: ComponentAccessor, mesh_refs: ComponentAccessor) -> bool:
		"""Group mesh components into contiguous LoD group/shader ranges.

		Owners may repeat. Selection, membership or grouping-key changes invalidate
		instance inputs; no component identity is retained across regrouping.
		Layout changes also refresh the retained selection. component_rows contains
		raw indices valid only until the next structural edit to the mesh store.
		When this returns True (or instance_order_version changes), rewrite all
		instance fields and custom attributes in the current entities/mesh_field order.
		Synchronize before recording any reset/culling/draw passes for the frame.
		"""
		ents = np.asarray(entities)
		if ents.ndim != 1: raise ValueError("Expected a 1D entity selection")
		version = (transforms, mesh_refs, transforms.membership_version, mesh_refs.layout_version, mesh_refs.version("lod_id", "shader_id"))
		regroup = version != self._batch_version or not np.array_equal(ents, self._selection)
		if regroup:
			refs = mesh_refs[ents]
			rows = refs.get_rows()
			order = self._group_instances(mesh_refs.owner_ids(refs), refs["lod_id"], refs["shader_id"])
			self._mesh_selection, self._mesh_order = refs, order
			self._array_keys = None
			self.component_rows = rows[order]
			self._selection = ents.copy()
			self._batch_version = version
		self.sync_resources()
		return regroup

	def sync_arrays(self, entities: EntityLike, lod_ids: ArrayLike, shader_ids: ArrayLike) -> bool:
		"""Group one draw instance per input row, without ECS accessors.

		Repeated entity IDs are labels and do not expand rows. instance_order maps
		compact output rows to input rows; reorder all instance data and metadata
		with it. Inputs are compared by value, including in-place edits. A True
		result invalidates instance inputs, just like sync_batches().
		"""
		ents = np.asarray(entities)
		lod_ids, shader_ids = np.asarray(lod_ids), np.asarray(shader_ids)
		if not lod_ids.size: lod_ids = lod_ids.astype(np.int64)
		if not shader_ids.size: shader_ids = shader_ids.astype(np.int64)
		if not (ents.ndim == lod_ids.ndim == shader_ids.ndim == 1 and len(ents) == len(lod_ids) == len(shader_ids)):
			raise ValueError("Expected equally sized 1D entity, LoD ID, and shader ID arrays")
		if lod_ids.dtype.kind not in "iu" or shader_ids.dtype.kind not in "iu":
			raise TypeError("LoD and shader IDs must be integers")
		keys = self._array_keys
		regroup = (keys is None or not np.array_equal(ents, self._selection)
			or not np.array_equal(lod_ids, keys[0]) or not np.array_equal(shader_ids, keys[1]))
		if regroup:
			self._group_instances(ents, lod_ids, shader_ids)
			self.component_rows = np.empty(0, dtype=np.uint32)
			self._mesh_selection = None
			self._batch_version = None
			self._selection = ents.copy()
			self._array_keys = (lod_ids.copy(), shader_ids.copy())
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
		for mesh_id, info in self.meshes.items():
			version = getattr(info.mesh, "version", None)
			if self._mesh_versions.get(mesh_id) != version:
				self._dirty_meshes.add(mesh_id)
				self._mesh_versions[mesh_id] = version
		if self._destinations_dirty or self._dirty_meshes or self._dirty_lod_distances:
			self.resource_version += 1
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
		layout = compile_destinations(len(ordered_entities), batches, self.lod_groups,
			self.meshes, self._shader_prepass)
		self.resource_version += 1
		for name, count in (
			("instances", len(ordered_entities)), ("frustum_candidates", len(ordered_entities)),
			("main_visible_instances", layout.main_count), ("prepass_visible_instances", layout.prepass_count),
		):
			self._reserve_buffer(name, count)
		for name, values in layout.arrays.items(): self._upload_array(name, values)
		self.source_batches, self.draw_batches = batches, layout.draw_batches
		self._mesh_commands, self.group_rows = layout.mesh_commands, layout.group_rows
		self._dirty_meshes.clear()
		self._sync_workgroups()
		self._destinations_dirty = False
		self._dirty_lod_distances.clear()

	def _sync_workgroups(self) -> None:
		# Source offsets live in batches; only dispatch-count changes rebuild this table.
		signature = tuple((batch.count + 63) // 64 for batch in self.source_batches)
		if signature == self._workgroup_signature: return
		workgroups = compile_workgroups(signature)
		self._upload_array("workgroups", workgroups)
		self.workgroup_count = len(workgroups)
		self._workgroup_signature = signature

	def invalidate_batches(self) -> None:
		"""Force regrouping and instance input invalidation on the next sync call."""
		self._batch_version = None
		self._array_keys = None
