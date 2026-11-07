"""Array-based drawing with named passes and independent per-view visibility."""

from dataclasses import dataclass

import numpy as np

from .draw_resources import _DrawResources
from .draw_culling import _CullingState, default_cull_shader
from .draw_layout import INDIRECT_DTYPE
from .RenderContext import BindGroup, BufferUsage, GpuBuffer, RenderPipeline


@dataclass
class ShaderPass:
	"""A named drawing pass and its bindings; group 1 binds instance data by default.
	Supply rendering uniforms and extra resources through bindings in other groups.

	Passes sharing visibility must agree on positions, coverage, and rasterization.
	Additional instance attributes can use storage in an extra group, indexed
	by visible_instances[instance_idx] in the current DrawBatches.entities order.
	Owners may repeat; use EcsDrawAdapter.mesh_field() for per-component attributes.
	"""
	pipeline: RenderPipeline
	bindings: tuple[BindGroup, ...] = ()
	instance_group: int | None = 1


@dataclass
class RenderShader:
	"""Named passes; grouping and visibility do not prescribe pass names."""
	passes: dict[str, ShaderPass]

	def __post_init__(self):
		self.passes = {name: spec for name, spec in self.passes.items() if spec is not None}
		if not self.passes: raise ValueError("Expected at least one shader pass")
		if any(not isinstance(name, str) or not name for name in self.passes):
			raise ValueError("Pass names must be nonempty strings")

	def __getitem__(self, name): return self.passes[name]


def standard_render_shader(shader, uniform_buffer=None, vertex_entry="vertex", fragment_entry="fragment"):
	"""Build matching depth/main passes sharing one uniform buffer and binding."""
	uniform_buffer = uniform_buffer if uniform_buffer is not None else shader.uniform_buffer()
	bindings = (shader.bind_group(0, uniforms=uniform_buffer),)
	return RenderShader({
		"main": ShaderPass(RenderPipeline(shader, vertex_entry=vertex_entry,
			fragment_entry=fragment_entry, depth_test="less-equal"), bindings),
		"prepass": ShaderPass(RenderPipeline(shader, vertex_entry=vertex_entry,
			fragment_entry=None), bindings),
	})


class DrawBatches:
	"""Shared geometry and instance resources. Views own visibility independently.

	Use sync(lod_ids, shader_ids, labels=...) with one row per instance. Instance
	packing is caller-owned; instance_order maps packed rows back to input rows.
	The default culler accepts InstanceData.MESH_INSTANCE_DTYPE. Custom layouts
	require matching shaders or caller-produced Visibility.write() data.
	"""
	def __init__(self, instance_dtype, *, cull_shader=None):
		self._resources = _DrawResources(instance_dtype)
		self._cull_shader = cull_shader if cull_shader is not None else default_cull_shader()
		self.shaders = {}
		self._instance_version = 0

	@property
	def buffers(self):
		return {name: self._resources.buffers[name] for name in ("instances", "mesh_metadata", "lod_distances", "workgroups")}

	@property
	def entities(self): return self._resources.entities

	@property
	def instance_order(self): return self._resources.instance_order

	@property
	def instance_dtype(self): return self._resources.instance_dtype

	@property
	def instance_order_version(self): return self._resources.instance_order_version + self._instance_version

	@property
	def draw_batches(self): return self._resources.draw_batches

	@property
	def layout_token(self): return self._resources.resource_version, self.instance_order_version

	def register_mesh(self, mesh_id, mesh, bounds=None): self._resources.register_mesh(mesh_id, mesh, bounds)
	def register_lod_group(self, group_id, lod_ids, distances=()): self._resources.register_lod_group(group_id, lod_ids, distances)
	def unregister_lod_group(self, group_id): self._resources.unregister_lod_group(group_id)
	def unregister_mesh(self, mesh_id): self._resources.unregister_mesh(mesh_id)

	def register_shader(self, shader_id, shader):
		if not isinstance(shader, RenderShader): raise TypeError("Expected RenderShader with named passes")
		for spec in shader.passes.values():
			if not isinstance(spec.pipeline, RenderPipeline): raise TypeError("Expected a RenderPipeline")
			if spec.instance_group is not None and any(binding.group == spec.instance_group for binding in spec.bindings):
				raise ValueError("Group 1 is reserved for default instance bindings; use instance_binding to override")
		self._resources.register_shader(shader_id)
		self.shaders[shader_id] = shader

	def sync(self, lod_ids, shader_ids, *, labels=None):
		from .RenderContext import RenderContext
		lod_ids, shader_ids = np.asarray(lod_ids), np.asarray(shader_ids)
		if lod_ids.ndim != 1 or shader_ids.ndim != 1: raise ValueError("Expected one-dimensional grouping keys")
		labels = np.arange(len(lod_ids), dtype=np.uint64) if labels is None else np.asarray(labels)
		changed = self._resources.sync_arrays(labels, lod_ids, shader_ids)
		if changed: RenderContext.record_stat("regroup_calls")
		return changed

	def invalidate_instances(self): self._instance_version += 1
	def sync_resources(self): self._resources.sync_resources()
	def write_instances(self, values): self._resources.write_instances(values)
	def write_instance_fields(self, **fields): self._resources.write_instance_fields(**fields)

	def create_visibility(self, *, cull_shader=None): return Visibility(self, cull_shader=cull_shader)

	def draw(self, render, visibility, pass_name="main", *, pass_override=None, instance_binding=None):
		from .RenderContext import RenderContext
		visibility.check(self)
		if self.draw_batches: RenderContext.require_features(("indirect-first-instance",))
		commands = visibility.commands
		for batch in self.draw_batches:
			if batch.mesh_id is None: continue
			spec = pass_override or self.shaders[batch.shader_id].passes.get(pass_name)
			if spec is None: continue
			binding = (instance_binding(self, visibility, spec) if instance_binding is not None else
				spec.pipeline.shader.bind_group(spec.instance_group, instances=self.buffers["instances"], visible_instances=visibility.visible_instances) if spec.instance_group is not None else None)
			render.set_pipeline(spec.pipeline)
			render.set_bind_groups(*((binding,) if binding is not None else ()), *spec.bindings)
			mesh = self._resources.meshes[batch.mesh_id].mesh
			render.set_vertex_buffer(0, mesh.vertex_buffer)
			render.set_index_buffer(mesh.index_buffer, format=mesh.index_format)
			render.draw_indexed_indirect(commands, batch.command_index * INDIRECT_DTYPE.itemsize)


class Visibility:
	"""One view's culling state and indirect outputs. Shared data stays in DrawBatches."""
	def __init__(self, draws, *, cull_shader=None):
		self.draws, self._token, self._stage = draws, None, None
		self._view = _CullingState(cull_shader or draws._cull_shader, draws.buffers)
		self.sync()

	def sync(self):
		self.draws.sync_resources()
		if self._token == self.draws.layout_token: return False
		source = self.draws._resources
		for name in ("source_batches", "draw_batches", "workgroup_count"):
			setattr(self._view, name, getattr(source, name))
		for name, buffer in self.draws.buffers.items(): self._view.buffers[name] = buffer
		for name in ("frustum_candidates", "prepass_visible_instances", "main_visible_instances", "batches", "prepass_draw_cmd", "main_draw_cmd"):
			if name in ("frustum_candidates", "prepass_visible_instances", "main_visible_instances"):
				self._view.buffers[name] = GpuBuffer(None, BufferUsage.STORAGE | BufferUsage.COPY_DST,
					dtype=source.buffers[name].content.dtype, count=source.buffers[name].content.size)
			else:
				content = source.buffers[name].content.copy()
				if name in self._view.buffers: self._view.buffers[name].write(content)
				else:
					usage = BufferUsage.STORAGE | BufferUsage.COPY_DST
					if name.endswith("draw_cmd"): usage |= BufferUsage.INDIRECT
					self._view.buffers[name] = GpuBuffer(content, usage)
		for field, value in zip(("workgroup_count", "batch_count", "command_count"), (source.workgroup_count, len(source.source_batches), len(source.draw_batches))):
			self._view.camera_params_buffer.content[field] = value
		self._view._camera_params_dirty = True
		self._view.bindings.clear()
		self._token, self._stage = self.draws.layout_token, None
		return True

	@property
	def commands(self):
		self.check(self.draws)
		return self._view.buffers[f"{self._stage}_draw_cmd"]

	@property
	def visible_instances(self):
		self.check(self.draws)
		return self._view.buffers[f"{self._stage}_visible_instances"]

	def check(self, draws):
		if draws is not self.draws: raise ValueError("Visibility belongs to a different draw list")
		if self._token != draws.layout_token: raise RuntimeError("Visibility is stale; prepare it again after regrouping")
		if self._stage is None: raise RuntimeError("Prepare visibility before drawing")

	def write(self, visible_instances, commands):
		"""Accept caller-produced visibility/indirect records in packed instance order."""
		self.sync()
		visible_instances, commands = np.asarray(visible_instances), np.asarray(commands)
		if visible_instances.ndim != 1 or visible_instances.dtype != np.dtype("u4"):
			raise ValueError("Expected one-dimensional uint32 visible indices")
		if commands.shape != (len(self.draws.draw_batches),) or commands.dtype != INDIRECT_DTYPE:
			raise ValueError("Expected one indirect record per draw destination")
		if np.any(visible_instances >= len(self.draws.entities)):
			raise ValueError("Visible index exceeds instance count")
		ends = commands["first_instance"].astype(np.uint64) + commands["instance_count"]
		if np.any(ends[commands["instance_count"] > 0] > len(visible_instances)):
			raise ValueError("Indirect instance range exceeds visible indices")
		self._view.buffers["main_visible_instances"].write(visible_instances)
		self._view.buffers["main_draw_cmd"].write(commands)
		self._stage = "main"

	def all_visible(self, *, lod=0):
		"""CPU/direct path: select one LoD per source group without running a culler."""
		from .RenderContext import RenderContext
		RenderContext.record_stat("direct_visibility_calls")
		self.sync()
		commands = self._view.buffers["main_draw_cmd"].content.copy()
		commands["instance_count"] = 0
		indices = []
		instance_offset = 0
		for i, batch in enumerate(self.draws._resources.source_batches):
			group = self.draws._resources.lod_groups[batch.lod_group_id]
			if not 0 <= lod < len(group.lod_ids): raise ValueError("LoD is outside a registered group")
			if group.lod_ids[lod] is None: continue
			command_id = int(self._view.buffers["batches"].content[i]["command_offset"]) + lod
			commands[command_id]["first_instance"] = instance_offset
			commands[command_id]["instance_count"] = batch.count
			indices.append(np.arange(batch.offset, batch.offset + batch.count, dtype=np.uint32))
			instance_offset += batch.count
		self.write(np.concatenate(indices) if indices else np.empty(0, dtype=np.uint32), commands[:len(self.draws.draw_batches)])
		return self


class FrustumCuller:
	def record(self, commands, visibility, *, camera_position, view_projection):
		from .RenderContext import RenderContext
		RenderContext.record_stat("frustum_visibility_calls")
		if visibility._stage == "main": visibility._token = None
		visibility.sync()
		view = visibility._view
		view.refresh_bindings(None)
		view.update_cull_camera(camera_position, view_projection)
		view.reset_draw_counts(commands)
		view.cull_instances(commands, "frustum")
		visibility._stage = "prepass"
		return visibility


class HiZCuller:
	def record(self, commands, visibility, *, hzb_view):
		from .RenderContext import RenderContext
		RenderContext.record_stat("hiz_visibility_calls")
		visibility.check(visibility.draws)
		if visibility._stage != "prepass": raise RuntimeError("Record frustum visibility before Hi-Z culling")
		visibility._view.refresh_bindings(hzb_view)
		visibility._view.cull_instances(commands, "hiz")
		visibility._stage = "main"
		return visibility
