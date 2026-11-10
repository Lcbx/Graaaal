"""Internal per-view GPU culling bindings, camera state, and command recording."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .Math import extract_frustum_planes
from .RenderContext import ComputePipeline, Shader


def default_cull_shader():
	return Shader(filepath=str(Path(__file__).with_name("shaders") / "cull.shader"), label="cull")


class _CullingState:
	"""One view's GPU culling state; no registry, grouping cache, or ECS selection."""
	def __init__(self, shader, buffers):
		self._cull_shader = shader
		self._cull_pipelines = {}
		self.buffers = dict(buffers)
		self.bindings = {}
		self.camera_params_buffer = shader.uniform_buffer("camera_params")
		self._camera_params_dirty = True
		self._camera_initialized = False
		self.source_batches = []
		self.draw_batches = []
		self.workgroup_count = 0

	def get_cull_pipeline(self, pipeline_name : str) -> ComputePipeline:
		if (pipeline := self._cull_pipelines.get(pipeline_name)) is None:
			pipeline = self._cull_pipelines[pipeline_name] = ComputePipeline(self._cull_shader, entry=pipeline_name, label=pipeline_name)
		return pipeline


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
		if hzb_view is None: return
		if bindings is None or bindings.resources["hzb_texture"] is not hzb_view:
			self.bindings["hiz"] = cull_shader.bind_group(1, camera_params=self.camera_params_buffer, hzb_texture=hzb_view, main_draw_cmd=buffers["main_draw_cmd"], main_visible_instances=buffers["main_visible_instances"])


	def _dispatch(self, commands, pipeline_name, groups, *bindings):
		if not groups: return
		x = min(groups, 65535)
		y = (groups + x - 1) // x
		if y > 65535: raise ValueError(f"{pipeline_name} exceeds GPU dispatch limits")
		pipeline = self.get_cull_pipeline(pipeline_name)
		self._upload_camera_params()
		with commands.compute_pass(label=pipeline_name) as compute:
			compute.set_pipeline(pipeline)
			for name in bindings: compute.set_bind_group(self.bindings[name])
			compute.dispatch(x, y)

	def reset_draw_counts(self, commands, pipeline_name="reset_draw_counts"):
		count = max(len(self.draw_batches), len(self.source_batches))
		self._dispatch(commands, pipeline_name, (count + 63) // 64, "reset_prepass", "reset_main")

	def cull_instances(self, commands, stage="frustum"):
		self._dispatch(commands, f"cull_{stage}", self.workgroup_count, f"cull_{stage}", stage)

	def update_cull_camera(self, camera_position: Sequence[float],
		view_projection: Sequence[Sequence[float]] | NDArray[Any]) -> None:
		"""Update when the camera changes; upload before the next reset/culling operation."""
		vp = np.asarray(view_projection, dtype=np.float32)
		position = np.asarray(camera_position, dtype=np.float32)
		if vp.shape != (4, 4) or position.shape != (3,) or not np.all(np.isfinite(vp)) or not np.all(np.isfinite(position)):
			raise ValueError("Expected a finite 4x4 camera matrix and vec3 position")
		buffer = self.camera_params_buffer
		if (self._camera_initialized and np.array_equal(buffer.content["view_proj"][0], vp)
			and np.array_equal(buffer.content["camera_position"][0, :3], position)):
			return
		planes = extract_frustum_planes(vp)
		if not np.all(np.isfinite(planes)):
			raise ValueError("Camera matrix produced nonfinite frustum planes")
		buffer.content["view_proj"] = vp
		buffer.content["planes"] = planes
		buffer.content["camera_position"] = [*position, 0.0]
		self._camera_initialized = True
		self._camera_params_dirty = True


	def _upload_camera_params(self) -> None:
		if not self._camera_params_dirty: return
		self.camera_params_buffer.upload_changed()
		self._camera_params_dirty = False
