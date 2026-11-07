from __future__ import annotations

import atexit
import gc
import os
import re
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from threading import Lock
from time import perf_counter as get_time, sleep
from typing import Any, Callable, ClassVar, Iterable, Iterator, ParamSpec, Sequence, TypeVar

import glfw  # type: ignore[import-untyped]
import numpy as np
import wgpu  # type: ignore[import-untyped]
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from wgpu import BufferUsage, TextureUsage  # type: ignore[import-untyped]
from wgpu.utils.glfw_present_info import get_glfw_present_info  # type: ignore[import-untyped]

from .array_utils import higher_pow2


Color = tuple[float, float, float, float]


class _RenderContext:
	"""Application loop singleton on top of glfw & wgpu                                                                          
	inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py                                             
	"""
	def __init__(self) -> None:
		self.canvas: wgpu.GPUCanvasContext | None = None
		self.adapter: wgpu.GPUAdapter
		self.device: wgpu.GPUDevice
		self.device_generation = 0
		self._mesh_arena = None
		self._statistics = None
		self.presentation_format: str

		self.window: Any = None
		self.window_dimensions: tuple[int, int] = (0, 0)
		self.aspect: float = 1.0

		self.target_frame_time: float | None = None
		self.frame_start: float = 0.0
		self.frame_time: float = 0.0

		self.depth_format: str = wgpu.TextureFormat.depth24plus
		self.depth: wgpu.GPUTextureView
		self._screen_view: wgpu.GPUTextureView | None = None

		self.idle_hook : Callable = lambda : None

		# event_name:[handlers]
		# handlers return True if no need to propagate further
		self.event_handlers: dict[str, list[Callable]] = {}
		self._glfw_initialized = False
		self._gc_manager: GCManager | None = None

	def init_window(
		self,
		w: int,
		h: int,
		title: str,
		highpower_gpu: bool = True,
		target_fps: int = 0,
		required_gpu_features: Sequence[str] = (),
		custom_gc:bool=True
	) -> None:
		"""Init a window and wgpu rendering context.

		:param highpower_gpu: use the high performance gpu if there are multiple
		:param target_fps: -1 is limitless, 0 is vsync, other values is fps limit
		:param required_gpu_features: wgpu features requested                                                                
		:param custom_gc: run the gc during idle time
		"""
		if target_fps < -1:
			raise ValueError("target_fps must be -1, 0, or positive")
		if self.window is not None:
			raise RuntimeError("A window is already initialized")
		if not glfw.init():
			raise RuntimeError("GLFW initialization failed")
		self._glfw_initialized = True
		atexit.unregister(self.cleanup)
		atexit.register(self.cleanup)
		try:
			self._init_window(w, h, title, highpower_gpu, target_fps, required_gpu_features, custom_gc)
		except BaseException:
			self.cleanup()
			raise

	def _init_window(self, w, h, title, highpower_gpu, target_fps, required_gpu_features, custom_gc):

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w - w)
		glfw.window_hint(glfw.POSITION_Y, 30)

		vsync = target_fps == 0
		if vsync:
			video_mode = glfw.get_video_mode(monitor)
			target_fps = video_mode.refresh_rate

		self.target_frame_time = None if target_fps == -1 else 1.0 / float(target_fps)

		self.window = glfw.create_window(w, h, title, None, None)
		if not self.window:
			self.window = None
			raise RuntimeError("GLFW window creation failed")
		self.setup_callbacks()

		self.setup_graphics(vsync=vsync, highpower_gpu=highpower_gpu, required_gpu_features=required_gpu_features)
		if custom_gc:
			self._gc_manager = GCManager()
			self.event_handlers["idle"].append(self._gc_manager)

		self.frame_start = get_time()

	def setup_graphics(self, *, vsync: bool, highpower_gpu: bool, required_gpu_features: Sequence[str] = ()) -> None:
		"""Setup gpu compute & render surface."""
		present_info = get_glfw_present_info(self.window, vsync=vsync)
		self.canvas = wgpu.gpu.get_canvas_context(present_info)
		self.setup_graphics_backend(highpower_gpu, required_gpu_features)
		self.presentation_format = self.canvas.get_preferred_format(self.adapter)
		self.canvas.configure(device=self.device, format=self.presentation_format)

	def setup_graphics_backend(self, highpower_gpu: bool, required_features: Sequence[str] = ()) -> None:
		"""Setup gpu compute, not necessarily with canvas output. Used notably for tests."""
		request_params = {
			"power_preference": "high-performance" if highpower_gpu else "low-power"
		}
		if self.canvas:
			request_params["canvas"] = self.canvas
		self.reset_mesh_arena()
		self.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		self.device = self.adapter.request_device_sync(required_features=required_features)
		self.device_generation += 1

	def setup_callbacks(self) -> None:
		glfw.set_key_callback(self.window, self.setup_glfw_event("key"))
		glfw.set_char_callback(self.window, self.setup_glfw_event("char"))
		glfw.set_mouse_button_callback(self.window, self.setup_glfw_event("mouse_click"))
		glfw.set_cursor_pos_callback(self.window, self.setup_glfw_event("mouse_move"))
		glfw.set_cursor_enter_callback(self.window, self.setup_glfw_event("mouse_enter"))
		glfw.set_scroll_callback(self.window, self.setup_glfw_event("mouse_scroll"))
		glfw.set_window_focus_callback(self.window, self.setup_glfw_event("focus"))
		self.idle_hook = self.setup_event("idle")

	def update_window_size(self, wh: tuple[int, int]) -> None:
		w, h = wh
		self.window_dimensions = wh
		assert self.canvas is not None
		self.canvas.set_physical_size(w, h)
		self.aspect = w / h
		self.depth = self.device.create_texture(
			size=(w, h, 1),
			format=self.depth_format,
			usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
		).create_view()

	def capture_mouse(self, capture: bool = True) -> None:
		"""Hide and keep the mouse inside the window."""
		glfw.set_input_mode(
			self.window,
			glfw.CURSOR,
			glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL,
		)

	def subscribe_event(self, channel_name: str, handler: Callable) -> None:
		self.event_handlers[channel_name].append(handler)

	def setup_glfw_event(
		self,
		channel_name: str,
		mapper: Callable | None = None,
	) -> Callable:
		
		def deflt_mapper(*args) -> tuple:
			return args[1:] # glfw first arg is window handle
		
		mapper = mapper or deflt_mapper

		return self.setup_event(channel_name, mapper)

	def setup_event(
		self,
		channel_name: str,
		mapper: Callable | None = None,
	) -> Callable:

		self.event_handlers[channel_name] = []

		def deflt_mapper(*args) -> tuple:
			return args

		mapper = mapper or deflt_mapper

		def handlers_call(*args) -> None:
			args = mapper(*args)
			for handler in self.event_handlers[channel_name]:
				if handler(*args):
					break

		return handlers_call

	def cleanup(self) -> None:
		self.device_generation += 1
		canvas, self.canvas = self.canvas, None
		window, self.window = self.window, None
		initialized, self._glfw_initialized = self._glfw_initialized, False
		self._screen_view = None
		self.window_dimensions = (0, 0)
		if hasattr(self, "depth"):
			del self.depth
		if self._gc_manager is not None:
			self._gc_manager.close()
			self._gc_manager = None
		self.event_handlers.clear()
		self.idle_hook = lambda: None
		self.reset_mesh_arena()
		try:
			if canvas is not None:
				canvas.unconfigure()
		finally:
			try:
				if window is not None:
					glfw.destroy_window(window)
			finally:
				if initialized:
					try:
						# work around https://github.com/glfw/glfw/issues/1766
						end_time = get_time() + 0.1
						while (remaining := end_time - get_time()) > 0:
							glfw.wait_events_timeout(remaining)
					finally:
						glfw.terminate()

	def window_loop(self) -> bool:
		"""Present the previous image, update window/input state, run gc, and continue the loop. """
		self.present()

		wh = glfw.get_framebuffer_size(self.window)
		if wh != self.window_dimensions and wh[0] > 0 and wh[1] > 0:
			self.update_window_size(wh)
		elif wh[0] <= 0 or wh[1] <= 0:
			self.window_dimensions = wh

		self.idle_hook()
		self._frame_pacing()
		glfw.poll_events()
		return not glfw.window_should_close(self.window)

	def present(self) -> None:
		assert self.canvas is not None
		self.canvas.present()
		self._screen_view = None

	_FRAME_WAIT_MARGIN = 0.001 # 1ms

	def _frame_pacing(self) -> None:
		now = get_time()
		self.frame_time = now - self.frame_start

		if self.target_frame_time and self.frame_time < self.target_frame_time:
			sleep_time = self.target_frame_time - self.frame_time - self._FRAME_WAIT_MARGIN
			if sleep_time > 0.0:
				sleep(sleep_time)
			wait_end = self.frame_start + self.target_frame_time
			while now < wait_end:
				now = get_time()

		self.frame_time = now - self.frame_start
		self.frame_start = now

	# commands -----------------------------------------------------------------

	@property
	def mesh_arena(self):
		if self._mesh_arena is None:
			from .MeshArena import MeshArena
			self._mesh_arena = MeshArena()
		return self._mesh_arena

	def reset_mesh_arena(self):
		if self._mesh_arena is not None:
			self._mesh_arena.close()
			self._mesh_arena = None

	def commands(self, label: str | None = None) -> CommandContext:
		return CommandContext(label=label)

	def command(self, label: str | None = None) -> wgpu.GPUCommandEncoder:
		"""Raw command encoder escape hatch."""
		return self.device.create_command_encoder(label=label or "")

	def submit(self, *command_buffers: wgpu.GPUCommandBuffer) -> None:
		self.device.queue.submit(list(command_buffers))

	def require_features(self, features: Sequence[str]) -> None:
		missing = set(features) - set(self.device.features)
		if missing: raise ValueError(f"Device is missing required GPU features: {', '.join(sorted(missing))}")

	def enable_statistics(self, enabled: bool = True):
		"""Enable/reset opt-in counters; no GPU synchronization is performed."""
		self._statistics = {} if enabled else None

	def record_stat(self, name: str, value: int = 1):
		if self._statistics is not None:
			self._statistics[name] = self._statistics.get(name, 0) + value

	def inspect_resources(self):
		return {"device_generation": self.device_generation,
			"mesh_arena": self._mesh_arena.inspect() if self._mesh_arena is not None else None,
			"statistics": dict(self._statistics or {})}

	# attachments ---------------------------------------------------------------

	def screen(
		self,
		*,
		clear: Color | None = None,
		store: bool = True,
	) -> ColorAttachment:
		if self._screen_view is None:
			assert self.canvas is not None
			self._screen_view = self.canvas.get_current_texture().create_view()
		return ColorAttachment(self._screen_view, self.presentation_format, clear, store, device_generation=self.device_generation)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(self.depth, self.depth_format, clear, store, read_only, device_generation=self.device_generation)

RenderContext = _RenderContext()

class GCManager:
	"""Run at most one collection per idle call, preferably in spare frame time.

	Time budgets are estimates, not limits on GC pauses. A due generation is
	collected after max_deferral seconds even if no frame has enough spare time.
	One manager owns the process-wide GC settings until close() is called.
	"""

	def __init__(self,
		thresholds  : tuple[int,int,int]         = (300, 15, 10),
		time_budgets: tuple[float, float, float] = (0.0005, 0.002, 0.005), # sec
		max_deferral: float = 1.0, # sec, measured while collections are due
	) -> None:
		if len(thresholds) != 3 or any(not isinstance(n, int) or n <= 0 for n in thresholds):
			raise ValueError("GC thresholds must be three positive integers")
		if len(time_budgets) != 3 or any(not isfinite(t) or t < 0 for t in time_budgets):
			raise ValueError("GC time budgets must be three finite nonnegative durations")
		if not isfinite(max_deferral) or max_deferral <= 0:
			raise ValueError("max_deferral must be a finite positive duration")
		self.time_budgets = time_budgets
		self.max_deferral = max_deferral
		self._pending_since: list[float | None] = [None, None, None]
		self._was_enabled = gc.isenabled()
		self._thresholds = gc.get_threshold()
		self._closed = False
		# Do not freeze process-wide objects: their ownership is not ours.
		gc.disable()
		gc.set_threshold(*thresholds)

	def close(self) -> None:
		if self._closed:
			return
		self._closed = True
		gc.set_threshold(*self._thresholds)
		if self._was_enabled:
			gc.enable()
		else:
			gc.disable()

	def __call__(self):
		if self._closed:
			return
		counts = gc.get_count()
		thresholds = gc.get_threshold()

		now = get_time()
		budget = (
			RenderContext.target_frame_time - (now - RenderContext.frame_start)
			if RenderContext.target_frame_time else float("inf")
		)
		for gen in range(3):
			if counts[gen] < thresholds[gen]:
				self._pending_since[gen] = None
			elif self._pending_since[gen] is None:
				self._pending_since[gen] = now

		for gen in range(2, -1, -1):
			since = self._pending_since[gen]
			if since is not None and (
				budget >= self.time_budgets[gen] or now - since >= self.max_deferral
			):
				gc.collect(gen)
				# Collecting a generation also collects all younger generations.
				for younger in range(gen + 1):
					self._pending_since[younger] = None
				break

	@classmethod
	def full_sweep(cls) -> None:
		"""Collect all generations without changing the process's frozen objects."""
		gc.collect(2)

	@classmethod
	def scene_transition(cls, new_scene_load: Callable) -> None:
		"""Release old scene references before calling; retain the new scene in the loader."""
		gc.collect(2)
		try:
			new_scene_load()
		finally:
			# Loading may promote temporary cycles; sweep every generation once.
			gc.collect(2)


class _OwnedResource:
	def _capture_device(self):
		if not hasattr(RenderContext, "device"):
			raise RuntimeError("Initialize RenderContext with init_window() or setup_graphics_backend() before creating GPU resources")
		self._device = RenderContext.device
		self._device_generation = RenderContext.device_generation
		RenderContext.record_stat(f"created.{type(self).__name__}")

	def _check_device(self):
		# Metadata-only pipeline specifications bind lazily on first GPU use.
		if not hasattr(self, "_device"): self._capture_device()
		if self._device is not RenderContext.device or self._device_generation != RenderContext.device_generation:
			raise RuntimeError("Resource belongs to an expired device generation")
		if getattr(self, "_closed", False): raise RuntimeError("Resource is closed")
		return self._device

	def close(self):
		# Release wrapper ownership without destroying in-flight WGPU handles.
		self._closed = True
		for name in ("handle", "module", "_variant"):
			self.__dict__.pop(name, None)
		if "_content" in self.__dict__: self._content = None
		if hasattr(self, "_cache"): self._cache.clear()


class CommandContext(_OwnedResource):
	"""One independent GPU command encoder.

	Multiple CommandContexts may be encoded independently and later submitted in the
	order in which their resulting command buffers should execute.
	"""

	def __init__(self, label: str | None = None):
		self._capture_device()
		self.handle = self._check_device().create_command_encoder(label=label or "")
		self.finished = False

	def render_pass(
		self,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	) -> RenderPass:
		self._check_device()
		return RenderPass(self, color=color, depth=depth, label=label)

	def compute_pass(self, *, label: str | None = None) -> ComputePass:
		self._check_device()
		return ComputePass(self, label=label)

	def clear_buffer(
		self,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		self._check_device()
		buffer._check_device()
		self.handle.clear_buffer(buffer.handle, offset, size)

	def copy_buffer(
		self,
		source: GpuBuffer,
		destination: GpuBuffer,
		*,
		source_offset: int = 0,
		destination_offset: int = 0,
		size: int | None = None,
	) -> None:
		self._check_device()
		source._check_device()
		destination._check_device()
		if size is None:
			size = min(source.handle.size - source_offset, destination.handle.size - destination_offset)
		self.handle.copy_buffer_to_buffer(
			source.handle,
			source_offset,
			destination.handle,
			destination_offset,
			size,
		)

	def finish(self, label: str | None = None) -> wgpu.GPUCommandBuffer:
		self._check_device()
		if self.finished:
			raise RuntimeError("CommandContext.finish() called more than once")
		self.finished = True
		return self.handle.finish(label=label or "")


class RenderPass:
	def __init__(
		self,
		commands: CommandContext,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	):
		self.commands = commands
		self.color = [color] if isinstance(color, ColorAttachment) else list(color)
		self.depth = depth
		self.label = label

		self.handle: wgpu.GPURenderPassEncoder | None = None
		self.pipeline: RenderPipeline | None = None

		self.bind_groups: dict[int, BindGroup] = {}
		self.vertex_buffers: dict[int, tuple[np.dtype, str]] = {}

		self._variant: _PipelineVariant | None = None
		self._variant_dirty = True
		self._bound_bind_groups: dict[int, tuple] = {}
		self._bound_vertex_buffers: dict[int, tuple] = {}
		self._bound_index_buffer: tuple | None = None

	@property
	def color_formats(self) -> tuple[str, ...]:
		return tuple(attachment.format for attachment in self.color)

	@property
	def depth_format(self) -> str | None:
		return self.depth.format if self.depth else None

	@property
	def sample_count(self) -> int:
		counts = {item.sample_count for item in self.color}
		if self.depth is not None: counts.add(self.depth.sample_count)
		if len(counts) > 1: raise ValueError("Render attachments must have matching sample counts")
		return next(iter(counts), 1)

	def __enter__(self) -> RenderPass:
		self.sample_count  # Validate attachment compatibility before encoding.
		if isinstance(self.commands, _OwnedResource): self.commands._check_device()
		for attachment in (*self.color, *((self.depth,) if self.depth else ())):
			if attachment.device_generation is not None and attachment.device_generation != RenderContext.device_generation:
				raise RuntimeError("Attachment belongs to an expired device generation")
		if self.handle is not None:
			raise RuntimeError("RenderPass is already active")

		self._variant = None
		self._variant_dirty = True
		self._bound_bind_groups.clear()
		self._bound_vertex_buffers.clear()
		self._bound_index_buffer = None
		self.vertex_buffers.clear()

		self.handle = self.commands.handle.begin_render_pass(
			label=self.label or "",
			color_attachments=[attachment.descriptor() for attachment in self.color],
			depth_stencil_attachment=self.depth.descriptor() if self.depth else None,
		)
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self._require_handle().end()
		self.handle = None

	def set_pipeline(self, pipeline: RenderPipeline) -> None:
		if pipeline is not self.pipeline:
			self.pipeline = pipeline
			self._variant_dirty = True

	def set_bind_group(self, bindings: BindGroup) -> None:
		self.bind_groups[bindings.group] = bindings

	def set_bind_groups(self, *bindings: BindGroup) -> None:
		"""Replace the complete binding set, dropping groups from prior materials."""
		groups = {binding.group: binding for binding in bindings}
		if len(groups) != len(bindings):
			raise ValueError("Each bind group index must be supplied only once")
		self.bind_groups = groups

	def set_vertex_buffer(
		self,
		slot: int,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
		step_mode: str = wgpu.VertexStepMode.vertex,
	) -> None:
		if isinstance(buffer, _OwnedResource): buffer._check_device()
		layout = (buffer.dtype, step_mode)
		if self.vertex_buffers.get(slot) != layout:
			self.vertex_buffers[slot] = layout
			self._variant_dirty = True
		handle = self._require_handle()
		key = (buffer.handle, offset, size)
		if self._bound_vertex_buffers.get(slot) != key:
			handle.set_vertex_buffer(slot, buffer.handle, offset, size)
			self._bound_vertex_buffers[slot] = key

	def set_index_buffer(
		self,
		buffer: GpuBuffer,
		*,
		format: str,
		offset: int = 0,
		size: int | None = None,
	) -> None:
		if isinstance(buffer, _OwnedResource): buffer._check_device()
		handle = self._require_handle()
		key = (buffer.handle, format, offset, size)
		if self._bound_index_buffer != key:
			handle.set_index_buffer(buffer.handle, format, offset, size)
			self._bound_index_buffer = key

	def draw(
		self,
		vertex_count: int,
		*,
		instance_count: int = 1,
		first_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		handle = self._prepare_pipeline()
		handle.draw(
			vertex_count,
			instance_count,
			first_vertex,
			first_instance,
		)

	def draw_indexed(
		self,
		index_count: int,
		*,
		instance_count: int = 1,
		first_index: int = 0,
		base_vertex: int = 0,
		first_instance: int = 0,
	) -> None:
		handle = self._prepare_pipeline()
		handle.draw_indexed(
			index_count,
			instance_count,
			first_index,
			base_vertex,
			first_instance,
		)

	def draw_indexed_indirect(
		self,
		indirect_buffer: GpuBuffer | wgpu.GPUBuffer,
		indirect_offset: int = 0,
	) -> None:
		if isinstance(indirect_buffer, _OwnedResource): indirect_buffer._check_device()
		handle = self._prepare_pipeline()
		handle.draw_indexed_indirect(indirect_buffer.handle if isinstance(indirect_buffer, GpuBuffer) else indirect_buffer, indirect_offset)

	def draw_mesh(
		self,
		mesh: Mesh,
		*,
		instances: GpuBuffer | None = None,
		instance_count: int | None = None,
		instance_offset: int = 0,
	) -> None:
		slot_count = 2 if instances is not None else 1
		for slot in tuple(self.vertex_buffers):
			if slot >= slot_count:
				del self.vertex_buffers[slot]
				self._variant_dirty = True

		vertex_start, vertex_count = mesh.vertex_range_bytes
		self.set_vertex_buffer(
			0,
			mesh.vertex_buffer,
			offset=vertex_start,
			size=vertex_count,
		)

		if instances is not None:
			self.set_vertex_buffer(
				1,
				instances,
				offset=instance_offset * instances.content.dtype.itemsize,
				step_mode=wgpu.VertexStepMode.instance,
			)

		index_start, index_count = mesh.index_range
		self.set_index_buffer(
			mesh.index_buffer,
			format=mesh.index_format,
			offset=index_start * mesh.index_dtype.itemsize,
			size=index_count * mesh.index_dtype.itemsize,
		)

		if instance_count is None:
			instance_count = (instances.count - instance_offset) if instances is not None else 1

		handle = self._prepare_pipeline()
		handle.draw_indexed(index_count, instance_count, 0, 0, 0)

	def _prepare_pipeline(self) -> wgpu.GPURenderPassEncoder:
		handle = self._require_handle()
		if self.pipeline is None:
			raise RuntimeError("No RenderPipeline has been set on this pass")
		if isinstance(self.pipeline, _OwnedResource): self.pipeline._check_device()

		if self._variant_dirty:
			slots = sorted(self.vertex_buffers)
			if slots != list(range(len(slots))):
				raise RuntimeError("Vertex buffer slots must be contiguous starting at 0")

			buffers = [self.vertex_buffers[i] for i in slots]
			variant = self.pipeline.get_variant(self, buffers)

			if variant is not self._variant:
				handle.set_pipeline(variant.handle)
				self._variant = variant
				self._bound_bind_groups.clear()

			self._variant_dirty = False

		assert self._variant is not None
		for group in getattr(self._variant, "empty_groups", ()):
			if group not in self.bind_groups:
				handle.set_bind_group(group, self._variant.empty_bind_group(group))
		# A retained BindGroup may refer to a resized buffer or replaced resource.
		for index, bindings in self.bind_groups.items():
			key = bindings.cache_key()
			if self._bound_bind_groups.get(index) == key: continue
			handle.set_bind_group(index, self._variant.bind_group(bindings))
			self._bound_bind_groups[index] = key
		return handle

	def _require_handle(self) -> wgpu.GPURenderPassEncoder:
		if self.handle is None: raise RuntimeError("RenderPass is not active")
		return self.handle


class ComputePass:
	def __init__(self, commands: CommandContext, *, label: str | None = None):
		self.commands = commands
		self.label = label
		self.handle: wgpu.GPUComputePassEncoder | None = None
		self.pipeline: ComputePipeline | None = None

	def __enter__(self) -> ComputePass:
		self.handle = self.commands.handle.begin_compute_pass(label=self.label or "")
		return self

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		self.end()

	def end(self) -> None:
		if self.handle is not None:
			self.handle.end()
			self.handle = None

	def set_pipeline(self, pipeline: ComputePipeline) -> None:
		self.pipeline = pipeline
		self._require_handle().set_pipeline(pipeline.handle)
		for group in pipeline._variant.empty_groups:
			self._require_handle().set_bind_group(group, pipeline._variant.empty_bind_group(group))

	def set_bind_group(self, bindings: BindGroup) -> None:
		if self.pipeline is None:
			raise RuntimeError("Set the ComputePipeline before its bind groups")
		self._require_handle().set_bind_group(bindings.group, self.pipeline._variant.bind_group(bindings))

	def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
		self._require_handle().dispatch_workgroups(x, y, z)

	def dispatch_indirect(self, buffer: GpuBuffer, offset: int = 0) -> None:
		self._require_handle().dispatch_workgroups_indirect(buffer.handle, offset)

	def _require_handle(self) -> wgpu.GPUComputePassEncoder:
		if self.handle is None: raise RuntimeError("ComputePass is not active")
		return self.handle


@dataclass(frozen=True)
class ColorAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: Color | None = None
	store: bool = True
	sample_count: int = 1
	resolve_target: wgpu.GPUTextureView | None = None
	device_generation: int | None = None

	def descriptor(self) -> wgpu.RenderPassColorAttachment:
		return wgpu.RenderPassColorAttachment(
			view=self.view,
			resolve_target=self.resolve_target,
			clear_value=self.clear or (0.0, 0.0, 0.0, 0.0),
			load_op="clear" if self.clear is not None else "load",
			store_op="store" if self.store else "discard",
		)


@dataclass(frozen=True)
class DepthAttachment:
	view: wgpu.GPUTextureView
	format: str
	clear: float | None = None
	store: bool = True
	read_only: bool = False
	sample_count: int = 1
	device_generation: int | None = None

	def descriptor(self) -> wgpu.RenderPassDepthStencilAttachment:
		return wgpu.RenderPassDepthStencilAttachment(
			view=self.view,
			depth_clear_value=self.clear or 0.0,
			depth_load_op="clear" if self.clear is not None else "load",
			depth_store_op="store" if self.store else "discard",
			depth_read_only=self.read_only,
		)

# numpy buffers ---------------------------------------------------------------

def _element_count(value):
	if isinstance(value, (bool, np.bool_)): raise TypeError("Buffer counts must be integers")
	value = integer_index(value)
	if value < 0: raise ValueError("Buffer counts must be nonnegative")
	return value

_VertexFormat: dict[Any, tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
}


def dtype_to_vertex_format(dtype: np.dtype) -> list | str:
	dtype = np.dtype(dtype)
	if dtype.fields is not None:
		res = []
		for name in dtype.names or ():
			field_dtype, offset = dtype.fields[name][:2]
			vformat = dtype_to_vertex_format(field_dtype)
			if isinstance(vformat, str):
				res.append((vformat, offset))
			else:
				for vf, local_offset in vformat:
					res.append((vf, offset + local_offset))
		return res

	if dtype.subdtype is None:
		try:
			prefix, scalar_size = _VertexFormat[dtype.type]
		except KeyError:
			raise TypeError(f"Unsupported vertex scalar dtype: {dtype}") from None
		format = getattr(wgpu.VertexFormat, prefix, None)
		if format is None:
			raise TypeError(f"Unsupported vertex format: {dtype}")
		return format

	base_dtype, shape = dtype.subdtype
	if len(shape) != 1:
		raise TypeError("Vertex attributes must be one-dimensional vectors")
	base_type = base_dtype.type
	count = shape[0]
	if count < 1 or count > 4:
		raise ValueError(
			f"VertexFormat arrays must have 1-4 components (found {count} for {dtype.subdtype})"
		)

	try:
		prefix, scalar_size = _VertexFormat[base_type]
	except KeyError:
		raise TypeError(f"Unsupported vertex base dtype: {base_type}") from None

	vformat = getattr(wgpu.VertexFormat, prefix if count == 1 else f"{prefix}x{count}", None)
	if vformat is None:
		raise TypeError(f"Unsupported vertex format: {dtype}")
	return vformat


class GpuBuffer(_OwnedResource):
	def __init__(
		self,
		content: np.ndarray | None,
		usage: wgpu.BufferUsage,
		upload: bool = True,
		label: str | None = None,
		dtype: Any = None,
		count: int = 0,
		copy: bool = False,
	):
		self._capture_device()
		if content is None:
			count = _element_count(count)
			if dtype is None or count < 0: raise ValueError("GPU-only buffers require dtype and nonnegative count")
			self._dtype, self._count, self._content = np.dtype(dtype), count, None
			if self._dtype.hasobject: raise TypeError("GPU buffers cannot contain Python objects")
		else:
			self.content = np.array(content, copy=True, order="C") if copy else content
		self.usage = usage
		self.label = label
		self.generation = 0

		if upload and self.content is not None:
			self.handle = self._check_device().create_buffer_with_data(
				label=label or "",
				data=self.content,
				usage=self.usage,
			)
		else:
			self.handle = self._check_device().create_buffer(
				label=label or "",
				size=max(4, _round_up(4, self.binding_size)),
				usage=self.usage,
			)

	@property
	def content(self): return self._content

	@content.setter
	def content(self, value):
		self._content = self._content_array(value)
		self._dtype, self._count = self._content.dtype, self._content.size

	@property
	def dtype(self): return self._dtype

	@property
	def count(self): return self._count

	@property
	def binding_size(self): return self.count * self.dtype.itemsize

	@property
	def capacity_bytes(self):
		self._check_device()
		return self.handle.size

	def set_count(self, count: int):
		"""Set the active scalar/record count within allocated and mirrored capacity."""
		count = _element_count(count)
		if count < 0 or count * self.dtype.itemsize > self.capacity_bytes:
			raise ValueError("Active count exceeds GPU capacity")
		if self.content is not None and count > self.content.size:
			raise ValueError("Active count exceeds CPU mirror")
		self._count = count

	def reserve(self, count: int) -> bool:
		"""Grow capacity while preserving active data; do not change active count."""
		count = _element_count(count)
		self._check_device()
		if count * self.dtype.itemsize <= self.capacity_bytes: return False
		if self.content is None and self.binding_size and (self.usage & (BufferUsage.COPY_SRC | BufferUsage.COPY_DST)) != (BufferUsage.COPY_SRC | BufferUsage.COPY_DST):
			raise ValueError("GPU-only preservation requires COPY_SRC and COPY_DST")
		if self.content is not None and not self.usage & BufferUsage.COPY_DST:
			raise ValueError("CPU-backed preservation requires COPY_DST")
		old = self.handle
		self.resize(count)
		if self.content is not None: self.upload()
		elif self.binding_size:
			encoder = self._device.create_command_encoder(label="reserve_buffer")
			encoder.copy_buffer_to_buffer(old, 0, self.handle, 0, _round_up(4, self.binding_size))
			self._device.queue.submit([encoder.finish()])
		return True

	@property
	def binding(self) -> wgpu.BufferBinding:
		self._check_device()
		return wgpu.BufferBinding(
			buffer=self.handle,
			offset=0,
			size=self.binding_size,
		)

	@property
	def resource_key(self) -> tuple[wgpu.GPUBuffer, int]:
		self._check_device()
		return (self.handle, self.binding_size)
	

	def resize(self, count: int) -> bool:
		"""Grow GPU capacity, discarding GPU data; callers must re-upload content."""
		count = _element_count(count)
		if count < 0:
			raise ValueError("Buffer count cannot be negative")
		self._check_device()
		required_size = max(4, _round_up(4, count * self.dtype.itemsize))
		if required_size <= self.handle.size:
			return False
		size = higher_pow2(required_size - 1)

		self.handle = self._check_device().create_buffer(
			label=self.label or "",
			size=size,
			usage=self.usage,
		)
		self.generation += 1
		return True

	def upload(self) -> None:
		self._upload_bytes(0, self.binding_size)

	def upload_range(self, start: int, count: int) -> None:
		if self.content.ndim != 1:
			raise ValueError("Element range uploads require a one-dimensional array")
		if start < 0 or count < 0 or start + count > self.content.size:
			raise ValueError("Upload range is outside CPU content")
		self._upload_bytes(start * self.content.itemsize, count * self.content.itemsize)

	@staticmethod
	def _content_array(data: np.ndarray) -> np.ndarray:
		array = np.asarray(data)
		if array.dtype.hasobject:
			raise TypeError("GPU buffers cannot contain Python objects")
		return np.ascontiguousarray(array)

	def _upload_bytes(self, offset: int, size: int) -> None:
		self._check_device()
		if self.content is None: raise RuntimeError("GPU-only buffers have no CPU data to upload")
		if not size:
			return
		if not self.usage & wgpu.BufferUsage.COPY_DST:
			raise ValueError("Uploads require COPY_DST buffer usage")
		start, end = offset // 4 * 4, _round_up(4, offset + size)
		if end > self.handle.size:
			raise ValueError("Upload exceeds GPU capacity; resize first")
		# Preserve adjacent CPU bytes when uploading sub-word ranges.
		raw = self.content.view(np.uint8).reshape(-1)
		if end <= raw.size:
			data = raw[start:end]
		else:
			data = np.zeros(end - start, dtype=np.uint8)
			available = min(end, raw.size) - start
			data[:available] = raw[start:start + available]
		self._check_device().queue.write_buffer(self.handle, start, data)
		RenderContext.record_stat("upload_calls")
		RenderContext.record_stat("upload_bytes", data.nbytes)

	def write(self, data: np.ndarray) -> None:
		"""Replace CPU content and upload it, growing GPU capacity if necessary."""
		content = self._content_array(data)
		if not self.usage & wgpu.BufferUsage.COPY_DST:
			raise ValueError("Writes require COPY_DST buffer usage")
		self.content = content
		self.resize(content.size)
		self.upload()

	def write_range(self, data: np.ndarray, *, offset: int = 0) -> None:
		"""Update CPU/GPU content at a byte offset without changing binding size."""
		data = self._content_array(data).view(np.uint8).reshape(-1)
		if offset < 0 or offset + data.size > self.content.nbytes:
			raise ValueError("Write range is outside CPU content")
		if not self.usage & wgpu.BufferUsage.COPY_DST:
			raise ValueError("Writes require COPY_DST buffer usage")
		if _round_up(4, offset + data.size) > self.handle.size:
			raise ValueError("Write exceeds GPU capacity; resize first")
		self.content.view(np.uint8).reshape(-1)[offset:offset + data.size] = data
		self._upload_bytes(offset, data.size)


class GpuBufferPool:
	def __init__(
		self,
		dtype: np.dtype,
		usage: wgpu.BufferUsage,
		initial_count: int = 0,
	):
		self.dtype = np.dtype(dtype)
		self.used = 0
		self.buffer = GpuBuffer(
			np.empty(initial_count, dtype=self.dtype),
			usage | wgpu.BufferUsage.COPY_DST,
			upload=False,
		)

	def alloc(self, data: np.ndarray) -> tuple[int, int]:
		start = self.used
		count = len(data)
		end = start + count

		self._ensure_capacity(end)
		self.buffer.content[start:end] = data
		self.buffer.upload_range(start, count)
		self.used = end
		return start, count

	def _ensure_capacity(self, count: int) -> None:
		old_content = self.buffer.content
		if count <= old_content.size:
			return

		new_count = higher_pow2(max(1, count))
		new_content = np.empty(new_count, dtype=self.dtype)
		new_content[:self.used] = old_content[:self.used]
		self.buffer.content = new_content

		if self.buffer.resize(new_count):
			self.buffer.upload_range(0, self.used)


class Mesh:
	index_formats = {np.dtype(np.uint16): "uint16", np.dtype(np.uint32): "uint32"}

	def __init__(self, vertex_data: np.ndarray, indices: np.ndarray, *, arena=None):
		self.vertex_dtype, self.index_dtype = vertex_data.dtype, indices.dtype
		if self.index_dtype not in self.index_formats: raise TypeError(f"Unsupported index dtype: {self.index_dtype}")
		self.arena = arena if arena is not None else RenderContext.mesh_arena
		self._vertices = self.arena.allocate(vertex_data, BufferUsage.VERTEX)
		try:
			self._indices = self.arena.allocate(indices, BufferUsage.INDEX)
		except BaseException:
			self._vertices.release()
			raise
		self.index_format = self.index_formats[self.index_dtype]

	@property
	def vertex_buffer(self):
		self._vertices.check()
		return self._vertices.pool.buffer

	@property
	def index_buffer(self):
		self._indices.check()
		return self._indices.pool.buffer

	@property
	def vertex_range(self): return self._vertices.range

	@property
	def index_range(self): return self._indices.range

	@property
	def index_count(self): return self.index_range[1]

	@property
	def version(self):
		return self._vertices.generation, self._indices.generation

	@property
	def vertices(self):
		start, count = self.vertex_range
		return self._vertices.pool.content[start:start + count]

	@property
	def vertex_range_bytes(self):
		start, count = self.vertex_range
		return start * self.vertex_dtype.itemsize, count * self.vertex_dtype.itemsize

	def release(self):
		self._vertices.release()
		self._indices.release()


class Texture(_OwnedResource):
	def __init__(
		self,
		size: tuple[int, int] | tuple[int, int, int],
		*,
		format: str,
		usage: wgpu.TextureUsage,
		mip_level_count: int = 1,
		sample_count: int = 1,
		dimension: str = "2d",
		label: str | None = None,
	):
		self._capture_device()
		if len(size) == 2:
			size = (*size, 1)
		self.size = size
		self.format = format
		self.usage = usage
		self.sample_count = sample_count
		self._view: TextureView | None = None
		self.handle = self._check_device().create_texture(
			label=label or "",
			size=size,
			dimension=dimension,
			format=format,
			usage=usage,
			mip_level_count=mip_level_count,
			sample_count=sample_count,
		)

	@property
	def binding(self) -> wgpu.GPUTextureView:
		return self.view().handle

	def view(self, **kwargs) -> TextureView:
		self._check_device()
		if kwargs:
			return TextureView(self, self.handle.create_view(**kwargs))
		if self._view is None:
			self._view = TextureView(self, self.handle.create_view())
		return self._view

	def color_attachment(
		self,
		*,
		clear: Color | None = None,
		store: bool = True,
		resolve_target: Texture | TextureView | None = None,
	) -> ColorAttachment:
		self._check_device()
		return ColorAttachment(self.binding, self.format, clear, store, self.sample_count,
			resolve_target.binding if isinstance(resolve_target, Texture) else resolve_target.handle if resolve_target else None,
			self._device_generation)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		self._check_device()
		return DepthAttachment(self.binding, self.format, clear, store, read_only, self.sample_count, self._device_generation)


@dataclass(frozen=True)
class TextureView:
	texture: Texture
	handle: wgpu.GPUTextureView

	@property
	def format(self) -> str:
		return self.texture.format


class Sampler(_OwnedResource):
	def __init__(self, **descriptor: Any):
		self._capture_device()
		self.handle = self._check_device().create_sampler(**descriptor)



# WGSL shader metadata --------------------------------------------------------

@dataclass(frozen=True)
class ShaderAttribute:
	name: str
	value: str | None = None

	def int_value(self) -> int:
		if self.value is None:
			raise ValueError(f"Shader attribute @{self.name} has no value")
		return int(self.value)


@dataclass(frozen=True)
class ShaderField:
	name: str
	type_name: str
	attributes: tuple[ShaderAttribute, ...] = ()

	def attribute(self, name: str) -> ShaderAttribute | None:
		return next(
			(attribute for attribute in self.attributes if attribute.name == name),
			None,
		)


@dataclass(frozen=True)
class ShaderStruct:
	name: str
	fields: tuple[ShaderField, ...]


@dataclass(frozen=True)
class BindingInfo:
	name: str
	group: int
	binding: int
	storage: str | None
	access: str | None
	type_name: str


@dataclass(frozen=True)
class EntryPointInfo:
	name: str
	stage: str
	parameters: tuple[ShaderField, ...]
	return_type: str | None
	return_attributes: tuple[ShaderAttribute, ...]


@dataclass(frozen=True)
class UniformInfo:
	name: str
	group: int
	binding: int
	struct_name: str
	dtype: np.dtype


@dataclass(frozen=True)
class VertexInfo:
	name: str
	location: int
	type_name: str
	dtype: np.dtype


@dataclass(frozen=True)
class _WGSLType:
	dtype: np.dtype
	align: int
	size: int


_SCALAR_TYPES = {
	"f32": (np.dtype(np.float32), 4, 4),
	"i32": (np.dtype(np.int32), 4, 4),
	"u32": (np.dtype(np.uint32), 4, 4),
}

_SCALAR_ALIAS = {
	"f": "f32",
	"i": "i32",
	"u": "u32",
}

def _round_up(alignment: int, value: int) -> int:
	return (value + alignment - 1) // alignment * alignment

def _strip_wgsl_comments(source: str) -> str:
	source = re.sub(r"/\*[\s\S]*?\*/", "", source)
	return re.sub(r"//.*", "", source)

def _normalise_wgsl_type(type_name: str) -> str:
	type_name = re.sub(r"\s+", "", type_name)
	type_name = re.sub(
		r"\bvec([234])([fiu])\b",
		lambda m: f"vec{m.group(1)}<{_SCALAR_ALIAS[m.group(2)]}>",
		type_name,
	)
	type_name = re.sub(
		r"\bmat([234])x([234])f\b",
		lambda m: f"mat{m.group(1)}x{m.group(2)}<f32>",
		type_name,
	)
	return type_name


def _wgsl_vector_type(type_name: str) -> tuple[np.dtype, int] | None:
	match = re.fullmatch(r"vec([234])<([fiu]32)>", _normalise_wgsl_type(type_name))
	if match is None: return None
	return _SCALAR_TYPES[match.group(2)][0], int(match.group(1))


class ShaderSource:
	_ATTRIBUTE: ClassVar[str] = r"@\w+(?:\s*\([^)]*\))?"
	_ATTRIBUTES: ClassVar[str] = rf"(?:{_ATTRIBUTE}\s*)*"
	_PARENS: ClassVar[str] = r"(?:[^()]|\([^()]*\))*"

	_attribute_pattern: ClassVar[re.Pattern[str]] = re.compile(
		r"@(?P<name>\w+)(?:\s*\(\s*(?P<value>[^)]*?)\s*\))?"
	)

	_struct_pattern: ClassVar[re.Pattern[str]] = re.compile(
		r"\bstruct\s+(?P<name>\w+)\s*\{(?P<body>.*?)\}\s*;?",
		re.S,
	)

	_field_pattern: ClassVar[re.Pattern[str]] = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})(?P<name>\w+)\s*:\s*"
		rf"(?P<type>.+?)"
		rf"(?=,\s*(?:{_ATTRIBUTE}\s*)*\w+\s*:|,\s*$|\s*$)",
		re.S,
	)

	_binding_pattern: ClassVar[re.Pattern[str]] = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})"
		r"var(?:\s*<\s*(?P<storage>[^>]+?)\s*>)?\s+"
		r"(?P<name>\w+)\s*:\s*(?P<type>[^;]+?)\s*;",
		re.S,
	)

	_entry_pattern: ClassVar[re.Pattern[str]] = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})fn\s+(?P<name>\w+)\s*"
		rf"\((?P<parameters>{_PARENS})\)\s*"
		rf"(?:->\s*(?P<return_attributes>{_ATTRIBUTES})"
		r"(?P<return_type>[^{]+?))?\s*\{",
		re.S,
	)

	_parameter_pattern: ClassVar[re.Pattern[str]] = re.compile(
		rf"(?P<attributes>{_ATTRIBUTES})(?P<name>\w+)\s*:\s*"
		rf"(?P<type>.+?)" 
		rf"(?=,\s*(?:{_ATTRIBUTE}\s*)*\w+\s*:|\s*$)",
		re.S,
	)

	_storage_pattern: ClassVar[re.Pattern[str]] = re.compile(
		r"^\s*(?P<storage>\w+)(?:\s*,\s*(?P<access>\w+))?\s*$"
	)

	_dsl_pattern: ClassVar[re.Pattern[str]] = re.compile(
		rf"^[ \t]*<(?P<struct>\w+)>\s*(?P<attributes>{_ATTRIBUTES})"
		r"(?P<name>\w+)\s*:\s*(?P<type>[^;]+);\s*(?:\r?\n)?",
		re.S | re.M
	)


	def __init__(self, source: str):

		code = _strip_wgsl_comments(source)
		self.structs  = self._parse_structs(code)
		self.bindings = self._parse_bindings(code)
		self.entry_points = self._parse_entry_points(code)
		self.source = self._process_dsl(code)
		#print(self.source)
		self.uniforms = self._uniforms()

	def used_bindings(self, entries):
		"""Conservative function-call traversal for automatic pipeline layouts.

		Explicit ShaderInterface.active_bindings handles interfaces requiring
		more precise control, such as constants eliminating conditional code.
		"""
		functions = {}
		for match in re.finditer(r"\bfn\s+(\w+)\s*\(", self.source):
			start = self.source.find("{", match.end())
			if start < 0: continue
			depth, end = 1, start + 1
			while end < len(self.source) and depth:
				depth += (self.source[end] == "{") - (self.source[end] == "}")
				end += 1
			functions[match.group(1)] = set(re.findall(r"\b\w+\b", self.source[start:end]))
		pending, visited, used = list(filter(None, entries)), set(), set()
		while pending:
			name = pending.pop()
			if name in visited: continue
			visited.add(name)
			words = functions.get(name, set())
			used.update(words.intersection(self.bindings))
			pending.extend(words.intersection(functions) - visited)
		return {name: self.bindings[name] for name in used}

	@property
	def vertex_dtype(self) -> np.dtype | None:
		entries = [entry.name for entry in self.entry_points.values() if entry.stage == "vertex"]

		if not entries:
			return None

		dtype = self.vertex_dtype_for(entries[0])

		for entry in entries[1:]:
			if self.vertex_dtype_for(entry) != dtype:
				raise ValueError(
					"Shader vertex entry points use different input layouts; "
					"use vertex_dtype_for(entry_point)"
				)

		return dtype

	def vertex_dtype_for(self, entry_point: str) -> np.dtype:
		return np.dtype([
			(field.name, field.dtype)
			for field in self.vertex_inputs(entry_point)
		])

	def vertex_inputs(self, entry_point: str) -> tuple[VertexInfo, ...]:
		try:
			entry = self.entry_points[entry_point]
		except KeyError:
			raise ValueError(f"Shader has no entry point {entry_point!r}") from None

		if entry.stage != "vertex":
			raise ValueError(
				f"Shader entry point {entry_point!r} is not a vertex shader"
			)

		fields = []

		for parameter in entry.parameters:
			if location := parameter.attribute("location"):
				fields.append(self._vertex_info(parameter, location))
				continue

			struct = self.structs.get(parameter.type_name)

			if struct is None:
				continue

			for field in struct.fields:
				if location := field.attribute("location"):
					fields.append(self._vertex_info(field, location))

		fields.sort(key=lambda field: field.location)

		locations = [field.location for field in fields]

		if len(locations) != len(set(locations)):
			raise ValueError(
				f"Vertex entry point {entry_point!r} contains duplicate @location values"
			)

		return tuple(fields)

	def uniform_dtype(self, struct_name: str) -> np.dtype:
		return self._uniform_struct_dtype(struct_name)

	def get_stage_entry_points(self) -> dict[str, list[str]]:
		entries: dict[str, list[str]] = {"vertex": [], "fragment": [], "compute": []}

		for entry in self.entry_points.values():
			entries[entry.stage].append(entry.name)

		return entries

	def _parse_structs(self, source: str) -> dict[str, ShaderStruct]:
		structs = {}
		for struct_match in self._struct_pattern.finditer(source):
			struct_name = struct_match.group("name")
			fields = []
			
			for field_match in self._field_pattern.finditer(struct_match.group("body")):
				attrs = self._attributes(field_match.group("attributes") or "")
				fields.append( ShaderField(
					name=field_match.group("name"),
					type_name=field_match.group("type").strip(),
					attributes=tuple(attrs.values())
				))
			structs[struct_name] = ShaderStruct(struct_name, tuple(fields))
			
		return structs

	def _parse_bindings(self, source: str) -> dict[str, BindingInfo]:
		bindings = {}

		for match in self._binding_pattern.finditer(source):
			attributes = self._attributes(match.group("attributes"))

			group = attributes.get("group")
			binding = attributes.get("binding")

			if group is None or binding is None:
				continue

			storage, access = self._storage(match.group("storage"))
			name = match.group("name")

			bindings[name] = BindingInfo(
				name=name,
				group=group.int_value(),
				binding=binding.int_value(),
				storage=storage,
				access=access,
				type_name=self._type_name(match.group("type")),
			)

		return bindings


	def _parse_entry_points(self, source: str) -> dict[str, EntryPointInfo]:
		entry_points = {}

		for match in self._entry_pattern.finditer(source):
			attributes = self._attributes(match.group("attributes"))

			stage = next(
				(stage for stage in ("vertex", "fragment", "compute") if stage in attributes),
				None,
			)

			if stage is None:
				continue

			parameters = tuple(
				self._field(parameter)
				for parameter in self._parameter_pattern.finditer(match.group("parameters"))
			)

			name = match.group("name")
			return_type = match.group("return_type")

			entry_points[name] = EntryPointInfo(
				name=name,
				stage=stage,
				parameters=parameters,
				return_type=self._type_name(return_type) if return_type else None,
				return_attributes=tuple(
					self._attributes(match.group("return_attributes") or "").values()
				),
			)

		return entry_points


	def _process_dsl(self, source: str) -> str:
		dsl_fields = defaultdict(list)

		def dsl_replacer(match):
			struct_name = match.group("struct")
			attr_str = match.group("attributes") or ""
			field_name = match.group("name")
			wgsl_type = match.group("type").strip()
			
			attrs = self._attributes(attr_str)
			dsl_fields[struct_name].append((field_name, wgsl_type, attrs, attr_str))

			#return "\n" * match.group(0).count('\n')
			return ''

		source = self._dsl_pattern.sub(dsl_replacer, source)

		if not dsl_fields:
			return source

		injected_wgsl = []
		
		for struct_name, fields in dsl_fields.items():
			parsed_fields = []
			lines = [f"struct {struct_name} {{"]
			
			is_io_struct = "Input" in struct_name or "Output" in struct_name
			loc_idx = 0
			
			for field_name, wgsl_type, attrs, attr_str in fields:
				
				# auto-assign @location for IO structs
				if is_io_struct:
					if "builtin" not in attrs and "location" not in attrs:
						attrs["location"] = ShaderAttribute("location", str(loc_idx))
						attr_str = f"@location({loc_idx}) {attr_str}".strip()
						loc_idx += 1
					elif "location" in attrs:
						loc_idx = attrs["location"].int_value() + 1
						
				lines.append(f"	{attr_str}{' ' if attr_str else ''}{field_name}: {wgsl_type},")
				parsed_fields.append(ShaderField(
					name=field_name,
					type_name=wgsl_type,
					attributes=tuple(attrs.values())
				))
				
			lines.append("};")
			injected_wgsl.append("\n".join(lines))
			
			self.structs[struct_name] = ShaderStruct(struct_name, tuple(parsed_fields))

		return "\n\n".join(injected_wgsl) + "\n\n" + source

	def _storage(self, text: str | None) -> tuple[str | None, str | None]:
		if text is None:
			return None, None

		match = self._storage_pattern.fullmatch(text)

		if match is None:
			raise ValueError(f"Cannot parse WGSL address space {text!r}")

		return match.group("storage"), match.group("access")

	def _field(self, match: re.Match) -> ShaderField:
		return ShaderField(
			name=match.group("name"),
			type_name=self._type_name(match.group("type")),
			attributes=tuple(self._attributes(match.group("attributes")).values()),
		)

	def _attributes(self, text: str) -> dict[str, ShaderAttribute]:
		return {
			match.group("name"): ShaderAttribute(
				match.group("name"),
				match.group("value"),
			)
			for match in self._attribute_pattern.finditer(text)
		}

	@staticmethod
	def _type_name(type_name: str) -> str:
		return re.sub(r"\s+", "", type_name)

	def _uniforms(self) -> dict[str, UniformInfo]:
		uniforms = {}

		for binding in self.bindings.values():
			if binding.storage != "uniform":
				continue

			if binding.type_name not in self.structs:
				raise TypeError(
					f"Uniform {binding.name!r} must refer to a named WGSL struct"
				)

			uniforms[binding.name] = UniformInfo(
				name=binding.name,
				group=binding.group,
				binding=binding.binding,
				struct_name=binding.type_name,
				dtype=self._uniform_struct_dtype(binding.type_name),
			)

		return uniforms

	def _uniform_struct_dtype(
		self,
		name: str,
		stack: tuple[str, ...] = (),
	) -> np.dtype:
		if name in stack:
			raise TypeError(
				f"Recursive WGSL struct {name!r} cannot be a uniform dtype"
			)

		try:
			struct = self.structs[name]
		except KeyError:
			raise TypeError(f"Unknown WGSL struct {name!r}") from None

		names = []
		formats = []
		offsets = []

		offset = 0
		struct_align = 16

		for field in struct.fields:
			field_type = self._uniform_type(field.type_name, stack + (name,))

			align = (
				max(16, field_type.align)
				if field.type_name in self.structs
				else field_type.align
			)

			offset = _round_up(align, offset)

			names.append(field.name)
			formats.append(field_type.dtype)
			offsets.append(offset)

			offset += field_type.size
			struct_align = max(struct_align, align)

		return np.dtype({
			"names": names,
			"formats": formats,
			"offsets": offsets,
			"itemsize": _round_up(struct_align, offset),
		})

	def _uniform_type(
		self,
		type_name: str,
		stack: tuple[str, ...],
	) -> _WGSLType:
		type_name = _normalise_wgsl_type(type_name)

		if type_name in _SCALAR_TYPES:
			dtype, align, size = _SCALAR_TYPES[type_name]
			return _WGSLType(dtype, align, size)

		if vector := _wgsl_vector_type(type_name):
			base, count = vector
			scalar_size = base.itemsize
			align = 8 if count == 2 else 16

			return _WGSLType(
				np.dtype((base, (count,))),
				align,
				scalar_size * count,
			)

		if matrix := re.fullmatch(r"mat([234])x([234])<f32>", type_name):
			columns, rows = map(int, matrix.groups())
			column = self._uniform_type(f"vec{rows}<f32>", stack)
			stride = _round_up(column.align, column.size)

			if stride != column.size:
				raise TypeError(
					f"{type_name} requires padding inside matrix columns; "
					"this uniform layout is not implemented yet"
				)

			return _WGSLType(
				np.dtype((np.float32, (columns, rows))),
				column.align,
				stride * columns,
			)

		if array := re.fullmatch(r"array<(.+),(\d+)>", type_name):
			member = self._uniform_type(array.group(1), stack)
			count = int(array.group(2))

			align = max(16, member.align)
			stride = _round_up(align, member.size)

			if stride != member.dtype.itemsize:
				raise TypeError(
					f"{type_name} requires per-element padding; "
					"this uniform layout is not implemented yet"
				)

			return _WGSLType(
				np.dtype((member.dtype, (count,))),
				align,
				stride * count,
			)

		if type_name in self.structs:
			dtype = self._uniform_struct_dtype(type_name, stack)
			return _WGSLType(dtype, 16, dtype.itemsize)

		raise TypeError(f"Unsupported WGSL uniform type {type_name!r}")

	def _vertex_info(
		self,
		field: ShaderField,
		location: ShaderAttribute,
	) -> VertexInfo:
		return VertexInfo(
			name=field.name,
			location=location.int_value(),
			type_name=field.type_name,
			dtype=self._vertex_type_dtype(field.type_name),
		)

	def _vertex_type_dtype(self, type_name: str) -> np.dtype:
		type_name = _normalise_wgsl_type(type_name)

		if type_name in _SCALAR_TYPES:
			return _SCALAR_TYPES[type_name][0]

		if vector := _wgsl_vector_type(type_name):
			base, count = vector
			return np.dtype((base, (count,)))

		raise TypeError(f"Unsupported WGSL vertex type {type_name!r}")


@dataclass
class ShaderInterface:
	"""Explicit reflection replacement. entries maps entry-point names to stages.

	Use BindingInfo/UniformInfo and explicit NumPy dtypes for layouts the built-in
	reflector cannot represent. All offsets and strides remain caller-specified.
	"""
	entries: dict[str, str]
	bindings: dict[str, BindingInfo]
	uniforms: dict[str, UniformInfo]
	inputs: dict[str, tuple[VertexInfo, ...]]
	source: str = ""
	active_bindings: dict[str, tuple[str, ...]] | None = None

	@property
	def entry_points(self):
		return {name: EntryPointInfo(name, stage, (), None, ()) for name, stage in self.entries.items()}

	def vertex_inputs(self, name): return self.inputs.get(name, ())

	def used_bindings(self, entries):
		if self.active_bindings is None: return self.bindings
		names = {name for entry in entries if entry for name in self.active_bindings.get(entry, ())}
		return {name: self.bindings[name] for name in names}


class Shader(_OwnedResource):

	def __init__(
		self,
		*,
		source: str | None = None,
		filepath: str | None = None,
		basedir: str | None = None,
		features: Sequence[str] | None = None,
		params: dict[str, Any] | None = None,
		label: str | None = None,
		interface: ShaderInterface | None = None,
		required_features: Sequence[str] = (),
	):
		self._capture_device()
		if required_features: RenderContext.require_features(required_features)
		if source is None and filepath is None:
			raise ValueError("Shader requires source= or filepath=")

		source = self._load_source(
			source,
			filepath,
			basedir,
			features,
			params,
		)

		self.info = ShaderSource(source) if interface is None else deepcopy(interface)
		if interface is not None: self.info.source = source

		try:
			self.module = self._check_device().create_shader_module(
				label=label or filepath or "shader",
				code=self.info.source,
			)
		except Exception as ex:
			print(self.info.source)
			raise ex from None

	def uniform_buffer(self, name: str | None = None) -> _UniformBuffer:
		"""Allocate a reflected uniform block; infer its name when there is only one."""
		self._check_device()
		if not self.info.uniforms:
			raise ValueError( "no uniforms were parsed" )

		if name is None:
			if len(self.info.uniforms) > 1:
				raise ValueError( "Uniform name is required when a shader has multiple uniform blocks" )

			name = next(iter(self.info.uniforms))

		return _UniformBuffer(self.info.uniforms[name], label=name)


	def bind_group(self, group: int, **resources: Any) -> BindGroup:
		self._check_device()
		return BindGroup(self, group, resources)

	def entry(
		self,
		stage: str,
		name: str | None = None,
	) -> str | None:
		entries = [
			entry.name for entry in self.info.entry_points.values()
			if entry.stage == stage and (name is None or entry.name == name)
		]

		if not entries:
			return None

		if len(entries) > 1:
			raise ValueError(
				f"Shader has multiple {stage} entry points; "
				"specify one explicitly"
			)

		return entries[0]

	def _load_source(
		self,
		source: str | None,
		filepath: str | None,
		basedir: str | None,
		features: Sequence[str] | None,
		params: dict[str, Any] | None,
	) -> str:
		if source is not None:
			text = source
		else:
			if filepath is None:
				raise ValueError("Shader requires source= or filepath=")
			with open(filepath, "r") as file:
				text = file.read()

		base = (
			basedir
			if basedir
			else [os.path.dirname(os.path.abspath(filepath)), os.path.join(os.path.dirname(__file__), "shaders")]
			if filepath
			else [os.path.join(os.path.dirname(__file__), "shaders")]
		)

		env = Environment(
			loader=FileSystemLoader(base),
			undefined=StrictUndefined,
			autoescape=False,
			keep_trailing_newline=True,
			trim_blocks=True,
			lstrip_blocks=True,
			line_statement_prefix="#",
		)

		template = env.from_string(text)
		feature_dict = DefaultFalseDict()

		if features is not None:
			feature_dict.update({
				feature: True
				for feature in features
			})

		rendered_wgsl = template.render(
			FEATURES=feature_dict,
			PARAMS=params or {},
		)
		
		return rendered_wgsl


class DefaultFalseDict(dict):
	def __missing__(self, key):
		return False


class _UniformBuffer(GpuBuffer):
	def __init__(self, info: UniformInfo, label="uniform"):
		self.info = info

		super().__init__(
			np.zeros(1, dtype=info.dtype),
			wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
			label=label
		)


# pipelines / bindings --------------------------------------------------------

class BindGroup:
	def __init__(
		self,
		shader: Shader,
		group: int,
		resources: dict[str, Any],
	):
		self.shader = shader
		self.group = group
		self.resources = resources

		for name in resources:
			binding = shader.info.bindings.get(name)
			if binding is None:
				raise KeyError(f"Shader has no binding named {name!r}")
			if binding.group != group:
				raise ValueError(
					f"Binding {name!r} belongs to group {binding.group}, not {group}"
				)

		self._bindings = tuple(sorted(
			(shader.info.bindings[name].binding, name) for name in resources
		))

	def cache_key(self) -> tuple:
 		return (
 			self.group,
 			tuple(
				(binding, self._resource_key(self.resources[name]))
				for binding, name in self._bindings
			),
		)
 
	def entries(self) -> list[wgpu.BindGroupEntry]:
		return [
			wgpu.BindGroupEntry(binding=binding, resource=self._resource(self.resources[name]))
			for binding, name in self._bindings
		]

	def _resource_key(self, resource: Any) -> Any:
		key = getattr(resource, "resource_key", None)
		if key is not None: return key
		resource = self._resource(resource)
		if isinstance(resource, dict):
			return tuple((name, self._resource_key(value)) for name, value in sorted(resource.items()))
		# Retain the GPU object itself: Python ids may be reused after collection.
		return resource

	def _resource(self, resource: Any) -> Any:
		if isinstance(resource, _OwnedResource): resource._check_device()
		if isinstance(resource, TextureView): resource.texture._check_device()
		if bind := getattr(resource, "binding", None):
			return bind
		if handle := getattr(resource, "handle", None):
			return handle
		return resource

class _PipelineVariant(_OwnedResource):
	# Bound retained GPU resources across buffer growth and texture replacement.
	_BIND_GROUP_CACHE_LIMIT = 128

	def __init__(
		self,
		handle: wgpu.GPURenderPipeline | wgpu.GPUComputePipeline,
		bindings: dict[str, BindingInfo] | None = None,
	):
		self._capture_device()
		self.handle = handle
		self._bind_groups: OrderedDict[tuple, wgpu.GPUBindGroup] = OrderedDict()
		self._lock = Lock()
		self.binding_slots = None if bindings is None else {(info.group, info.binding) for info in bindings.values()}
		groups = {info.group for info in (bindings or {}).values()}
		self.empty_groups = tuple(group for group in range(max(groups, default=-1) + 1) if group not in groups)

	def empty_bind_group(self, group):
		self._check_device()
		key = ("empty", group)
		if key not in self._bind_groups:
			self._bind_groups[key] = self._check_device().create_bind_group(
				layout=self.handle.get_bind_group_layout(group), entries=[])
		return self._bind_groups[key]

	def bind_group(self, bindings: BindGroup) -> wgpu.GPUBindGroup:
		self._check_device()
		key = bindings.cache_key()
		with self._lock:
			cached = self._bind_groups.get(key)
			if cached is not None:
				RenderContext.record_stat("bind_group_cache_hits")
				self._bind_groups.move_to_end(key)
				return cached
			cached = self._check_device().create_bind_group(
				layout=self.handle.get_bind_group_layout(bindings.group),
				entries=[entry for entry in bindings.entries() if self.binding_slots is None or (bindings.group, entry["binding"]) in self.binding_slots],
			)
			self._bind_groups[key] = cached
			RenderContext.record_stat("bind_group_cache_misses")
			if len(self._bind_groups) > self._BIND_GROUP_CACHE_LIMIT:
				self._bind_groups.popitem(last=False)
			return cached


class RenderPipeline(_OwnedResource):
	"""Render pipeline specification.

	A concrete GPURenderPipeline is cached for each combination of inferred vertex
	layouts and render-target formats. The user never has to specify those layouts.
	Shader and pipeline settings are construction-time values; create a new
	RenderPipeline to change them. Do not mutate blend settings after construction.
	"""

	def __init__(
		self,
		shader: Shader,
		*,
		vertex_entry: str | None = None,
		fragment_entry: str | None = "auto",
		topology: str = wgpu.PrimitiveTopology.triangle_list,
		cull_mode: str = wgpu.CullMode.back,
		front_face: str = wgpu.FrontFace.ccw,
		depth_test: str | None = wgpu.CompareFunction.less,
		depth_write: bool = True,
		depth_bias: int = 0,
		depth_bias_slope_scale: float = 0.0,
		blend: wgpu.BlendState | None = None,
		label: str | None = None,
		targets: Sequence[dict[str, Any]] | None = None,
		sample_count: int = 1,
		layout: Any = "auto",
		depth_stencil: dict[str, Any] | None = None,
		vertex_formats: dict[str, str] | None = None,
		required_features: Sequence[str] = (),
	):
		if sample_count not in (1, 4): raise ValueError("sample_count must be 1 or 4")
		if required_features: RenderContext.require_features(required_features)
		self.targets = tuple(deepcopy(target) for target in targets) if targets is not None else None
		self.sample_count, self.layout = sample_count, layout
		self.depth_stencil = deepcopy(depth_stencil) if depth_stencil is not None else None
		self.vertex_formats = dict(vertex_formats or {})
		self.shader = shader
		if hasattr(shader, '_device'):
			self._device, self._device_generation = shader._device, shader._device_generation
		self.vertex_entry = shader.entry("vertex", vertex_entry)
		# "auto" infers a sole fragment entry; None disables it for depth passes.
		self.fragment_entry = None if fragment_entry is None else shader.entry("fragment", None if fragment_entry == "auto" else fragment_entry)
		self.topology = topology
		self.cull_mode = cull_mode
		self.front_face = front_face
		self.depth_test = depth_test
		self.depth_write = depth_write
		self.depth_bias = depth_bias
		self.depth_bias_slope_scale = depth_bias_slope_scale
		self.blend = blend
		self.label = label

		if self.vertex_entry is None:
			raise ValueError("RenderPipeline shader has no @vertex entry point")

		vertex_inputs = shader.info.vertex_inputs(self.vertex_entry)

		self.vertex_inputs = {
			field.name: field
			for field in vertex_inputs
		}

		if len(self.vertex_inputs) != len(vertex_inputs):
			raise ValueError(
				f"Vertex entry point {self.vertex_entry!r} contains "
				"duplicate input field names"
			)

		self._cache: dict[tuple, _PipelineVariant] = {}
		self._cache_lock = Lock()


	def get_variant(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> _PipelineVariant:
		self._check_device()
		key = (
			tuple(
				(np.dtype(dtype), step_mode)
				for dtype, step_mode in buffers
			),
			render_pass.color_formats,
			render_pass.depth_format,
			getattr(render_pass, "sample_count", 1),
		)

		cached = self._cache.get(key)
		if cached is not None:
			RenderContext.record_stat("pipeline_cache_hits")
			return cached

		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is None:
				RenderContext.record_stat("pipeline_cache_misses")
				cached = _PipelineVariant(
					self._make_pipeline(render_pass, buffers),
					self.shader.info.used_bindings((self.vertex_entry, self.fragment_entry)) if self.layout == "auto" else self.shader.info.bindings,
				)
				self._cache[key] = cached

		return cached

	def _make_pipeline(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> wgpu.GPURenderPipeline:
		if getattr(render_pass, "sample_count", 1) != self.sample_count:
			raise ValueError("Pipeline and render attachments must have matching sample counts")
		if self.targets is not None and len(self.targets) != len(render_pass.color_formats):
			raise ValueError("Color target settings must match the render attachment count")
		provided_vertex_input = set()
		layouts = []

		for dtype, step_mode in buffers:
			if dtype.fields is None:
				raise TypeError(f"Vertex buffer dtype must be structured, got {dtype}")

			attributes = []

			for name in dtype.names or ():
				field_dtype, offset = dtype.fields[name][:2]
				shader_input = self.vertex_inputs.get(name)

				if shader_input is None:
					continue

				if name in provided_vertex_input:
					raise ValueError(
						f"Vertex attribute {name!r} is provided by multiple buffers"
					)

				field_dtype = np.dtype(field_dtype)
				shader_dtype = shader_input.dtype


				field_size = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
				shader_size = int(np.prod(shader_dtype.shape)) if shader_dtype.shape else 1

				vertex_format = self.vertex_formats.get(name)
				if vertex_format is not None:
					match = re.fullmatch(r"([us])norm(8|16)x([24])", vertex_format)
					if match:
						kind, bits, count = match.groups()
						if (field_dtype.base.kind != ("u" if kind == "u" else "i") or field_dtype.base.itemsize * 8 != int(bits) or field_size != int(count) or shader_dtype.base.kind != "f"):
							raise TypeError(f"Vertex field {name!r} does not match normalized format {vertex_format}")
					elif vertex_format != dtype_to_vertex_format(field_dtype):
						raise TypeError(f"Vertex field {name!r} does not match format {vertex_format}")
				if (
					field_size != shader_size
					or (vertex_format is None and field_dtype.base.kind != shader_dtype.base.kind)
				):
					raise TypeError(
						f"Vertex attribute {name!r} has dtype {field_dtype}; "
						f"shader expects {shader_input.type_name}"
					)

				attributes.append(wgpu.VertexAttribute(
					shader_location=shader_input.location,
					offset=offset,
					format=vertex_format or dtype_to_vertex_format(field_dtype),
				))
				provided_vertex_input.add(name)

			layouts.append(wgpu.VertexBufferLayout(
				array_stride=dtype.itemsize,
				step_mode=step_mode,
				attributes=attributes,
			))

		missing = self.vertex_inputs.keys() - provided_vertex_input
		if missing:
			raise ValueError(
				f"Vertex shader {self.vertex_entry!r} requires attributes "
				f"not provided by the draw buffers: {', '.join(sorted(missing))}"
			)

		fragment = None

		#print(render_pass.handle.label, self.vertex_entry, self.fragment_entry)

		if render_pass.color_formats and self.fragment_entry is None:
			raise ValueError("Rendering to a color target requires an @fragment entry point")

		# A depth-only pass may still need fragment alpha tests and discard.
		# fragment_entry=None explicitly requests a vertex-only depth pass.
		if self.fragment_entry is not None:
			targets = []
			for index, format in enumerate(render_pass.color_formats):
				settings = {"blend": self.blend} if self.targets is None else dict(self.targets[index])
				if settings.pop("format", format) != format: raise ValueError("Configured target format does not match render attachment")
				targets.append(wgpu.ColorTargetState(format=format, **settings))
			fragment = wgpu.FragmentState(
				module=self.shader.module,
				entry_point=self.fragment_entry,
				targets=targets,
			)

		depth_stencil = None

		if self.depth_test is not None:
			if render_pass.depth_format is None:
				raise ValueError(
					"Pipeline has depth testing enabled but pass has no depth target"
				)

			depth_stencil = wgpu.DepthStencilState(
				format=render_pass.depth_format,
				depth_write_enabled=self.depth_write,
				depth_compare=self.depth_test,
				depth_bias=self.depth_bias,
				depth_bias_slope_scale=self.depth_bias_slope_scale,
			)
		if self.depth_stencil is not None:
			if render_pass.depth_format is None: raise ValueError("Depth/stencil settings require a depth target")
			settings = dict(self.depth_stencil)
			if settings.pop("format", render_pass.depth_format) != render_pass.depth_format:
				raise ValueError("Configured depth/stencil format does not match attachment")
			depth_stencil = dict(format=render_pass.depth_format, **settings)

		return self._check_device().create_render_pipeline(
			label=self.label or "",
			layout=self.layout,
			multisample={"count": self.sample_count, "mask": 0xffffffff, "alpha_to_coverage_enabled": False},
			vertex=wgpu.VertexState(
				module=self.shader.module,
				entry_point=self.vertex_entry,
				buffers=layouts,
			),
			primitive=wgpu.PrimitiveState(
				topology=self.topology,
				cull_mode=self.cull_mode,
				front_face=self.front_face,
			),
			depth_stencil=depth_stencil,
			fragment=fragment,
		)


class ComputePipeline(_OwnedResource):
	def __init__(
		self,
		shader: Shader,
		*,
		entry: str | None = None,
		label: str | None = None,
		layout: Any = "auto",
		required_features: Sequence[str] = (),
	):
		if required_features: RenderContext.require_features(required_features)
		self.shader = shader
		if hasattr(shader, '_device'):
			self._device, self._device_generation = shader._device, shader._device_generation
		self.entry = shader.entry("compute", entry)

		if self.entry is None:
			raise ValueError("ComputePipeline shader has no @compute entry point")

		handle = self._check_device().create_compute_pipeline(
			label=label or "",
			layout=layout,
			compute=wgpu.ProgrammableStage(
				module=shader.module,
				entry_point=self.entry,
			),
		)
		self._variant = _PipelineVariant(handle,
			shader.info.used_bindings((self.entry,)) if layout == "auto" else shader.info.bindings)

	@property
	def handle(self) -> wgpu.GPUComputePipeline:
		self._check_device()
		return self._variant.handle




# compatibility-ish helpers ---------------------------------------------------

def create_depth_framebuffer(
	width: int,
	height: int,
	format: str = wgpu.TextureFormat.depth24plus,
) -> Texture:
	return Texture(
		(width, height),
		format=format,
		usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
	)


def create_depth_sampler() -> Sampler:
	return Sampler(
		address_mode_u=wgpu.AddressMode.clamp_to_edge,
		address_mode_v=wgpu.AddressMode.clamp_to_edge,
		address_mode_w=wgpu.AddressMode.clamp_to_edge,
		compare=wgpu.CompareFunction.less_equal,
		min_filter=wgpu.FilterMode.linear,
		mag_filter=wgpu.FilterMode.linear,
	)
