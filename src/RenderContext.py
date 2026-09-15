from __future__ import annotations

import time, os, re, gc
from time import sleep, perf_counter as get_time
from threading import Lock
from dataclasses import dataclass
from typing import Any, Sequence, Iterator, Iterable, Callable, ParamSpec, TypeVar, ClassVar
from collections import defaultdict, OrderedDict

import glfw  # type: ignore[import-untyped]
import atexit
import wgpu  # type: ignore[import-untyped]
from wgpu.utils.glfw_present_info import get_glfw_present_info  # type: ignore[import-untyped]
from wgpu import BufferUsage, TextureUsage  # type: ignore[import-untyped]

import numpy as np

from glob import glob
from jinja2 import Environment, FileSystemLoader, StrictUndefined


Color = tuple[float, float, float, float]

# useful for buffer resizing
# 0->1, 1->2, 2->4, 3->4, 4->8, 5->8
def higher_pow2(n: int | np.uint32 | np.uint64) -> int:
	return 1 << int(n).bit_length()


class _RenderContext:
	"""Application loop singleton on top of glfw & wgpu                                                                          
	inspired by https://github.com/pygfx/rendercanvas/blob/main/rendercanvas/glfw.py                                             
	"""
	def __init__(self) -> None:
		self.canvas: wgpu.GPUCanvasContext | None = None
		self.adapter: wgpu.GPUAdapter
		self.device: wgpu.GPUDevice
		self.presentation_format: str

		self.window: Any = None
		self.windowDimensions: tuple[int, int] = (0, 0)
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

	def init_window(
		self,
		w: int,
		h: int,
		title: str,
		highpower_gpu: bool = True,
		target_fps: int = 0,
		required_gpu_features:list = [],
		custom_gc:bool=True
	) -> None:
		"""Init a window and wgpu rendering context.

		:param highpower_gpu: use the high performance gpu if there are multiple
		:param target_fps: -1 is limitless, 0 is vsync, other values is fps limit
		:param required_gpu_features: wgpu features requested                                                                
		:param custom_gc: run the gc during idle time
		"""
		glfw.init()
		atexit.register(self.cleanup)

		glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

		monitor = glfw.get_primary_monitor()
		monitor_x, monitor_y, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
		glfw.window_hint(glfw.POSITION_X, monitor_w - w)
		glfw.window_hint(glfw.POSITION_Y, 30)

		if target_fps == 0:
			video_mode = glfw.get_video_mode(monitor)
			target_fps = video_mode.refresh_rate

		if target_fps != -1:
			self.target_frame_time = 1.0 / float(target_fps)

		self.window = glfw.create_window(w, h, title, None, None)
		self.setup_callbacks()
		if custom_gc:
			self.event_handlers["idle"].append(GC_Manager())

		self.setup_graphics(vsync=target_fps == 0, highpower_gpu=highpower_gpu, required_gpu_features=required_gpu_features)

		self.frame_start = get_time()

	def setup_graphics(self, *, vsync: bool, highpower_gpu: bool, required_gpu_features:list=[]) -> None:
		"""Setup gpu compute & render surface."""
		present_info = get_glfw_present_info(self.window, vsync=vsync)
		self.canvas = wgpu.gpu.get_canvas_context(present_info)
		self.setup_graphics_backend(highpower_gpu, required_gpu_features)
		self.presentation_format = self.canvas.get_preferred_format(self.adapter)
		self.canvas.configure(device=self.device, format=self.presentation_format)

	def setup_graphics_backend(self, highpower_gpu: bool, required_features:list=[]) -> None:
		"""Setup gpu compute, not necessarily with canvas output. Used notably for tests."""
		request_params = {
			"power_preference": "high-performance" if highpower_gpu else "low-power"
		}
		if self.canvas:
			request_params["canvas"] = self.canvas
		self.adapter = wgpu.gpu.request_adapter_sync(**request_params)
		self.device = self.adapter.request_device_sync(required_features=required_features)

	def setup_callbacks(self) -> None:
		glfw.set_key_callback(self.window, self.setup_glfw_event("key"))
		glfw.set_char_callback(self.window, self.setup_glfw_event("char"))
		glfw.set_mouse_button_callback(self.window, self.setup_glfw_event("mouse_click"))
		glfw.set_cursor_pos_callback(self.window, self.setup_glfw_event("mouse_move"))
		glfw.set_cursor_enter_callback(self.window, self.setup_glfw_event("mouse_enter"))
		glfw.set_scroll_callback(self.window, self.setup_glfw_event("mouse_scroll"))
		self.idle_hook = self.setup_event("idle")

	def update_window_size(self, wh: tuple[int, int]) -> None:
		w, h = wh
		self.windowDimensions = wh
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
		if self.canvas:
			self.canvas.unconfigure()

		if self.window is not None:
			glfw.destroy_window(self.window)

		# work around https://github.com/glfw/glfw/issues/1766
		end_time = get_time() + 0.1
		while get_time() < end_time:
			glfw.wait_events_timeout(end_time - get_time())
		glfw.terminate()

	def window_loop(self) -> bool:
		"""Present the previous image, update window/input state, run gc, and continue the loop. """
		self.present()

		wh = glfw.get_framebuffer_size(self.window)
		if wh != self.windowDimensions and wh[0] > 0 and wh[1] > 0:
			self.update_window_size(wh)

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

		self.frame_start = now

	# commands -----------------------------------------------------------------

	def commands(self, label: str | None = None) -> CommandContext:
		return CommandContext(label=label)

	def command(self, label: str | None = None) -> wgpu.GPUCommandEncoder:
		"""Raw command encoder escape hatch."""
		return self.device.create_command_encoder(label=label or "")

	def submit(self, *command_buffers: wgpu.GPUCommandBuffer) -> None:
		self.device.queue.submit(list(command_buffers))

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
		return ColorAttachment(self._screen_view, self.presentation_format, clear, store)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(self.depth, self.depth_format, clear, store, read_only)

RenderContext = _RenderContext()

class GC_Manager:
	""" decides whether to run gc, called by 'idle' hook
	automatically instanciated by RenderContext.init_window by default
	"""

	def __init__(self,
		thresholds  : tuple[int,int,int]         = (300, 15, 10),
		time_budgets: tuple[float, float, float] = (0.0005, 0.002, 0.005), # sec
	) -> None:

		# collect then freeze, assuming initial window setup is there to stay 
		gc.collect(0)
		gc.freeze()

		gc.disable()
		gc.set_threshold(*thresholds)
		self.time_budgets = time_budgets

	def __call__(self):
		counts = gc.get_count()
		thresholds = gc.get_threshold()

		#print(f'{counts=}, {thresholds=}')

		max_gen = -1
		
		if RenderContext.target_frame_time:
			budget = RenderContext.target_frame_time - (get_time() - RenderContext.frame_start)
			for gen in range(max_gen, -1, -1):
				if budget >= self.time_budgets[gen]:
					max_gen = gen
					break
		else:
			max_gen = 2 # there are 3 generations : 0, 1, 2

		#print(f'{max_gen=}')
		for gen in range(max_gen, -1, -1):
			if counts[gen] >= thresholds[gen]:
				gc.collect(gen)
				print("collected gen", gen)
				break

	@classmethod
	def full_sweep() -> None:
		gc.unfreeze()
		gc.collect(2)
		gc.freeze()

	@classmethod
	def scene_transition(new_scene_load: Callable) -> None:
		""" collects old scene, mark new scene for keeping """
		gc.unfreeze()
		gc.collect(2)  # Sweeps old scene
		load()
		gc.collect(0)  # Sweeps temporary load junk
		gc.collect(0)  # called twice since some stuff survives single sweeps
		gc.freeze()    # exclude new scene from sweeps


class CommandContext:
	"""One independent GPU command encoder.

	Multiple CommandContexts may be encoded independently and later submitted in the
	order in which their resulting command buffers should execute.
	"""

	def __init__(self, label: str | None = None):
		self.handle = RenderContext.device.create_command_encoder(label=label or "")
		self.finished = False

	def render_pass(
		self,
		*,
		color: ColorAttachment | Sequence[ColorAttachment] = (),
		depth: DepthAttachment | None = None,
		label: str | None = None,
	) -> RenderPass:
		return RenderPass(self, color=color, depth=depth, label=label)

	def compute_pass(self, *, label: str | None = None) -> ComputePass:
		return ComputePass(self, label=label)

	def clear_buffer(
		self,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
	) -> None:
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

		self.bind_groups: dict[int, BindGroup | wgpu.GPUBindGroup] = {}
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

	def __enter__(self) -> RenderPass:
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

	def set_bind_group(self, index: int, bindings: BindGroup | wgpu.GPUBindGroup) -> None:
		if isinstance(bindings, BindGroup) and bindings.group != index:
			raise ValueError(f"BindGroup belongs to group {bindings.group}, not group {index}")
		
		if self.bind_groups.get(index) is bindings:
			return

		self.bind_groups[index] = bindings

	def set_vertex_buffer(
		self,
		slot: int,
		buffer: GpuBuffer,
		*,
		offset: int = 0,
		size: int | None = None,
		step_mode: str = wgpu.VertexStepMode.vertex,
	) -> None:
		layout = (buffer.content.dtype, step_mode)
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
		handle = self._prepare_pipeline()
		handle.draw_indexed_indirect( indirect_buffer.handle, indirect_offset )

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
			instance_count = (instances.content.size - instance_offset) if instances is not None else 1

		handle = self._prepare_pipeline()
		handle.draw_indexed(index_count, instance_count, 0, 0, 0)

	def _prepare_pipeline(self) -> wgpu.GPURenderPassEncoder:
		handle = self._require_handle()
		if self.pipeline is None:
			raise RuntimeError("No RenderPipeline has been set on this pass")

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
		# A retained BindGroup may refer to a resized buffer or replaced resource.
		for index, bindings in self.bind_groups.items():
			key = bindings.cache_key() if isinstance(bindings, BindGroup) else (bindings,)
			if self._bound_bind_groups.get(index) == key: continue
			if isinstance(bindings, BindGroup):
				bindings = self._variant.bind_group(bindings)
			handle.set_bind_group(index, bindings)
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

	def set_bind_group(self, index: int, bindings: BindGroup | wgpu.GPUBindGroup ) -> None:
		if self.pipeline is None:
			raise RuntimeError("Set the ComputePipeline before its bind groups")
		if isinstance(bindings, BindGroup) and bindings.group != index:
			raise ValueError( f"BindGroup belongs to group {bindings.group}, not {index}" )
		if isinstance(bindings, BindGroup):
			bindings = self.pipeline._variant.bind_group(bindings)
		self._require_handle().set_bind_group(index, bindings)

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

	def descriptor(self) -> wgpu.RenderPassColorAttachment:
		return wgpu.RenderPassColorAttachment(
			view=self.view,
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

	def descriptor(self) -> wgpu.RenderPassDepthStencilAttachment:
		return wgpu.RenderPassDepthStencilAttachment(
			view=self.view,
			depth_clear_value=self.clear or 0.0,
			depth_load_op="clear" if self.clear is not None else "load",
			depth_store_op="store" if self.store else "discard",
			depth_read_only=self.read_only,
		)

# numpy buffers ---------------------------------------------------------------

_VertexFormat: dict[Any, tuple[str, int]] = {
	np.int8:	("sint8",   1),
	np.uint8:	("uint8",   1),
	np.int16:	("sint16",  2),
	np.uint16:	("uint16",  2),
	np.int32:	("sint32",  4),
	np.uint32:	("uint32",  4),
	np.int64:	("sint64",  8),
	np.uint64:	("uint64",  8),
	np.uint64:	("uint32",  4),
	np.float16:	("float16", 2),
	np.float32:	("float32", 4),
	np.float64:	("float64", 8),
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
		return getattr(wgpu.VertexFormat, prefix)

	base_dtype, shape = dtype.subdtype
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

	vformat = getattr(wgpu.VertexFormat, prefix if count == 1 else f"{prefix}x{count}")
	if len(shape) == 1:
		return vformat
	if len(shape) == 2:
		row_count = shape[1]
		return [(vformat, scalar_size * count * i) for i in range(row_count)]

	raise TypeError(
		f"Only 1D or 2D array dtypes are valid vertex attributes (found {shape})"
	)


class GpuBuffer:
	def __init__(
		self,
		content: np.ndarray,
		usage: wgpu.BufferUsage,
		upload: bool = True,
		label: str | None = None,
	):
		self.content = np.asarray(content)
		self.usage = usage
		self.label = label
		self.generation = 0

		if upload:
			self.handle = RenderContext.device.create_buffer_with_data(
				label=label or "",
				data=self.content,
				usage=self.usage,
			)
		else:
			self.handle = RenderContext.device.create_buffer(
				label=label or "",
				size=max(4, self.content.nbytes),
				usage=self.usage,
			)

	@property
	def binding(self) -> wgpu.BufferBinding:
		return wgpu.BufferBinding(
			buffer=self.handle,
			offset=0,
			size=self.content.nbytes,
		)

	@property
	def resource_key(self) -> tuple[wgpu.GPUBuffer, int]:
		return (self.handle, self.content.nbytes)
	

	def resize(self, count: int) -> bool:
		size = higher_pow2(max(4, count * self.content.itemsize))
		if size <= self.handle.size:
			return False

		self.handle = RenderContext.device.create_buffer(
			label=self.label or "",
			size=size,
			usage=self.usage,
		)
		self.generation += 1
		return True

	def upload(self) -> None:
		RenderContext.device.queue.write_buffer(self.handle, 0, self.content)

	def upload_range(self, start: int, count: int) -> None:
		if count <= 0:
			return
		uploaded = self.content[start:start + count]
		RenderContext.device.queue.write_buffer(
			buffer=self.handle,
			buffer_offset=start * self.content.itemsize,
			data=uploaded,
		)

	def write(self, data: np.ndarray, *, offset: int = 0) -> None:
		self.content = np.asarray(data)
		RenderContext.device.queue.write_buffer(self.handle, offset, self.content)


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
	""" a specific mesh's data, gpu bugger pools are stored at class level for similar vertex layouts """

	vertex_buffers: ClassVar[dict[np.dtype, GpuBufferPool]] = {}
	index_buffers: ClassVar[dict[np.dtype, GpuBufferPool]] = {}

	index_formats: ClassVar[dict[np.dtype, str]] = {
		np.dtype(np.uint16): wgpu.IndexFormat.uint16,
		np.dtype(np.uint32): wgpu.IndexFormat.uint32,
	}

	def __init__(self, vertex_data: np.ndarray, indices: np.ndarray):
		self.index_count = indices.size
		self.vertex_dtype = np.dtype(vertex_data.dtype)
		self.index_dtype = np.dtype(indices.dtype)

		if self.index_dtype not in self.index_formats:
			raise TypeError(f"Unsupported index dtype: {self.index_dtype}")

		vertex_pool = self.vertex_buffers.get(self.vertex_dtype)
		if vertex_pool is None:
			vertex_pool = GpuBufferPool(self.vertex_dtype, wgpu.BufferUsage.VERTEX)
			self.vertex_buffers[self.vertex_dtype] = vertex_pool

		index_pool = self.index_buffers.get(self.index_dtype)
		if index_pool is None:
			index_pool = GpuBufferPool(self.index_dtype, wgpu.BufferUsage.INDEX)
			self.index_buffers[self.index_dtype] = index_pool

		self.vertex_buffer = vertex_pool.buffer
		self.vertex_range = vertex_pool.alloc(vertex_data)

		self.index_buffer = index_pool.buffer
		self.index_range = index_pool.alloc(indices)
		self.index_format = self.index_formats[self.index_dtype]

	@property
	def vertices(self) -> np.ndarray:
		start, count = self.vertex_range
		return self.vertex_buffer.content[start:start + count]

	@property
	def vertex_range_bytes(self) -> tuple[int,int]:
		return tuple(np.asarray(self.vertex_range) * self.vertex_dtype.itemsize)


# textures --------------------------------------------------------------------

class Texture:
	def __init__(
		self,
		size: tuple[int, int] | tuple[int, int, int],
		*,
		format: str,
		usage: wgpu.TextureUsage,
		mip_level_count: int = 1,
		dimension: str = "2d",
		label: str | None = None,
	):
		if len(size) == 2:
			size = (*size, 1)
		self.size = size
		self.format = format
		self.usage = usage
		self._view: TextureView | None = None
		self.handle = RenderContext.device.create_texture(
			label=label or "",
			size=size,
			dimension=dimension,
			format=format,
			usage=usage,
			mip_level_count=mip_level_count,
		)

	@property
	def binding(self) -> wgpu.GPUTextureView:
		return self.view().handle

	def view(self, **kwargs) -> TextureView:
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
	) -> ColorAttachment:
		return ColorAttachment(self.binding, self.format, clear, store)

	def depth_attachment(
		self,
		*,
		clear: float | None = None,
		store: bool = True,
		read_only: bool = False,
	) -> DepthAttachment:
		return DepthAttachment(self.binding, self.format, clear, store, read_only)


@dataclass(frozen=True)
class TextureView:
	texture: Texture
	handle: wgpu.GPUTextureView

	@property
	def format(self) -> str:
		return self.texture.format


class Sampler:
	def __init__(self, **descriptor: Any):
		self.handle = RenderContext.device.create_sampler(**descriptor)


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
		self.structs  = self._parse_structs(source)
		self.bindings = self._parse_bindings(source)
		self.entry_points = self._parse_entry_points(source)
		self.source = self._process_dsl(code)
		#print(self.source)
		self.uniforms = self._uniforms()

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


class Shader:

	def __init__(
		self,
		*,
		source: str | None = None,
		filepath: str | None = None,
		basedir: str | None = None,
		features: Sequence[str] | None = None,
		params: dict[str, Any] | None = None,
		label: str | None = None,
	):
		if source is None and filepath is None:
			raise ValueError("Shader requires source= or filepath=")

		source = self._load_source(
			source,
			filepath,
			basedir,
			features,
			params,
		)

		self.info = ShaderSource(source)

		try:
			self.module = RenderContext.device.create_shader_module(
				label=label or filepath or "shader",
				code=self.info.source,
			)
		except Exception as ex:
			print(self.info.source)
			raise ex from None

	def UniformBuffer(self, name: str | None = None) -> _UniformBuffer:
		if not self.info.uniforms:
			raise ValueError( "no uniforms were parsed" )

		if name is None:
			if len(self.info.uniforms) > 1:
				raise ValueError( "Uniform name is required when a shader has multiple uniform blocks" )

			name = next(iter(self.info.uniforms))

		return _UniformBuffer(self.info.uniforms[name], label=name)

	def bind_group(self, group: int, **resources: Any) -> BindGroup:
		return BindGroup(self, group, resources)

	def entry(
		self,
		stage: str,
		name: str | None = None,
	) -> str | None:
		entries = [
			entry.name for entry in self.info.entry_points.values()
			if entry.stage == stage and entry.name == name
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
			else os.path.dirname(os.path.abspath(filepath))
			if filepath
			else glob("**/shaders/", recursive=True)
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

	def cache_key(self) -> tuple:
		return (
			self.group,
			tuple(
				(
					self.shader.info.bindings[name].binding,
					self._resource_key(resource),
				)
				for name, resource in sorted(self.resources.items(), key=lambda item: self.shader.info.bindings[item[0]].binding)
			),
		)

	def entries(self) -> list[wgpu.BindGroupEntry]:
		return [
			wgpu.BindGroupEntry(
				binding=self.shader.info.bindings[name].binding,
				resource=self._resource(resource),
			)
			for name, resource in self.resources.items()
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
		if bind := getattr(resource, "binding", None):
			return bind
		if handle := getattr(resource, "handle", None):
			return handle
		return resource

class _PipelineVariant:
	# Bound retained GPU resources across buffer growth and texture replacement.
	_BIND_GROUP_CACHE_LIMIT = 128

	def __init__(
		self,
		handle: wgpu.GPURenderPipeline | wgpu.GPUComputePipeline,
	):
		self.handle = handle
		self._bind_groups: OrderedDict[tuple, wgpu.GPUBindGroup] = OrderedDict()
		self._lock = Lock()

	def bind_group(self, bindings: BindGroup) -> wgpu.GPUBindGroup:
		key = bindings.cache_key()
		with self._lock:
			cached = self._bind_groups.get(key)
			if cached is not None:
				self._bind_groups.move_to_end(key)
				return cached
			cached = RenderContext.device.create_bind_group(
				layout=self.handle.get_bind_group_layout(bindings.group),
				entries=bindings.entries(),
			)
			self._bind_groups[key] = cached
			if len(self._bind_groups) > self._BIND_GROUP_CACHE_LIMIT:
				self._bind_groups.popitem(last=False)
			return cached



class RenderPipeline:
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
		fragment_entry: str | None = None,
		topology: str = wgpu.PrimitiveTopology.triangle_list,
		cull_mode: str = wgpu.CullMode.back,
		front_face: str = wgpu.FrontFace.ccw,
		depth_test: str | None = wgpu.CompareFunction.less,
		depth_write: bool = True,
		depth_bias: int = 0,
		depth_bias_slope_scale: float = 0.0,
		blend: wgpu.BlendState | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.vertex_entry = shader.entry("vertex", vertex_entry)
		self.fragment_entry = shader.entry("fragment", fragment_entry)
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
		key = (
			tuple(
				(np.dtype(dtype), step_mode)
				for dtype, step_mode in buffers
			),
			render_pass.color_formats,
			render_pass.depth_format,
		)

		cached = self._cache.get(key)
		if cached is not None:
			return cached

		with self._cache_lock:
			cached = self._cache.get(key)
			if cached is None:
				cached = _PipelineVariant(
					self._make_pipeline(render_pass, buffers)
				)
				self._cache[key] = cached

		return cached

	def _make_pipeline(
		self,
		render_pass: RenderPass,
		buffers: Sequence[tuple[np.dtype, str]],
	) -> wgpu.GPURenderPipeline:
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

				if (
					field_size != shader_size
					or field_dtype.base.kind != shader_dtype.base.kind
				):
					raise TypeError(
						f"Vertex attribute {name!r} has dtype {field_dtype}; "
						f"shader expects {shader_input.type_name}"
					)

				attributes.append(wgpu.VertexAttribute(
					shader_location=shader_input.location,
					offset=offset,
					format=dtype_to_vertex_format(field_dtype),
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

		if render_pass.color_formats:
			if self.fragment_entry is None:
				raise ValueError(
					"Rendering to a color target requires an @fragment entry point"
				)

			fragment = wgpu.FragmentState(
				module=self.shader.module,
				entry_point=self.fragment_entry,
				targets=[
					wgpu.ColorTargetState(
						format=format,
						blend=self.blend,
					)
					for format in render_pass.color_formats
				],
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

		return RenderContext.device.create_render_pipeline(
			label=self.label or "",
			layout="auto",
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



class ComputePipeline:
	def __init__(
		self,
		shader: Shader,
		*,
		entry: str | None = None,
		label: str | None = None,
	):
		self.shader = shader
		self.entry = shader.entry("compute", entry)

		if self.entry is None:
			raise ValueError("ComputePipeline shader has no @compute entry point")

		handle = RenderContext.device.create_compute_pipeline(
			label=label or "",
			layout="auto",
			compute=wgpu.ProgrammableStage(
				module=shader.module,
				entry_point=self.entry,
			),
		)
		self._variant = _PipelineVariant(handle)

	@property
	def handle(self) -> wgpu.GPUComputePipeline:
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