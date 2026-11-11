from __future__ import annotations

import gc
import os
from time import perf_counter as get_time
from typing import Callable, ClassVar

import numpy as np


class WatchTimer:
	nesting : ClassVar[int] = 0
	timers: ClassVar[list[WatchTimer]] = []
	report: ClassVar[str] = ''
	print_region: ClassVar[WatchTimer | None] = None
	
	def __init__(self, region:str, print_:bool=False, *, frame_time: float | None = None):
		self.region = region
		self.frame_time = frame_time
		if print_:
			WatchTimer.print_region = self
	
	def __enter__(self) -> None:
		self.start_time = get_time()
		self._nesting = WatchTimer.nesting
		WatchTimer.nesting += 1
		for i,t in enumerate(WatchTimer.timers):
			if t.region == self.region:
				WatchTimer.timers[i] = self
				return
		WatchTimer.timers.append(self)

	def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
		WatchTimer.nesting -= 1
		self.message = self.get_message()

		if WatchTimer.print_region is self:
			print(WatchTimer.capture())
	
	def get_message(self) -> str:
		percentage = f" ({self.elapsed_percent():.0f}%)" if self.frame_time else ""
		return '  ' * self._nesting + f'{self.region} : {self.elapsed_ms():.1f}ms' + percentage
	
	def elapsed_ms(self) -> float:
		return (get_time() - self.start_time) * 1000.0

	def elapsed_percent(self) -> float:
		ft = (self.frame_time or 0) + 0.00001
		return (get_time() - self.start_time) / ft * 100.0
	
	@staticmethod
	def capture() -> str:
		WatchTimer.report = '\n'.join( list(map(
			lambda t: t.message if hasattr(t, 'message') else t.get_message(),
			WatchTimer.timers))
		)
		WatchTimer.timers.clear()
		return WatchTimer.report


def setup_gc_monitor() -> None:
	print(f'{gc.get_threshold()=}')
	#gc.set_threshold(1000, 30, 2) # deflt 2000, 10, 10
	_gc_start: float | None = None

	def gc_probe(phase:str, info:dict) -> None:
		nonlocal _gc_start
		if phase == "start":
			_gc_start = get_time()
		elif phase == "stop" and _gc_start is not None:
			dt_ms = (get_time() - _gc_start) * 1000
			if not any(info.values()): return
			print(
				f"GC gen={info['generation']} "
				f"collected={info['collected']} "
				f"uncollectable={info['uncollectable']} "
				f"time={dt_ms:.3f} ms"
			)

	gc.callbacks.append(gc_probe)


def setup_memory_monitor() -> Callable:
	import psutil  # type: ignore[import-untyped]  # NOTE: this is not a built-in lib
	process = psutil.Process(os.getpid())

	import tracemalloc
	tracemalloc.start()

	def print_report() -> None:
		rss = process.memory_info().rss
		print(f"process RSS: {rss / 1024 / 1024:.1f} MiB")
		current, peak = tracemalloc.get_traced_memory()
		print(f"traced memory: {current / 1024 / 1024:.2f} MiB")

	return print_report


def get_draw_counts(renderer, passes=None) -> tuple[int, int]:
	"""Return (drawable batches, submitted draw calls) without GPU visibility readback.

	DrawBatches defaults to the "main" pass; pass ("prepass", "main") to count
	both. MaterialRenderer defaults to all draws in record(), including fullscreen
	composition, for a nonzero output size. Compute dispatches are excluded.
	"""
	from .DrawBatches import DrawBatches
	from .MaterialRenderer import MaterialRenderer

	if isinstance(renderer, MaterialRenderer):
		draws = renderer.draws
	elif isinstance(renderer, DrawBatches):
		draws = renderer
	else:
		raise TypeError("Expected DrawBatches or MaterialRenderer")
	batches = [batch for batch in draws.draw_batches if batch.mesh_id is not None]
	if passes is not None or isinstance(renderer, DrawBatches):
		passes = ("main",) if passes is None else (passes,) if isinstance(passes, str) else tuple(passes)
		return len(batches), sum(name in draws.shaders[batch.shader_id].passes for batch in batches for name in passes)

	oit = renderer.transparency
	opaque = sum(batch.shader_id == 0 for batch in batches)
	transparent = sum(batch.shader_id == 1 for batch in batches)
	filtered = oit.bilateral_filter.kernel_size != 1
	draw_calls = opaque + transparent * (1 + int(filtered) + oit.winner_count) + 1
	if filtered and oit.debug_view in ("final", "reconstruction", "filter_support"):
		if oit.reconstruction == "fragment":
			draw_calls += (oit.reconstruction_passes - 1) * oit.sample_count
	return len(batches), draw_calls


def get_shader_sources(renderer) -> list[str]:
	"""Return unique expanded WGSL sources of this renderer's registered shaders.

	Includes its culling shader and, for MaterialRenderer, shading/composition
	shaders. External passes such as a separately owned HZB are excluded.
	"""
	from .DrawBatches import DrawBatches
	from .MaterialRenderer import MaterialRenderer

	if isinstance(renderer, MaterialRenderer):
		draws = renderer.draws
	elif isinstance(renderer, DrawBatches):
		draws = renderer
	else:
		raise TypeError("Expected DrawBatches or MaterialRenderer")
	shaders = [draws._cull_shader]
	for shader in draws.shaders.values():
		shaders.extend(spec.pipeline.shader for spec in shader.passes.values())
	if isinstance(renderer, MaterialRenderer):
		oit = renderer.transparency
		shaders.extend((oit.shading_shader, oit.compose_shader, oit.pair_shader, oit.reconstruct_shader))
	return list(dict.fromkeys(shader.info.source for shader in shaders if shader is not None))
