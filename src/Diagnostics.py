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

