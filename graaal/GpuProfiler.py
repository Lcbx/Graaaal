"""Opt-in pass-boundary GPU timestamps; one submitted/read frame at a time."""
from .RenderContext import BufferUsage, GpuBuffer, RenderContext
import numpy as np


class GpuProfiler:
	def __init__(self, max_passes=64):
		if type(max_passes) is not int or max_passes < 1:
			raise ValueError("max_passes must be a positive integer")
		if "timestamp-query" not in RenderContext.device.features:
			raise ValueError("Request timestamp-query when creating the device to profile GPU passes")
		self.device = RenderContext.device
		self.generation = RenderContext.device_generation
		self.capacity = max_passes
		self.queries = self.device.create_query_set(type="timestamp", count=2*max_passes)
		self.results = GpuBuffer(None, BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC,
			dtype=np.uint64, count=2*max_passes)
		self.labels = []
		self.recording = self.pending = self.closed = False
		self.last_frame_ms = None

	def _check(self):
		if self.closed or self.device is not RenderContext.device or self.generation != RenderContext.device_generation:
			raise RuntimeError("GPU profiler is closed or belongs to an expired device")

	def wrap(self, commands):
		self._check()
		if isinstance(commands, _ProfiledCommands): raise ValueError("Commands already have a GPU profiler")
		commands._check_device()
		if self.recording or self.pending:
			raise RuntimeError("Finish, submit and read the previous profiled frame first")
		self.labels = []
		self.recording = True
		return _ProfiledCommands(commands, self)

	def _timestamps(self, label):
		self._check()
		if not self.recording: raise RuntimeError("Profiled commands have already finished")
		index = len(self.labels)
		if index >= self.capacity: raise ValueError("GPU profiler pass capacity exceeded")
		self.labels.append(label or f"pass_{index}")
		return dict(query_set=self.queries, beginning_of_pass_write_index=2*index,
			end_of_pass_write_index=2*index+1)

	def read(self):
		"""Synchronize/read the submitted frame. Values are milliseconds per label."""
		self._check()
		if not self.pending: raise RuntimeError("Finish and submit a profiled frame before reading")
		result = {}
		self.last_frame_ms = 0.0
		if self.labels:
			values = np.frombuffer(self.device.queue.read_buffer(self.results.handle, size=16*len(self.labels)), np.uint64)
			self.last_frame_ms = float(values[-1]-values[0])/1e6
			for index, label in enumerate(self.labels):
				result[label] = result.get(label, 0.0)+float(values[2*index+1]-values[2*index])/1e6
		self.pending = False
		return result

	def close(self):
		if self.closed: return
		self.queries.destroy()
		self.results.close()
		self.closed = True


class _ProfiledCommands:
	def __init__(self, commands, profiler):
		self.commands, self.profiler = commands, profiler

	def __getattr__(self, name):
		return getattr(self.commands, name)

	def render_pass(self, **kwargs):
		return _ProfiledPass(self.commands.render_pass(**kwargs), self.profiler)

	def compute_pass(self, **kwargs):
		return _ProfiledPass(self.commands.compute_pass(**kwargs), self.profiler)

	def finish(self, label=None):
		self.profiler._check()
		if not self.profiler.recording: raise RuntimeError("Profiled commands already finished")
		count = len(self.profiler.labels)*2
		if count: self.commands.handle.resolve_query_set(self.profiler.queries, 0, count, self.profiler.results.handle, 0)
		result = self.commands.finish(label=label)
		self.profiler.recording = False
		self.profiler.pending = True
		return result


class _ProfiledPass:
	def __init__(self, scope, profiler):
		self.scope, self.profiler = scope, profiler

	def __getattr__(self, name):
		return getattr(self.scope, name)

	def __enter__(self):
		if self.scope.timestamp_writes is not None: raise ValueError("Pass already has timestamp writes")
		# Allocate on entry, not construction: callers may prepare scopes, enter
		# them in another order, or never enter an optional scope at all.
		self.scope.timestamp_writes = self.profiler._timestamps(self.scope.label)
		return self.scope.__enter__()

	def __exit__(self, *args):
		try: return self.scope.__exit__(*args)
		finally: self.scope.timestamp_writes = None
