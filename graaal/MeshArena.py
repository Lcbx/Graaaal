"""Releasable CPU-backed mesh allocations owned through RenderContext."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class MeshAllocation:
    pool: _Pool
    start: int
    count: int
    generation: int = 0
    live: bool = True

    def check(self):
        self.pool.arena.check()
        if not self.live: raise RuntimeError("Mesh allocation has been released")

    @property
    def range(self):
        self.check()
        return self.start, self.count

    def release(self):
        if self.live:
            self.pool.free.append((self.start, self.count))
            self.live = False
            self.generation += 1
            self.pool.detach = True
            self.pool.allocations.remove(self)


class _Pool:
    def __init__(self, arena, dtype, usage):
        self.arena, self.dtype, self.usage = arena, np.dtype(dtype), usage
        self.buffer = None
        self.content = np.empty(0, dtype=dtype)
        self.used = 0
        self.free = []
        self.allocations = []
        self.detach = False

    def _replace_buffer(self):
        from .RenderContext import BufferUsage, GpuBuffer
        self.buffer = GpuBuffer(self.content, self.usage | BufferUsage.COPY_DST)
        self.detach = False

    def alloc(self, data):
        count = len(data)
        self.free.sort()
        merged = []
        for start, size in self.free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else: merged.append((start, size))
        self.free = merged
        start = self.used
        for i, (offset, size) in enumerate(self.free):
            if size >= count:
                start = offset
                self.free.pop(i)
                if size > count: self.free.append((offset + count, size - count))
                break
        end = start + count
        if end > len(self.content):
            grown = np.zeros(1 << max(0, end - 1).bit_length(), dtype=self.dtype)
            grown[:self.used] = self.content[:self.used]
            self.content = grown
            self.detach = True
        self.content[start:end] = data
        self.used = max(self.used, end)
        if self.buffer is None or self.detach:
            # Copy-on-reuse keeps previously recorded GPU commands valid.
            # WGPU retains the old handles for recorded/submitted work.
            self._replace_buffer()
        else:
            self.buffer.upload_range(start, count)
        allocation = MeshAllocation(self, start, count)
        self.allocations.append(allocation)
        return allocation

    def compact(self):
        live = [item for item in self.allocations if item.live]
        values = [self.content[item.start:item.start + item.count].copy() for item in live]
        offset = 0
        for item, data in zip(live, values):
            self.content[offset:offset + item.count] = data
            if item.start != offset:
                item.start = offset
                item.generation += 1
            offset += item.count
        self.used, self.allocations, self.free = offset, live, []
        self._replace_buffer()


class MeshArena:
    def __init__(self):
        from .RenderContext import RenderContext
        self.device = RenderContext.device
        self.device_generation = RenderContext.device_generation
        self.pools = {}
        self.closed = False

    def check(self):
        from .RenderContext import RenderContext
        if self.closed: raise RuntimeError("Mesh arena is closed")
        if self.device is not RenderContext.device or self.device_generation != RenderContext.device_generation:
            raise RuntimeError("Mesh arena belongs to an expired device generation")

    def allocate(self, data, usage):
        self.check()
        data = np.ascontiguousarray(data)
        if data.ndim != 1 or not len(data): raise ValueError("Expected nonempty one-dimensional mesh data")
        key = data.dtype, usage
        if key not in self.pools: self.pools[key] = _Pool(self, data.dtype, usage)
        return self.pools[key].alloc(data)

    def compact(self):
        self.check()
        for pool in self.pools.values(): pool.compact()

    def inspect(self):
        return {"capacity_bytes": sum(pool.content.nbytes for pool in self.pools.values()),
            "active_bytes": sum(item.count * pool.dtype.itemsize for pool in self.pools.values() for item in pool.allocations if item.live),
            "allocations": sum(item.live for pool in self.pools.values() for item in pool.allocations)}

    def close(self):
        if self.closed: return
        self.closed = True
        for pool in self.pools.values():
            for allocation in pool.allocations: allocation.live = False
            if pool.buffer is not None: pool.buffer.close()
            pool.content = np.empty(0, dtype=pool.dtype)
        self.pools.clear()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
