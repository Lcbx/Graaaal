"""Pure NumPy grouping, independent of ECS and GPU modules."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceBatch:
    lod_group_id: int
    shader_id: int
    offset: int
    count: int


@dataclass(frozen=True)
class DrawList:
    order: np.ndarray
    batches: tuple[SourceBatch, ...]

    @classmethod
    def build(cls, lod_ids, shader_ids):
        lod_ids, shader_ids = np.asarray(lod_ids), np.asarray(shader_ids)
        if lod_ids.ndim != 1 or shader_ids.ndim != 1 or len(lod_ids) != len(shader_ids):
            raise ValueError("Expected equally sized 1D LoD and shader IDs")
        if len(lod_ids) and (lod_ids.dtype.kind not in "iu" or shader_ids.dtype.kind not in "iu"):
            raise TypeError("LoD and shader IDs must be integers")
        order = np.lexsort((lod_ids, shader_ids)).astype(np.intp)
        lod_ids, shader_ids = lod_ids[order], shader_ids[order]
        changes = (lod_ids[1:] != lod_ids[:-1]) | (shader_ids[1:] != shader_ids[:-1])
        starts = np.r_[0, np.flatnonzero(changes) + 1] if len(order) else np.empty(0, dtype=np.intp)
        ends = np.r_[starts[1:], len(order)] if len(order) else starts
        batches = tuple(SourceBatch(int(lod_ids[a]), int(shader_ids[a]), int(a), int(b-a)) for a, b in zip(starts, ends))
        order.flags.writeable = False
        return cls(order, batches)
