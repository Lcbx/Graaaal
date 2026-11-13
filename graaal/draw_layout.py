"""NumPy record layouts shared by draw compilation and GPU culling shaders."""

from dataclasses import dataclass

import numpy as np


MESH_METADATA_DTYPE = np.dtype([
	("box_center", "<f4", 4), ("box_extents", "<f4", 4),
	("lod_offset", "<u4"), ("lod_count", "<u4"), ("padding", "<u4", 2),
])
FRUSTUM_CANDIDATE_DTYPE = np.dtype([("instance_id", "<u4"), ("command_id", "<u4")])
BATCH_DTYPE = np.dtype([
	("group_id", "<u4"), ("instance_offset", "<u4"), ("instance_count", "<u4"),
	("command_offset", "<u4"), ("prepass", "<u4"), ("frustum_count", "<u4"),
])
WORKGROUP_DTYPE = np.dtype([("batch_id", "<u4"), ("instance_offset", "<u4")])
INDIRECT_DTYPE = np.dtype([
	("index_count", "<u4"), ("instance_count", "<u4"), ("first_index", "<u4"),
	("base_vertex", "<i4"), ("first_instance", "<u4"),
])


MeshId = int | None


@dataclass(frozen=True)
class LodGroup:
	lod_ids: tuple[MeshId, ...]
	distances: tuple[float, ...]


@dataclass
class DrawBatch:
	"""Main-pass destination; offset/count describe its reserved index region."""
	mesh_id: MeshId
	shader_id: int
	offset: int
	count: int
	command_index: int
