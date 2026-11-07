"""Pure NumPy compilation of draw destinations, bounds, and culling workgroups.

The caller supplies registries and instance counts; no buffers or GPU commands
are created here. Retained registries and uploads live in draw_resources.
"""

from dataclasses import dataclass

import numpy as np

from .draw_layout import (
    BATCH_DTYPE,
    INDIRECT_DTYPE,
    MESH_METADATA_DTYPE,
    WORKGROUP_DTYPE,
    DrawBatch,
)


@dataclass
class CompiledLayout:
	"""Owned arrays and destination lookup tables ready for inspection or upload."""
	arrays: dict[str, np.ndarray]
	draw_batches: list[DrawBatch]
	group_rows: dict[int, int]
	mesh_commands: dict[int, list[int]]
	main_count: int
	prepass_count: int


def group_bounds(group, meshes):
	"""Conservative local bounds enclosing every nonempty LoD in a group."""
	infos = [meshes[mesh_id] for mesh_id in group.lod_ids if mesh_id is not None]
	box_min = np.min([info.box_center - info.box_extents for info in infos], axis=0)
	box_max = np.max([info.box_center + info.box_extents for info in infos], axis=0)
	return (box_min + box_max) * 0.5, (box_max - box_min) * 0.5


def compile_destinations(instance_count, batches, lod_groups, meshes, shader_prepass):
	"""Build destination arrays without mutating input registries or GPU state."""
	used_groups = sorted({batch.lod_group_id for batch in batches})
	group_rows = {group_id: i for i, group_id in enumerate(used_groups)}
	for batch in batches:
		if batch.lod_group_id not in lod_groups: raise KeyError(f"Unregistered LoD group {batch.lod_group_id}")
		if batch.shader_id not in shader_prepass: raise KeyError(f"Unregistered shader_id {batch.shader_id}")
	metadata = np.zeros(len(used_groups), dtype=MESH_METADATA_DTYPE)
	distances : list[float] = []
	for group_id, row in group_rows.items():
		group = lod_groups[group_id]
		center, extents = group_bounds(lod_groups[group_id], meshes)
		metadata[row]["box_center"][:3], metadata[row]["box_extents"][:3] = center, extents
		metadata[row]["lod_offset"] = len(distances)
		metadata[row]["lod_count"] = len(group.lod_ids)
		distances.extend((0.0, *group.distances))
	params = np.zeros(len(batches), dtype=BATCH_DTYPE)
	draw_batches     : list[DrawBatch]      = []
	commands         : list[tuple[int,...]] = []
	prepass_commands : list[tuple[int,...]] = []
	main_count = prepass_count = 0
	mesh_commands : dict[int, list[int]]= {}
	for batch_id, batch in enumerate(batches):
		prepass = shader_prepass[batch.shader_id]
		params[batch_id] = (group_rows[batch.lod_group_id], batch.offset, batch.count, len(commands), prepass, 0)
		for mesh_id in lod_groups[batch.lod_group_id].lod_ids:
			command_index = len(commands)
			draw_batches.append(DrawBatch(mesh_id, batch.shader_id, main_count, batch.count, command_index))
			if mesh_id is None:
				command = (0, 0, 0, 0)
				commands.append((*command, main_count))
			else:
				mesh = meshes[mesh_id].mesh
				command = (mesh.index_count, 0, mesh.index_range[0], mesh.vertex_range[0])
				commands.append((*command, main_count))
				mesh_commands.setdefault(mesh_id, []).append(command_index)
			prepass_commands.append((*command, prepass_count if prepass else 0))
			main_count += batch.count
			if prepass: prepass_count += batch.count
	if max(
			main_count,
			prepass_count,
			len(commands),
			instance_count
		) > np.iinfo(np.uint32).max:
		raise ValueError("Draw destinations exceed uint32 addressing")
	return CompiledLayout(
		arrays={
			"mesh_metadata": metadata,
			"lod_distances": np.asarray(distances, dtype="<f4"),
			"batches": params,
			"prepass_draw_cmd": np.asarray(prepass_commands, dtype=INDIRECT_DTYPE),
			"main_draw_cmd": np.asarray(commands, dtype=INDIRECT_DTYPE),
		},
		draw_batches=draw_batches, group_rows=group_rows, mesh_commands=mesh_commands,
		main_count=main_count, prepass_count=prepass_count,
	)


def compile_workgroups(counts):
	"""Build the flattened culling work table from per-source dispatch counts."""
	group_counts = np.asarray(counts, dtype=np.int64)
	workgroups = np.zeros(int(group_counts.sum()), dtype=WORKGROUP_DTYPE)
	if len(group_counts):
		workgroups["batch_id"] = np.repeat(np.arange(len(group_counts)), group_counts)
		starts = np.cumsum(group_counts) - group_counts
		workgroups["instance_offset"] = (np.arange(len(workgroups)) - np.repeat(starts, group_counts)) * 64
	return workgroups
