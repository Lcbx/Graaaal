from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .Assets import load_gltf_scene


VERTEX_DTYPE = np.dtype([
	("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2),
	("barycentric", "<f4", 2), ("triangle", "<u4"),
])
SHADING_VERTEX_DTYPE = np.dtype([
	("position", "<f4", 4), ("normal", "<f4", 4), ("uv", "<f4", 4),
])
MATERIAL_DTYPE = np.dtype([
	("base_color", "<f4", 4), ("emission", "<f4", 4),
	("surface", "<f4", 4), ("opacity_rgb", "<f4", 4), ("flags", "<u4", 4),
])


@dataclass
class Primitive:
	"""Packed raster geometry with a material, global triangle offset and stable ID."""
	vertices: np.ndarray
	material_id: int
	triangle_offset: int
	stable_id: int


def load_transparency_scene(path: str | Path, *, mask_materials=()):
	"""Load static glTF into raster primitives, GPU material records and images.

	Returns (primitives, materials, images, shading_vertices), packed for the
	transparency shaders. This prepares CPU arrays only; no device is needed.
	Use load_gltf_scene() for plain geometry and original glTF materials.
	mask_materials explicitly overrides named materials to MASK for cutout assets
	incorrectly authored as BLEND; ordinary BLEND is never automatically sharpened.
	"""
	loaded, materials, images = load_gltf_scene(path)
	records = np.zeros(len(materials), dtype=MATERIAL_DTYPE)
	for i, material in enumerate(materials):
		pbr = material.pbrMetallicRoughness if material else None
		records[i]["base_color"] = (pbr.baseColorFactor if pbr else None) or (1, 1, 1, 1)
		records[i]["emission"][:3] = (material.emissiveFactor if material else None) or (0, 0, 0)
		records[i]["surface"] = (
			pbr.roughnessFactor if pbr and pbr.roughnessFactor is not None else 1,
			pbr.metallicFactor if pbr and pbr.metallicFactor is not None else 1,
			{"OPAQUE": 0, "MASK": 1, "BLEND": 2}[(material.alphaMode or "OPAQUE") if material else "OPAQUE"],
			material.alphaCutoff if material and material.alphaCutoff is not None else 0.5,
		)
		# RGB opacity is separate from color; scalar glTF alpha uses (1,1,1).
		records[i]["opacity_rgb"] = (1, 1, 1, 1)
		records[i]["flags"] = (i, bool(material and material.doubleSided), 0, 0)
		if material and material.name in mask_materials:
			records[i]["surface"][2] = 1

	primitives, shading_vertices = [], []
	triangle_offset = 0
	for stable_id, primitive in enumerate(loaded):
		vertices = np.zeros(len(primitive.vertices), dtype=VERTEX_DTYPE)
		stored = np.zeros(len(vertices), dtype=SHADING_VERTEX_DTYPE)
		for name, width in (("position", 3), ("normal", 3), ("uv", 2)):
			vertices[name] = primitive.vertices[name]
			stored[name][:, :width] = vertices[name]
		triangle_count = len(vertices)//3
		vertices["barycentric"] = np.tile(((1, 0), (0, 1), (0, 0)), (triangle_count, 1))
		vertices["triangle"] = np.repeat(np.arange(triangle_count), 3)
		primitives.append(Primitive(vertices, primitive.material_id, triangle_offset, stable_id))
		shading_vertices.append(stored)
		triangle_offset += triangle_count
	return primitives, records, images, np.concatenate(shading_vertices)

