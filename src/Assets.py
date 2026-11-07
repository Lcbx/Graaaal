from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np


if TYPE_CHECKING:
	from pygltflib import GLTF2, Material

def gltf_buffer_bytes(gltf: GLTF2, index: int, base_dir=".") -> bytes:
	"""Read a GLB, base64 data URI, or external buffer relative to the asset."""
	buffer = gltf.buffers[index]
	if buffer.uri is None:
		return gltf.binary_blob()
	if buffer.uri.startswith("data:"):
		return base64.b64decode(buffer.uri.split(",", 1)[1])
	return (Path(base_dir) / buffer.uri).read_bytes()


def gltf_image_bytes(gltf: GLTF2, image, base_dir=".") -> bytes | None:
	"""Read encoded image bytes from a URI or buffer view, without decoding."""
	if image.uri:
		if image.uri.startswith("data:"):
			return base64.b64decode(image.uri.split(",", 1)[1])
		return (Path(base_dir) / image.uri).read_bytes()
	if image.bufferView is not None:
		view = gltf.bufferViews[image.bufferView]
		start = view.byteOffset or 0
		return gltf_buffer_bytes(gltf, view.buffer, base_dir)[start:start + view.byteLength]
	return None


def read_gltf_accessor(gltf: GLTF2, index: int, base_dir=".") -> np.ndarray:
	"""Read interleaved/normalized glTF attributes without opening a GPU device."""
	accessor = gltf.accessors[index]
	if accessor.sparse is not None:
		raise ValueError("Sparse glTF accessors are not supported")
	dtype = np.dtype({5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2",
		5125: "<u4", 5126: "<f4"}[accessor.componentType])
	components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}[accessor.type]
	if accessor.bufferView is None:
		values = np.zeros((accessor.count, components), dtype=dtype)
	else:
		view = gltf.bufferViews[accessor.bufferView]
		# Matrix columns are padded to four bytes, including byte/short matrices.
		columns = int(accessor.type[-1]) if accessor.type.startswith("MAT") else 1
		rows = components//columns
		column_stride = ((rows*dtype.itemsize + 3)//4)*4 if columns > 1 else rows*dtype.itemsize
		values = np.ndarray(
			(accessor.count, columns, rows), dtype=dtype,
			buffer=gltf_buffer_bytes(gltf, view.buffer, base_dir),
			offset=(view.byteOffset or 0) + (accessor.byteOffset or 0),
			strides=(view.byteStride or columns*column_stride, column_stride, dtype.itemsize),
		).reshape(accessor.count, components).copy()
	if accessor.normalized and dtype.kind in "iu":
		values = values.astype(np.float32) / np.iinfo(dtype).max
		if dtype.kind == "i":
			values = np.maximum(values, -1.0)
	return values


def gltf_node_matrix(node) -> np.ndarray:
	"""Column-vector glTF local transform (including nonuniform/mirrored scale)."""
	if node.matrix is not None:
		return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
	x, y, z, w = node.rotation or (0, 0, 0, 1)
	rotation = np.array([
		[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
		[2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
		[2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
	])
	matrix = np.eye(4)
	matrix[:3, :3] = rotation @ np.diag(node.scale or (1, 1, 1))
	matrix[:3, 3] = node.translation or (0, 0, 0)
	return matrix


GLTF_VERTEX_DTYPE = np.dtype([
	("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2),
])


@dataclass
class GltfPrimitive:
	"""World-space triangle-list vertices and an index into the scene materials."""
	vertices: np.ndarray
	material_id: int


def load_gltf_scene(path: str | Path) -> tuple[list[GltfPrimitive], list[Material | None], list[np.ndarray]]:
	"""Load a static glTF hierarchy as CPU triangle lists, materials and RGBA images.

	Bake node transforms, fix reflected winding, transform/normalize normals,
	and generate flat normals or zero UVs when absent. Each material gets a
	base-color image (white when absent); a final None slot represents glTF's
	default material. Material objects retain their original PBR/alpha settings.
	No renderer-specific fields, GPU resources, or scene paths are required.
	"""
	from PIL import Image

	path = Path(path)
	from pygltflib import GLTF2
	gltf = GLTF2().load(str(path))
	materials = [*gltf.materials, None]
	images = []
	for material in materials:
		pbr = material.pbrMetallicRoughness if material else None
		if pbr and pbr.baseColorTexture:
			info = pbr.baseColorTexture
			if info.texCoord != 0 or info.extensions:
				raise ValueError("Base-color textures require TEXCOORD_0 without transforms")
			texture = gltf.textures[info.index]
			data = gltf_image_bytes(gltf, gltf.images[texture.source], path.parent)
			if data is None:
				raise ValueError("Base-color texture has no image data")
			with Image.open(io.BytesIO(data)) as image:
				images.append(np.array(image.convert("RGBA")))
		else:
			images.append(np.full((1, 1, 4), 255, dtype=np.uint8))

	primitives = []
	def visit(node_id, parent):
		node = gltf.nodes[node_id]
		if node.skin is not None:
			raise ValueError("Static glTF loading does not support skinned nodes")
		matrix = parent @ gltf_node_matrix(node)
		if node.mesh is not None:
			for primitive in gltf.meshes[node.mesh].primitives:
				if primitive.mode not in (None, 4):
					raise ValueError("Static glTF loading requires triangle-list primitives")
				attributes = primitive.attributes
				positions = read_gltf_accessor(gltf, attributes.POSITION, path.parent)
				indices = (read_gltf_accessor(gltf, primitive.indices, path.parent).reshape(-1)
					if primitive.indices is not None else np.arange(len(positions)))
				if len(indices) % 3:
					raise ValueError("Triangle index count must be divisible by three")
				if not len(indices):
					continue
				if np.linalg.det(matrix[:3, :3]) < 0:
					indices = indices.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
				vertices = np.zeros(len(indices), dtype=GLTF_VERTEX_DTYPE)
				vertices["position"] = (positions @ matrix[:3, :3].T + matrix[:3, 3])[indices]
				if attributes.NORMAL is not None:
					normals = read_gltf_accessor(gltf, attributes.NORMAL, path.parent)
					vertices["normal"] = (normals @ np.linalg.inv(matrix[:3, :3]))[indices]
				else:
					triangles = vertices["position"].reshape(-1, 3, 3)
					vertices["normal"] = np.repeat(np.cross(
						triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), 3, axis=0)
				vertices["normal"] /= np.maximum(np.linalg.norm(vertices["normal"], axis=1, keepdims=True), 1e-8)
				if attributes.TEXCOORD_0 is not None:
					vertices["uv"] = read_gltf_accessor(gltf, attributes.TEXCOORD_0, path.parent)[indices]
				material_id = primitive.material if primitive.material is not None else len(materials)-1
				primitives.append(GltfPrimitive(vertices, material_id))
		for child in node.children or ():
			visit(child, matrix)

	if gltf.scenes:
		roots = gltf.scenes[gltf.scene if gltf.scene is not None else 0].nodes
	else:
		children = {child for node in gltf.nodes for child in node.children or ()}
		roots = [i for i in range(len(gltf.nodes)) if i not in children]
	for root in roots:
		visit(root, np.eye(4))
	if not primitives:
		raise ValueError("The glTF scene has no triangles")
	return primitives, materials, images


def load_blue_noise(path: str | Path) -> np.ndarray:
	"""Load single-channel 16-bit PNG thresholds without sRGB conversion.

	Use quantization-bin midpoints so thresholds stay strictly inside (0, 1).
	"""
	from PIL import Image

	with Image.open(path) as image:
		samples = np.asarray(image)
		if samples.ndim != 2 or samples.dtype != np.uint16:
			raise ValueError("Blue noise must be a single-channel 16-bit PNG")
		return np.ascontiguousarray((samples.astype(np.float32) + 0.5) / 65536)


def load_gltf_first_mesh(glb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Read the first primitive in local space, without hierarchy/material baking."""
	path = Path(glb_path)
	from pygltflib import GLTF2
	gltf = GLTF2().load(str(path))
	primitive = gltf.meshes[0].primitives[0]
	attributes = primitive.attributes
	positions = read_gltf_accessor(gltf, attributes.POSITION, path.parent).astype(np.float32)
	count = len(positions)
	indices = (read_gltf_accessor(gltf, primitive.indices, path.parent)
		if primitive.indices is not None else np.arange(count)).astype(
			np.uint32 if count > 65536 else np.uint16)
	normals = (read_gltf_accessor(gltf, attributes.NORMAL, path.parent).astype(np.float32)
		if attributes.NORMAL is not None else np.zeros_like(positions))
	uvs = (read_gltf_accessor(gltf, attributes.TEXCOORD_0, path.parent).astype(np.float16)
		if attributes.TEXCOORD_0 is not None else np.zeros((count, 2), dtype=np.float16))
	return positions, normals, uvs, indices


def load_gltf_first_mesh_interleaved(glb_path: str) -> Tuple[np.ndarray,np.ndarray]:
	( pos, nor, uv, idx ) = load_gltf_first_mesh(glb_path)
	verts = interleave_mesh_position_normal_uv(pos, nor, uv)
	return verts, idx.astype(np.uint32).reshape(-1)


def interleave_mesh_position_normal_uv(positions, normals, uvs):
	vertex_dtype = np.dtype([
			("position", np.float32, (3,)),
			("normal",   np.float32, (3,)), # float16x3 does not exist in wgpu
			("uv",       np.float16, (2,)),
		], align=False,   # important: keep it tightly packed (no padding surprises)
	)
	#print("stride:", vertex_dtype.itemsize)
	#print("offsets:", {n: vertex_dtype.fields[n][1] for n in vertex_dtype.names})

	verts = np.empty(len(positions), dtype=vertex_dtype)
	verts["position"] = positions
	verts["normal"]   = normals
	verts["uv"]		  = uvs
	return verts

