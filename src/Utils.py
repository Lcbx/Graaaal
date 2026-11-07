from __future__ import annotations

from RenderContext import *
from RenderContext import Mesh, RenderContext

import gc, os
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter as get_time

from functools import wraps
from mypy_extensions import mypyc_attr
from typing import Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable, ParamSpec, TypeVar, ClassVar

import numpy as np
from pyrr import Matrix44 as Mat4, Quaternion  # type: ignore[import-untyped]
from pygltflib import GLTF2, Material  # type: ignore[import-untyped]


# pyrr-style Vector classes

def _array_accessor(index:int=0) -> property:
	return property(
		lambda v: v[index],
		lambda v, y: v.__setitem__(index, y),
	)

@mypyc_attr(native_class=False)
class _baseVector(np.ndarray):
	_count: ClassVar[int]
	_base_dtype: ClassVar[np.dtype] = np.dtype(np.float32)

	def __new__(cls, *args):
		values = args[0] if len(args) == 1 and not np.isscalar(args[0]) else args
		if len(values) != cls._count:
			raise ValueError(f'got passed {len(values)} when {cls._count} were expected')
		return np.asarray(values, dtype=cls._base_dtype).view(cls)

	@classmethod
	def storage_dtype(cls) -> np.dtype:
		return np.dtype((cls._base_dtype, (cls._count,)))

@mypyc_attr(native_class=False)
class Vec2(_baseVector):
	_count = 2
	x = _array_accessor(0)
	y = _array_accessor(1)

@mypyc_attr(native_class=False)
class Vec3(_baseVector):
	_count = 3
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)

@mypyc_attr(native_class=False)
class Vec4(_baseVector):
	_count = 4
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)
	w = _array_accessor(3)


# caches last return value and retrieves it based on args
P = ParamSpec("P")
R = TypeVar("R")
def cache_last(key: Callable[..., object]) -> Callable[[Callable[P, R]], Callable[P, R]]:
	missing = object()

	def decorator(fn: Callable[P, R]) -> Callable[P, R]:
		cache_key = f"__cache_key_{id(fn)}"
		cache_val = f"__cache_val_{id(fn)}"

		@wraps(fn)
		def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
			self = args[0]
			current_key = key(*args, **kwargs)

			cached = getattr(self, cache_key, missing)
			if cached == current_key:
				return getattr(self, cache_val)

			value = fn(*args, **kwargs)
			setattr(self, cache_key, current_key)
			setattr(self, cache_val, value)

			return value
		return wrapped
	return decorator


@mypyc_attr(native_class=False)
class Camera:
	def __init__(self, position: tuple, target: tuple, up: tuple, fovy_deg:float, near:float=0.1, far:float=1000.0, perspective:bool=True):
		self.position = Vec3(position)
		self.target = Vec3(target)
		self.up = Vec3(up)
		self.fovy_deg = fovy_deg
		self.near = near
		self.far = far
		self.perspective = perspective

	@cache_last(lambda self: (tuple(self.position), tuple(self.target), tuple(self.up)) )
	def view(self) -> Mat4:
		return Camera.calc_view(self.position, self.target, self.up)
	
	@staticmethod
	def calc_view(position, target:Vec3, up:Vec3) -> Mat4:
		return Mat4.look_at(position, target, up)
	
	# NOTE: we could split this class into orth and perspective cameras	
	@cache_last(lambda self, aspect: (self.fovy_deg, aspect, self.near, self.far))
	def projection(self, aspect:float) -> Mat4:
		if self.perspective: return Camera.perspective_projection(self.fovy_deg, aspect, self.near, self.far )
		top = self.fovy_deg * 0.5
		right = top*aspect
		left = -right
		bottom = -top
		return Camera.orthogonal_projection(left, right, bottom, top, self.near, self.far)

	@cache_last(lambda self: (tuple(self.position), tuple(self.target)) )
	def direction(self):
		l = self.position - self.target
		return l / np.linalg.norm(l)
	
	# can't use pyrr projections since it follows opengl convetions
	# opengl has ndc -1->1, wgpu is 0->1

	@staticmethod
	def perspective_projection(fovy_deg:float, aspect:float, near:float, far:float) -> Mat4:
		f = 1.0/np.tan(fovy_deg*3.14159/180.0 *0.5)
		far_factor = far/(near-far)
		mat = np.array([
			[f/aspect, 0.0,            0.0,  0.0],
			[0.0,        f,            0.0,  0.0],
			[0.0,      0.0,     far_factor, -1.0],
			[0.0,      0.0, far_factor*near, 0.0],
		], dtype=np.float32)
		#return np.ascontiguousarray(mat, dtype=np.float32)
		return mat

	@staticmethod
	def orthogonal_projection(left:float, right:float, bottom:float, top:float, near:float, far:float) -> Mat4:

		rml = right - left
		rpl = right + left
		tmb = top - bottom
		tpb = top + bottom
		fn = far - near

		mat = np.array([
			[2.0/rml,       0.0,      0.0, 0.0],
			[0.0,       2.0/tmb,      0.0, 0.0],
			[0.0,           0.0,  -1.0/fn, 0.0],
			[-rpl/rml, -tpb/tmb, -near/fn, 1.0],
		], dtype=np.float32)

		return mat

def extract_frustum_planes(viewProjectionMatrix):
	vp = np.asarray(viewProjectionMatrix)
	planes = np.zeros((6, 4), dtype=np.float32)
	planes[0], planes[1] = vp[:, 3] + vp[:, 0], vp[:, 3] - vp[:, 0]
	planes[2], planes[3] = vp[:, 3] + vp[:, 1], vp[:, 3] - vp[:, 1]
	planes[4], planes[5] = vp[:, 2], vp[:, 3] - vp[:, 2]
	for i in range(6):
		norm = np.linalg.norm(planes[i, :3])
		if norm > 0: planes[i] /= norm
	return planes



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


gltf_vertex_dtype = np.dtype([
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
				vertices = np.zeros(len(indices), dtype=gltf_vertex_dtype)
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


@dataclass
class Primitive:
	"""Packed raster geometry with a material, global triangle offset and stable ID."""
	vertices: np.ndarray
	material_id: int
	triangle_offset: int
	stable_id: int


def load_scene(path: str | Path):
	"""Load static glTF into raster primitives, GPU material records and images.

	Returns (primitives, materials, images, shading_vertices), packed for the
	transparency shaders. This prepares CPU arrays only; no device is needed.
	Use load_gltf_scene() for plain geometry and original glTF materials.
	"""
	# Resolve shared shader layouts at call time to avoid an import cycle through DrawBatches.
	from tranparency_pass import MATERIAL_DTYPE, SHADING_VERTEX_DTYPE, VERTEX_DTYPE
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


def build_material_texture_array(images, materials, size):
	"""Upload source images, then resample/filter/correct/encode mips on the GPU.

	Uncorrected premultiplied linear RGBA32F scratch mips retain tiny alpha values.
	MASK coverage search uses GPU reductions; no CPU filtering or read-back occurs.
	Encoded bytes are copied from a storage texture into the sRGB sampling array.
	This also works with WGPU backends that do not support mutable texture views.
	"""
	width, height = size
	levels = max(width, height).bit_length()
	result = Texture((*size, len(images)), format="rgba8unorm-srgb", mip_level_count=levels,
		usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="material_mip_array")
	encoded = Texture((*size, 1), format="rgba8unorm", mip_level_count=levels,
		usage=TextureUsage.STORAGE_BINDING | TextureUsage.COPY_SRC, label="encoded_mip_scratch")
	shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "material_mipmaps.shader"))
	pipelines = {entry: ComputePipeline(shader, entry=entry) for entry in (
		"decode", "resample", "init_source", "measure_source", "begin_search", "measure_candidate", "update_search", "encode")}
	def dispatch(compute, entry, resources, x=1, y=1):
		compute.set_pipeline(pipelines[entry])
		compute.set_bind_group(shader.bind_group(0, **resources))
		compute.dispatch(x, y)

	for layer, (image, material) in enumerate(zip(images, materials)):
		source_height, source_width, _ = image.shape
		source_size = (source_width, source_height)
		upload = Texture(source_size, format="rgba8unorm-srgb",
			usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": upload.handle}, np.ascontiguousarray(image),
			{"bytes_per_row": source_width*4, "rows_per_image": source_height}, (*source_size, 1))
		scratch_usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
		original = Texture(source_size, format="rgba32float", usage=scratch_usage)
		working = Texture(size, format="rgba32float", usage=scratch_usage, mip_level_count=levels)
		mode, factor = int(material["surface"][2]), float(material["base_color"][3])
		cutoff = float(material["surface"][3])/factor if factor > 0 else 0
		preserve = mode == 1 and 0 < cutoff <= 1
		params = shader.UniformBuffer("params")
		params.content["info"] = (0, mode, preserve, 0)
		params.content["cutoff"] = (cutoff, 0, 0, 0)
		params.upload()
		state = GpuBuffer(np.zeros(12, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		cmd = RenderContext.commands(f"material_mips_{layer}")
		with cmd.compute_pass(label="decode_and_source_coverage") as compute:
			dispatch(compute, "decode", dict(params=params, source=upload, linear_destination=original),
				(source_width+7)//8, (source_height+7)//8)
			if preserve:
				dispatch(compute, "init_source", dict(source=original, coverage=state))
				dispatch(compute, "measure_source", dict(params=params, source=original, coverage=state),
					(source_width+7)//8, (source_height+7)//8)
		previous = original.view()
		for level in range(levels):
			mip_width, mip_height = max(1, width >> level), max(1, height >> level)
			current = working.view(base_mip_level=level, mip_level_count=1, dimension="2d")
			destination = encoded.view(base_mip_level=level, mip_level_count=1, dimension="2d-array")
			groups = ((mip_width+7)//8, (mip_height+7)//8)
			with cmd.compute_pass(label=f"generate_mip_{level}") as compute:
				dispatch(compute, "resample", dict(source=previous, linear_destination=current), *groups)
				if preserve:
					dispatch(compute, "begin_search", dict(source=current, coverage=state))
					# Bracket by doubling, then bisect. Converged searches skip sample
					# work; fixed bounds cover RGBA8 source alpha through tiny mips.
					for _ in range(56):
						dispatch(compute, "measure_candidate", dict(params=params, source=current, coverage=state), *groups)
						dispatch(compute, "update_search", dict(params=params, coverage=state))
				dispatch(compute, "encode", dict(params=params, source=current,
					encoded_destination=destination, coverage=state), *groups)
			cmd.handle.copy_texture_to_texture(
				{"texture": encoded.handle, "mip_level": level},
				{"texture": result.handle, "mip_level": level, "origin": (0, 0, layer)},
				(mip_width, mip_height, 1))
			previous = current
		RenderContext.submit(cmd.finish())
	return result


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


class WatchTimer:
	nesting : ClassVar[int] = 0
	timers: ClassVar[list[WatchTimer]] = []
	report: ClassVar[str] = ''
	print_region: ClassVar[WatchTimer | None] = None
	
	def __init__(self, region:str, print_:bool=False):
		self.region = region
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
		return ('  ' * self._nesting + f'{self.region} : { self.elapsed_ms() :.1f}ms ({ self.elapsed_percent() :.0f}%)')	
	
	def elapsed_ms(self) -> float:
		return (get_time() - self.start_time) * 1000.0

	def elapsed_percent(self) -> float:
		ft = RenderContext.frame_time + 0.00001
		return (get_time() - self.start_time) / ft * 100.0
	
	@staticmethod
	def capture() -> str:
		WatchTimer.report = '\n'.join( list(map(
			lambda t: t.message if hasattr(t, 'message') else t.get_message(),
			WatchTimer.timers))
		)
		WatchTimer.timers.clear()
		return WatchTimer.report

	#def display(x, y, size, color):
	#	rl.DrawText(WatchTimer.report.encode(), x, y, size, color)


def make_cube_mesh():
	return Mesh(interleave_mesh_position_normal_uv(
		positions = np.array((
		# +Z (front)
		(-0.5,-0.5,+0.5), (+0.5,-0.5,+0.5), (+0.5,+0.5,+0.5), (-0.5,+0.5,+0.5),
		# -Z (back)
		(+0.5,-0.5,-0.5), (-0.5,-0.5,-0.5), (-0.5,+0.5,-0.5), (+0.5,+0.5,-0.5),
		# +X (right)
		(+0.5,-0.5,+0.5), (+0.5,-0.5,-0.5), (+0.5,+0.5,-0.5), (+0.5,+0.5,+0.5),
		# -X (left)
		(-0.5,-0.5,-0.5), (-0.5,-0.5,+0.5), (-0.5,+0.5,+0.5), (-0.5,+0.5,-0.5),
		# +Y (top)
		(-0.5,+0.5,+0.5), (+0.5,+0.5,+0.5), (+0.5,+0.5,-0.5), (-0.5,+0.5,-0.5),
		# -Y (bottom)
		(-0.5,-0.5,-0.5), (+0.5,-0.5,-0.5), (+0.5,-0.5,+0.5), (-0.5,-0.5,+0.5),
	), dtype=np.float32),
		normals=np.array(
			([ Vec3(  0.0, 0.0, 1.0 ) ] * 4) +   # +Z
			([ Vec3(  0.0, 0.0,-1.0 ) ] * 4) +   # -Z
			([ Vec3(  1.0, 0.0, 0.0 ) ] * 4) +   # +X
			([ Vec3( -1.0, 0.0, 0.0 ) ] * 4) +   # -X
			([ Vec3(  0.0, 1.0, 0.0 ) ] * 4) +   # +Y
			([ Vec3(  0.0,-1.0, 0.0 ) ] * 4),    # -Y
			dtype=np.float32),
		uvs = np.array([ (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 6, dtype=np.float32)
	),
	indices =np.array([
		0, 1, 2,  2, 3, 0,		 # +Z
		4, 5, 6,  6, 7, 4,		 # -Z
		8, 9, 10,  10, 11, 8,	 # +X
		12, 13, 14,  14, 15, 12, # -X
		16, 17, 18,  18, 19, 16, # +Y
		20, 21, 22,  22, 23, 20, # -Y
	], dtype=np.uint32)
)

mesh_instance_dtype = np.dtype([
	("iPosition",  np.float32, 3), # 12 bytes
	("iTint",      np.uint32,  1), # 4 bytes  (fills align 16 gap)
	("iRotation",  np.uint32,  2), # 8 bytes  (snorm16 mapped to [-1, 1])
	("iScale",     np.uint32,  2), # 8 bytes  (4x float16: x, y, z, 0)
])


# quantize quaternion to int16 snorm [-32767, 32767]
def pack_quaternion(rot):
	rot_arr = np.asarray(rot)
	#if rot_arr.dtype == object: rot_arr = np.stack(rot_arr)
	snorm16 = np.round(np.clip(rot_arr, -1.0, 1.0) * 32767.0).astype(np.int16)
	u32_view = snorm16.view(np.uint32)
	return u32_view

# put 3 f16 into 2 u32
# NOTE: could pack something else in last 16bits
def pack_scale(scale):
	scl_arr = np.asarray(scale, dtype=np.float32)
	f16 = np.zeros((*scl_arr.shape[:-1], 4), dtype=np.float16)
	f16[..., :3] = scl_arr
	return f16.view(np.uint32)


def make_instances(position, rotation, scale, tint):
	n = len(position["x"])

	out = np.empty(n, dtype=mesh_instance_dtype)

	out["iPosition"] = np.column_stack([
		position["x"],
		position["y"],
		position["z"],
	]).astype(np.float32, copy=False)

	out["iRotation"] = np.column_stack([
		rotation["x"],
		rotation["y"],
		rotation["z"],
		rotation["w"],
	]).astype(np.float16, copy=False)

	out["iScale"][:, :3] = np.column_stack([
		scale["x"],
		scale["y"],
		scale["z"]
	]).astype(np.float16, copy=False)

	out["iTint"][:,0] = tint['value'].astype(np.uint32, copy=False)

	return out


def linear_to_srgb(x):
	x = np.asarray(x, dtype=np.float32)
	return np.where(
		x <= 0.0031308,
		x * 12.92,
		1.055 * np.power(x, 1.0 / 2.4) - 0.055,
	)

_RGBA_SHIFT = np.array([0, 8, 16, 24], dtype=np.uint32)
def pack_rgba8_srgb(rgba):
	rgba = np.array([ *linear_to_srgb(rgba[:3]), rgba[3]])
	rgba8 = np.rint(rgba * 255).astype(np.uint32)
	return np.uint32(np.sum(rgba8 << _RGBA_SHIFT))

"""
import timeit
import math
import collections

def test(f):
	for i in range(1, 1000001): f(i)

def timetest(f):
	print('{}: {}'.format(timeit.timeit(lambda: test(f), number=10), f.__name__))
"""

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
	import psutil # type: ignore[import-untyped]  # NOTE: this is not a built-in lib
	process = psutil.Process(os.getpid())

	import tracemalloc
	tracemalloc.start()

	def print_report() -> None:
		rss = process.memory_info().rss
		print(f"process RSS: {rss / 1024 / 1024:.1f} MiB")
		current, peak = tracemalloc.get_traced_memory()
		print(f"traced memory: {current / 1024 / 1024:.2f} MiB")

	return print_report

