from __future__ import annotations

from RenderContext import *
from RenderContext import Mesh, RenderContext

import gc, os
from time import perf_counter as get_time

from functools import wraps
from mypy_extensions import mypyc_attr
from typing import Any, Type, Sequence, Iterator, Iterable, List, Dict, Tuple, Callable, ParamSpec, TypeVar, ClassVar

import numpy as np
from pyrr import Matrix44 as Mat4, Quaternion  # type: ignore[import-untyped]
from pygltflib import GLTF2, BufferView, Accessor  # type: ignore[import-untyped]


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



def _get_data_from_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
	acc: Accessor = gltf.accessors[accessor_index]
	bv: BufferView = gltf.bufferViews[acc.bufferView]
	buf = gltf.buffers[bv.buffer]

	if buf.uri is None:  # GLB
		bin_chunk = gltf.binary_blob()
	else:
		raise NotImplementedError("External buffers not handled")

	b = bv.byteOffset or 0
	e = b + (bv.byteLength or 0)
	view_bytes = memoryview(bin_chunk)[b:e]

	type_num_comps = {
		"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16
	}[acc.type]

	np_dtype = {
		5120: np.int8,
		5121: np.uint8,
		5122: np.int16,
		5123: np.uint16,
		5125: np.uint32,
		5126: np.float32
	}[acc.componentType]

	stride = bv.byteStride or (np.dtype(np_dtype).itemsize * type_num_comps)
	count = acc.count
	offset = acc.byteOffset or 0
	tight = np.dtype(np_dtype).itemsize * type_num_comps

	arr = np.frombuffer(view_bytes, dtype=np_dtype, count=count * type_num_comps, offset=offset)
	if stride != tight:
		rec = np.empty((count, type_num_comps), dtype=np_dtype)
		base = offset
		for i in range(count):
			start = base + i * stride
			rec[i] = np.frombuffer(view_bytes, dtype=np_dtype, count=type_num_comps, offset=start)
		return rec
	else:
		return arr.reshape(count, type_num_comps)



def load_gltf_first_mesh(glb_path: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	gltf = GLTF2().load(glb_path)
	mesh = gltf.meshes[0]
	prim = mesh.primitives[0]
	
	#print(f'{mesh=}')

	pos =   _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
	pos_count = pos.shape[0]
	idx =   _get_data_from_accessor(gltf, prim.indices).astype(
		np.uint32 if pos_count.bit_length() > 16 else np.uint16
	)
	nor = ( _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32)
		if prim.attributes.NORMAL is not None
		else np.zeros_like( pos.shape, dtype=np.float32)
	)
	uv  = ( _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float16)
		if prim.attributes.TEXCOORD_0 is not None
		else np.zeros( (pos_count, 2), dtype=np.float16)
	)

	return pos, nor, uv, idx


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


""" # not perfect
def load_gltf_meshes(program : Program, glb_path: str)-> IndexedVertexList:
	gltf = GLTF2().load(glb_path)

	meshes = []
	for mesh in gltf.meshes:
		for prim in mesh.primitives:
			pos = _get_data_from_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
			nor = _get_data_from_accessor(gltf, prim.attributes.NORMAL).astype(np.float32) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
			uv  = _get_data_from_accessor(gltf, prim.attributes.TEXCOORD_0).astype(np.float32) if prim.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0],2), dtype=np.float32)
			idx = _get_data_from_accessor(gltf, prim.indices)
			if idx.dtype != np.uint32:
				idx = idx.astype(np.uint32)

		model = Mesh(program, pos, nor, uv, idx)
		meshes.append(model)

	return meshes
"""

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

