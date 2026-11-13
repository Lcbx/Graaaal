from __future__ import annotations

from typing import ClassVar

import numpy as np

from .utils import cache_last


def _array_accessor(index:int=0) -> property:
	return property(
		lambda v: v[index],
		lambda v, y: v.__setitem__(index, y),
	)


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


class Vec2(_baseVector):
	_count = 2
	x = _array_accessor(0)
	y = _array_accessor(1)


class Vec3(_baseVector):
	_count = 3
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)


class Vec4(_baseVector):
	_count = 4
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)
	w = _array_accessor(3)


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
	def view(self) -> np.ndarray:
		return Camera.calc_view(self.position, self.target, self.up)
	
	@staticmethod
	def calc_view(position, target:Vec3, up:Vec3) -> np.ndarray:
		position, target, up = (np.asarray(value, dtype=np.float64) for value in (position, target, up))
		forward = target - position
		if not np.all(np.isfinite([position, target, up])) or np.linalg.norm(forward) == 0:
			raise ValueError("Expected finite camera values with distinct position and target")
		forward /= np.linalg.norm(forward)
		side = np.cross(forward, up)
		if np.linalg.norm(side) == 0:
			raise ValueError("Camera up must not be parallel to the view direction")
		side /= np.linalg.norm(side)
		up = np.cross(side, forward)
		matrix = np.eye(4, dtype=np.float32)
		matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = side, up, -forward
		matrix[3, :3] = (-np.dot(side, position), -np.dot(up, position), np.dot(forward, position))
		return matrix
	
	# NOTE: we could split this class into orth and perspective cameras	
	@cache_last(lambda self, aspect: (self.perspective, self.fovy_deg, aspect, self.near, self.far))
	def projection(self, aspect:float) -> np.ndarray:
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
	def perspective_projection(fovy_deg:float, aspect:float, near:float, far:float) -> np.ndarray:
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
	def orthogonal_projection(left:float, right:float, bottom:float, top:float, near:float, far:float) -> np.ndarray:

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


def extract_frustum_planes(view_projection):
	vp = np.asarray(view_projection)
	planes = np.zeros((6, 4), dtype=np.float32)
	planes[0], planes[1] = vp[:, 3] + vp[:, 0], vp[:, 3] - vp[:, 0]
	planes[2], planes[3] = vp[:, 3] + vp[:, 1], vp[:, 3] - vp[:, 1]
	planes[4], planes[5] = vp[:, 2], vp[:, 3] - vp[:, 2]
	for i in range(6):
		norm = np.linalg.norm(planes[i, :3])
		if norm > 0: planes[i] /= norm
	return planes

