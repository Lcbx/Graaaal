from __future__ import annotations

import numpy as np


MESH_INSTANCE_DTYPE = np.dtype([
	("iPosition",  np.float32, 3), # 12 bytes
	("iTint",      np.uint32,  1), # 4 bytes  (fills align 16 gap)
	("iRotation",  np.uint32,  2), # 8 bytes  (snorm16 mapped to [-1, 1])
	("iScale",     np.uint32,  2), # 8 bytes  (4x float16: x, y, z, 0)
])


def pack_quaternion(rot):
	rot_arr = np.asarray(rot)
	#if rot_arr.dtype == object: rot_arr = np.stack(rot_arr)
	snorm16 = np.round(np.clip(rot_arr, -1.0, 1.0) * 32767.0).astype(np.int16)
	u32_view = snorm16.view(np.uint32)
	return u32_view


def pack_scale(scale):
	scl_arr = np.asarray(scale, dtype=np.float32)
	f16 = np.zeros((*scl_arr.shape[:-1], 4), dtype=np.float16)
	f16[..., :3] = scl_arr
	return f16.view(np.uint32)


def make_instances(position, rotation, scale, tint):
	n = len(position["x"])

	out = np.empty(n, dtype=MESH_INSTANCE_DTYPE)

	out["iPosition"] = np.column_stack([
		position["x"],
		position["y"],
		position["z"],
	]).astype(np.float32, copy=False)

	out["iRotation"] = pack_quaternion(np.column_stack([
		rotation["x"],
		rotation["y"],
		rotation["z"],
		rotation["w"],
	]).astype(np.float32, copy=False))

	out["iScale"] = pack_scale(np.column_stack([
		scale["x"],
		scale["y"],
		scale["z"]
	]).astype(np.float32, copy=False))

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

