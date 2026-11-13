from __future__ import annotations

import numpy as np

from .Assets import interleave_mesh_position_normal_uv
from .Math import Vec3


def make_cube_mesh():
	from .RenderContext import Mesh
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

