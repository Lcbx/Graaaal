"""Reusable CPU loading independent of the transparency example's GPU layouts."""
import base64
import json

import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2

from graaal.Assets import load_gltf_first_mesh, load_gltf_scene, read_gltf_accessor


@pytest.mark.parametrize("source", ["external", "data", "glb"])
def test_accessor_reads_interleaved_normalized_data_and_offsets(tmp_path, source):
	data = b"HEAD" + bytes([99, 99, 255, 128, 0, 99, 0, 64, 255, 99])
	uri = "buffer.bin"
	if source == "external":
		(tmp_path / uri).write_bytes(data)
	elif source == "data":
		uri = "data:application/octet-stream;base64," + base64.b64encode(data).decode()
	else:
		uri = None
	gltf = GLTF2.from_dict({
		"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}],
		"buffers": [{"byteLength": len(data), "uri": uri}],
		"bufferViews": [{"buffer": 0, "byteOffset": 4, "byteLength": 10, "byteStride": 4}],
		"accessors": [{"bufferView": 0, "byteOffset": 2, "componentType": 5121,
			"normalized": True, "count": 2, "type": "VEC3"}],
	})
	if source == "glb":
		gltf.set_binary_blob(data)
	values = read_gltf_accessor(gltf, 0, tmp_path)
	np.testing.assert_allclose(values, [[1, 128/255, 0], [0, 64/255, 1]])
	values[0] = 0  # Decoded accessor output is independently writable.
	assert read_gltf_accessor(gltf, 0, tmp_path)[0, 0] == 1


def test_accessor_reads_padded_matrix_columns():
	data = bytes([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0])
	gltf = GLTF2.from_dict({
		"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}],
		"buffers": [{"byteLength": len(data)}],
		"bufferViews": [{"buffer": 0, "byteLength": len(data)}],
		"accessors": [{"bufferView": 0, "componentType": 5121, "count": 1, "type": "MAT3"}],
	})
	gltf.set_binary_blob(data)
	np.testing.assert_array_equal(read_gltf_accessor(gltf, 0), [list(range(1, 10))])


def test_scene_loader_bakes_hierarchy_preserves_materials_and_decodes_images(tmp_path):
	positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
	(tmp_path / "mesh.bin").write_bytes(positions.tobytes())
	pixels = np.array([[[240, 80, 10, 128]]], dtype=np.uint8)
	Image.fromarray(pixels).save(tmp_path / "color.png")
	document = {
		"asset": {"version": "2.0"},
		"buffers": [{"byteLength": positions.nbytes, "uri": "mesh.bin"}],
		"bufferViews": [{"buffer": 0, "byteLength": positions.nbytes}],
		"accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
		"images": [{"uri": "color.png"}], "textures": [{"source": 0}],
		"materials": [{"alphaMode": "BLEND", "doubleSided": True,
			"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "roughnessFactor": 0.3}}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
		"nodes": [{"translation": [2, 3, 4], "children": [1]}, {"mesh": 0, "scale": [-2, 1, 1]}],
		"scenes": [{"nodes": [0]}], "scene": 0,
	}
	path = tmp_path / "scene.gltf"
	path.write_text(json.dumps(document))
	primitives, materials, images = load_gltf_scene(path)
	assert len(primitives) == 1 and primitives[0].material_id == 0
	assert primitives[0].vertices.dtype.names == ("position", "normal", "uv")
	np.testing.assert_allclose(primitives[0].vertices["position"], [[2, 3, 4], [2, 4, 4], [0, 3, 4]])
	np.testing.assert_allclose(primitives[0].vertices["normal"], [[0, 0, 1]]*3)
	np.testing.assert_array_equal(primitives[0].vertices["uv"], 0)
	assert materials[0].alphaMode == "BLEND" and materials[0].doubleSided
	assert materials[0].pbrMetallicRoughness.roughnessFactor == 0.3
	assert materials[-1] is None
	np.testing.assert_array_equal(images[0], pixels)
	np.testing.assert_array_equal(images[-1], 255)
	# The lightweight loader shares buffer handling but keeps local coordinates.
	local_positions, normals, uvs, indices = load_gltf_first_mesh(str(path))
	np.testing.assert_array_equal(local_positions, positions)
	assert normals.shape == (3, 3) and uvs.shape == (3, 2)
	np.testing.assert_array_equal(indices, [0, 1, 2])
