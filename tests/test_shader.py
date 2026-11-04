import numpy as np
import pytest

from RenderContext import *
from pyrr import Matrix44 as Mat4


RenderContext.setup_graphics_backend(highpower_gpu=False)

shader1 = Shader(filepath="tests/test.shader")
shader2 = Shader(filepath="tests/test.shader", features={"BIAS"})
shader3 = Shader(filepath="tests/test.shader", features={"BIAS"}, params={"bias": 0.01} )

bias_decl = "const bias: f32 = "
func_decl = "fn plus_one(bias: f32) -> f32"
vertex_bias = "position.z *= plus_one(bias);"


def test_shader_variants():
	assert bias_decl not in shader1.info.source
	assert "position.z *=" not in shader1.info.source

	assert bias_decl + "0.0001" in shader2.info.source
	assert vertex_bias in shader2.info.source

	assert bias_decl + "0.01" in shader3.info.source
	assert vertex_bias in shader3.info.source

	for shader in (shader1, shader2, shader3):
		assert func_decl in shader.info.source if shader is not shader1 else func_decl not in shader.info.source

	assert shader1.info.vertex_dtype == np.dtype([
		("vertexPosition", np.float32, (3,)),
		("vertexNormal", np.float32, (3,)),
		("vertexTexCoord", np.float32, (2,)),
	])


def test_uniform_buffer():
	u = shader1.UniformBuffer()

	u.content["mvp"] = Mat4.from_scale([1.0, 1.0, 1.0], dtype=np.float32)
	u.upload()