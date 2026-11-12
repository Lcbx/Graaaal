import numpy as np
import pytest

from numpyecs.RenderContext import *
from pyrr import Matrix44 as Mat4


@pytest.fixture(autouse=True)
def shaders():
	global shader1, shader2, shader3
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
	u = shader1.uniform_buffer()

	u.content["mvp"] = Mat4.from_scale([1.0, 1.0, 1.0], dtype=np.float32)
	u.upload()


@pytest.mark.parametrize("count", [1, 4])
def test_sample_texture_shorthand_compiles(count):
	shader = Shader(source='''
sample_texture<0, 0> color: f32;
sample_texture<0, 1> ids: u32;
sample_texture<0, 2> depth: depth;
sample_texture<0, 3, layered> shaded: f32;
texture<0, 4> output: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
	let pixel = vec2i(id.xy);
	let value = sample_texture_load(color, pixel, id.z)
		+ sample_texture_load(color, array<vec2i, 2>(vec2i({{ PARAMS.x }}, 0), vec2i(1))[0], id.z)
		+ vec4f(sample_texture_load(ids, pixel, id.z))
		+ vec4f(sample_texture_load(depth, pixel, id.z))
		+ sample_texture_load(shaded, pixel, id.z, layered=true);
	textureStore(output, pixel, value);
}
''', params={"sample_count": count, "x": 0})
	assert set(shader.info.used_bindings(("main",))) == {"color", "ids", "depth", "shaded", "output"}
