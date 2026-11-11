"""Jinja shader declaration metadata, independent of GPU creation."""
import numpy as np
import pytest

from numpyecs.RenderContext import Shader, ShaderSource


def render(source, **params):
	loader = Shader.__new__(Shader)
	code, declarations = loader._render_source(source, None, None, None, params)
	return ShaderSource(code, declarations=declarations)


def test_struct_uniform_and_vertex_metadata_follow_jinja_declarations():
	info = render('''
<Frame> view: mat4x4f;
<Frame> color: vec4f;
uniform<0, 0> frame: Frame;
<VertexInput> position: vec3f;
<VertexInput> @location(3) normal: vec3f;
<VertexInput> uv: vec2f;
<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> @location(0) @interpolate(flat) color: vec4f;
@vertex fn vertex(input: VertexInput) -> VertexOutput { return VertexOutput(vec4f(input.position, 1), frame.color); }
@fragment fn fragment(input: VertexOutput) -> @location(0) vec4f { return input.color; }
''')
	assert set(info.structs) == {"Frame", "VertexInput", "VertexOutput"}
	assert set(info.bindings) == {"frame"}
	assert info.uniforms["frame"].dtype.itemsize == 80
	assert info.vertex_dtype_for("vertex") == np.dtype([
		("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2)])
	assert [field.location for field in info.vertex_inputs("vertex")] == [0, 3, 4]
	assert "@location(4) uv: vec2f" in info.source
	assert "@interpolate(flat) color: vec4f" in info.source
	assert set(info.used_bindings(("vertex",))) == {"frame"}


def test_jinja_conditions_and_included_macros_share_registration():
	info = render('''
#include "draw_common.shaderlib"
#from "draw_common.shaderlib" import instance_bindings
{{ instance_bindings() }}
#if PARAMS.extra
{{ binding("extra", "texture_2d<f32>", 2, 0) }}
#endif
@vertex fn vertex(@location(0) position: vec3f) -> @builtin(position) vec4f { return vec4f(position, 1); }
''', extra=False)
	assert len(info.structs["MeshInstance"].fields) == 4
	assert len(info.bindings) == 3
	assert "extra" not in info.bindings
	assert render('{{ binding("extra", "texture_2d<f32>", 2, 0) }}').bindings["extra"].binding == 0


def test_shader_bindings_are_not_reached_through_field_names():
	info = render('''
<Surface> image: vec4f;
{{ binding("image", "texture_2d<f32>", 0, 0) }}
fn load_image() -> vec4f { return textureLoad(image, vec2i(0), 0); }
@compute @workgroup_size(1) fn main() {
	var surface: Surface;
	let value = surface.image;
}
''')
	assert info.used_bindings(("main",)) == {}
	assert set(info.used_bindings(("load_image",))) == {"image"}


@pytest.mark.parametrize("source,match", [
	('{{ field("Frame", "invalid-name", "f32") }}', "identifier"),
	('{{ field("Frame", "x", "f32") }}{{ field("Frame", "x", "u32") }}', "Conflicting"),
	('{{ binding("a", "f32", 0, 1) }}{{ binding("b", "f32", 0, 1) }}', "binding slot"),
	('{{ binding("a", "f32", 0, 1, access="read") }}', "access"),
])
def test_invalid_jinja_shader_declarations(source, match):
	with pytest.raises(ValueError, match=match):
		render(source)


def test_uniforms_require_known_struct_and_can_be_explicitly_overridden():
	with pytest.raises(TypeError, match="named WGSL struct"):
		render('{{ binding("frame", "array<f32, 4>", 0, 0, storage="uniform") }}')
	with pytest.raises(TypeError, match="Unsupported WGSL uniform"):
		render('{{ field("Frame", "nested", "Unknown") }}{{ binding("frame", "Frame", 0, 0, storage="uniform") }}')


def test_shorthand_declares_struct_fields_and_resource_bindings():
	loader = Shader.__new__(Shader)
	code, declarations = loader._render_source('''
<Frame> view: mat4x4f;
<Frame> color: vec4f = (1, 0.5, 0, 1);
uniform<0, 0, read> frame: Frame;
storage<1, 0, read_write> output: array<vec4f>;
<VertexInput> position: vec3f;
<VertexInput> @location(3) normal: vec3f;
<VertexInput> uv: vec2f;
<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> @location(0) @interpolate(flat) color: vec4f;
@vertex fn vertex(input: VertexInput) -> VertexOutput {
	return VertexOutput(vec4f(input.position, 1), frame.color);
}
''', None, None, None, None)
	info = ShaderSource(code, declarations=declarations)
	assert declarations.defaults["Frame"]["color"] == (1, 0.5, 0, 1)
	assert info.defaults["Frame"]["color"] == (1, 0.5, 0, 1)
	assert info.uniforms["frame"].dtype.itemsize == 80
	assert info.bindings["frame"].access is None
	assert (info.bindings["output"].group, info.bindings["output"].access) == (1, "read_write")
	assert [field.location for field in info.vertex_inputs("vertex")] == [0, 3, 4]
	assert "@interpolate(flat) color: vec4f" in code
	assert "<Frame>" not in code


def test_shorthand_respects_jinja_conditions_and_inline_templates():
	loader = Shader.__new__(Shader)
	code, declarations = loader._render_source('''
#if PARAMS.enabled
<Data> value: u32 = 2 if PARAMS.enabled else 0;
uniform<0, 1> data: Data;
#endif
#if not PARAMS.enabled
storage<0, 2, read> unused: array<u32>;
#endif
''', None, None, None, {"enabled": True})
	assert declarations.defaults["Data"] == {"value": 2}
	assert set(declarations.bindings) == {"data"}
	assert "unused" not in code


@pytest.mark.parametrize("declaration,match", [
	("uniform<0, 0, read_write> invalid: Frame;", "read-only"),
	("storage<0, 0, write> invalid: array<u32>;", "Storage access"),
	("<Frame> @unknown(1) x: f32;", "Invalid WGSL shorthand"),
	("<Frame> x: f32", "Invalid WGSL shorthand"),
	("<Frame> @location(foo) x: f32;", "Invalid @location"),
])
def test_invalid_shorthand_is_rejected(declaration, match):
	with pytest.raises(ValueError, match=match):
		render(declaration)


def test_shorthand_repeated_includes_reuse_identical_fields():
	info = render('''
#include "draw_common.shaderlib"
#from "draw_common.shaderlib" import instance_bindings
{{ instance_bindings() }}
''')
	assert len(info.structs["MeshInstance"].fields) == 4
	assert set(info.bindings) == {"uniforms", "instances", "visible_instances"}


def test_shorthand_conflicting_defaults_are_rejected():
	with pytest.raises(ValueError, match="Conflicting default"):
		render("<Frame> a: f32 = 1;\n<Frame> a: f32 = 2;")
