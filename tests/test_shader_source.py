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


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("dtype", ["f32", "i32", "u32", "depth"])
@pytest.mark.parametrize("layered", [False, True])
def test_sample_texture_macros_and_shorthand(count, dtype, layered):
	options = {"sample_count": count, "dtype": dtype, "layered": layered}
	explicit = render('''
#from "utils_texture.shaderlib" import sample_texture_type, sample_texture_load with context
{{ binding('image', sample_texture_type(PARAMS.dtype, layered=PARAMS.layered), 1, 2) }}
fn load(pixel: vec2i, sample: u32) -> vec4f {
	return vec4f({{ sample_texture_load('image', 'pixel', 'sample', layered=PARAMS.layered) }});
}
''', **options)
	layout = ", layered" if layered else ""
	shorthand = render(f'''
sample_texture<1, 2{layout}> image: {dtype};
fn load(pixel: vec2i, sample: u32) -> vec4f {{
	return vec4f(sample_texture_load(image, pixel, sample, layered={str(layered).lower()}));
}}
''', **options)
	prefix = "texture_depth_" if dtype == "depth" else "texture_"
	expected_type = prefix+("multisampled_" if count > 1 and not layered else "")+"2d"
	if count > 1 and layered: expected_type += "_array"
	if dtype != "depth": expected_type += f"<{dtype}>"
	expected_read = "textureLoad(image, pixel, "+("i32(sample)" if count > 1 else "0")+(", 0" if count > 1 and layered else "")+")"
	for info in (explicit, shorthand):
		assert info.bindings["image"].type_name == expected_type
		assert (info.bindings["image"].group, info.bindings["image"].binding) == (1, 2)
		assert expected_read in info.source
		assert set(info.used_bindings(("load",))) == {"image"}
		assert "fn texture_lod_from_gradients" not in info.source


def test_texture_shorthand_supports_regular_and_storage_texture_types():
	loader = Shader.__new__(Shader)
	code, declarations = loader._render_source('''
texture<1, 0> source: texture_2d<u32>; // source image
texture<1, 1> target: texture_storage_2d<rgba16float, write>;
''', None, None, None, None)
	info = ShaderSource(code, declarations=declarations)
	assert info.bindings["source"].type_name == "texture_2d<u32>"
	assert info.bindings["target"].type_name == "texture_storage_2d<rgba16float, write>"
	assert "// source image" in code


def test_sample_texture_shorthand_handles_nested_multiline_expressions_and_conditions():
	info = render('''
#if PARAMS.enabled
sample_texture<1, 2, layered> image: f32;
#endif
#if not PARAMS.enabled
sample_texture<1, 3> unused: f32;
#endif
fn load(id: vec3u) -> vec4f {
	return sample_texture_load(
		image,
		vec2i(clamp(id.xy, vec2u(0), vec2u(7))),
		select(0u, id.z, id.z > 0u),
		layered=true
	).xyzw;
}
''', enabled=True, sample_count=4)
	assert set(info.bindings) == {"image"}
	assert "textureLoad(image, vec2i(clamp(id.xy, vec2u(0), vec2u(7))), i32(select(0u, id.z, id.z > 0u)), 0).xyzw" in info.source
	assert set(info.used_bindings(("load",))) == {"image"}


def test_sample_texture_shorthand_defaults_to_single_sample_and_preserves_comments():
	loader = Shader.__new__(Shader)
	code, declarations = loader._render_source('''
sample_texture<1, 0> image: f32;
// sample_texture_load(this is not a call
/* sample_texture_load(neither is this */
fn load() -> vec4f { return sample_texture_load(image, vec2i(0), 0u); }
''', None, None, None, None)
	info = ShaderSource(code, declarations=declarations)
	assert info.bindings["image"].type_name == "texture_2d<f32>"
	assert "return textureLoad(image, vec2i(0), 0);" in info.source
	assert "// sample_texture_load(this is not a call" in code
	assert "/* sample_texture_load(neither is this */" in code


@pytest.mark.parametrize("source,match", [
	("sample_texture<0, 0> image: f16;", "Invalid sample_texture type"),
	("sample_texture<0, 0, write> image: f32;", "layout"),
	("sample_texture<0, 0> image: f32", "Invalid WGSL shorthand"),
	("texture<0, 0, write> image: texture_2d<f32>;", "access argument"),
	("let x = sample_texture_load(image, pixel);", "requires texture"),
	("let x = sample_texture_load(image, pixel, sample, layered=maybe);", "layered must be"),
	("let x = sample_texture_load(image, pixel, sample;", "Unclosed"),
	("let x = sample_texture_load(image, pixel], sample);", "Unbalanced"),
])
def test_invalid_texture_shorthand_is_rejected(source, match):
	with pytest.raises(ValueError, match=match):
		render(source)


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("pixel,sample,expected_pixel,expected_sample", [
	("vec2i({{ PARAMS.x }}, {{ PARAMS.y }})", "{{ PARAMS.sample }}u", "vec2i(2, 3)", "1u"),
	("array<vec2i, 2>(vec2i(0), vec2i(1))[0]", "0u", "array<vec2i, 2>(vec2i(0), vec2i(1))[0]", "0u"),
	("array<array<vec2i, 2>, 1>(array<vec2i, 2>(vec2i(0), vec2i(1)))[0][0]", "0u", "array<array<vec2i, 2>, 1>(array<vec2i, 2>(vec2i(0), vec2i(1)))[0][0]", "0u"),
	("vec2i(select(0, 1, 0 < 1), select(0, 1, 2 > 1))", "0u", "vec2i(select(0, 1, 0 < 1), select(0, 1, 2 > 1))", "0u"),
])
def test_sample_texture_load_template_arguments(count, pixel, sample, expected_pixel, expected_sample):
	info = render("sample_texture<0, 0> image: f32;\nfn load() -> vec4f { return sample_texture_load(image, "+pixel+", "+sample+"); }", sample_count=count, x=2, y=3, sample=1)
	index = "i32("+expected_sample+")" if count > 1 else "0"
	assert "textureLoad(image, "+expected_pixel+", "+index+")" in info.source
	assert "{{" not in info.source
