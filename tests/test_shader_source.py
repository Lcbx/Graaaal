"""Shared WGSL reflection spans and hygienic transformations, without materials."""
import pytest

from numpyecs.RenderContext import RenderContext, Shader, ShaderSource


def test_spans_and_initializers_address_expanded_source_with_nested_comments():
	info = ShaderSource('''
// fn comment_only() { braces are not code }
/* outer /* nested */ const fake = 0; */
<Frame> color: vec3f;
const color = vec3f(0.1, 0.2, 0.3);
fn helper(color: vec3f,) -> vec3f { return color; }
''')
	assert set(info.declarations) == {"Frame", "color", "helper"}
	constant = info.declarations["color"]
	assert constant.kind == "const"
	assert constant.initializer == "vec3f(0.1, 0.2, 0.3)"
	for declaration in info.declarations.values():
		assert info.source[declaration.name_span.start:declaration.name_span.end] == declaration.name
	function = info.functions["helper"]
	parameter = function.parameters[0]
	member = info.structs["Frame"].fields[0]
	for item in (parameter, member, function):
		assert info.source[item.name_span.start:item.name_span.end] == item.name
	assert info.source[function.body_start] == "{"
	transformed = ShaderSource(info.rename_symbols({"color": "shared_color", "helper": "apply"}))
	assert transformed.functions["apply"].parameters[0].name == "color"
	assert transformed.structs["Frame"].fields[0].name == "color"
	assert set(transformed.declarations) == {"Frame", "shared_color", "apply"}


def test_shared_renaming_preserves_attribute_enums_and_compiles():
	info = ShaderSource('''
const alignment = 16u;
const e = 1.0e-2;
const x1p = 0x1p-1f;
const flat = 0.0;
const position = 0.25+e+x1p+flat;
struct Vertex {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) @align(alignment) value: f32,
}
@vertex fn vertex(@builtin(vertex_index) index: u32,) -> Vertex {
    return Vertex(vec4f(position, 0.0, 0.0, 1.0), f32(index));
}
''')
	renamed = info.rename_symbols({"Vertex": "VertexOutput", "position": "offset", "alignment": "size", "vertex": "main",
		"e": "epsilon", "x1p": "hex_value", "flat": "interpolation"})
	updated = ShaderSource(renamed)
	assert updated.structs["VertexOutput"].fields[0].attribute("builtin").value == "position"
	assert updated.structs["VertexOutput"].fields[1].attribute("align").value == "size"
	RenderContext.setup_graphics_backend(True)
	shader = Shader(source=renamed)
	assert shader.entry("vertex") == "main"


def test_shared_renaming_rejects_module_collisions():
	info = ShaderSource("const first = 1.0; const second = 2.0;")
	with pytest.raises(ValueError, match="collide"):
		info.rename_symbols({"first": "second"})
	with pytest.raises(ValueError, match="identifiers"):
		info.rename_symbols({"first": "bad-name"})


def test_binding_reachability_uses_shared_function_bodies():
	info = ShaderSource('''
@group(0) @binding(0) var<storage, read> used: array<f32>;
@group(0) @binding(1) var<storage, read> unused: array<f32>;
/* fn fake() { unused[0]; /* nested */ } */
fn helper(value: array<f32, u32(max(1, 2))>) -> f32 { return used[0]+value[0]; }
@compute @workgroup_size(1) fn main() { let value = helper(array<f32, 2>(1.0, 2.0)); }
''')
	assert set(info.used_bindings(["main"])) == {"used"}
