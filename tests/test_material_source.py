"""Plain WGSL parameter reflection, material contracts and linked-family shading."""
import numpy as np
import pytest

from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.Materials import MaterialBinding, MaterialLibrary
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, ShaderSource, Texture, TextureUsage
from numpyecs.TransparencyData import VERTEX_DTYPE


def test_reflection_exposes_material_functions_without_treating_them_as_entry_points():
	info = ShaderSource('''
fn helper(value: vec3f,) -> f32 { return value.x; }
fn generic(value: ptr<function, array<vec3f, 2>>,) {}
@compute @workgroup_size(1) fn main() {}
''')
	assert set(info.functions) == {"helper", "generic", "main"}
	assert set(info.entry_points) == {"main"}
	assert info.functions["helper"].parameters[0].type_name == "vec3f"
	assert info.functions["helper"].return_type == "f32"
	assert info.functions["generic"].parameters[0].type_name == "ptr<function,array<vec3f,2>>"


def test_source_parameters_keep_shader_defaults_and_instance_scopes(tmp_path, monkeypatch):
	path = tmp_path / "custom.wgsl"
	path.write_text('''
struct MaterialParameters { color: vec3<f32>, count: u32, enabled: bool, roughness: f32, }
struct InstanceParameters { phase: f32, grazing_fade: bool, }
const default_color = vec3<f32>(0.25f);
const default_count: u32 = 0x10u;
const default_enabled = true;
const default_phase = -1.5e-2;
''')
	monkeypatch.chdir(tmp_path)
	shader = DefaultMaterialShader(filepath=path)
	assert shader.parameters["color"].default == (0.25,)*3
	assert shader.parameters["count"].default == 16
	assert shader.parameters["enabled"].default is True
	assert shader.parameters["roughness"].default == 0.7
	assert shader.parameters["phase"].default == -0.015
	assert shader.parameters["grazing_fade"].scope == "instance"
	library = MaterialLibrary([shader])
	shared = {"count": 2}
	a = library.pack([MaterialBinding(shader, shared), MaterialBinding(shader, shared, {"phase": 1.0})])
	assert len(a[0]) == 1  # schema changes do not break dictionary identity sharing
	np.testing.assert_array_equal(a[3], [0, 0])
	with pytest.raises(ValueError, match="shared"):
		library.pack([MaterialBinding(shader, overrides={"count": 3})])
	shader.source = "changed"
	with pytest.raises(ValueError, match="layout"):
		library.pack([])


@pytest.mark.parametrize("source, message", [
	('fn shade() -> vec3f { return vec3f(1.0); }', "signature"),
	('struct MaterialParameters { base_color: f32, }', "base_color"),
	('struct MaterialParameters { value: f32, } struct InstanceParameters { value: f32, }', "duplicate"),
	('struct MaterialParameters { value: mat4x4f, }', "parameter type"),
	('struct MaterialParameters { value: f32, } const default_value = 1.0+2.0;', "default_value"),
	('struct Params { value: f32, }', "provided"),
	('struct Other { value: vec4f, } @group(0) @binding(0) var<uniform> another: Other;', "renderer-provided"),
	('@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0); }', "renderer-provided"),
	('fn default_lighting() {}', "reserved"),
	('#extends "default_material.shader"', "plain WGSL"),
])
def test_invalid_material_contracts_fail_before_gpu_creation(source, message):
	with pytest.raises(ValueError, match=message):
		DefaultMaterialShader(source=source)


@pytest.mark.parametrize("count", [1, 4])
def test_linked_plain_families_share_transparency_and_update_instance_opacity(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	source = '''
struct MaterialParameters { color: vec3f, }
struct InstanceParameters { coverage: f32, }
const default_color = vec3f(1.0, 0.0, 0.0);
const default_coverage = 1.0;
const weights = array<f32, 2>(0.5, 0.5);
const factor = 2.0;
// Comments may mention {{ template syntax }} in an otherwise plain WGSL module.
const constant = weights[0];
const_assert(constant == 0.5);
// The member 'shade' must survive namespacing of the function with that name.
struct Layer { color: vec3f, shade: f32, }
fn scoped(grazing: f32, grazing_0: f32,) -> f32 {
    let default_lighting = grazing+grazing_0;
    return default_lighting;
}
fn evaluate(color: vec3f) -> vec3f {
    let layer = Layer(color, 1.0);
    var sum = vec3f(0.0);
    for (var i = 0u; i < 2u; i++) { sum += layer.color*weights[i]; }
    // Local and parameter names must not be rewritten as default helper names,
    // or collide with generated names for either of these linked families.
    let grazing = 1.0;
    let grazing_0 = 2.0;
    let grazing_1 = 3.0;
    var gain = scoped(grazing, grazing_0);
    { let grazing = grazing_1; gain += grazing; }
    for (var grazing = 0u; grazing < 2u; grazing++) { gain += f32(grazing); }
    gain += grazing;
    // The for-local must expire; this factor refers to the module constant.
    for (var factor = 0u; factor < 1u; factor++) { gain += f32(factor); }
    let factor = factor; // initializer resolves the outer module constant
    return sum*layer . shade*(gain/8.0)*(factor/2.0);
}
fn surface(p: Params,) -> Material { return default_surface(p); }
fn shade(material: Material, p: Params, uv: vec2f, position: vec3f, normal: vec3f, lod: f32,) -> vec3f {
    // These locals must not capture calls to the namespaced evaluate helper.
    let evaluate_0 = 1.0;
    let evaluate_1 = 1.0;
    return evaluate(p.color)*evaluate_0*evaluate_1;
}
fn opacity(material: Material, p: Params, alpha: vec3f, uv: vec2f, position: vec3f, lod: f32,) -> vec3f {
    return default_opacity(material, p, alpha, uv, position, lod)*p.coverage;
}
fn grazing(p: Params,) -> vec3f { return default_grazing(p); }
'''
	left = DefaultMaterialShader(transparent=True, source=source)
	right = DefaultMaterialShader(transparent=True, source=source, default_material={"color": (0, 1, 0)})
	vertices = np.zeros(6, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-1.4, -2, 0), (1.4, -2, 0), (1.4, 2, 0), (-1.4, -2, 0), (1.4, 2, 0), (-1.4, 2, 0)]
	vertices["normal"] = (0, 0, 1)
	items = [DrawInstance(MaterialMesh(vertices, shader), position=(x, 0, 0)) for shader, x in ((left, -1.4), (right, 1.4))]
	renderer = MaterialRenderer(items, target_format="rgba8unorm", sample_count=count)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((64, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (64, 32), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 32, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(32, 64, 4).copy()
	before = render()
	assert before[16, 16, 0] > 180 and before[16, 16, 1] < 5
	assert before[16, 48, 1] > 180 and before[16, 48, 0] < 5
	compiled = renderer.transparency.shader
	items[0].overrides["coverage"] = 0.0
	after = render()
	assert not np.array_equal(before[16, 16], after[16, 16])
	np.testing.assert_array_equal(before[16, 48], after[16, 48])
	assert renderer.transparency.shader is compiled
