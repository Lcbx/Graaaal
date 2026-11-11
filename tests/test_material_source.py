"""Jinja material declarations, inheritance, and mixed-family rendering."""
import numpy as np
import pytest

from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.Materials import MaterialBinding, MaterialLibrary
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.TransparencyData import VERTEX_DTYPE


SOURCE = '''
#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> color: vec3f = (1, 0, 0);
<InstanceParameters> coverage: f32 = 1;
<MaterialParameters> count: u32 = 16;
<MaterialParameters> enabled: bool = true;
#endblock
#block lighting
	return p.color;
#endblock
#block opacity
	return alpha*p.coverage;
#endblock
'''


def test_jinja_source_parameters_keep_shader_defaults_and_instance_scopes():
	shader = DefaultMaterialShader(transparent=True, source=SOURCE)
	assert shader.parameters["color"].default == (1, 0, 0)
	assert shader.parameters["count"].default == 16
	assert shader.parameters["enabled"].default is True
	assert shader.parameters["coverage"].scope == "instance"
	assert shader.parameters["roughness"].default == 0.7
	library = MaterialLibrary([shader])
	shared = {"count": 2}
	a = library.pack([MaterialBinding(shader, shared), MaterialBinding(shader, shared, {"coverage": 0.5})])
	assert len(a[0]) == 1
	np.testing.assert_array_equal(a[3], [0, 0])
	with pytest.raises(ValueError, match="shared"):
		library.pack([MaterialBinding(shader, overrides={"count": 3})])
	shader.source = "changed"
	with pytest.raises(ValueError, match="layout"):
		library.pack([])


@pytest.mark.parametrize("declaration", [
	'<MaterialParameters> base_color: f32 = 1;',
	'<MaterialParameters> color: f64 = 1;',
	'<MaterialParameters> color: vec3f = (1, 2);',
	'<InvalidParameters> color: f32 = 1;',
])
def test_invalid_jinja_material_parameters_fail_before_gpu_creation(declaration):
	with pytest.raises(ValueError):
		DefaultMaterialShader(source='#extends "default_material.shader"\n#block parameters\n{{ super() }}\n'+declaration+'\n#endblock')



@pytest.mark.parametrize("count", [1, 4])
def test_linked_jinja_families_share_transparency_and_update_instance_opacity(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	left = DefaultMaterialShader(transparent=True, source=SOURCE)
	right = DefaultMaterialShader(transparent=True, source=SOURCE, default_material={"color": (0, 1, 0)})
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
