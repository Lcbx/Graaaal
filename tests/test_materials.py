"""Dictionary semantics, shader inheritance and mixed-family GPU composition."""
import numpy as np
import pytest

from numpyecs.Materials import MaterialBinding, MaterialLibrary, MaterialShader, Parameter
from numpyecs.DefaultMaterial import DefaultMaterialShader, TRANSPARENCY_INTERFACE
from numpyecs.MaterialRenderer import MaterialRenderer, MaterialMesh, DrawInstance
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage, Shader, RenderPipeline
from numpyecs.TransparencyData import VERTEX_DTYPE
from numpyecs.Math import Camera
from scenes.material_example import make_instances, sphere_vertices


def test_shared_dicts_defaults_and_instance_scopes(tmp_path):
	path = tmp_path / "shared.shaderlib"
	path.write_text('''#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> gain: f32 = 2;
<InstanceParameters> phase: f32 = 0;
#endblock
''')
	shader = DefaultMaterialShader(template=path)
	shared = {"gain": 4}
	bindings = [MaterialBinding(shader, shared, {"phase": 1}), MaterialBinding(shader, shared), MaterialBinding(shader)]
	library = MaterialLibrary([shader])
	core, material, instance, ids, offsets = library.pack(bindings)
	assert len(core) == 2
	np.testing.assert_array_equal(ids, [0, 0, 1])
	assert offsets[1] > offsets[0]
	shared["gain"] = 8
	assert not np.array_equal(library.pack(bindings)[1], material)
	np.testing.assert_array_equal(library.pack(bindings)[2], instance)
	shader.default_material = {"gain": 3}
	assert not np.array_equal(library.pack(bindings)[1], material)
	with pytest.raises(ValueError, match="shared"):
		library.pack([MaterialBinding(shader, overrides={"gain": 5})])
	with pytest.raises(ValueError, match="Unknown"):
		library.pack([MaterialBinding(shader, {"typo": 1})])


@pytest.mark.parametrize("kind,value", [("f32", float("nan")), ("vec3f", (1, 2)), ("u32", -1), ("bool", 1)])
def test_parameters_reject_invalid_types_and_values(kind, value):
	with pytest.raises(ValueError): Parameter(kind, value)


def test_jinja_material_keeps_default_features_and_adds_typed_parameters(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> custom: f32 = 0.2;
#endblock
#block lighting
	return vec3f(p.custom);
#endblock
''')
	source = MaterialLibrary([shader]).source
	assert "return vec3f(p.custom)" in source
	assert "fn default_lighting_0" in source
	assert "fn opacity_0" in source
	assert "custom: f32" in source
	assert shader.parameters["custom"].default == 0.2
	assert shader.parameters["alpha_mode"].default == 2


def test_material_source_and_template_are_mutually_exclusive():
	with pytest.raises(ValueError, match="exactly one"):
		DefaultMaterialShader(template="default_material.shader", source="")
	with pytest.raises(ValueError, match="exactly one"):
		MaterialShader(interface=TRANSPARENCY_INTERFACE)


def test_child_template_owns_defaults_scopes_and_route_dependent_values(tmp_path):
	path = tmp_path / "child.shaderlib"
	path.write_text('''#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> roughness: f32 = 0.15;
<InstanceParameters> grazing_fade: bool = false;
<InstanceParameters> rim_power: f32 = 3.0;
#endblock
''')
	opaque = DefaultMaterialShader(template=path)
	transparent = DefaultMaterialShader(template=path, transparent=True)
	assert opaque.parameters["alpha_mode"].default == 0
	assert transparent.parameters["alpha_mode"].default == 2
	assert transparent.parameters["roughness"].default == 0.15
	assert transparent.parameters["base_color"].default == (1, 1, 1, 1)
	assert transparent.parameters["grazing_fade"].scope == "instance"
	library = MaterialLibrary([transparent])
	packed = library.pack([MaterialBinding(transparent, overrides={"rim_power": 5, "grazing_fade": True})])
	assert packed[2].size > 0
	assert "rim_power: f32" in library.source
	assert "parameter(" not in library.source  # declarations never reach WGSL


@pytest.mark.parametrize("declaration", [
	'<MaterialParameters> bad-name: f32 = 1;',
	'<MaterialParameters> value: f64 = 1;',
	'<BadParameters> value: f32 = 1;',
	'<MaterialParameters> value: vec3f = (1, 2);',
])
def test_invalid_shader_declarations_fail_before_gpu_creation(tmp_path, declaration):
	path = tmp_path / "bad.shaderlib"
	path.write_text('#extends "material_surface.shaderlib"\n#block parameters\n'+declaration+'\n#endblock\n')
	with pytest.raises(ValueError):
		MaterialShader(interface=TRANSPARENCY_INTERFACE, template=path)



def test_layout_changes_require_relinking_but_default_edits_do_not():
	shader = DefaultMaterialShader()
	library = MaterialLibrary([shader])
	shader.default_material = {"roughness": 0.2}
	library.pack([MaterialBinding(shader)])
	with pytest.raises(TypeError):
		shader.parameters["added"] = Parameter("f32", 1)
	with pytest.raises(AttributeError):
		shader.parameters = {}
	shader.transparent = True
	with pytest.raises(ValueError, match="layout"):
		library.pack([MaterialBinding(shader)])


def test_builtin_types_and_grazing_intervals_are_checked(tmp_path):
	path = tmp_path / "bad.shaderlib"
	path.write_text('''#extends "default_material.shader"
#block parameters
{{ super() }}
<MaterialParameters> base_color: f32 = 1;
#endblock
''')
	with pytest.raises(ValueError, match="base_color"):
		DefaultMaterialShader(template=path)
	shader = DefaultMaterialShader(transparent=True)
	library = MaterialLibrary([shader])
	with pytest.raises(ValueError, match="Grazing"):
		library.pack([MaterialBinding(shader, {"grazing_start": 85, "grazing_end": 85+1e-9})])


def read(texture):
	w, h = texture.size[:2]
	data = RenderContext.device.queue.read_texture({"texture": texture.handle}, {"bytes_per_row": w*4}, (w, h, 1))
	return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4).copy()


def test_example_sphere_winding_matches_outward_shading_normals():
	vertices = sphere_vertices()
	positions = vertices["position"].reshape(-1, 3, 3)
	normals = vertices["normal"].reshape(-1, 3, 3).mean(axis=1)
	geometric = np.cross(positions[:, 1]-positions[:, 0], positions[:, 2]-positions[:, 0])
	valid = np.linalg.norm(geometric, axis=1) > 1e-7
	assert np.all(np.sum(geometric[valid]*normals[valid], axis=1) > 0)


@pytest.mark.parametrize("count", [1, 4])
def test_gpu_shared_and_inherited_materials_compose_and_update_without_recompiling(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	instances, shared, opaque = make_instances()
	noise = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
	renderer = MaterialRenderer(instances, blue_noise=noise, target_format="rgba8unorm", sample_count=count)
	camera = Camera((0, 2.5, 7), (0, 0, 0), (0, 1, 0), 50, near=0.05, far=100)
	output = Texture((96, 72), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (96, 72), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		return read(output)
	before = render()
	assert np.ptp(before[..., :3]) > 100
	assert len(renderer.headers.content) == 5  # shared glass pair occupies one record
	assert len(renderer.draws.draw_batches) == 4  # different instance values stay batched
	shader = renderer.transparency.shader
	instances[-1].mesh.shader.default_material["stripe_frequency"] = 3.0
	assert not np.array_equal(before, render())
	shared["base_color"] = (0.9, 0.05, 0.1, 0.65)
	instances[3].overrides["opacity_multiplier"] = 0.3
	instances[2].overrides["grazing_fade"] = False
	after = render()
	assert not np.array_equal(before, after)
	assert renderer.transparency.shader is shader
	opaque.default_material = {"base_color": (0, 0, 1, 1)}
	assert not np.array_equal(after, render())
	# A different instance shader changes routing, retaining shared winner addressing.
	instances[0].shader = instances[4].mesh.shader
	assert not np.array_equal(after, render())
	renderer.set_sample_count(5-count)
	assert np.ptp(render()[..., :3]) > 100
	baseline = renderer.transparency.shader
	renderer.set_noise_footprint_balancing(True)
	assert renderer.transparency.shader is not baseline
	assert renderer.draws.shaders[1]["main"].pipeline is renderer.geometry[1].render_shader["main"].pipeline
	assert np.ptp(render()[..., :3]) > 100
	renderer.set_noise_footprint_balancing(False)
	assert renderer.transparency.shader is baseline


def test_example_shared_highlights_are_visible_and_default_replacement_is_local():
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	instances, shared, opaque = make_instances()
	noise = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
	renderer = MaterialRenderer(instances, blue_noise=noise, target_format="rgba8unorm")
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50, near=0.05, far=100)
	output = Texture((256, 192), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (256, 192), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		return read(output)
	def crop(image, index):
		clip = np.array((*instances[index].position, 1)) @ camera.view() @ camera.projection(256/192)
		x, y = (clip[:2]/clip[3]*[0.5, -0.5]+0.5)*[256, 192]
		return image[int(y)-24:int(y)+24, int(x)-24:int(x)+24, :3].astype(np.int16)
	for name, low, high in (("specular", 0, 8), ("roughness", 0.02, 1)):
		shared[name] = low
		before = render()
		shared[name] = high
		after = render()
		for index in (2, 3):
			change = np.abs(crop(after, index)-crop(before, index))
			assert np.count_nonzero(change.max(axis=2) > 8) > 20
	before = render()
	opaque.default_material = {"base_color": (0.1, 0.1, 1, 1)}
	after = render()
	assert not np.array_equal(crop(before, 0), crop(after, 0))
	np.testing.assert_array_equal(crop(before, 1), crop(after, 1))


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("local_opaque", [False, True])
@pytest.mark.parametrize("winner_count", [1, 2])
@pytest.mark.parametrize("force", [False, True])
def test_material_record_preserves_external_opaque_color_and_occlusion(count, local_opaque, winner_count, force):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	usage = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
	color = Texture((32, 32), format="rgba16float", usage=usage, sample_count=count)
	depth = Texture((32, 32), format="depth32float", usage=usage, sample_count=count)
	noise = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
	vertices = np.zeros(3, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-100, -100, 0), (100, -100, 0), (0, 100, 0)]
	vertices["normal"] = (0, 0, 1)
	transparent = DefaultMaterialShader(transparent=True,
		default_material={"base_color": (0, 0, 0, 0.5), "emission": (1, 0, 0), "metallic": 1})
	items = [DrawInstance(MaterialMesh(vertices, transparent))]
	if local_opaque:
		opaque = DefaultMaterialShader(default_material={"base_color": (0, 0, 0, 1), "emission": (0, 1, 0), "metallic": 1})
		items.append(DrawInstance(MaterialMesh(vertices, opaque), position=(0, 0, 1)))
	renderer = MaterialRenderer(items, blue_noise=noise, target_format="rgba8unorm", sample_count=count,
		opaque_color=color, opaque_depth=depth, winner_count=winner_count, force_single_contributor=force)
	# Another renderer supplies blue HDR color and a left-half foreground occluder.
	shader = Shader(source='''
        <Output> @location(0) color: vec4f;
        <Output> @builtin(frag_depth) depth: f32;
        @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            return vec4f(vec2f(f32((i << 1u) & 2u), f32(i & 2u))*2.0-1.0, 0.0, 1.0);
        }
        @fragment fn fragment(@builtin(position) p: vec4f) -> Output {
            return Output(vec4f(0.0, 0.1, 0.5, 1.0), select(1.0, 0.2, p.x < 16.0));
        }
    ''')
	pipeline = RenderPipeline(shader, cull_mode="none", depth_test="always", sample_count=count)
	output = Texture((32, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands()
	with commands.render_pass(color=color.color_attachment(clear=(0, 0, 0, 1)), depth=depth.depth_attachment(clear=1)) as render:
		render.set_pipeline(pipeline)
		render.draw(3)
	renderer.record(commands, (32, 32), Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50), output.color_attachment())
	RenderContext.submit(commands.finish())
	pixels = read(output)
	def encoded(hdr):
		linear = np.array(hdr)/(1+np.array(hdr))
		return np.rint(np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055)*255)
	# External depth occludes both local geometry types on the left.
	np.testing.assert_allclose(pixels[16, 8, :3], encoded((0, 0.1, 0.5)), atol=1)
	# On the right, add opaque geometry or compose red transparency over blue.
	np.testing.assert_allclose(pixels[16, 24, :3], encoded((0, 1, 0) if local_opaque else (0.5, 0.05, 0.25)), atol=1)
