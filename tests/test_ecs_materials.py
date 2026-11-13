"""ECS material lifecycle, stable surface IDs and versioned extraction."""
from unittest.mock import Mock
import numpy as np

from graaal.ECS import ECS
from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance, MaterialTransform
from graaal.Math import Vec3
from graaal.Materials import MaterialShader
from graaal.MaterialRenderer import MaterialRenderer, MaterialMesh
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from scenes.material_example import make_world, sphere_vertices


def renderer(meshes, **kwargs):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	noise = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
	return MaterialRenderer(meshes=meshes, blue_noise=noise, target_format="rgba8unorm", frame_fields={"time": ("f32", 0.0)}, **kwargs)


def test_minimal_jinja_shader_requires_no_default_material_fields():
	shader = MaterialShader( transparent=True, source='''
#extends "material_surface.shaderlib"
#block parameters
<MaterialParameters> color: vec4f = (0.1, 0.3, 0.8, 0.6);
#endblock
#block surface
    var m = Material();
    m.base_color = p.color;
    m.opacity_rgb = vec4f(1.0);
    m.flags = vec4u(0u, 1u, 0u, 0u);
    return m;
#endblock
#block lighting
    return material.base_color.rgb;
#endblock
''')
	assert set(shader.parameters) == {"color"}
	r = renderer([MaterialMesh(sphere_vertices(), shader)])
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	[entity] = world.create()
	world.add(entity, MaterialInstance(r.meshes[0]), MaterialTransform())
	adapter = EcsMaterialAdapter(r, world)
	assert adapter.sync()
	from graaal.Math import Camera
	output = Texture((32, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands()
	r.record(commands, (32, 32), Camera((0, 0, 4), (0, 0, 0), (0, 1, 0), 50), output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (32, 32, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(32, 32, 4)
	assert pixels[16, 16, 2] > pixels[16, 16, 0]+40
	assert "grazing_start" not in shader.parameters


def test_ecs_versions_mutations_spawn_despawn_and_compaction(monkeypatch):
	world, entities, meshes, shared, opaque = make_world()
	r = renderer(meshes)
	adapter = EcsMaterialAdapter(r, world)
	pack = Mock(wraps=r.library.pack)
	monkeypatch.setattr(r.library, "pack", pack)
	assert adapter.sync()
	assert pack.call_count == 1
	refs = world.get(MaterialInstance)
	initial = {int(owner): int(sid) for owner, sid in zip(refs.owner_ids(refs[entities]), refs[entities].surface_id)}
	assert all(initial.values()) and len(set(initial.values())) == len(entities)
	assert dict(zip(map(int, r.draws.entities), map(int, r.metadata.content[:len(entities), 2]))) == initial
	assert not adapter.sync()
	assert pack.call_count == 1
	world.get(MaterialTransform)[entities[0]].position = Vec3(-2, 3, 0)
	assert adapter.sync() and pack.call_count == 1
	shared["roughness"] = 0.8
	assert not adapter.sync()  # nested object changes require explicit notification
	adapter.touch_materials()
	assert adapter.sync() and pack.call_count == 2
	assert not adapter.sync()
	world.delete(entities[0])
	assert adapter.sync()
	world.get(MaterialInstance)._store._defragment(world.get(MaterialInstance)._store._capacity)
	assert adapter.sync()
	live = world.query(MaterialInstance).entities()
	assert {int(e): int(s) for e, s in zip(refs.owner_ids(refs[live]), refs[live].surface_id)} == {e: s for e, s in initial.items() if e != int(entities[0])}
	assert dict(zip(map(int, r.draws.entities), map(int, r.metadata.content[:len(live), 2]))) == {e: s for e, s in initial.items() if e != int(entities[0])}
	[new] = world.create()
	world.add(new, MaterialTransform(), MaterialInstance(meshes[0]), MaterialInstance(meshes[1]))
	assert adapter.sync()
	new_ids = refs[new].surface_id
	assert len(new_ids) == 2 and len(set(map(int, new_ids))) == 2
	assert new_ids.min() > max(initial.values())
	from graaal.Math import Camera
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50)
	output = Texture((48, 48), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		r.record(commands, (48, 48), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 192}, (48, 48, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(48, 48, 4)
	assert np.ptp(render()[..., :3]) > 0
	world.delete(world.entities())
	assert adapter.sync()
	assert len(r.draws.entities) == 0
	assert not adapter.sync()
	# Empty worlds still clear and compose, without stale winners or invalid bindings.
	image = render()
	assert np.all(image == image[0, 0])


def test_dictionary_touch_and_default_replacement_invalidate_materials(monkeypatch):
	world, entities, meshes, shared, opaque = make_world()
	r = renderer(meshes)
	adapter = EcsMaterialAdapter(r, world)
	adapter.sync()
	before = r.parameters["material_parameters"].content.copy()
	opaque.default_material = {"base_color": (0, 0, 1, 1)}
	world.get(MaterialInstance).touch("material")
	assert adapter.sync()
	assert not np.array_equal(before, r.parameters["material_parameters"].content)
	before_ids = world.get(MaterialInstance)[entities].surface_id.copy()
	world.get(MaterialInstance)[entities[0]].shader = meshes[-1].shader
	assert adapter.sync()
	np.testing.assert_array_equal(world.get(MaterialInstance)[entities].surface_id, before_ids)
