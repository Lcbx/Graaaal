"""BLEND defaults and deterministic depth-writing MASK at native MSAA samples."""
import numpy as np
import pytest

from graaal.DefaultMaterial import DefaultMaterialShader
from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.TransparencyData import VERTEX_DTYPE
from test_transparency_multiple_winners import read, render
from test_transparency_exact_nearest import encoded


def quad():
	vertices = np.zeros(6, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
	vertices["normal"] = (0, 0, 1)
	return vertices


def unlit():
	return DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block lighting
	return p.emission;
#endblock
''')


def test_default_is_banded_with_one_pair_and_single_sample_tail():
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	renderer = MaterialRenderer([DrawInstance(MaterialMesh(quad(), unlit()))], sample_count=4, target_format="rgba8unorm")
	oit = renderer.transparency
	assert oit.banded_tail and oit.exact_nearest and oit.reconstruction == "kawase"
	assert oit.kawase_passes == 1 and oit.tail_sample_count == 1
	assert not oit.banded_proxy and not oit.banded_subset
	assert np.isfinite(render(renderer, (17, 19))).all()
	assert [entry for entry, *_ in oit.kawase_stages] == ["down", "up", "finish"]


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("alpha", [.49, .5, .51])
def test_mask_uses_cutoff_and_occludes_blended_geometry(samples, alpha):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	mesh = MaterialMesh(quad(), unlit())
	mask = DrawInstance(mesh, material={"alpha_mode": 1, "alpha_cutoff": .5, "base_color": (1, 1, 1, alpha), "emission": (1, 0, 0)})
	blend = DrawInstance(mesh, position=(0, 0, -.3), material={"base_color": (1, 1, 1, .5), "emission": (0, 0, 1)})
	renderer = MaterialRenderer([mask, blend], sample_count=samples, target_format="rgba8unorm")
	image = render(renderer, (31, 23))
	expected = encoded((1, 0, 0) if alpha >= .5 else np.array([0, 0, .5])+.5*np.array([.035, .05, .08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, (23, 31, 3)), atol=1)
	assert any(batch.shader_id == 0 and batch.mesh_id is not None for batch in renderer.draws.draw_batches)
	assert any(batch.shader_id == 1 and batch.mesh_id is not None for batch in renderer.draws.draw_batches)
	# At most the one BLEND sheet is captured exactly; neither contributes a tail.
	np.testing.assert_array_equal(read(renderer.transparency.band_payloads, np.uint32), 0)
	for phase in (1, 3):
		renderer.set_kawase_passes(phase)
		np.testing.assert_array_equal(render(renderer, (31, 23)), image)
	mask.material["alpha_mode"] = 2
	assert not np.array_equal(render(renderer, (31, 23)), image)
	mask.material["alpha_mode"] = 1
	np.testing.assert_array_equal(render(renderer, (31, 23)), image)


@pytest.mark.parametrize("samples", [1, 4])
def test_alpha_cutoff_is_evaluated_at_msaa_sample_positions(samples):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	vertices = quad()
	vertices["uv"][:, 0] = .5+.4*vertices["position"][:, 0]
	vertices["uv"][:, 1] = .5
	texture = Texture((2, 1, 1), format="rgba8unorm-srgb", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	texels = np.array([[[255, 255, 255, 0], [255, 255, 255, 255]]], dtype=np.uint8)
	RenderContext.device.queue.write_texture({"texture": texture.handle}, texels, {"bytes_per_row": 8}, (2, 1, 1))
	# Threshold at screen x=8.55. Pixel x=8 is rejected at its center; two of
	# four native hardware samples lie beyond it. A pixel-rate discard fails this.
	cutoff = .5+.8*((8.55/16*2-1)*3*np.tan(np.deg2rad(55)/2))
	item = DrawInstance(MaterialMesh(vertices, unlit()), material={"alpha_mode": 1, "alpha_cutoff": cutoff, "emission": (1, 0, 0)})
	renderer = MaterialRenderer([item], base_textures=texture.view(dimension="2d-array"), sample_count=samples, target_format="rgba8unorm")
	image = render(renderer, (16, 16))
	coverage = 0 if samples == 1 else .5
	hdr = coverage*np.array([1, 0, 0])+(1-coverage)*np.array([.035, .05, .08])
	np.testing.assert_allclose(image[0, :, 8, :3], np.broadcast_to(encoded(hdr), (16, 3)), atol=1)
	np.testing.assert_allclose(image[0, :, 7, :3], np.broadcast_to(encoded(np.array([.035, .05, .08])), (16, 3)), atol=1)
	np.testing.assert_allclose(image[0, :, 9, :3], np.broadcast_to(encoded(np.array([1, 0, 0])), (16, 3)), atol=1)


def test_default_material_route_resolves_shared_defaults_and_instance_mode():
	shader = DefaultMaterialShader(transparent=True, default_material={"alpha_mode": 1})
	assert not shader.is_blended()
	assert not shader.is_blended({"base_color": (1, 1, 1, 1)})
	assert shader.is_blended({"alpha_mode": 2})
	assert not shader.is_blended({"alpha_mode": 2}, {"alpha_mode": 1})


def test_route_uses_uploaded_values_when_shared_material_is_not_dirty():
	from graaal.Materials import MaterialBinding, MaterialLibrary
	shader = DefaultMaterialShader(transparent=True)
	material = {"alpha_mode": 1}
	binding = MaterialBinding(shader, material)
	library = MaterialLibrary([shader])
	library.pack([binding])
	material["alpha_mode"] = 2
	library.pack([binding], dirty_materials=[])
	assert not shader.is_blended(library.material_values(binding))
	library.pack([binding], dirty_materials=[material])
	assert shader.is_blended(library.material_values(binding))
	with pytest.raises(TypeError): library.material_values(binding)["alpha_mode"] = 1


def test_ecs_mask_mode_edit_updates_route_and_keeps_surface_identity():
	from graaal.ECS import ECS
	from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance, MaterialTransform
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	mesh = MaterialMesh(quad(), unlit())
	material = {"alpha_mode": 1, "base_color": (1, 1, 1, .75), "emission": (1, 0, 0)}
	renderer = MaterialRenderer(meshes=[mesh], target_format="rgba8unorm")
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	[entity] = world.create()
	world.add(entity, MaterialInstance(mesh, material=material), MaterialTransform())
	adapter = EcsMaterialAdapter(renderer, world)
	assert adapter.sync()
	identity = renderer.metadata.content[0, 2]
	assert any(batch.shader_id == 0 and batch.mesh_id is not None for batch in renderer.draws.draw_batches)
	material["alpha_mode"] = 2
	adapter.touch_materials(material)
	assert adapter.sync()
	assert renderer.metadata.content[0, 2] == identity
	assert any(batch.shader_id == 1 and batch.mesh_id is not None for batch in renderer.draws.draw_batches)


def test_unused_correlated_noise_is_lazy_until_switching_to_stochastic():
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	renderer = MaterialRenderer([DrawInstance(MaterialMesh(quad(), unlit()))], target_format="rgba8unorm")
	oit = renderer.transparency
	image = render(renderer, (17, 19))
	oit.set_noise_mode("triplanar_correlated")
	assert oit._noise_hierarchy is None and oit._noise_hierarchy_second is None
	np.testing.assert_array_equal(render(renderer, (17, 19)), image)
	renderer.set_tail_allocation("stochastic")
	assert oit._noise_hierarchy is not None
	assert oit._noise_hierarchy_second is None
	assert np.isfinite(render(renderer, (17, 19))).all()
