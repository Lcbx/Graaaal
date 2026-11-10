"""Opt-in frame time and agreement between deformed raster/deferred geometry."""
from unittest.mock import Mock

import numpy as np
import pytest

from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.EcsMaterials import EcsMaterialAdapter
from numpyecs.Materials import MaterialBinding, MaterialLibrary
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.TransparencyData import VERTEX_DTYPE
from numpyecs.TransparencyPass import BilateralFilter
from scenes.material_example import make_world


ANIMATED_SOURCE = '''
struct FrameParameters { time: f32, }
fn vertex(p: Params, position: vec3f, normal: vec3f, uv: vec2f) -> MaterialVertex {
    var v = default_vertex(p, position, normal, uv);
    v.position.x += p.time*(0.4+0.3*position.y*position.y);
    v.normal = vec3f(position.y*position.y+0.2*p.time, 0.0, 1.0);
    v.uv += vec2f(0.1*p.time, 0.0);
    return v;
}
fn shade(material: Material, p: Params, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
    return vec3f(0.3+position.x*0.2, 0.2+uv.x*0.25, 0.2+normal.x*0.25);
}
'''


def test_time_is_opt_in_and_does_not_occupy_material_or_instance_slots():
	plain = DefaultMaterialShader()
	animated = DefaultMaterialShader(source=ANIMATED_SOURCE)
	assert "time" not in plain.parameters
	assert animated.parameters["time"].scope == "frame"
	library = MaterialLibrary([plain, animated])
	_, shared, instances, _, offsets = library.pack([MaterialBinding(plain), MaterialBinding(animated)])
	assert len(shared) == 2*library.parameter_layouts[plain]["material"].slot_count
	assert len(instances) == 2*library.parameter_layouts[plain]["instance"].slot_count
	assert offsets[1] == library.parameter_layouts[plain]["instance"].slot_count
	for binding in (MaterialBinding(animated, {"time": 1}), MaterialBinding(animated, overrides={"time": 1})):
		with pytest.raises(ValueError, match="renderer-owned"):
			library.pack([binding])
	for declaration in ("struct FrameParameters { time: vec2f, }", "struct FrameParameters { unknown: f32, }"):
		with pytest.raises(ValueError, match="Unsupported frame"):
			DefaultMaterialShader(source=declaration)


@pytest.mark.parametrize("count", [1, 4])
def test_deformed_raster_and_deferred_shading_match_at_fixed_times(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	vertices = np.zeros(12, dtype=VERTEX_DTYPE)
	rows = []
	for lower, upper in ((-0.8, 0), (0, 0.8)):
		rows.extend(((-0.6, lower, 0), (0.6, lower, 0), (0.6, upper, 0),
			(-0.6, lower, 0), (0.6, upper, 0), (-0.6, upper, 0)))
	vertices["position"] = rows
	vertices["normal"] = (0, 0, 1)
	vertices["uv"] = (vertices["position"][:, :2]+1)*0.5
	renderers = [MaterialRenderer([DrawInstance(MaterialMesh(vertices,
		DefaultMaterialShader(transparent=transparent, source=ANIMATED_SOURCE),
		bounds=((0.4, 0, 0), (1.0, 0.8, 0.1))))], target_format="rgba8unorm",
		sample_count=count, bilateral_filter=BilateralFilter(kernel_size=1)) for transparent in (False, True)]
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((96, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render(renderer, seconds):
		commands = RenderContext.commands()
		renderer.record(commands, (96, 64), camera, output.color_attachment(), time=seconds)
		RenderContext.submit(commands.finish())
		raw = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 384}, (96, 64, 1))
		return np.frombuffer(raw, dtype=np.uint8).reshape(64, 96, 4).copy()
	first = None
	for seconds in (0.0, 1.0):
		opaque, transparent = [render(renderer, seconds) for renderer in renderers]
		np.testing.assert_allclose(opaque.astype(float), transparent, atol=1)
		if first is None: first = opaque
		else: assert np.count_nonzero(np.any(first != opaque, axis=2)) > 100
		np.testing.assert_array_equal(render(renderers[1], seconds), transparent)


def test_ecs_animation_updates_frame_time_without_repacking_materials(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	world, _, meshes, _, _ = make_world()
	renderer = MaterialRenderer(meshes=meshes, target_format="rgba8unorm")
	adapter = EcsMaterialAdapter(renderer, world)
	assert adapter.sync()
	pack = Mock(wraps=renderer.library.pack)
	monkeypatch.setattr(renderer.library, "pack", pack)
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50)
	output = Texture((96, 72), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	compiled = renderer.transparency.shader
	for seconds in (0.0, 1.0):
		assert not adapter.sync()
		commands = RenderContext.commands()
		renderer.record(commands, (96, 72), camera, output.color_attachment(), time=seconds)
		RenderContext.submit(commands.finish())
		assert renderer.transparency.uniforms.content["time"] == seconds
	pack.assert_not_called()
	assert renderer.transparency.shader is compiled
