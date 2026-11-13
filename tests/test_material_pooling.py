"""Packed field fidelity, incremental uploads, and cross-pipeline binding reuse."""
from unittest.mock import Mock
from pathlib import Path

import numpy as np
import pytest

from scenes.material_example import make_instances, make_world, sphere_vertices
from graaal.DefaultMaterial import DefaultMaterialShader
from graaal.EcsMaterials import EcsMaterialAdapter
from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Materials import MaterialBinding, MaterialLibrary, MaterialShader
from graaal.Math import Camera
from graaal.RenderContext import (BindGroupLayout, BufferUsage, ComputePipeline,
	GpuBuffer, PipelineLayout, RenderContext, Shader, Texture, TextureUsage, wgpu)


def test_example_pools_pack_scalar_lanes_and_preserve_shader_declarations():
	items, _, _ = make_instances()
	library = MaterialLibrary([item.mesh.shader for item in items])
	bindings = [MaterialBinding(item.mesh.shader, item.material if item.material is not None else item.mesh.material, item.overrides) for item in items]
	headers, shared, overrides, ids, offsets = library.pack(bindings)
	assert headers.nbytes == 80 and shared.nbytes == 512 and overrides.nbytes == 192
	np.testing.assert_array_equal(ids, [0, 1, 2, 2, 3, 4])
	assert len(set(offsets)) == 6
	source, _ = Shader.__new__(Shader)._render_source(None, str(Path(__file__).resolve().parents[1]/"graaal/shaders/stochastic_geometry.shader"), None, None, {"material_library": library})
	assert "struct Materials" in source and "struct InstanceOverrides" in source
	assert "fn vertex_3" in source and "fn shade_3" in source
	assert "struct Params3" in source
	assert "fn vertex_3" in source and "fn vertex_0" in source


def test_targeted_updates_reuse_other_records_and_invalidate_inherited_defaults(monkeypatch):
	shader = DefaultMaterialShader()
	red, green = {"base_color": (1, 0, 0, 1)}, {"base_color": (0, 1, 0, 1)}
	library = MaterialLibrary([shader])
	bindings = [MaterialBinding(shader, red), MaterialBinding(shader, green)]
	initial = library.pack(bindings)
	layout = library.parameter_layouts[shader]["material"]
	pack = Mock(wraps=layout.pack)
	monkeypatch.setattr(layout, "pack", pack)
	red["roughness"] = 0.2
	updated = library.pack(bindings, dirty_materials=[red])
	assert pack.call_count == 1
	np.testing.assert_array_equal(initial[0], updated[0])
	np.testing.assert_array_equal(initial[1][layout.slot_count:], updated[1][layout.slot_count:])
	shader.default_material["specular"] = 3.0
	inherited = library.pack(bindings, dirty_materials=[shader.default_material])
	assert pack.call_count == 3 and not np.array_equal(inherited[1], updated[1])
	shader.default_material = {"specular": 4.0}
	replaced = library.pack(bindings, dirty_materials=[])
	assert pack.call_count == 5 and not np.array_equal(replaced[1], inherited[1])


def test_shared_explicit_bindings_refresh_after_buffer_growth_and_respect_close():
	RenderContext.setup_graphics_backend(True)
	shader = Shader(source='''
storage<0, 0, read_write> words: array<u32>;
@compute @workgroup_size(1) fn main() { words[0] += arrayLength(&words); }
''')
	group = BindGroupLayout([dict(binding=0, visibility=wgpu.ShaderStage.COMPUTE, buffer=dict(type="storage"))])
	layout = PipelineLayout([group])
	first, second = [ComputePipeline(shader, layout=layout) for _ in range(2)]
	buffer = GpuBuffer(np.zeros(1, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC)
	bindings = shader.bind_group(0, words=buffer)
	original = first._variant.bind_group(bindings)
	assert second._variant.bind_group(bindings) is original
	buffer.write(np.zeros(16, dtype=np.uint32))
	grown = second._variant.bind_group(bindings)
	assert grown is not original and first._variant.bind_group(bindings) is grown
	commands = RenderContext.commands()
	with commands.compute_pass() as compute:
		for pipeline in (first, second):
			compute.set_pipeline(pipeline)
			compute.set_bind_group(bindings)
			compute.dispatch(1)
	# Dropping cached bindings leaves recorded work and the layout usable.
	group.clear_cache()
	assert not group._bind_groups
	assert second._variant.bind_group(bindings) is not grown
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_buffer(buffer.handle, 0, 4)
	assert np.frombuffer(data, dtype=np.uint32)[0] == 32
	group.close()
	with pytest.raises(RuntimeError, match="closed"):
		first._variant.bind_group(bindings)
	assert not group._bind_groups


@pytest.mark.parametrize("count", [1, 4])
def test_animation_and_dictionary_edits_upload_words_and_share_frame_binding(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	world, _, meshes, shared, _ = make_world()
	renderer = MaterialRenderer(meshes=meshes, target_format="rgba8unorm", sample_count=count, frame_fields={"time": ("f32", 0.0)})
	adapter = EcsMaterialAdapter(renderer, world)
	adapter.sync()
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50)
	output = Texture((48, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	def record(seconds):
		commands = RenderContext.commands()
		renderer.record(commands, (48, 32), camera, output.color_attachment(), frame={"time": seconds})
		RenderContext.submit(commands.finish())
	record(0)
	try:
		RenderContext.enable_statistics()
		record(1)
		stats = RenderContext.inspect_resources()["statistics"]
		assert stats["upload_calls"] == 1 and stats["upload_bytes"] == 4
		RenderContext.enable_statistics()
		record(1)
		assert RenderContext.inspect_resources()["statistics"].get("upload_calls", 0) == 0
		shared["roughness"] = 0.8
		adapter.touch_materials(shared)
		assert adapter.sync()
		assert RenderContext.inspect_resources()["statistics"]["upload_bytes"] == 4
		oit = renderer.transparency
		opaque = renderer.geometry[0].render_shader["main"]
		variant = next(iter(opaque.pipeline._cache.values()))
		assert variant.bind_group(opaque.bindings[0]) is oit.shade_pipeline._variant.bind_group(oit.shade_frame)
		compose = next(iter(oit.compose_pipeline._cache.values()))
		assert variant.bind_group(opaque.bindings[0]) is compose.bind_group(oit.compose_frame)
		assert "light_dir" not in oit.uniforms.dtype.names
		assert oit.lighting_uniforms.content.nbytes == 16
		np.testing.assert_allclose(oit.lighting_uniforms.content["light_dir"][0], [0.4, 0.8, 0.3, 0])
		assert oit.shader.info.source.count("fn evaluate_lighting(") == 1
		frame_layout = variant.layout.groups[0]
		renderer.set_sample_count(5-count)
		record(1)
		assert oit.shade_pipeline._variant.layout.groups[0] is frame_layout
	finally:
		RenderContext.enable_statistics(False)


@pytest.mark.parametrize("count", [1, 4])
def test_packed_vectors_uint_and_bool_reach_the_gpu_without_bit_loss(count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	shader = MaterialShader( source='''
#extends "material_surface.shaderlib"
#block parameters
<MaterialParameters> color: vec3f = (0.1, 0.2, 0.3);
<MaterialParameters> pair: vec2f = (0.4, 0.5);
<MaterialParameters> gain: f32 = 0.6;
<MaterialParameters> key: u32 = 4275878552;
<MaterialParameters> enabled: bool = true;
<InstanceParameters> other: vec2f = (0.7, 0.8);
<InstanceParameters> valid: bool = true;
#endblock
#block helpers
@must_use fn unused_helper_{{ PREFIX }}() -> f32 { return 0.0; }
#endblock
#block surface
    var material: Material;
    material.base_color = vec4f(1.0);
    material.opacity_rgb = vec4f(1.0);
    material.flags = vec4u(0u, 1u, 0u, 0u);
    return material;
#endblock
#block lighting
    if p.key != 0xfedcba98u || !p.enabled || !p.valid { return vec3f(0.0); }
    return p.color+vec3f(p.pair, p.gain)+vec3f(p.other, 0.0);
#endblock
#block opacity
    return alpha;
#endblock
#block grazing
    return vec3f(0.0);
#endblock
''')
	renderer = MaterialRenderer([DrawInstance(MaterialMesh(sphere_vertices(), shader))],
		target_format="rgba8unorm", sample_count=count)
	assert "@must_use fn unused_helper_0" in renderer.transparency.shader.info.source
	output = Texture((32, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands()
	renderer.record(commands, (32, 32), Camera((0, 0, 4), (0, 0, 0), (0, 1, 0), 50), output.color_attachment())
	RenderContext.submit(commands.finish())
	raw = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (32, 32, 1))
	pixel = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 4)[16, 16, :3]
	hdr = np.array([1.2, 1.5, 0.9])
	linear = hdr/(1+hdr)
	expected = np.rint((1.055*linear**(1/2.4)-0.055)*255)
	np.testing.assert_allclose(pixel, expected, atol=1)
