"""Procedural, offscreen use without importing a scene or loading asset files."""
from unittest.mock import Mock

import numpy as np
import pytest

from graaal.RenderContext import BufferUsage, GpuBuffer, RenderContext, Texture, TextureUsage
from graaal.Math import Camera
from graaal.InstanceData import MESH_INSTANCE_DTYPE, pack_scale, pack_quaternion
from graaal.TransparencyPass import MATERIAL_DTYPE, SHADING_VERTEX_DTYPE, VERTEX_DTYPE, TransparencyPass


@pytest.mark.parametrize("noise_mode", ["uv_blue_lerp", "uv_blue_correlated"])
@pytest.mark.parametrize("sample_count", [1, 4])
def test_procedural_transparency_records_into_caller_commands_and_output(monkeypatch, tmp_path, noise_mode, sample_count):
	RenderContext.setup_graphics_backend(True)
	# Shader files and includes must resolve without the repository as cwd.
	monkeypatch.chdir(tmp_path)
	material = np.zeros(1, dtype=MATERIAL_DTYPE)
	material["base_color"] = (0, 0, 0, 0.5)
	material["emission"] = (1, 0, 0, 0)
	material["surface"] = (1, 1, 2, 0)
	material["opacity_rgb"] = (1, 1, 1, 0)
	material["flags"] = (0, 1, 0, 0)
	texture_usage = TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST
	white = Texture((1, 1, 1), format="rgba8unorm-srgb", usage=texture_usage)
	noise = Texture((1, 1), format="r32float", usage=texture_usage)
	queue = RenderContext.device.queue
	queue.write_texture({"texture": white.handle}, np.full(4, 255, dtype=np.uint8), {"bytes_per_row": 4}, (1, 1, 1))
	queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
	vertices = np.zeros(3, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-100, -100, 0), (100, -100, 0), (0, 100, 0)]
	vertices["normal"] = (0, 0, 1)
	vertices["barycentric"] = [(1, 0), (0, 1), (0, 0)]
	stored = np.zeros(3, dtype=SHADING_VERTEX_DTYPE)
	stored["position"][:, :3] = vertices["position"]
	stored["normal"][:, :3] = vertices["normal"]
	storage = BufferUsage.STORAGE | BufferUsage.COPY_DST
	oit = TransparencyPass(materials=GpuBuffer(material, storage),
		base_textures=white.view(dimension="2d-array"), vertices=GpuBuffer(stored, storage),
		blue_noise=noise, target_format="rgba8unorm", noise_mode=noise_mode, sample_count=sample_count, winner_count=1, tail_allocation="stochastic")
	metadata = GpuBuffer(np.array([[0, 0, 42, 0]], dtype=np.uint32), storage)
	vertex_buffer = GpuBuffer(vertices, BufferUsage.VERTEX | BufferUsage.COPY_DST)
	instances = GpuBuffer(np.zeros(1, dtype=MESH_INSTANCE_DTYPE), storage)
	instances.content["iScale"] = pack_scale([1, 1, 1])
	instances.content["iRotation"] = pack_quaternion([0, 0, 0, 1])
	instances.upload()
	geometry = oit.create_geometry_passes(metadata, transparent=True, instances=instances)
	visible = GpuBuffer(np.zeros(1, dtype=np.uint32), storage)
	instance_bindings = oit.shader.bind_group(1, instances=instances, visible_instances=visible)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=0.01, far=100)
	with pytest.raises(RuntimeError, match="Resize"):
		oit.update_camera(camera)
	submit = Mock(wraps=RenderContext.submit)
	monkeypatch.setattr(RenderContext, "submit", submit)
	for width, height in ((16, 16), (24, 8)):
		output = Texture((width, height), format="rgba8unorm",
			usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		oit.resize((width, height))
		assert oit.size == (width, height)
		# Idempotent/minimized resize retains valid targets.
		previous = oit.winner
		oit.resize((width, height))
		oit.resize((0, 0))
		assert oit.winner is previous
		oit.update_camera(camera)
		commands = RenderContext.commands("procedural_transparency")
		with oit.opaque_pass(commands, clear=(0, 0, 0, 1)):
			pass  # This caller has no opaque geometry.
		for begin, shader_pass in (
			(oit.guide_pass, geometry.guide),
			(oit.transmittance_pass, geometry.render_shader['transmittance']),
			(oit.selection_pass, geometry.render_shader['main']),
		):
			with begin(commands) as render:
				render.set_pipeline(shader_pass.pipeline)
				render.set_bind_groups(*shader_pass.bindings, instance_bindings)
				render.set_vertex_buffer(0, vertex_buffer)
				render.draw(3)
		oit.shade_winner(commands, geometry)
		oit.compose(commands, output.color_attachment())
		submit.assert_not_called()
		RenderContext.submit(commands.finish())
		data = queue.read_texture({"texture": output.handle}, {"bytes_per_row": width*4}, (width, height, 1))
		pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
		# Half-opaque red wins everywhere with the supplied constant threshold.
		assert np.all(pixels[:, :, 0] > 100)
		assert np.all(pixels[:, :, 1:3] < 50)
		submit.reset_mock()
