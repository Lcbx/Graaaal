from dataclasses import replace

import numpy as np
import pytest

from graaal.AntialiasingPass import AntialiasingPass, AntialiasingSettings
from graaal.RenderContext import RenderContext, Texture, TextureUsage


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb"])
def test_antialiasing_preserves_flat_color_and_processes_edges(mode, format):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingPass(target_format=format)
	size = (32, 32)
	aa.resize(size)
	aa.source = Texture(size, format=format, usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((32, 32, 4), dtype=np.uint8)
	image[:, :, 3] = 137
	image[:, :, :3] = (np.arange(32)[None, :] <= np.arange(32)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 128}, (*size, 1))
	output = Texture(size, format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("antialiasing_test")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128, "rows_per_image": 32}, (*size, 1))
	result = np.frombuffer(data, dtype=np.uint8).reshape(32, 32, 4)
	assert np.all(result[2, 20, :3] == 0)
	assert np.all(result[28, 2, :3] == 255)
	assert np.all(result[:, :, 3] == 137)
	assert np.any((result[:, :, 0] > 0) & (result[:, :, 0] < 255))


def test_antialiasing_settings_validation():
	settings = AntialiasingSettings()
	for values in ({"lxaa_edge_threshold": -1}, {"faaa_edge_threshold": 10}, {"faaa_quality": 0}, {"faaa_show_edges": 1}, {"ddaa2_strength": -1}, {"ddaa2_threshold_min": float("nan")}, {"ddaa2_edge_steps": 6}):
		with pytest.raises(ValueError): replace(settings, **values)


def test_ddaa2_disabled_diffusion_and_search_preserve_image():
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingPass(target_format="rgba8unorm", settings=AntialiasingSettings(ddaa2_strength=0, ddaa2_edge_steps=0))
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :, 3] = 255
	image[:, :, :3] = (np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("ddaa2_settings")
	aa.record(commands, output.color_attachment(), "ddaa2")
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
	assert np.array_equal(np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4), image)


def test_faaa_edge_display_and_quality_settings():
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingPass(target_format="rgba8unorm")
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :, 3] = 255
	image[:, :, :3] = (np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	for quality in range(1, 10):
		aa.set_settings(replace(aa.settings, faaa_quality=quality, faaa_show_edges=True))
		commands = RenderContext.commands("faaa_edges")
		aa.record(commands, output.color_attachment(), "faaa")
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
		pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
		assert np.any(np.all(pixels[:, :, :3] == (0, 255, 0), axis=2))
