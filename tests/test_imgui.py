"""Input/lifecycle regressions and real offscreen ImGui rendering."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock

import glfw
import numpy as np
import pytest
import wgpu
from imgui_bundle import imgui
import wgpu.utils.imgui as backend_module

from numpyecs.ImGui import ImGui
from numpyecs.RenderContext import RenderContext
from scenes.prototype import ActorCache, Damageable, Attributes, ActorMetadata, create_world, draw_actors
from scenes.transparency_example import BilateralFilter, FilterSettings, GrazingFade


@pytest.fixture
def ui(monkeypatch):
	outer = imgui.create_context()
	sizes = NS(logical=(600, 450), physical=(1200, 900))
	monkeypatch.setattr(glfw, "get_window_size", lambda window: sizes.logical)
	monkeypatch.setattr(glfw, "get_framebuffer_size", lambda window: sizes.physical)
	monkeypatch.setattr(glfw, "get_cursor_pos", lambda window: (20, 30))
	monkeypatch.setattr(glfw, "get_key", lambda *args: glfw.RELEASE)
	def backend(*args):
		io = imgui.get_io()
		io.set_ini_filename("")
		io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
		return NS(io=io, render=Mock())
	monkeypatch.setattr(backend_module, "ImguiWgpuBackend", backend)
	@contextmanager
	def render_pass(**kwargs):
		yield NS(handle="pass")
	commands = NS(render_pass=Mock(side_effect=render_pass), finish=lambda: "commands")
	context = NS(window=object(), canvas=object(), device=object(), presentation_format="rgba8unorm",
		frame_time=1/60, event_handlers={}, screen=Mock(return_value=NS(view=NS(size=(1200, 900, 1)))),
		commands=Mock(return_value=commands), submit=Mock())
	gui = ImGui(context)
	try:
		yield gui, sizes, commands, outer
	finally:
		gui.close()
		imgui.destroy_context(outer)


def test_input_captures_before_scene_handlers_and_restores_context(ui):
	gui, _, _, outer = ui
	io = NS(add_key_event=Mock(), add_input_character=Mock(), add_mouse_wheel_event=Mock(),
		add_mouse_button_event=Mock(), add_mouse_pos_event=Mock(), add_focus_event=Mock(),
		want_text_input=False)
	gui._backend.io = io
	# The first callback must stop propagation when a widget owns the input.
	for channel in ("key", "char", "mouse_scroll", "mouse_click"):
		gui.context.event_handlers[channel].append(Mock())
	io.want_capture_keyboard = True
	io.want_capture_mouse = True
	for channel, args in (
		("key", (glfw.KEY_LEFT, 0, glfw.PRESS, glfw.MOD_SHIFT)),
		("char", (ord("é"),)), ("mouse_scroll", (2, -3)),
		("mouse_click", (glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)),
	):
		for handler in gui.context.event_handlers[channel]:
			if handler(*args):
				break
		gui.context.event_handlers[channel][-1].assert_not_called()
	io.add_key_event.assert_any_call(imgui.Key.left_arrow, True)
	io.add_key_event.assert_any_call(imgui.Key.mod_shift, True)
	io.add_input_character.assert_called_once_with(ord("é"))
	io.add_mouse_wheel_event.assert_called_once_with(2, -3)
	io.add_mouse_button_event.assert_called_once_with(1, True)
	io.want_capture_keyboard = False
	assert not gui._on_key(glfw.KEY_LEFT, 0, glfw.RELEASE, 0)
	io.add_key_event.assert_any_call(imgui.Key.left_arrow, False)
	gui._on_focus(False)
	io.add_focus_event.assert_called_once_with(False)
	assert imgui.get_current_context() == outer


def test_render_uses_screen_overlay_updates_dpi_and_detaches_only_own_handlers(ui):
	gui, sizes, commands, outer = ui
	draw = Mock()
	gui.render(draw)
	draw.assert_called_once()
	assert tuple(gui.io.display_size) == (600, 450)
	assert tuple(gui.io.display_framebuffer_scale) == (2, 2)
	gui.context.screen.assert_called_once_with(clear=None)
	commands.render_pass.assert_called_once_with(color=gui.context.screen.return_value, label="imgui")
	gui.context.submit.assert_called_once_with("commands")
	assert imgui.get_current_context() == outer
	sizes.logical, sizes.physical = (800, 400), (1200, 800)
	gui.context.screen.return_value.view.size = (1200, 800, 1)
	gui.context.frame_time = 0.02
	gui.render(draw, clear=(0, 0, 0, 1))
	assert tuple(gui.io.display_framebuffer_scale) == (1.5, 2)
	assert gui.io.delta_time == pytest.approx(0.02)
	sizes.physical = (0, 0)
	gui.render(draw)
	assert draw.call_count == 2  # Minimized windows do not start a frame.
	other = Mock()
	gui.context.event_handlers["key"].append(other)
	gui.close()
	gui.close()
	assert gui.context.event_handlers["key"] == [other]
	assert all(not handlers for name, handlers in gui.context.event_handlers.items() if name != "key")
	assert imgui.get_current_context() == outer
	with pytest.raises(RuntimeError, match="closed"):
		gui.render(draw)


def test_draw_failure_finishes_imgui_frame_and_allows_next_frame(ui):
	gui, _, _, outer = ui
	def fail():
		raise ValueError("widget failed")
	with pytest.raises(ValueError, match="widget failed"):
		gui.render(fail)
	assert imgui.get_current_context() == outer
	gui.render(lambda: None)


def test_resize_between_scene_and_overlay_uses_acquired_target_dimensions(ui):
	gui, sizes, commands, _ = ui
	# GLFW already reports the new window; the scene acquired the old surface.
	sizes.logical, sizes.physical = (2560, 1369), (2560, 1369)
	def draw():
		# Resizing can continue even while building the widgets.
		sizes.logical, sizes.physical = (1920, 1080), (1920, 1080)
	gui.render(draw)
	data = gui._backend.render.call_args.args[0]
	assert tuple(data.display_size) == (2560, 1369)
	np.testing.assert_allclose(np.array(data.display_size)*np.array(data.framebuffer_scale), (1200, 900), atol=0.001)
	gui.context.screen.assert_called_once_with(clear=None)
	commands.render_pass.assert_called_once_with(color=gui.context.screen.return_value, label="imgui")


def test_both_scenes_render_real_imgui_draw_data_to_gpu(monkeypatch):
	RenderContext.setup_graphics_backend(True)
	device = RenderContext.device
	texture = device.create_texture(size=(1200, 900, 1), format="rgba8unorm",
		usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
	monkeypatch.setattr(glfw, "get_window_size", lambda window: (1200, 900))
	monkeypatch.setattr(glfw, "get_framebuffer_size", lambda window: (1200, 900))
	monkeypatch.setattr(glfw, "get_cursor_pos", lambda window: (1190, 890))
	# Use the real RenderContext command/attachment APIs with an offscreen view.
	monkeypatch.setattr(RenderContext, "window", object())
	monkeypatch.setattr(RenderContext, "canvas", object())
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	monkeypatch.setattr(RenderContext, "_screen_view", texture.create_view())
	world = create_world()
	cache = ActorCache(world, world.query(Damageable, Attributes, ActorMetadata))
	cache.update()
	renderer = NS(bilateral_filter=BilateralFilter(), debug_view="final", sample_count=1,
		noise_mode="uv_blue_lerp", noise_scale=1, effective_noise_scale=1, noise_footprint_balance=0,
		noise_footprint_balancing=False, grazing_fade=GrazingFade())
	settings = FilterSettings(renderer)
	try:
		with ImGui() as gui:
			for draw in (lambda: draw_actors(cache), settings.draw):
				# A first-use window may defer layout for one frame.
				for _ in range(2):
					gui.render(draw, clear=(0, 0, 0, 1))
				data = device.queue.read_texture({"texture": texture}, {"bytes_per_row": 1200*4}, (1200, 900, 1))
				pixels = np.frombuffer(data, dtype=np.uint8).reshape(900, 1200, 4)
				assert np.count_nonzero(np.any(pixels[:, :, :3] > 0, axis=2)) > 10000
				assert np.max(pixels[:, :, :3]) > 150  # Includes text, not just a background.
	finally:
		texture.destroy()


@pytest.mark.parametrize("logical,physical", [((2560, 1369), (2560, 1369)), ((40, 30), (40, 30)), ((80, 60), (160, 120))])
def test_real_gpu_clipping_matches_surface_during_resize_and_dpi_changes(monkeypatch, logical, physical):
	RenderContext.setup_graphics_backend(True)
	device = RenderContext.device
	texture = device.create_texture(size=(80, 60, 1), format="rgba8unorm",
		usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
	monkeypatch.setattr(glfw, "get_window_size", lambda window: logical)
	monkeypatch.setattr(glfw, "get_framebuffer_size", lambda window: physical)
	monkeypatch.setattr(glfw, "get_cursor_pos", lambda window: (0, 0))
	monkeypatch.setattr(RenderContext, "window", object())
	monkeypatch.setattr(RenderContext, "canvas", object())
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	# The acquired view, rather than GLFW or cached window dimensions, is authoritative.
	monkeypatch.setattr(RenderContext, "window_dimensions", physical)
	monkeypatch.setattr(RenderContext, "_screen_view", texture.create_view())
	def draw():
		imgui.get_background_draw_list().add_rect_filled((0, 0), imgui.get_io().display_size, 0xffffffff)
	try:
		with ImGui() as gui:
			gui.render(draw, clear=(0, 0, 0, 1))
			data = device.queue.read_texture({"texture": texture}, {"bytes_per_row": 80*4}, (80, 60, 1))
			np.testing.assert_array_equal(np.frombuffer(data, dtype=np.uint8), 255)
	finally:
		texture.destroy()
