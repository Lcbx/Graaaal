"""Packaged default thresholds, resource sharing and device-generation safety."""
from unittest.mock import Mock
import weakref
import gc

import numpy as np

from numpyecs import TextureProcessing as textures
from numpyecs.Assets import DEFAULT_BLUE_NOISE_PATH
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.MaterialRenderer import MaterialRenderer
from scenes.material_example import make_instances


def test_packaged_default_matches_stored_thresholds_without_cwd_or_pillow(monkeypatch, tmp_path):
	samples = np.load(DEFAULT_BLUE_NOISE_PATH, allow_pickle=False)
	expected = (samples.astype(np.float32)+0.5)/65536
	RenderContext.setup_graphics_backend(True)
	monkeypatch.setattr(textures, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.chdir(tmp_path)
	import sys
	monkeypatch.setitem(sys.modules, "PIL", None)
	tile = textures.get_default_blue_noise()
	assert textures.get_default_blue_noise() is tile
	data = RenderContext.device.queue.read_texture({"texture": tile.handle}, {"bytes_per_row": 256}, (64, 64, 1))
	np.testing.assert_array_equal(np.frombuffer(data, dtype=np.float32).reshape(64, 64), expected)
	assert 0 < expected.min() < expected.max() < 1


def test_default_texture_is_recreated_for_new_devices_and_closed_resources():
	RenderContext.setup_graphics_backend(True)
	first = textures.get_default_blue_noise()
	first.close()
	second = textures.get_default_blue_noise()
	assert second is not first
	RenderContext.setup_graphics_backend(True)
	third = textures.get_default_blue_noise()
	assert third is not second
	third._check_device()
	assert textures.get_default_blue_noise() is third


def test_default_texture_cache_does_not_keep_gpu_resources_alive():
	RenderContext.setup_graphics_backend(True)
	tile = textures.get_default_blue_noise()
	reference = weakref.ref(tile)
	del tile
	gc.collect()
	assert reference() is None


def test_material_renderers_share_default_and_respect_explicit_override(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	instances, _, _ = make_instances()
	first = MaterialRenderer(instances, target_format="rgba8unorm")
	second = MaterialRenderer(instances, target_format="rgba8unorm")
	assert first.transparency.noise is second.transparency.noise
	assert first.transparency.noise.size == (64, 64, 1)
	custom = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING)
	loader = Mock(side_effect=AssertionError("Explicit overrides must bypass the default loader"))
	monkeypatch.setattr(textures, "get_default_blue_noise", loader)
	third = MaterialRenderer(instances, blue_noise=custom, target_format="rgba8unorm")
	assert third.transparency.noise is custom
	loader.assert_not_called()
