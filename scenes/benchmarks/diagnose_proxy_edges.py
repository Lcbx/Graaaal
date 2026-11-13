"""Measure negative tail colors at a proxy discontinuity with constant lighting.

Run with python -m scenes.benchmarks.diagnose_proxy_edges.
"""
import json

import numpy as np

from graaal.MaterialRenderer import MaterialRenderer
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal import TransparencyPass as transparency_module
from scenes.benchmarks.profile_banded_tail import make_scene, read_color


def run(size=(256, 192), matching_proxy=False):
	"""Return proxy-off/on measurements for constant-color overlapping quads."""
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	original_texture = transparency_module.Texture
	def readable(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1: usage |= TextureUsage.COPY_SRC
		return Texture(*args, usage=usage, **kwargs)
	transparency_module.Texture = readable
	try:
		items, camera, *_ = make_scene(size)
		for index, item in enumerate(items):
			item.material["emission"] = (.05, .05, .05)
			color = (.05, .05, .05) if matching_proxy else ((1., 1., 1.) if index == 1 else (.1, .1, .1))
			item.material["base_color"] = (*color, .5)
		renderer = MaterialRenderer(items, target_format="rgba8unorm", banded_proxy=False)
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		rows = []
		for samples, native_tail in ((1, False), (4, False), (4, True)):
			renderer.set_sample_count(samples)
			renderer.set_banded_options(tail_msaa=native_tail)
			for pairs in (0, 1, 2):
				renderer.set_kawase_passes(pairs)
				for proxy in (False, True):
					renderer.set_banded_options(proxy=proxy)
					commands = RenderContext.commands()
					renderer.record(commands, size, camera, output.color_attachment())
					RenderContext.submit(commands.finish())
					tail = read_color(renderer.transparency.reconstruction_color)[..., :3]
					pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size[0]*4, "rows_per_image": size[1]}, output.size)
					image = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 4)
					rows.append(dict(samples=samples, native_tail=native_tail, pairs=pairs, proxy=proxy,
						tail_min=float(tail.min()), negative_tail_samples=int(np.any(tail < -.01, axis=-1).sum()),
						black_output_pixels=int(np.all(image[..., :3] == 0, axis=-1).sum())))
		return rows
	finally:
		transparency_module.Texture = original_texture
		RenderContext.cleanup()


if __name__ == "__main__": print(json.dumps(run(), indent=2))
