import numpy as np
import pytest

from graaal.DefaultMaterial import DefaultMaterialShader
from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.TransparencyData import VERTEX_DTYPE


@pytest.mark.parametrize("sample_count", [1, 4])
@pytest.mark.parametrize("alpha_mode", [0, 1, 2])
def test_material_backfaces_default_to_hidden_and_double_sided_is_live(alpha_mode, sample_count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	shader = DefaultMaterialShader(transparent=alpha_mode == 2)
	material = {"base_color": (0, 0, 0, 0.75), "emission": (1, 0, 0), "alpha_mode": alpha_mode}
	vertices = np.zeros(3, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-10, -10, 0), (10, -10, 0), (0, 10, 0)]
	vertices["normal"] = (0, 0, 1)
	front = MaterialMesh(vertices, shader)
	back = MaterialMesh(vertices[::-1].copy(), shader)
	item = DrawInstance(back, material=material)
	renderer = MaterialRenderer([item], meshes=[front, back], target_format="rgba8unorm", sample_count=sample_count)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	background = np.array((0.035, 0.05, 0.08))
	background /= 1+background
	background = np.rint((1.055*background**(1/2.4)-0.055)*255)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (16, 16), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64}, (16, 16, 1))
		return np.frombuffer(pixels, dtype=np.uint8).reshape(16, 16, 4)[8, 8, :3].copy()
	for tail in (("banded", "stochastic") if alpha_mode == 2 else ("banded",)):
		renderer.set_tail_allocation(tail)
		compiled = renderer.transparency.shader
		item.mesh = back
		material.pop("double_sided", None)
		np.testing.assert_allclose(render(), background, atol=1)
		material["double_sided"] = True
		assert render()[0] > 150
		material["double_sided"] = False
		np.testing.assert_allclose(render(), background, atol=1)
		item.mesh = front
		assert render()[0] > 150
		assert renderer.transparency.shader is compiled
