"""Reusable hierarchical depth-pyramid generation."""

from pathlib import Path
from typing import Any

from .RenderContext import BindGroup, ComputePipeline, Shader, Texture, TextureUsage


class HZB:
	"""Build a hierarchical depth pyramid independently of draw batching or culling."""
	def __init__(self,
		shader: Shader | None = None,
		pipelines: tuple[ComputePipeline, ComputePipeline] | None = None
	) -> None:
		self._shader = shader or Shader(filepath=str(Path(__file__).with_name("shaders") / "hzb.shader"), label="hzb")
		self._pipelines = pipelines or (
			ComputePipeline(self._shader, entry="reduce_depth", label="hzb_depth"),
			ComputePipeline(self._shader, entry="main", label="hzb")
		)
		self.size: tuple[int, int] | tuple[()] = ()
		self.depth_texture: Texture | None = None
		self.texture: Texture | None = None
		self.view: Any = None
		self.passes: list[tuple[ComputePipeline, BindGroup, int, int]] = []

	def resize(self, size: tuple[int, int]) -> bool:
		shader, pipelines = self._shader, self._pipelines
		width, height = size
		if width <= 0 or height <= 0 or self.size == size: return False
		usage = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
		self.depth_texture = Texture(size, format="depth32float", usage=usage, label="prepass_depth")
		width, height = max(1, width // 2), max(1, height // 2)
		num_mips = max(width, height).bit_length()
		usage = TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING
		self.texture = Texture((width, height), format="r32float", usage=usage, mip_level_count=num_mips, label="hzb_pyramid")
		self.view = self.texture.view()
		views = [self.texture.view(base_mip_level=i, mip_level_count=1) for i in range(num_mips)]
		bindings = shader.bind_group(0, src_prepass=self.depth_texture.view(), dst_depth=views[0])
		self.passes = [(pipelines[0], bindings, width, height)]
		for mip in range(1, num_mips):
			width, height = max(1, width // 2), max(1, height // 2)
			bindings = shader.bind_group(0, src_depth=views[mip - 1], dst_depth=views[mip])
			self.passes.append((pipelines[1], bindings, width, height))
		self.size = size
		return True

	def build(self, cmd: Any) -> None:
		for mip, (pipeline, bindings, width, height) in enumerate(self.passes):
			with cmd.compute_pass(label=f"hzb_mip_{mip}") as cp:
				cp.set_pipeline(pipeline)
				cp.set_bind_group(bindings)
				cp.dispatch((width + 15) // 16, (height + 15) // 16)
