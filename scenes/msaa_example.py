"""Rotating triangle: press 1 for no MSAA or 4 for 4x MSAA."""

def run_scene():
	from numpyecs.RenderContext import (
		RenderContext,
		RenderPipeline,
		Shader,
		Texture,
		TextureUsage,
	)

	RenderContext.init_window(800, 600, "RenderContext MSAA | 1: off | 4: 4x MSAA")
	sample_count = 4
	msaa_texture = None
	try:
		shader = Shader(source="""
			struct Frame {
				angle: f32,
				aspect: f32,
			}
			@group(0) @binding(0) var<uniform> frame: Frame;

			@vertex
			fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
				let vertices = array<vec2f, 3>(
					vec2f(-0.8, -0.8), vec2f(0.8, -0.8), vec2f(0.0, 0.8)
				);
				let c = cos(frame.angle);
				let s = sin(frame.angle);
				let p = vertices[i];
				let rotated = vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
				// Fit the full rotation and preserve proportions when resizing.
				let scale = 0.8 * vec2f(min(1.0 / frame.aspect, 1.0), min(frame.aspect, 1.0));
				return vec4f(rotated * scale, 0.0, 1.0);
			}

			@fragment
			fn fs() -> @location(0) vec4f {
				return vec4f(1.0, 0.5, 0.1, 1.0);
			}
		""", label="msaa")
		pipelines = {
			count: RenderPipeline(
				shader,
				depth_test=None,
				depth_write=False,
				sample_count=count,
				label=f"msaa_{count}x",
			)
			for count in (1, 4)
		}
		uniforms = shader.uniform_buffer()
		bindings = shader.bind_group(0, frame=uniforms)
		start_time = RenderContext.frame_start

		def on_char(code):
			nonlocal sample_count
			if chr(code) not in ("1", "4"):
				return False
			count = int(chr(code))
			if count != sample_count:
				sample_count = count
				print(f"Sample count: {sample_count}x")
			return True

		RenderContext.subscribe_event("char", on_char)

		while RenderContext.window_loop():
			size = RenderContext.window_dimensions
			if size[0] <= 0 or size[1] <= 0:
				continue

			if sample_count > 1 and (msaa_texture is None or msaa_texture.size[:2] != size):
				if msaa_texture is not None:
					msaa_texture.close()
				msaa_texture = Texture(
					size,
					format=RenderContext.presentation_format,
					usage=TextureUsage.RENDER_ATTACHMENT,
					sample_count=sample_count,
					label="msaa_color",
				)

			uniforms.content["angle"] = (RenderContext.frame_start - start_time) * 0.8
			uniforms.content["aspect"] = RenderContext.aspect
			uniforms.upload()

			clear = (0.04, 0.06, 0.10, 1.0)
			if sample_count == 1:
				color = RenderContext.screen(clear=clear)
			else:
				color = msaa_texture.color_attachment(
					clear=clear,
					store=False,
					resolve_target=RenderContext.screen(),
				)
			commands = RenderContext.commands("msaa")
			with commands.render_pass(
				color=color,
				label="msaa",
			) as render:
				render.set_pipeline(pipelines[sample_count])
				render.set_bind_group(bindings)
				render.draw(3)
			RenderContext.submit(commands.finish())
	finally:
		if msaa_texture is not None:
			msaa_texture.close()
		RenderContext.cleanup()


if __name__ == "__main__":
	run_scene()
