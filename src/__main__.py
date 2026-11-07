"""Installed-package examples: python -m numpyecs [--offscreen | --window]."""

import argparse
from pathlib import Path

import numpy as np

from .ECS import ECS, array_field, columns, component


@component
class Position:
    value: np.ndarray = array_field(np.float32, shape=(3,))


def simulate(steps=60):
    world = ECS()
    positions, = world.register(Position)
    entities = world.create(2)
    world.add(entities, columns(Position, value=[[-0.5, 0, 0], [0.5, 0, 0]]))
    velocity = np.array([0.1, 0, 0], dtype=np.float32)
    for _ in range(steps):
        with positions.edit_dense("value") as fields:
            fields["value"] += velocity / 60
    return positions[entities].value


def render_example(*, window=False, frames=1):
    from .RenderContext import Mesh, RenderContext, RenderPipeline, Shader, Texture, TextureUsage
    from .DrawBatches import DrawBatches, RenderShader, ShaderPass
    from .InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale

    if window:
        RenderContext.init_window(640, 480, "NumPy ECS procedural example",
            required_gpu_features=["indirect-first-instance"])
    else:
        RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
    try:
        shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "procedural_example.shader"))
        pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
        vertices = np.zeros(3, dtype=[("position", "f4", 3)])
        vertices["position"] = [[-0.2, -0.2, 0], [0.2, -0.2, 0], [0, 0.2, 0]]
        draws = DrawBatches(MESH_INSTANCE_DTYPE)
        draws.register_mesh(1, Mesh(vertices, np.arange(3, dtype=np.uint32)))
        draws.register_lod_group(1, (1,))
        draws.register_shader(0, RenderShader({"color": ShaderPass(pipeline)}))
        draws.sync([1, 1], [0, 0])
        visibility = draws.create_visibility()
        output = None if window else Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
        for _ in range(frames):
            if window and not RenderContext.window_loop(): break
            draws.write_instance_fields(iPosition=simulate(), iTint=0,
                iRotation=pack_quaternion([0, 0, 0, 1]), iScale=pack_scale([1, 1, 1]))
            visibility.all_visible()
            commands = RenderContext.commands("example")
            target = RenderContext.screen(clear=(0, 0, 0, 1)) if window else output.color_attachment(clear=(0, 0, 0, 1))
            with commands.render_pass(color=target) as render:
                draws.draw(render, visibility, "color")
            RenderContext.submit(commands.finish())
        if window: RenderContext.present()
        else:
            pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
            if not np.any(np.frombuffer(pixels, dtype=np.uint8).reshape(64, 64, 4)[:, :, 0]):
                raise RuntimeError("Example produced no colored pixels")
            print("Offscreen example rendered two procedural instances")
    finally:
        RenderContext.cleanup()


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offscreen", action="store_true")
    mode.add_argument("--window", action="store_true")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.frames <= 0: parser.error("--frames must be positive")
    if args.offscreen or args.window: render_example(window=args.window, frames=args.frames)
    else: print(simulate(args.frames))


if __name__ == "__main__": main()
