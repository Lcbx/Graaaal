from scenes.benchmarks.benchmark_loop import *
import scenes.benchmarks.benchmark_loop as loop

import numpy as np
from pyray import *

from numpyecs.ECS import *


@component
class Position:
    x: float
    y: float
    z: float


@component
class Velocity:
    x: float
    y: float
    z: float


@component
class Mesh:
    color: object


@component
class BoundingBox:
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float


world = ECS()
world.register(Position, Velocity, Mesh, BoundingBox)
movement_query = world.query(Position, Velocity)
render_query = world.query(Position, Mesh, BoundingBox)

ground = world.create()
world.add(
    ground,
    Position(0, -0.51, 0),
    Mesh(LIGHTGRAY),
    BoundingBox(
        -25, 0, -25,
        25, 1, 25,
    ),
)

def random_values(low: int, high: int, count: int) -> np.ndarray:
    return np.fromiter(
        (get_random_value(low, high) for _ in range(count)),
        dtype=np.float64,
        count=count,
    )


def add_cubes() -> None:
    count = 250
    ents = world.create(count)

    pos = np.column_stack((
        random_values(-SPACE_SIZE, SPACE_SIZE, count),
        random_values(0, 25, count),
        random_values(-SPACE_SIZE, SPACE_SIZE, count),
    ))

    velocity = (
        random_values(-4, 4, count),
        np.zeros(count),
        random_values(-4, 4, count),
    )

    bbox = (
        random_values(-CUBE_MAX, 0, count),
        random_values(-CUBE_MAX, 0, count),
        random_values(-CUBE_MAX, 0, count),
        random_values(1, CUBE_MAX, count),
        random_values(1, CUBE_MAX, count),
        random_values(1, CUBE_MAX, count),
    )

    colors = np.empty(count, dtype=object)
    colors[:] = [rnd_color() for _ in range(count)]

    world.add(
        ents,
        Position, pos,
        Velocity, velocity,
        BoundingBox, bbox,
        Mesh, colors,
    )


def draw() -> None:
    matched = render_query.fetch()
    pos = matched[Position].selection
    mesh = matched[Mesh].selection
    bbox = matched[BoundingBox].selection

    pos_vec = pos.vector()
    bmins = bbox.vector('x_min', 'y_min', 'z_min')
    bmaxs = bbox.vector('x_max', 'y_max', 'z_max')

    sizes = bmaxs - bmins
    centers = pos_vec + (bmaxs + bmins) * 0.5

    for center, size, color in zip(centers, sizes, mesh.color):
        draw_cube(
            tuple(center),
            size[0],
            size[1],
            size[2],
            color,
        )


def update(frameTime: float) -> None:
    matched = movement_query.fetch()
    pos = matched[Position].selection
    vel = matched[Velocity].selection

    pos.set_vector(pos.vector() + vel.vector() * frameTime)


def cubes_len() -> int:
    return len(world)


loop.add_cubes = add_cubes
loop.draw = draw
loop.update = update
loop.cubes_len = cubes_len
if __name__ == "__main__":
    loop.run()
