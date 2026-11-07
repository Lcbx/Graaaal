from scenes.benchmarks.benchmark_loop import *
import scenes.benchmarks.benchmark_loop as loop
from pyray import *
from dataclasses import dataclass

@dataclass
class Cube:
    position: Vector3
    velocity: Vector3
    color : Color
    BoundingBox_min : Vector3
    BoundingBox_max : Vector3


ground = Cube(
    Vector3(0,-0.51,0),
    Vector3(0,0,0),
    LIGHTGRAY,
    Vector3(-25,0,-25),
    Vector3(25,1,25)
)

cubes = [ground]

def add_cubes():
    for e in range(250):
        cubes.append(Cube(
            Vector3(
                get_random_value(-SPACE_SIZE, SPACE_SIZE),
                get_random_value(0, 25),
                get_random_value(-SPACE_SIZE, SPACE_SIZE) ),
            Vector3(
                get_random_value(-4, 4),
                0,
                get_random_value(-4, 4) ),
            rnd_color(),
            Vector3(get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0), get_random_value(-CUBE_MAX, 0) ),
            Vector3(get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX), get_random_value(1, CUBE_MAX) ),
        ))

def draw():
    for c in cubes:
        center = vector3_scale( vector3_add(c.BoundingBox_min, c.BoundingBox_max), 0.5)
        size = vector3_subtract(c.BoundingBox_max, c.BoundingBox_min)
        draw_cube(
            vector3_add(c.position, center),
            size.x, size.y, size.z,
            c.color
        )

def update(frameTime):
    for c in cubes:
        c.position = vector3_add(c.position, vector3_scale(c.velocity, frameTime))

def cubes_len():
    return len(cubes)

loop.add_cubes = add_cubes
loop.draw = draw
loop.update = update
loop.cubes_len = cubes_len
if __name__ == "__main__":
    loop.run()

