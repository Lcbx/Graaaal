from RenderContext import *
from Utils import *
from ECS import *

from math import cos, sin
import random as rd

@component
class Position:
	x: float; y: float; z: float


@component
class Velocity:
	x: float; y: float; z: float


@component
class Rotation:
	x: float; y: float; z: float; w: float


@component
class Scale:
	x: float; y: float; z: float


@component
class Tint:
	value: np.uint32


world = ECS()
positions, velocities, rotations, scales, tints = world.register(
	Position, Velocity, Rotation, Scale, Tint
)

CUBE_COUNT = 1000
SPACE_SIZE = 180
CUBE_MAX_SIDE = 7

ground = world.create()
world.add(
	ground,
	Position(0, -0.51, 0),
	Rotation(*Quaternion()),
	Scale(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE),
	Tint(pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])),
)

cube_entities = world.create(CUBE_COUNT)
cube_count = cube_entities.size

cube_positions = np.column_stack((
	np.random.randint(-SPACE_SIZE, SPACE_SIZE, cube_count),
	np.random.randint(0, 20, cube_count),
	np.random.randint(-SPACE_SIZE, SPACE_SIZE, cube_count),
))
cube_velocities = np.column_stack((
	np.random.randint(-4, 4, cube_count),
	np.zeros(cube_count),
	np.random.randint(-4, 4, cube_count),
))
cube_rotations = np.asarray([
	Quaternion.from_axis_rotation( (0.0,1.0,0.0), rd.random() * 3.143 )
	for _ in range(cube_count)
], dtype=np.float32)


cube_scales = np.column_stack((
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	np.random.randint(1, CUBE_MAX_SIDE, cube_count),
))
cube_tints = np.asarray([
	pack_rgba8_srgb([rd.random(), rd.random(), rd.random(), 1.0])
	for _ in range(cube_count)
], dtype=np.uint32)

world.add(
	cube_entities,
	Position, cube_positions,
	Velocity, cube_velocities,
	Rotation, cube_rotations,
	Scale, cube_scales,
	Tint, cube_tints,
)

WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + wgpu"

camera_dist = 30.0
camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)

light_camera = Camera(
	position=(-30.0, 30.0, 25.0),
	target=(0.0, 0.0, -20.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=90.0,
	near=1.0,
	far=300.0,
	perspective=False,
)

RenderContext.init_window(WINDOW_W, WINDOW_H, TITLE, target_fps=-1)
# RenderContext.capture_mouse()

shader = Shader(filepath='scenes/shaders/simple.shader', label="simple")
main_pipeline = RenderPipeline(
	shader,
	vertex_entry="vertex",
	fragment_entry="fragment",
	label="main",
)
shadow_pipeline = RenderPipeline(
	shader,
	vertex_entry="shadow_vertex",
	depth_bias=2,
	depth_bias_slope_scale=2.0,
	label="shadow",
)

vertices, indices = load_gltf_first_mesh_interleaved(
	"scenes/resources/rooftop_utility_pole.glb"
)
model_mesh = Mesh(vertices, indices)
model_instances = np.zeros(1, mesh_instance_dtype)
model_instances[0]["iPosition"] = Vec3(15.0, 0.0, 15.0)
model_instances[0]["iTint"] = pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])
model_instances[0]["iRotation"] = pack_quaternion(Quaternion())
model_instances[0]["iScale"] = pack_scale([10.0] * 3)
model_instances[0]["iPosition"] = Vec3(15.0, 0.0, 15.0)
model_instance_buffer = GpuBuffer(
	model_instances,
	BufferUsage.VERTEX | BufferUsage.COPY_DST,
)

cube_mesh = make_cube_mesh()
render_entities = world.where(Position, Rotation, Scale, Tint)

def make_cube_instances(entities):
	instances = np.empty(entities.size, dtype=mesh_instance_dtype)
	instances["iPosition"] = positions[entities].vector()
	instances["iTint"][:, 0] = tints[entities].value
	instances["iRotation"] = pack_quaternion(rotations[entities].vector())
	instances["iScale"] = pack_scale(scales[entities].vector())
	return instances

cube_instance_buffer = GpuBuffer(
	make_cube_instances(render_entities),
	BufferUsage.VERTEX | BufferUsage.COPY_DST,
)

uniform_buffer = shader.UniformBuffer()

shadow_texture = create_depth_framebuffer(1024, 1024)
shadow_view = shadow_texture.view()
shadow_sampler = create_depth_sampler()

uniform_bindings = shader.bind_group( 0, uniforms=uniform_buffer)
shadow_bindings  = shader.bind_group( 1,
	shadow_map     = shadow_view,
	shadow_sampler = shadow_sampler
)


def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)


def scroll_callback(xoff, yoff):
	global camera_dist, camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	camera.position = Vec3(cp.x, cp.y - y_factor * yoff, cp.z)
	return True


RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)

fps_frames = 0
start_t = get_time()
fps_print_timestamp = start_t

while RenderContext.window_loop():
	now = RenderContext.frame_start

	fps_frames += 1
	if now - fps_print_timestamp >= 1.0:
		print(f"fps {fps_frames}")
		fps_frames = 0
		fps_print_timestamp = now

	# orbit update
	elapsed = now - start_t
	cam_ang = elapsed * 0.5
	camera.position = Vec3(
		cos(cam_ang) * camera_dist,
		camera.position.y,
		sin(cam_ang) * camera_dist,
	)

	# cubes movement
	pv = world.where(Position, Velocity)
	p = positions[pv]
	v = velocities[pv]
	p_vec, v_vec = p.vector(), v.vector()
	p_vec += v_vec * RenderContext.frame_time

	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	p.set_vector(p_vec)
	v.set_vector(v_vec)

	cube_instance_buffer.content = make_cube_instances(render_entities)
	cube_instance_buffer.resize(cube_instance_buffer.content.size)

	view = camera.view()
	proj = camera.projection(RenderContext.aspect)
	light_view = light_camera.view()
	light_proj = light_camera.projection(RenderContext.aspect)

	uniform_buffer.content["view"] = view
	uniform_buffer.content["proj"] = proj
	uniform_buffer.content["light_dir"] = [*light_camera.direction(), 0.0]
	uniform_buffer.content["light_view_proj"] = light_view @ light_proj

	cube_instance_buffer.upload()
	uniform_buffer.upload()

	with (shadow_cmd := RenderContext.commands("shadow")).render_pass(
		depth=shadow_texture.depth_attachment(clear=1.0),
		label="shadow",
	) as rp:
		rp.set_pipeline(shadow_pipeline)
		rp.set_bind_group(uniform_bindings)

		rp.draw_mesh(model_mesh, instances=model_instance_buffer)
		rp.draw_mesh(cube_mesh, instances=cube_instance_buffer)

	with (main_cmd := RenderContext.commands("main")).render_pass(
		color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)),
		depth=RenderContext.depth_attachment(clear=1.0),
		label="main",
	) as rp:
		rp.set_pipeline(main_pipeline)
		rp.set_bind_group(uniform_bindings)
		rp.set_bind_group(shadow_bindings)

		rp.draw_mesh(model_mesh, instances=model_instance_buffer)
		rp.draw_mesh(cube_mesh, instances=cube_instance_buffer)

	RenderContext.submit(
		shadow_cmd.finish(),
		main_cmd.finish(),
	)
