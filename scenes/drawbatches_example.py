from RenderContext import *
from Utils import *
from ECS import *
from DrawBatches import *

from math import cos, sin
import random as rd


@component
class Transform:
	position: Vec3
	scale: Vec3
	rotation: Vec4 # Quaternion

@component
class Velocity:
	x: float; y: float; z: float

@component(multiples=True)
class MeshRef:
	shader_id: np.uint32
	lod_id: np.uint32
	tint: np.uint32

world = ECS()
transforms, velocities, mesh_refs = world.register(
	Transform, Velocity, MeshRef
)

CUBE_COUNT = 1000
SPACE_SIZE = 180
CUBE_MAX_SIDE = 7
WINDOW_W, WINDOW_H = 1200, 1200
TITLE = "glfw + wgpu - Split Compute Hi-Z Pipeline"

camera_dist = 30.0
main_camera = Camera(
	position=(-20.0, 70.0, 25.0),
	target=(0.0, 10.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)
observer_camera = Camera(
	position=(140.0, 170.0, 140.0),
	target=(0.0, 0.0, 0.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=60.0,
	near=0.1,
	far=1000.0,
)
active_camera = main_camera

light_camera = Camera(
	position=(-30.0, 30.0, 25.0),
	target=(0.0, 0.0, -20.0),
	up=(0.0, 1.0, 0.0),
	fovy_deg=90.0,
	near=1.0,
	far=300.0,
	perspective=False,
)

RenderContext.init_window(
	WINDOW_W, WINDOW_H, TITLE,
	target_fps=-1,
	required_gpu_features=["indirect-first-instance"]
)

# --- Shaders & Pipelines ---
shader = Shader(filepath='scenes/shaders/complex.shader', label="complex")
uniform_buffer = shader.UniformBuffer()
render_shader = standard_RenderShader(shader, uniform_buffer)

# used to split between observer camera and observer camera uniforms
prepass_uniform_buffer = shader.UniformBuffer()
render_shader.prepass.bindings = ((0, shader.bind_group(0, uniforms=prepass_uniform_buffer)),)

# --- Draw registries & HZB ---
render_data = DrawBatches(mesh_instance_dtype)
render_data.register_shader(0, render_shader)

hzb = HZB()
instance_version = None

vertices, indices = load_gltf_first_mesh_interleaved(
	"scenes/resources/rooftop_utility_pole.glb"
)

model_mesh = Mesh(vertices, indices)
cube_mesh = make_cube_mesh()

render_data.register_mesh(0, model_mesh)
render_data.register_mesh(1, cube_mesh)

# to test LoD system, pole disappears after some distance
render_data.register_lod_group(0, (0, None), (100.0,))
render_data.register_lod_group(1, (1,))


[ground] = world.create()
world.add(
	ground,
	Transform(
		Vec3(0, -0.51, 0),
		Vec3(2 * SPACE_SIZE, 1, 2 * SPACE_SIZE),
		Quaternion(),
	),
	MeshRef(0, 1, pack_rgba8_srgb([0.5, 0.5, 0.5, 1.0])),
)

[model_entity] = world.create()
world.add(
	model_entity,
	Transform(Vec3(15.0, 0.0, 15.0), Vec3(10.0, 10.0, 10.0), Quaternion()),
)
world.add([model_entity, model_entity],
	MeshRef,
	[ 
	 (0, 0, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])), 
	 (0, 1, pack_rgba8_srgb([0.3, 0.5, 0.7, 1.0])),
	]
)

def create_cubes(count):
	cube_entities = world.create(count)
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
		Quaternion.from_axis_rotation((0.0, 1.0, 0.0), rd.random() * 3.143)
		for _ in range(cube_count)
	], dtype=np.float32)

	cube_scales = np.column_stack((
		np.random.randint(1, CUBE_MAX_SIDE, cube_count),
		np.random.randint(1, CUBE_MAX_SIDE, cube_count),
		np.random.randint(1, CUBE_MAX_SIDE, cube_count),
	))

	cube_meshrefs = np.column_stack((
		np.zeros(cube_count, dtype=np.uint32),
		np.full(cube_count, 1, dtype=np.uint32),
		np.asarray([
			pack_rgba8_srgb([rd.random(), rd.random(), rd.random(), 1.0])
			for _ in range(cube_count)
		], dtype=np.uint32),
	))

	world.add(
		cube_entities,
		Transform, (cube_positions, cube_scales, cube_rotations),
		Velocity, cube_velocities,
		MeshRef, cube_meshrefs,
	)

deleted_count = 2 # start after ground and pole mesh
def delete_cubes(count):
	global deleted_count
	world.delete(range(deleted_count, deleted_count+count))
	deleted_count += count


create_cubes(CUBE_COUNT)


def camera_system(camera, elapsed, camera_dist):
	cam_ang = elapsed * 0.5
	camera.position = Vec3(cos(cam_ang) * camera_dist, camera.position.y, sin(cam_ang) * camera_dist)


def movement_system(world, dt):
	pv = world.where(Transform, Velocity)
	p, v = transforms[pv], velocities[pv]
	p_vec, v_vec = p.position, v.vector()
	p_vec += v_vec * dt

	mask_x = np.abs(p_vec[:, 0]) > SPACE_SIZE
	mask_z = np.abs(p_vec[:, 2]) > SPACE_SIZE
	v_vec[mask_x, 0] *= -1
	v_vec[mask_z, 2] *= -1
	p_vec[mask_x, 0] = np.sign(p_vec[mask_x, 0]) * 0.99 * SPACE_SIZE
	p_vec[mask_z, 2] = np.sign(p_vec[mask_z, 2]) * 0.99 * SPACE_SIZE

	p.position = p_vec
	v.set_vector(v_vec)


def update_instances(world, data):
	global instance_version
	
	entities = world.where(Transform, MeshRef)
	data.sync_batches(entities, transforms, mesh_refs)
	version = (data.instance_order_version, transforms.version("position"), transforms.version("rotation"), transforms.version("scale"), mesh_refs.version("tint"))
	if instance_version == version: return
	previous = instance_version or (None,) * 5
	order_changed = previous[0] != version[0]
	changed = tuple(order_changed or old != new for old, new in zip(previous[1:], version[1:]))
	if len(data.entities):
		p = transforms[data.entities] if any(changed[:3]) else None
		fields = {}
		if changed[0]: fields["iPosition"] = p.position
		if changed[1]: fields["iRotation"] = pack_quaternion(p.rotation)
		if changed[2]: fields["iScale"] = pack_scale(p.scale)
		if changed[3]: fields["iTint"] = data.mesh_field("tint").reshape(-1, 1)
		data.write_instance_fields(**fields)
	instance_version = version


def update_cameras(data, culling_camera, rendering_camera, light_dir):
	for buffer, camera in (
		(prepass_uniform_buffer, culling_camera),
		(uniform_buffer, rendering_camera)
	):
		view = camera.view()
		proj = camera.projection(RenderContext.aspect)
		buffer.content["view"] = view
		buffer.content["proj"] = proj
		buffer.content["light_dir"] = [*light_dir, 0.0]
		buffer.upload()
		if camera is culling_camera:
			vp = view @ proj

	data.update_cull_camera(culling_camera.position, vp)


def render_system(world, culling_camera, rendering_camera):
	width, height = RenderContext.windowDimensions
	if width <= 0 or height <= 0: return
	update_instances(world, render_data)

	hzb.resize((width, height))
	render_data.refresh_bindings(hzb.view)
	update_cameras(render_data, culling_camera, rendering_camera, light_camera.direction())
	cmd = RenderContext.commands("frame_commands")
	render_data.reset_draw_counts(cmd)

	# --- Step 1: Frustum Culling Pass ---
	render_data.cull_instances(cmd, "frustum")

	# --- Step 2: Depth Prepass ---
	with cmd.render_pass(color=(), depth=hzb.depth_texture.depth_attachment(clear=1.0), label="depth_prepass") as rp:
		render_data.draw(rp, "prepass")

	# --- Step 3: Compute Hi-Z Downsampling ---
	hzb.build(cmd)

	# --- Step 4: Hi-Z Occlusion Culling Pass ---
	render_data.cull_instances(cmd, "hiz")

	# --- Step 5: Main Render Pass ---
	depth = hzb.depth_texture.depth_attachment(clear=None if rendering_camera is culling_camera else 1.0)
	with cmd.render_pass(color=RenderContext.screen(clear=(0.02, 0.02, 0.03, 1.0)), depth=depth, label="main_pass") as rp:
		render_data.draw(rp, "main")
	RenderContext.submit(cmd.finish())


def clamp(val, val_min, val_max):
	return min(max(val, val_min), val_max)

def scroll_callback(xoff, yoff):
	global camera_dist, main_camera
	camera_dist = clamp(camera_dist - 5.0 * yoff, 5.0, 100.0)
	cp = main_camera.position
	y_factor = 1.0 if cp.y < 15.0 else 3.0
	main_camera.position = Vec3(cp.x, cp.y - y_factor * yoff, cp.z)
	return True

def on_char(code):
	global active_camera
	if chr(code) == " ":
		active_camera = observer_camera if active_camera is main_camera else main_camera
		print("switch")
	return True

RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)
RenderContext.event_handlers["char"].append(on_char)

fps_frames = 0
start_t = get_time()
fps_print_timestamp = start_t

while RenderContext.window_loop():
	now = RenderContext.frame_start

	fps_frames += 1
	if (big_frame := now - fps_print_timestamp >= 1.0):
		create_cubes(10)
		delete_cubes(3)

	elapsed = now - start_t
	camera_system(main_camera, elapsed, camera_dist)
	movement_system(world, RenderContext.frame_time)
	with WatchTimer("render system"):
		render_system(world, main_camera, active_camera)

	if big_frame:
		print(f"fps {fps_frames}")
		print("render instance count", render_data.entities.size)
		print(WatchTimer.capture())

		# reset counter
		fps_frames = 0
		fps_print_timestamp = now
