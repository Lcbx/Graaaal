"""Single-winner stochastic OIT, using DrawBatches for every geometry pass.

Run: python main.py scenes/transparency_example.py [--frames 120]
Space toggles orbit, scrolling zooms, R reverses transparent submission order.
3 and 5 select the bilateral reconstruction kernel size.
D cycles diagnostic views; A toggles adaptation; F shows/hides ImGui settings.
Drag settings or Ctrl+click to type; Up/Down and Left/Right also select/edit.
Imports only prepare CPU helpers; graphics are initialized by run_scene().
"""

import base64
from dataclasses import asdict, dataclass, replace
import io
import json
from math import cos, sin
from pathlib import Path

import numpy as np
import glfw
from PIL import Image
from pygltflib import GLTF2

from ECS import ECS, component
from DrawBatches import DrawBatches, RenderShader, ShaderPass
from RenderContext import (
	BufferUsage, ComputePipeline, GpuBuffer, Mesh, RenderContext,
	RenderPipeline, Sampler, Shader, Texture, TextureUsage,
)
from Utils import Camera, Vec3, Vec4, mesh_instance_dtype, pack_quaternion, pack_scale


PATH = Path(__file__).parent / "resources/small_scene.glb"
SHADERS = Path(__file__).parent / "shaders"
BLUE_NOISE_PATH = Path(__file__).parent / "resources/blue_noise/HDR_L_0.png"
TEXTURE_SIZE = 512
DEBUG_VIEWS = ("final", "transmittance", "winner", "reconstruction", "filter_support")
VERTEX_DTYPE = np.dtype([
	("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2),
	("barycentric", "<f4", 2), ("triangle", "<u4"),
])
SHADING_VERTEX_DTYPE = np.dtype([
	("position", "<f4", 4), ("normal", "<f4", 4), ("uv", "<f4", 4),
])
MATERIAL_DTYPE = np.dtype([
	("base_color", "<f4", 4), ("emission", "<f4", 4),
	("surface", "<f4", 4), ("opacity_rgb", "<f4", 4),
	("flags", "<u4", 4),
])


@component
class Transform:
	position: Vec3
	scale: Vec3
	rotation: Vec4


@component
class MeshRef:
	shader_id: np.uint32
	lod_id: np.uint32
	triangle_offset: np.uint32
	material_id: np.uint32
	stable_id: np.uint32


@dataclass
class Primitive:
	vertices: np.ndarray
	material_id: int
	triangle_offset: int
	stable_id: int


@dataclass(frozen=True)
class BilateralFilter:
	"""Reconstruction footprint and spatial/depth/transmittance edge tolerances.

	Spatial sigma is in pixels; depth sigma is relative to view-space depth,
	with a world-space minimum. Near/far sigma sets interpolate with view depth.
	Transmittance sigma operates on linear RGBA; its far weight can fade to zero.
	"""
	kernel_size: int = 5
	# Retain the original reconstruction support near cutout-shaped BLEND edges.
	spatial_sigma: float = 2**0.5
	depth_sigma: float = 0.02
	transmittance_sigma: float = (1/24)**0.5
	min_depth_sigma: float = 0.01
	adaptive: bool = True
	filter_near: float = 10.0
	filter_far: float = 50.0
	far_spatial_sigma: float = 2.0
	far_depth_sigma: float = 0.05
	far_transmittance_sigma: float = 0.5
	far_transmittance_weight: float = 0.0

	def __post_init__(self):
		if not isinstance(self.kernel_size, int) or self.kernel_size not in (3, 5):
			raise ValueError("Bilateral kernel_size must be 3 or 5")
		for name in ("spatial_sigma", "depth_sigma", "transmittance_sigma", "min_depth_sigma",
			"far_spatial_sigma", "far_depth_sigma", "far_transmittance_sigma"):
			value = getattr(self, name)
			if not np.isfinite(value) or value <= 0:
				raise ValueError(f"Bilateral {name} must be finite and positive")
		if not isinstance(self.adaptive, bool):
			raise ValueError("Bilateral adaptive must be a boolean")
		if not (np.isfinite(self.filter_near) and np.isfinite(self.filter_far)
			and 0 <= self.filter_near < self.filter_far):
			raise ValueError("Bilateral distances must satisfy 0 <= filter_near < filter_far")
		if not np.isfinite(self.far_transmittance_weight) or not 0 <= self.far_transmittance_weight <= 1:
			raise ValueError("Bilateral far_transmittance_weight must be between zero and one")


def buffer_bytes(gltf, index, base_dir):
	buffer = gltf.buffers[index]
	if buffer.uri is None:
		return gltf.binary_blob()
	if buffer.uri.startswith("data:"):
		return base64.b64decode(buffer.uri.split(",", 1)[1])
	return (Path(base_dir) / buffer.uri).read_bytes()


def get_image_bytes(gltf, image, base_dir=None):
	if image.uri:
		if image.uri.startswith("data:"):
			return base64.b64decode(image.uri.split(",", 1)[1])
		return (Path(base_dir) / image.uri).read_bytes()
	if image.bufferView is not None:
		view = gltf.bufferViews[image.bufferView]
		start = view.byteOffset or 0
		return buffer_bytes(gltf, view.buffer, base_dir)[start:start + view.byteLength]
	return None


def read_accessor(gltf, index, base_dir):
	"""Read interleaved/normalized glTF attributes without opening a GPU device."""
	accessor = gltf.accessors[index]
	if accessor.sparse is not None:
		raise ValueError("This example does not support sparse glTF accessors")
	dtype = np.dtype({5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2",
		5125: "<u4", 5126: "<f4"}[accessor.componentType])
	components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor.type]
	if accessor.bufferView is None:
		values = np.zeros((accessor.count, components), dtype=dtype)
	else:
		view = gltf.bufferViews[accessor.bufferView]
		values = np.ndarray(
			(accessor.count, components), dtype=dtype,
			buffer=buffer_bytes(gltf, view.buffer, base_dir),
			offset=(view.byteOffset or 0) + (accessor.byteOffset or 0),
			strides=(view.byteStride or components * dtype.itemsize, dtype.itemsize),
		).copy()
	if accessor.normalized and dtype.kind in "iu":
		values = values.astype(np.float32) / np.iinfo(dtype).max
		if dtype.kind == "i":
			values = np.maximum(values, -1.0)
	return values


def node_matrix(node):
	"""Column-vector glTF local transform (including nonuniform/mirrored scale)."""
	if node.matrix is not None:
		return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
	x, y, z, w = node.rotation or (0, 0, 0, 1)
	rotation = np.array([
		[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
		[2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
		[2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
	])
	matrix = np.eye(4)
	matrix[:3, :3] = rotation @ np.diag(node.scale or (1, 1, 1))
	matrix[:3, 3] = node.translation or (0, 0, 0)
	return matrix


def build_material_texture_array(images, materials, size):
	"""Upload source images, then resample/filter/correct/encode mips on the GPU.

	Uncorrected premultiplied linear RGBA32F scratch mips retain tiny alpha values.
	MASK coverage search uses GPU reductions; no CPU filtering or read-back occurs.
	Encoded bytes are copied from a storage texture into the sRGB sampling array.
	This also works with WGPU backends that do not support mutable texture views.
	"""
	width, height = size
	levels = max(width, height).bit_length()
	result = Texture((*size, len(images)), format="rgba8unorm-srgb", mip_level_count=levels,
		usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="material_mip_array")
	encoded = Texture((*size, 1), format="rgba8unorm", mip_level_count=levels,
		usage=TextureUsage.STORAGE_BINDING | TextureUsage.COPY_SRC, label="encoded_mip_scratch")
	shader = Shader(filepath=str(SHADERS / "material_mipmaps.shader"))
	pipelines = {entry: ComputePipeline(shader, entry=entry) for entry in (
		"decode", "resample", "init_source", "measure_source", "begin_search", "measure_candidate", "update_search", "encode")}
	def dispatch(compute, entry, resources, x=1, y=1):
		compute.set_pipeline(pipelines[entry])
		compute.set_bind_group(shader.bind_group(0, **resources))
		compute.dispatch(x, y)

	for layer, (image, material) in enumerate(zip(images, materials)):
		source_height, source_width, _ = image.shape
		source_size = (source_width, source_height)
		upload = Texture(source_size, format="rgba8unorm-srgb",
			usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": upload.handle}, np.ascontiguousarray(image),
			{"bytes_per_row": source_width*4, "rows_per_image": source_height}, (*source_size, 1))
		scratch_usage = TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING
		original = Texture(source_size, format="rgba32float", usage=scratch_usage)
		working = Texture(size, format="rgba32float", usage=scratch_usage, mip_level_count=levels)
		mode, factor = int(material["surface"][2]), float(material["base_color"][3])
		cutoff = float(material["surface"][3])/factor if factor > 0 else 0
		preserve = mode == 1 and 0 < cutoff <= 1
		params = shader.UniformBuffer("params")
		params.content["info"] = (0, mode, preserve, 0)
		params.content["cutoff"] = (cutoff, 0, 0, 0)
		params.upload()
		state = GpuBuffer(np.zeros(12, dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		cmd = RenderContext.commands(f"material_mips_{layer}")
		with cmd.compute_pass(label="decode_and_source_coverage") as compute:
			dispatch(compute, "decode", dict(params=params, source=upload, linear_destination=original),
				(source_width+7)//8, (source_height+7)//8)
			if preserve:
				dispatch(compute, "init_source", dict(source=original, coverage=state))
				dispatch(compute, "measure_source", dict(params=params, source=original, coverage=state),
					(source_width+7)//8, (source_height+7)//8)
		previous = original.view()
		for level in range(levels):
			mip_width, mip_height = max(1, width >> level), max(1, height >> level)
			current = working.view(base_mip_level=level, mip_level_count=1, dimension="2d")
			destination = encoded.view(base_mip_level=level, mip_level_count=1, dimension="2d-array")
			groups = ((mip_width+7)//8, (mip_height+7)//8)
			with cmd.compute_pass(label=f"generate_mip_{level}") as compute:
				dispatch(compute, "resample", dict(source=previous, linear_destination=current), *groups)
				if preserve:
					dispatch(compute, "begin_search", dict(source=current, coverage=state))
					# Bracket by doubling, then bisect. Converged searches skip sample
					# work; fixed bounds cover RGBA8 source alpha through tiny mips.
					for _ in range(56):
						dispatch(compute, "measure_candidate", dict(params=params, source=current, coverage=state), *groups)
						dispatch(compute, "update_search", dict(params=params, coverage=state))
				dispatch(compute, "encode", dict(params=params, source=current,
					encoded_destination=destination, coverage=state), *groups)
			cmd.handle.copy_texture_to_texture(
				{"texture": encoded.handle, "mip_level": level},
				{"texture": result.handle, "mip_level": level, "origin": (0, 0, layer)},
				(mip_width, mip_height, 1))
			previous = current
		RenderContext.submit(cmd.finish())
	return result


def load_scene(path=PATH):
	"""Bake static node hierarchies and expand triangles for portable barycentrics.

	Each node primitive remains an independent DrawBatches mesh. Baking also makes
	the culling bounds exact for arbitrary glTF node matrices, including shear.
	"""
	path = Path(path)
	gltf = GLTF2().load(str(path))
	materials = list(gltf.materials) + [None]  # glTF's default material
	records = np.zeros(len(materials), dtype=MATERIAL_DTYPE)
	images = []
	for i, material in enumerate(materials):
		pbr = material.pbrMetallicRoughness if material else None
		mode = (material.alphaMode or "OPAQUE") if material else "OPAQUE"
		records[i]["base_color"] = (pbr.baseColorFactor if pbr else None) or (1, 1, 1, 1)
		records[i]["emission"][:3] = (material.emissiveFactor if material else None) or (0, 0, 0)
		records[i]["surface"] = (
			pbr.roughnessFactor if pbr and pbr.roughnessFactor is not None else 1,
			pbr.metallicFactor if pbr and pbr.metallicFactor is not None else 1,
			{"OPAQUE": 0, "MASK": 1, "BLEND": 2}[mode],
			material.alphaCutoff if material and material.alphaCutoff is not None else 0.5,
		)
		# RGB opacity is separate from color; scalar glTF alpha uses (1,1,1).
		records[i]["opacity_rgb"] = (1, 1, 1, 1)
		records[i]["flags"] = (i, bool(material and material.doubleSided), 0, 0)
		image = Image.new("RGBA", (1, 1), "white")
		if pbr and pbr.baseColorTexture:
			texture_info = pbr.baseColorTexture
			if texture_info.texCoord != 0 or texture_info.extensions:
				raise ValueError("This example supports base-color textures on TEXCOORD_0 without transforms")
			texture = gltf.textures[texture_info.index]
			image = Image.open(io.BytesIO(get_image_bytes(gltf, gltf.images[texture.source], path.parent))).convert("RGBA")
		# Preserve source coverage before the texture-array resize and mip build.
		images.append(np.asarray(image))

	primitives, shading_vertices = [], []
	vertex_offset = 0
	def visit(node_id, parent):
		nonlocal vertex_offset
		node = gltf.nodes[node_id]
		if node.skin is not None:
			raise ValueError("This static example does not support skinned glTF nodes")
		matrix = parent @ node_matrix(node)
		if node.mesh is not None:
			for primitive in gltf.meshes[node.mesh].primitives:
				if primitive.mode not in (None, 4):
					raise ValueError("This example requires triangle-list glTF primitives")
				positions = read_accessor(gltf, primitive.attributes.POSITION, path.parent)
				indices = (read_accessor(gltf, primitive.indices, path.parent).reshape(-1)
					if primitive.indices is not None else np.arange(len(positions)))
				if len(indices) % 3:
					raise ValueError("Triangle index count must be divisible by three")
				indices = indices.reshape(-1, 3).copy()
				# Baking a reflection requires reversing winding as well as normals.
				if np.linalg.det(matrix[:3, :3]) < 0:
					indices = indices[:, [0, 2, 1]]
				indices = indices.reshape(-1)
				if not len(indices):
					continue
				vertices = np.zeros(len(indices), dtype=VERTEX_DTYPE)
				vertices["position"] = (positions @ matrix[:3, :3].T + matrix[:3, 3])[indices]
				if primitive.attributes.NORMAL is not None:
					normals = read_accessor(gltf, primitive.attributes.NORMAL, path.parent)
					vertices["normal"] = (normals @ np.linalg.inv(matrix[:3, :3]))[indices]
				else:
					tri = vertices["position"].reshape(-1, 3, 3)
					vertices["normal"] = np.repeat(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), 3, axis=0)
				vertices["normal"] /= np.maximum(np.linalg.norm(vertices["normal"], axis=1, keepdims=True), 1e-8)
				if primitive.attributes.TEXCOORD_0 is not None:
					vertices["uv"] = read_accessor(gltf, primitive.attributes.TEXCOORD_0, path.parent)[indices]
				vertices["barycentric"] = np.tile(((1, 0), (0, 1), (0, 0)), (len(indices)//3, 1))
				vertices["triangle"] = np.repeat(np.arange(len(indices)//3), 3)
				material_id = primitive.material if primitive.material is not None else len(materials)-1
				primitives.append(Primitive(vertices, material_id, vertex_offset//3, len(primitives)))
				stored = np.zeros(len(indices), dtype=SHADING_VERTEX_DTYPE)
				stored["position"][:, :3] = vertices["position"]
				stored["normal"][:, :3] = vertices["normal"]
				stored["uv"][:, :2] = vertices["uv"]
				shading_vertices.append(stored)
				vertex_offset += len(indices)
		for child in node.children or ():
			visit(child, matrix)
	if gltf.scenes:
		roots = gltf.scenes[gltf.scene if gltf.scene is not None else 0].nodes
	else:
		children = {child for node in gltf.nodes for child in node.children or ()}
		roots = [i for i in range(len(gltf.nodes)) if i not in children]
	for root in roots:
		visit(root, np.eye(4))
	if not primitives:
		raise ValueError("The glTF scene has no triangles")
	return primitives, records, images, np.concatenate(shading_vertices)


def load_blue_noise(path=BLUE_NOISE_PATH):
	"""Load the bundled tileable 16-bit linear noise without sRGB conversion.

	Use quantization-bin midpoints so thresholds stay strictly inside (0, 1).
	The original texture and CC0 dedication are in resources/blue_noise/.
	"""
	with Image.open(path) as image:
		samples = np.asarray(image)
		if samples.ndim != 2 or samples.dtype != np.uint16:
			raise ValueError("Blue noise must be a single-channel 16-bit PNG")
		return np.ascontiguousarray((samples.astype(np.float32) + 0.5) / 65536)


class GeometryBatches:
	def __init__(self, primitives, materials, shader, uniforms, shared, transparent):
		self.world = ECS()
		transforms, refs = self.world.register(Transform, MeshRef)
		self.data = DrawBatches(mesh_instance_dtype)
		# This scene copies frustum results into the second pass's destinations.
		# Give only the two copy sources the additional usage before registration.
		for name in ("prepass_visible_instances", "prepass_draw_cmd"):
			buffer = self.data.buffers[name]
			self.data.buffers[name] = GpuBuffer(buffer.content,
				buffer.usage | BufferUsage.COPY_SRC, upload=False, label=name)
		frame_bindings = shader.bind_group(0, frame=uniforms)
		if transparent:
			multiply = {"color": {"src_factor": "zero", "dst_factor": "src", "operation": "add"},
				"alpha": {"src_factor": "zero", "dst_factor": "src-alpha", "operation": "add"}}
			first = RenderPipeline(shader, vertex_entry="vertex", fragment_entry="transmittance",
				cull_mode="none", depth_write=False, blend=multiply, label="transmittance")
			last = RenderPipeline(shader, vertex_entry="vertex", fragment_entry="select_winner",
				cull_mode="none", label="select_winner")
		else:
			first = last = RenderPipeline(shader, vertex_entry="vertex", fragment_entry="opaque",
				cull_mode="none", label="opaque")
		self.first_pass = ShaderPass(first)
		self.last_pass = ShaderPass(last)
		self.data.register_shader(0, RenderShader(self.last_pass, self.first_pass))
		for primitive in primitives:
			if (materials[primitive.material_id]["surface"][2] == 2) != transparent:
				continue
			mesh_id = primitive.stable_id
			mesh = Mesh(primitive.vertices, np.arange(len(primitive.vertices), dtype=np.uint32))
			self.data.register_mesh(mesh_id, mesh)
			self.data.register_lod_group(mesh_id, (mesh_id,))
			[entity] = self.world.create()
			self.world.add(entity, Transform(Vec3(0, 0, 0), Vec3(1, 1, 1), Vec4(0, 0, 0, 1)),
				MeshRef(0, mesh_id, primitive.triangle_offset, primitive.material_id, primitive.stable_id))
		self.data.sync_batches(self.world.query(Transform, MeshRef).entities(), transforms, refs)
		self.data.write_instance_fields(iPosition=(0, 0, 0), iTint=0,
			iRotation=pack_quaternion(np.array([0, 0, 0, 1], dtype=np.float32)),
			iScale=pack_scale(np.ones(3, dtype=np.float32)))
		# Metadata follows compact DrawBatches instance order, never entity order.
		metadata = np.zeros((max(1, len(self.data.entities)), 4), dtype=np.uint32)
		for column, name in enumerate(("triangle_offset", "material_id", "stable_id")):
			metadata[:len(self.data.entities), column] = self.data.mesh_field(name)
		self.metadata = GpuBuffer(metadata, BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self.first_pass.bindings = (frame_bindings, shader.bind_group(2, **shared, metadata=self.metadata))
		self.last_pass.bindings = self.first_pass.bindings
		self.guide_pass = None
		if transparent:
			self.guide_pass = ShaderPass(RenderPipeline(shader, vertex_entry="vertex",
				fragment_entry="transparent_depth", cull_mode="none", label="transparent_depth"), self.first_pass.bindings)

	def prepare(self, commands, camera, view_projection, dummy_hzb):
		self.data.refresh_bindings(dummy_hzb)
		self.data.update_cull_camera(camera.position, view_projection)
		self.data.reset_draw_counts(commands)
		self.data.cull_instances(commands, "frustum")
		# All batches participate in the first pass. Reuse its visibility and
		# indirect commands for selection; transparent culling never uses itself
		# as an occluder. No Hi-Z pass or sorting is needed for this small scene.
		for source, destination in (("prepass_visible_instances", "main_visible_instances"),
			("prepass_draw_cmd", "main_draw_cmd")):
			commands.copy_buffer(self.data.buffers[source], self.data.buffers[destination])


class TransparencyRenderer:
	def __init__(self, path=PATH, *, bilateral_filter=None, blue_noise_path=BLUE_NOISE_PATH):
		primitives, materials, images, vertices = load_scene(path)
		# Ignore a large flat ground plane when framing and scaling filter distance.
		objects = [p.vertices["position"] for p in primitives if np.ptp(p.vertices["position"][:, 1]) > 0.01]
		positions = np.concatenate(objects or [p.vertices["position"] for p in primitives])
		self.center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
		self.radius = max(float(np.linalg.norm(np.ptp(positions, axis=0))) * 0.8, 1.0)
		self.shader = Shader(filepath=str(SHADERS / "stochastic_geometry.shader"))
		self.shading_shader = Shader(filepath=str(SHADERS / "stochastic_shade.shader"))
		self.compose_shader = Shader(filepath=str(SHADERS / "stochastic_compose.shader"),
			params={"srgb_surface": getattr(RenderContext, "presentation_format", "").endswith("-srgb")})
		self.uniforms = self.shader.UniformBuffer("frame")
		self.bilateral_uniforms = self.compose_shader.UniformBuffer("bilateral")
		self.debug_view = "final"
		self.set_bilateral_filter(bilateral_filter if bilateral_filter is not None else
			BilateralFilter(filter_near=self.radius, filter_far=self.radius*4))
		self.shade = ComputePipeline(self.shading_shader, entry="shade_winner")
		self.compose = RenderPipeline(self.compose_shader, vertex_entry="fullscreen",
			fragment_entry="compose", depth_test=None, depth_write=False, cull_mode="none")
		material_buffer = GpuBuffer(materials, BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self.vertices = GpuBuffer(vertices, BufferUsage.STORAGE | BufferUsage.COPY_DST)
		textures = build_material_texture_array(images, materials, (TEXTURE_SIZE, TEXTURE_SIZE))
		noise = load_blue_noise(blue_noise_path)
		noise_size = (noise.shape[1], noise.shape[0])
		self.noise = Texture(noise_size, format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": self.noise.handle}, noise,
			{"bytes_per_row": noise.shape[1]*4}, (*noise_size, 1))
		self.shared = dict(materials=material_buffer, base_textures=textures.view(dimension="2d-array"),
			material_sampler=Sampler(address_mode_u="repeat", address_mode_v="repeat",
				mag_filter="linear", min_filter="linear", mipmap_filter="linear"))
		self.opaque = GeometryBatches(primitives, materials, self.shader, self.uniforms, self.shared, False)
		self.transparent = GeometryBatches(primitives, materials, self.shader, self.uniforms, self.shared, True)
		self.shade_resources = self.shading_shader.bind_group(2, **self.shared,
			metadata=self.transparent.metadata, vertices=self.vertices)
		self.shade_frame = self.shading_shader.bind_group(0, frame=self.uniforms)
		self.size = None
		self.camera = Camera(tuple(self.center + (self.radius, self.radius*0.65, self.radius)),
			tuple(self.center), (0, 1, 0), 55, near=max(0.01, self.radius/1000), far=self.radius*100)

	def set_bilateral_filter(self, options):
		"""Change reconstruction settings without recreating pipelines or targets."""
		self.bilateral_uniforms.content["near_sigmas"] = (
			options.spatial_sigma, options.depth_sigma, options.transmittance_sigma, options.min_depth_sigma,
		)
		self.bilateral_uniforms.content["far_sigmas"] = (
			options.far_spatial_sigma, options.far_depth_sigma, options.far_transmittance_sigma, options.far_transmittance_weight,
		)
		self.bilateral_uniforms.content["distance_range"] = (options.filter_near, options.filter_far, 0, 0)
		self.bilateral_uniforms.content["kernel"] = (options.kernel_size//2, DEBUG_VIEWS.index(self.debug_view), options.adaptive, 0)
		self.bilateral_uniforms.upload()
		self.bilateral_filter = options

	def set_debug_view(self, view):
		"""Inspect scalar transmittance, raw winners, or reconstructed contribution."""
		if view not in DEBUG_VIEWS:
			raise ValueError(f"Debug view must be one of {DEBUG_VIEWS}")
		self.debug_view = view
		self.set_bilateral_filter(self.bilateral_filter)

	def resize(self, size):
		if size == self.size:
			return
		attachment = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING
		self.main_color = Texture(size, format="rgba16float", usage=attachment, label="opaque_hdr")
		self.depth = Texture(size, format="depth32float", usage=attachment, label="opaque_depth")
		self.transparent_depth = Texture(size, format="depth32float", usage=attachment, label="transparent_guide_depth")
		self.transmittance = Texture(size, format="rgba16float", usage=attachment, label="transmittance")
		self.selection_depth = Texture(size, format="depth32float", usage=attachment, label="selection_depth")
		self.winner = Texture(size, format="rgba32uint", usage=attachment, label="winner_payload")
		self.winner_color = Texture(size, format="rgba16float",
			usage=TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING, label="winner_hdr")
		self.transparent.last_pass.bindings = self.transparent.first_pass.bindings + (
			self.shader.bind_group(3, opaque_depth=self.depth, blue_noise=self.noise),)
		self.transparent.guide_pass.bindings = self.transparent.first_pass.bindings + (
			self.shader.bind_group(3, opaque_depth=self.depth),)
		self.shade_targets = self.shading_shader.bind_group(1, winners=self.winner, winner_color=self.winner_color)
		self.compose_resources = self.compose_shader.bind_group(0, main_color=self.main_color,
			transmittance=self.transmittance, winner_color=self.winner_color, opaque_depth=self.depth,
			transparent_depth=self.transparent_depth, frame=self.uniforms, bilateral=self.bilateral_uniforms)
		# Only frustum culling is dispatched; refresh_bindings still needs a valid
		# float texture for its unused Hi-Z binding.
		self.dummy_hzb = Texture((1, 1), format="r32float", usage=TextureUsage.TEXTURE_BINDING).view()
		self.size = size

	def render(self):
		width, height = RenderContext.windowDimensions
		if width <= 0 or height <= 0:
			return
		self.resize((width, height))
		camera = self.camera
		view, projection = camera.view(), camera.projection(width/height)
		for name, value in dict(view=view, proj=projection, light_dir=(0.4, 0.8, 0.3, 0),
			camera_position=(*camera.position, 1), clip=(camera.near, camera.far, 0, 0)).items():
			self.uniforms.content[name] = value
		self.uniforms.upload()
		commands = RenderContext.commands("stochastic_transparency")
		for geometry in (self.opaque, self.transparent):
			geometry.prepare(commands, camera, view @ projection, self.dummy_hzb)
		with commands.render_pass(color=self.main_color.color_attachment(clear=(0.035, 0.05, 0.08, 1)),
			depth=self.depth.depth_attachment(clear=1), label="1_opaque") as render:
			self.opaque.data.draw(render, "prepass")
		# This deterministic guide includes low-alpha fragments and pixels with no
		# stochastic winner. It never writes opaque depth or selection depth.
		with commands.render_pass(depth=self.transparent_depth.depth_attachment(clear=1),
			label="2a_transparent_depth_guide") as render:
			self.transparent.data.draw(render, "main", pass_override=self.transparent.guide_pass)
		with commands.render_pass(color=self.transmittance.color_attachment(clear=(1, 1, 1, 1)),
			depth=self.depth.depth_attachment(), label="2_transmittance") as render:
			self.transparent.data.draw(render, "prepass")
		with commands.render_pass(color=self.winner.color_attachment(clear=(0, 0, 0, 0)),
			depth=self.selection_depth.depth_attachment(clear=1), label="3_select_winner") as render:
			self.transparent.data.draw(render, "main")
		with commands.compute_pass(label="4_shade_winner") as compute:
			compute.set_pipeline(self.shade)
			for bindings in (self.shade_frame, self.shade_targets, self.shade_resources):
				compute.set_bind_group(bindings)
			compute.dispatch((width+7)//8, (height+7)//8)
		with commands.render_pass(color=RenderContext.screen(), label="5_reconstruct_compose") as render:
			render.set_pipeline(self.compose)
			render.set_bind_groups(self.compose_resources)
			render.draw(3)
		RenderContext.submit(commands.finish())


# name, displayed label, increment, minimum, maximum
SETTINGS = (
	("adaptive", "Distance adaptation", None, None, None),
	("kernel_size", "Kernel size", None, None, None),
	("filter_near", "Near distance", 1.0, 0.0, None),
	("filter_far", "Far distance", 1.0, 0.0001, None),
	("spatial_sigma", "Near spatial sigma", 0.1, 0.01, None),
	("far_spatial_sigma", "Far spatial sigma", 0.1, 0.01, None),
	("depth_sigma", "Near relative depth sigma", 0.005, 0.0001, None),
	("far_depth_sigma", "Far relative depth sigma", 0.005, 0.0001, None),
	("transmittance_sigma", "Near transmittance sigma", 0.05, 0.001, None),
	("far_transmittance_sigma", "Far transmittance sigma", 0.05, 0.001, None),
	("min_depth_sigma", "Minimum depth tolerance", 0.001, 0.000001, None),
	("far_transmittance_weight", "Far transmittance weight", 0.05, 0.0, 1.0),
)


class FilterSettings:
	"""ImGui filter controls, keyboard shortcuts, and validated JSON profiles."""

	def __init__(self, renderer, profile_path=None):
		self.renderer = renderer
		self.defaults = renderer.bilateral_filter
		self.profile_path = Path(profile_path) if profile_path else Path(__file__).with_name("transparency_filter.json")
		self.selected = 0
		self.visible = True
		self.message = ""

	def move(self, direction):
		self.selected = (self.selected+direction) % len(SETTINGS)
		self.message = ""

	def adjust(self, direction, fine=False):
		name, _, step, _, _ = SETTINGS[self.selected]
		options = self.renderer.bilateral_filter
		current = getattr(options, name)
		if name == "adaptive":
			value = direction > 0
		elif name == "kernel_size":
			value = 5 if direction > 0 else 3
		else:
			value = round(current + direction*step*(0.1 if fine else 1), 9)
		self.set_value(name, value)

	def reset(self):
		self.renderer.set_bilateral_filter(self.defaults)
		self.message = "Restored startup settings"

	def save(self, path=None):
		path = Path(path) if path else self.profile_path
		path.write_text(json.dumps(asdict(self.renderer.bilateral_filter), indent=2)+"\n", encoding="utf-8")
		self.message = f"Saved {path.name}"

	def load(self, path=None):
		path = Path(path) if path else self.profile_path
		options = type(self.defaults)(**json.loads(path.read_text(encoding="utf-8")))
		self.renderer.set_bilateral_filter(options)
		self.message = f"Loaded {path.name}"

	def handle_key(self, key, scancode, action, mods):
		if action not in (glfw.PRESS, glfw.REPEAT):
			return False
		if key == glfw.KEY_F:
			if action == glfw.PRESS:
				self.visible = not self.visible
			return True
		if key in (glfw.KEY_F5, glfw.KEY_F9):
			if action == glfw.PRESS:
				try:
					self.save() if key == glfw.KEY_F5 else self.load()
				except (OSError, TypeError, ValueError) as error:
					self.message = str(error)
			return True
		if not self.visible:
			return False
		if key in (glfw.KEY_UP, glfw.KEY_DOWN):
			self.move(-1 if key == glfw.KEY_UP else 1)
		elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
			self.adjust(-1 if key == glfw.KEY_LEFT else 1, bool(mods & glfw.MOD_SHIFT))
		elif key == glfw.KEY_HOME:
			self.reset()
		else:
			return False
		return True

	def set_value(self, name, value):
		"""Apply the same valid ranges to mouse widgets and keyboard edits."""
		options = self.renderer.bilateral_filter
		_, _, _, minimum, maximum = next(row for row in SETTINGS if row[0] == name)
		if name not in ("adaptive", "kernel_size"):
			if not np.isfinite(value):
				return
			value = max(value, minimum)
			if maximum is not None:
				value = min(value, maximum)
			if name == "filter_near":
				value = min(value, options.filter_far-min(0.0001, options.filter_far*0.5))
			elif name == "filter_far":
				value = max(value, options.filter_near+0.0001)
		self.renderer.set_bilateral_filter(replace(options, **{name: value}))
		self.message = ""

	def draw(self):
		from imgui_bundle import imgui

		if not self.visible:
			return
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((460, 560), imgui.Cond_.first_use_ever)
		expanded, self.visible = imgui.begin("Transparency settings", self.visible)
		try:
			if not expanded:
				return
			changed, index = imgui.combo("Diagnostic view", DEBUG_VIEWS.index(self.renderer.debug_view), DEBUG_VIEWS)
			if changed:
				self.renderer.set_debug_view(DEBUG_VIEWS[index])
			imgui.separator()
			for index, (name, label, step, minimum, maximum) in enumerate(SETTINGS):
				value = getattr(self.renderer.bilateral_filter, name)
				label = ("> " if index == self.selected else "") + label + "###" + name
				if name == "adaptive":
					changed, value = imgui.checkbox(label, value)
				elif name == "kernel_size":
					changed, selection = imgui.combo(label, 0 if value == 3 else 1, ("3 x 3", "5 x 5"))
					value = (3, 5)[selection]
				else:
					changed, value = imgui.drag_float(label, value, step, minimum, maximum or 0.0, "%.6g")
				if changed:
					self.selected = index
					self.set_value(name, value)
			imgui.separator()
			if imgui.button("Reset (Home)"):
				self.reset()
			for label, action in (("Save (F5)", self.save), ("Load (F9)", self.load)):
				imgui.same_line()
				if imgui.button(label):
					try:
						action()
					except (OSError, TypeError, ValueError) as error:
						self.message = str(error)
			imgui.text_wrapped("Drag values or Ctrl+click to type. Shift gives fine keyboard steps. F hides/shows this window.")
			if self.message:
				imgui.text_wrapped(self.message)
		finally:
			imgui.end()


def run_scene():
	from Imgui import Imgui

	RenderContext.init_window(1200, 900, "Stochastic transparency | F: settings | D: debug | Space: orbit",
		required_gpu_features=["indirect-first-instance"])
	renderer = TransparencyRenderer()
	settings = FilterSettings(renderer)
	angle, orbit = 0.65, False
	distance = renderer.radius * 1.4
	def scroll(x, y):
		nonlocal distance
		distance = np.clip(distance * np.exp(-0.1*y), renderer.radius*0.15, renderer.radius*8)
		return True
	def character(code):
		nonlocal orbit
		if chr(code) == " ":
			orbit = not orbit
		elif chr(code).lower() == "r":
			renderer.transparent.data.draw_batches.reverse()
		elif chr(code) in ("3", "5"):
			size = int(chr(code))
			renderer.set_bilateral_filter(replace(renderer.bilateral_filter, kernel_size=size))
			print(f"Bilateral filter: {size}x{size}")
		elif chr(code).lower() == "d":
			index = (DEBUG_VIEWS.index(renderer.debug_view)+1) % len(DEBUG_VIEWS)
			renderer.set_debug_view(DEBUG_VIEWS[index])
			print(f"View: {renderer.debug_view}")
		elif chr(code).lower() == "a":
			options = renderer.bilateral_filter
			renderer.set_bilateral_filter(replace(options, adaptive=not options.adaptive))
			print(f"Distance-adaptive filter: {renderer.bilateral_filter.adaptive}")
		return True
	RenderContext.event_handlers["mouse_scroll"].append(scroll)
	RenderContext.event_handlers["char"].append(character)
	RenderContext.event_handlers["key"].append(settings.handle_key)
	try:
		with Imgui() as gui:
			while RenderContext.window_loop():
				if orbit:
					angle += RenderContext.frame_time * 0.2
				renderer.camera.position = Vec3(renderer.center + distance * np.array([cos(angle), 0.5, sin(angle)]))
				renderer.render()
				gui.render(settings.draw)
	finally:
		RenderContext.event_handlers["mouse_scroll"].remove(scroll)
		RenderContext.event_handlers["char"].remove(character)
		RenderContext.event_handlers["key"].remove(settings.handle_key)


if __name__ == "__main__":
	run_scene()
