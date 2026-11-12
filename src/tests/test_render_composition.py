from pathlib import Path
import numpy as np
import pytest
from numpyecs import DrawBatches as db
from numpyecs import TransparencyPass as tp
from numpyecs import RenderContext as rc
from numpyecs.InstanceData import MESH_INSTANCE_DTYPE, pack_scale, pack_quaternion
from numpyecs.Math import Camera


def read(texture, dtype, channels):
    width, height, _ = texture.size
    raw = rc.RenderContext.device.queue.read_texture({"texture": texture.handle},
        {"bytes_per_row": width * np.dtype(dtype).itemsize * channels}, texture.size)
    return np.frombuffer(raw, dtype=dtype).reshape(height, width, channels)


def test_draw_mesh_uses_gpu_generated_instances_with_offset():
    rc.RenderContext.setup_graphics_backend(True)
    shader = rc.Shader(source='''
        storage<0, 0, read_write> positions: array<vec2f>;
        @compute @workgroup_size(1) fn generate() {
            positions[0] = vec2f(-0.5, 0.0);
            positions[1] = vec2f(0.5, 0.0);
            positions[2] = vec2f(0.0, 0.5);
        }
        @vertex fn vertex(@location(0) position: vec2f, @location(1) offset: vec2f) -> @builtin(position) vec4f {
            return vec4f(position + offset, 0.0, 1.0);
        }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
    ''')
    vertices = np.zeros(3, dtype=[("position", "f4", 2)])
    vertices["position"] = [[-0.2, -0.2], [0.2, -0.2], [0, 0.2]]
    mesh = rc.Mesh(vertices, np.arange(3, dtype=np.uint32))
    instances = rc.GpuBuffer(None, rc.BufferUsage.STORAGE | rc.BufferUsage.VERTEX,
        dtype=np.dtype([("offset", "f4", 2)]), count=3)
    output = rc.Texture((32, 32), format="rgba8unorm",
        usage=rc.TextureUsage.RENDER_ATTACHMENT | rc.TextureUsage.COPY_SRC)
    compute_pipeline = rc.ComputePipeline(shader)
    render_pipeline = rc.RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
    commands = rc.RenderContext.commands("generated_instances")
    with commands.compute_pass() as compute:
        compute.set_pipeline(compute_pipeline)
        compute.set_bind_group(shader.bind_group(0, positions=instances))
        compute.dispatch(1)
    with commands.render_pass(color=output.color_attachment(clear=(0, 0, 0, 1))) as render:
        render.set_pipeline(render_pipeline)
        render.draw_mesh(mesh, instances=instances, instance_offset=1, instance_count=1)
    rc.RenderContext.submit(commands.finish())
    pixels = read(output, np.uint8, 4)
    assert np.any(pixels[:, 16:, 0])
    assert not np.any(pixels[:, :16, 0])
    assert instances.content is None


def test_two_view_visibility_and_named_passes_share_instances(monkeypatch):
    rc.RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
    monkeypatch.setattr(db, "GpuBuffer", lambda content, usage, **kw:
        rc.GpuBuffer(content, usage | rc.BufferUsage.COPY_SRC, **kw))
    draws = db.DrawBatches(MESH_INSTANCE_DTYPE)
    shader = rc.Shader(source='''
        #include "draw_common.shaderlib"
        #include "utils.shaderlib"
        #include "instance_transform.shaderlib"
        storage<1, 0, read> instances: array<MeshInstance>;
        storage<1, 1, read> visible_instances: array<u32>;
        @vertex fn vertex(@location(0) position: vec3f, @builtin(instance_index) index: u32) -> @builtin(position) vec4f {
            return vec4f(instance_position(instances[visible_instances[index]], position), 1.0);
        }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
    ''', basedir=str(Path(db.__file__).with_name("shaders")))
    pipeline = rc.RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
    draws.register_shader(0, db.RenderShader({"color": db.ShaderPass(pipeline)}))
    vertices = np.zeros(3, dtype=[("position", "f4", 3)])
    vertices["position"] = [(-0.15, -0.15, 0), (0.15, -0.15, 0), (0, 0.15, 0)]
    mesh = rc.Mesh(vertices, np.arange(3, dtype=np.uint32))
    draws.register_mesh(1, mesh)
    draws.register_lod_group(10, (1,))
    assert draws.sync([10, 10], [0, 0], labels=["left", "right"])
    draws.write_instance_fields(iPosition=[[-0.5, 0, 0.5], [0.5, 0, 0.5]], iTint=0,
        iRotation=pack_quaternion([0, 0, 0, 1]), iScale=pack_scale([1, 1, 1]))
    first, second = draws.create_visibility(), draws.create_visibility()
    for attribute in ("commands", "visible_instances"):
        with pytest.raises(RuntimeError, match="Prepare visibility"):
            getattr(first, attribute)
    shifted = np.eye(4)
    shifted[3, 0] = 1.4
    commands = rc.RenderContext.commands("two_views")
    db.FrustumCuller().record(commands, first, camera_position=[0, 0, 0], view_projection=np.eye(4))
    db.FrustumCuller().record(commands, second, camera_position=[0, 0, 0], view_projection=shifted)
    output = rc.Texture((32, 32), format="rgba8unorm", usage=rc.TextureUsage.RENDER_ATTACHMENT | rc.TextureUsage.COPY_SRC)
    with commands.render_pass(color=output.color_attachment(clear=(0, 0, 0, 1))) as render:
        draws.draw(render, second, "color")
    rc.RenderContext.submit(commands.finish())
    for visibility, expected in ((first, 2), (second, 1)):
        raw = rc.RenderContext.device.queue.read_buffer(visibility.commands.handle)
        assert np.frombuffer(raw, dtype=db.INDIRECT_DTYPE, count=1)["instance_count"][0] == expected
    pixels = read(output, np.uint8, 4)
    assert np.count_nonzero(pixels[:, :16, 0]) > 0
    assert not np.any(pixels[:, 16:, 0])
    direct = draws.create_visibility().all_visible()
    assert direct.visible_instances is not first.visible_instances
    draws.sync([10], [0])
    with pytest.raises(RuntimeError, match="stale"): first.check(draws)
    for attribute in ("commands", "visible_instances"):
        with pytest.raises(RuntimeError, match="stale"):
            getattr(first, attribute)


def test_explicit_uniform_layout_bypasses_reflector_and_gpu_only_buffer():
    rc.RenderContext.setup_graphics_backend(True)
    source = '''
        struct Frame { matrix: mat3x3f, }
        uniform<0, 0> frame: Frame;
        storage<0, 1, read_write> result: array<f32>;
        @compute @workgroup_size(1) fn run() { result[0] = frame.matrix[1].y; }
    '''
    dtype = np.dtype({"names": ["column0", "column1", "column2"],
        "formats": [(np.float32, 3)] * 3, "offsets": [0, 16, 32], "itemsize": 48})
    interface = rc.ShaderInterface(entries={"run": "compute"}, bindings={
        "frame": rc.BindingInfo("frame", 0, 0, "uniform", None, "Frame"),
        "result": rc.BindingInfo("result", 0, 1, "storage", "read_write", "array<f32>")},
        uniforms={"frame": rc.UniformInfo("frame", 0, 0, "Frame", dtype)}, inputs={})
    shader = rc.Shader(source=source, interface=interface)
    uniforms = shader.uniform_buffer()
    uniforms.content["column1"] = [4, 5, 6]
    uniforms.upload()
    result = rc.GpuBuffer(None, rc.BufferUsage.STORAGE | rc.BufferUsage.COPY_SRC,
        dtype=np.float32, count=1)
    pipeline = rc.ComputePipeline(shader)
    commands = rc.RenderContext.commands()
    with commands.compute_pass() as compute:
        compute.set_pipeline(pipeline)
        compute.set_bind_group(shader.bind_group(0, frame=uniforms, result=result))
        compute.dispatch(1)
    rc.RenderContext.submit(commands.finish())
    assert np.frombuffer(rc.RenderContext.device.queue.read_buffer(result.handle), dtype=np.float32)[0] == 5


@pytest.mark.parametrize("resolve_target_kind", ["texture", "view", "attachment"])
def test_msaa_resolve_normalized_attributes_and_per_attachment_state(resolve_target_kind):
    rc.RenderContext.setup_graphics_backend(True)
    shader = rc.Shader(source='''
        <VertexOutput> @builtin(position) position: vec4f;
        <VertexOutput> @location(0) color: vec4f;
        <FragmentOutput> @location(0) first: vec4f;
        <FragmentOutput> @location(1) second: vec4f;
        @vertex fn vertex(@location(0) position: vec2f, @location(1) color: vec4f) -> VertexOutput {
            return VertexOutput(vec4f(position, 0.0, 1.0), color);
        }
        @fragment fn fragment(input: VertexOutput) -> FragmentOutput {
            return FragmentOutput(input.color, vec4f(0.0, 1.0, 0.0, 1.0));
        }
    ''')
    pipeline = rc.RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none", sample_count=4,
        vertex_formats={"color": "unorm8x4"},
        targets=[{"format": "rgba8unorm", "write_mask": 9}, {"write_mask": 15}])
    vertices = np.zeros(3, dtype=[("position", "f4", 2), ("color", "u1", 4)])
    vertices["position"] = [[-1, -1], [1, -1], [0, 1]]
    vertices["color"] = [255, 128, 64, 255]
    buffer = rc.GpuBuffer(vertices, rc.BufferUsage.VERTEX | rc.BufferUsage.COPY_DST)
    resolved = [rc.Texture((16, 16), format="rgba8unorm", usage=rc.TextureUsage.RENDER_ATTACHMENT | rc.TextureUsage.COPY_SRC) for _ in range(2)]
    multisampled = [rc.Texture((16, 16), format="rgba8unorm", usage=rc.TextureUsage.RENDER_ATTACHMENT, sample_count=4) for _ in range(2)]
    targets = [texture.view() if resolve_target_kind == "view" else
        texture.color_attachment() if resolve_target_kind == "attachment" else texture for texture in resolved]
    commands = rc.RenderContext.commands()
    with commands.render_pass(color=[a.color_attachment(clear=(0, 0, 0, 1), resolve_target=b) for a, b in zip(multisampled, targets)]) as render:
        render.set_pipeline(pipeline)
        render.set_vertex_buffer(0, buffer)
        render.draw(3)
    rc.RenderContext.submit(commands.finish())
    np.testing.assert_array_equal(read(resolved[0], np.uint8, 4)[8, 8], [255, 0, 0, 255])
    np.testing.assert_array_equal(read(resolved[1], np.uint8, 4)[8, 8], [0, 255, 0, 255])
    with pytest.raises(ValueError, match="required GPU features"):
        rc.RenderContext.require_features(["not-a-device-feature"])


def test_gpu_only_preserving_reserve_and_opt_in_upload_counters():
    rc.RenderContext.setup_graphics_backend(True)
    gpu = rc.GpuBuffer(None, rc.BufferUsage.COPY_SRC | rc.BufferUsage.COPY_DST,
        dtype=np.uint32, count=4)
    expected = np.arange(4, dtype=np.uint32)
    rc.RenderContext.device.queue.write_buffer(gpu.handle, 0, expected)
    assert gpu.reserve(40)
    assert gpu.count == 4 and gpu.binding_size == 16
    actual = rc.RenderContext.device.queue.read_buffer(gpu.handle, 0, 16)
    np.testing.assert_array_equal(np.frombuffer(actual, dtype=np.uint32), expected)
    rc.RenderContext.enable_statistics()
    try:
        buffer = rc.GpuBuffer(expected, rc.BufferUsage.COPY_DST)
        buffer.upload()
        buffer.upload()
        counters = rc.RenderContext.inspect_resources()["statistics"]
        assert counters["upload_calls"] == 2 and counters["upload_bytes"] == 32
        assert counters["created.GpuBuffer"] == 1
    finally:
        rc.RenderContext.enable_statistics(False)


def test_transparency_external_opaque_inputs_and_transformed_reconstruction(monkeypatch):
    rc.RenderContext.setup_graphics_backend(True)
    monkeypatch.setattr(tp, "Texture", lambda *args, usage, **kw:
        rc.Texture(*args, usage=usage | rc.TextureUsage.COPY_SRC, **kw))
    size = (32, 32)
    attachment = rc.TextureUsage.RENDER_ATTACHMENT | rc.TextureUsage.TEXTURE_BINDING
    opaque = rc.Texture(size, format="rgba16float", usage=attachment)
    depth = rc.Texture(size, format="depth32float", usage=attachment)
    texture_usage = rc.TextureUsage.TEXTURE_BINDING | rc.TextureUsage.COPY_DST
    white = rc.Texture((1, 1, 1), format="rgba8unorm-srgb", usage=texture_usage)
    noise = rc.Texture((1, 1), format="r32float", usage=texture_usage)
    queue = rc.RenderContext.device.queue
    queue.write_texture({"texture": white.handle}, np.full(4, 255, dtype=np.uint8), {"bytes_per_row": 4}, (1, 1, 1))
    queue.write_texture({"texture": noise.handle}, np.array([0.25], dtype=np.float32), {"bytes_per_row": 4}, (1, 1, 1))
    vertices = np.zeros(3, dtype=tp.VERTEX_DTYPE)
    vertices["position"] = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    normal = np.array([1, 0, 1]) / np.sqrt(2)
    normals = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    vertices["normal"] = normals
    vertices["barycentric"] = [[1, 0], [0, 1], [0, 0]]
    stored = np.zeros(3, dtype=tp.SHADING_VERTEX_DTYPE)
    stored["position"][:, :3], stored["normal"][:, :3] = vertices["position"], vertices["normal"]
    material = np.zeros(1, dtype=tp.MATERIAL_DTYPE)
    material["base_color"] = [1, 1, 1, 0.5]
    material["opacity_rgb"] = 1
    material["surface"] = [1, 1, 2, 0]
    material["flags"] = [0, 1, 0, 0]
    storage = rc.BufferUsage.STORAGE | rc.BufferUsage.COPY_DST
    material_source = '''fn shade(material: Material, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
        return vec3f(0.25 + 0.05 * position.x, 0.25 + 0.25 * normal.x, 0.25 + 0.25 * normal.z);
    }'''
    oit = tp.TransparencyPass(materials=rc.GpuBuffer(material, storage), base_textures=white.view(dimension="2d-array"),
        vertices=rc.GpuBuffer(stored, storage), blue_noise=noise, target_format="rgba8unorm",
        opaque_color=opaque, opaque_depth=depth, material_source=material_source, winner_count=1, tail_allocation="stochastic")
    instance = np.zeros(1, dtype=MESH_INSTANCE_DTYPE)
    angle = np.pi / 6
    scale, translation = np.array([1.2, 0.6, 1.8]), np.array([0.2, 0, 0.1])
    instance["iPosition"] = translation
    instance["iScale"] = pack_scale(scale)
    instance["iRotation"] = pack_quaternion([0, np.sin(angle / 2), 0, np.cos(angle / 2)])
    instances = rc.GpuBuffer(instance, storage)
    metadata = rc.GpuBuffer(np.array([[0, 0, 42, 0]], dtype=np.uint32), storage)
    geometry = oit.create_geometry_passes(metadata, transparent=True, instances=instances)
    opaque_geometry = oit.create_geometry_passes(metadata, transparent=False, instances=instances)
    binding = oit.shader.bind_group(1, instances=instances, visible_instances=rc.GpuBuffer(np.zeros(1, dtype=np.uint32), storage))
    vertex_buffer = rc.GpuBuffer(vertices, rc.BufferUsage.VERTEX | rc.BufferUsage.COPY_DST)
    oit.resize(size)
    assert oit.main_color is opaque and oit.depth is depth
    oit.update_camera(Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 70))
    commands = rc.RenderContext.commands()
    raster_color = rc.Texture(size, format="rgba16float", usage=attachment | rc.TextureUsage.COPY_SRC)
    raster_depth = rc.Texture(size, format="depth32float", usage=attachment)
    with commands.render_pass(color=raster_color.color_attachment(clear=(0, 0, 0, 1)), depth=raster_depth.depth_attachment(clear=1)) as render:
        spec = opaque_geometry.render_shader["main"]
        render.set_pipeline(spec.pipeline)
        render.set_bind_groups(*spec.bindings, binding)
        render.set_vertex_buffer(0, vertex_buffer)
        render.draw(3)
    # Opaque rendering belongs to another renderer in this workflow.
    with commands.render_pass(color=opaque.color_attachment(clear=(0, 0, 0.25, 1)), depth=depth.depth_attachment(clear=1)): pass
    for begin, spec in ((oit.guide_pass, geometry.guide), (oit.transmittance_pass, geometry.render_shader["transmittance"]), (oit.selection_pass, geometry.render_shader["main"])):
        with begin(commands) as render:
            render.set_pipeline(spec.pipeline)
            render.set_bind_groups(*spec.bindings, binding)
            render.set_vertex_buffer(0, vertex_buffer)
            render.draw(3)
    oit.shade_winner(commands, geometry)
    output = rc.Texture(size, format="rgba8unorm", usage=rc.TextureUsage.RENDER_ATTACHMENT)
    oit.compose(commands, output.color_attachment())
    rc.RenderContext.submit(commands.finish())
    payload, colors = read(oit.winner, np.uint32, 4), read(oit.winner_color, np.float16, 4)
    mask = payload[:, :, 0] > 0
    assert np.count_nonzero(mask) > 20
    xy = np.stack((payload[:, :, 2] & 65535, payload[:, :, 2] >> 16), axis=-1) / 65535
    bary = np.maximum(np.concatenate((xy, 1-xy.sum(axis=-1, keepdims=True)), axis=-1), 0)
    bary /= bary.sum(axis=-1, keepdims=True)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    position = (bary @ vertices["position"] * scale) @ rotation.T + translation
    expected_normal = ((bary @ normals) / scale) @ rotation.T
    expected_normal /= np.linalg.norm(expected_normal, axis=-1, keepdims=True)
    face = payload[:, :, 3].copy().view(np.float16).reshape(32, 32, 2)[:, :, 1]
    expected = np.stack((0.25 + 0.05 * position[:, :, 0],
        0.25 + 0.25 * expected_normal[:, :, 0] * np.sign(face),
        0.25 + 0.25 * expected_normal[:, :, 2] * np.sign(face)), axis=-1)
    np.testing.assert_allclose(colors[mask, :3], expected[mask], atol=0.001)
    np.testing.assert_allclose(colors[mask, :3], read(raster_color, np.float16, 4)[mask, :3], atol=0.001)
