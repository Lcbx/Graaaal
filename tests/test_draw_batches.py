"""CPU regression tests for DrawBatches
This checks CPU layout and requested uploads, not WGSL execution or actual GPU behavior.
"""

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


from numpyecs.ECS import ECS, Entity, component
from numpyecs.RenderContext import BindGroup


class Buffer:

	def __init__(self, content, usage=0, upload=True, label=None, *, dtype=None, count=0):
		content = np.zeros(count, dtype=dtype) if content is None else content
		self.content = content
		self.uploads = []
		self.gpu = np.zeros_like(content)
		if upload:
			self.upload()

	def resize(self, capacity):
		self.gpu = np.zeros(capacity, dtype=self.content.dtype)

	def upload(self):
		self.upload_range(0, len(self.content))

	def upload_range(self, offset, count):
		self.gpu[offset:offset + count] = self.content[offset:offset + count]
		self.uploads.append((offset, count))

	def write(self, content):
		self.content = content.copy()
		self.resize(len(content))
		self.upload()


class Shader:

	def __init__(self, **kwargs):
		pass

	def uniform_buffer(self, name):
		return Buffer(np.zeros(1, dtype=[('workgroup_count', 'u4'), ('batch_count', 'u4'), ('command_count', 'u4')]))

	def bind_group(self, *args, **kwargs):
		return NS(group=args[0], resources=kwargs)


class RenderPipeline:
	def __init__(self):
		self.shader = Shader()

class Selection:

	def __init__(self, parent, rows):
		self.parent, self.rows = (parent, rows)

	def get_rows(self):
		return self.rows

	def __getitem__(self, key):
		return self.parent.fields[key][self.rows]

class Refs:

	def __init__(self, owners, lod, shader):
		self.owners = np.array(owners, dtype=np.uint64)
		self.fields = {'lod_id': np.array(lod), 'shader_id': np.array(shader), 'value': np.arange(len(owners), dtype=np.uint32)}
		self.v = 0
		self.reads = 0

	def __getitem__(self, entities):
		self.reads += 1
		return Selection(self, np.concatenate([np.flatnonzero(self.owners == e) for e in entities]).astype(np.intp) if len(entities) else np.empty(0, dtype=np.intp))

	def owner_ids(self, rows):
		if isinstance(rows, Selection): rows = rows.get_rows()
		return self.owners[rows].copy()

	@property
	def layout_version(self):
		return self.v

	def version(self, *args):
		return self.v

def check(d):
	n = len(d.entities)
	assert [b.offset for b in d.source_batches] == list(np.cumsum([0] + [b.count for b in d.source_batches[:-1]])) if d.source_batches else n == 0
	assert sum((b.count for b in d.source_batches)) == n
	for batch, params in zip(d.source_batches, d.buffers['batches'].content):
		assert params['instance_offset'] == batch.offset and params['instance_count'] == batch.count
	offset = 0
	for b, c in zip(d.draw_batches, d.buffers['main_draw_cmd'].content):
		assert b.offset == offset and c['first_instance'] == offset
		offset += b.count
	assert d.workgroup_count == sum(((b.count + 63) // 64 for b in d.source_batches))
	assert d.buffers['instances'].content.size >= n

@pytest.fixture
def m(monkeypatch):
	# Load a separate package so mocked resources cannot leak into real GPU tests.
	prefix = "_numpyecs_drawing_under_test"
	package = types.ModuleType(prefix)
	package.__path__ = [str(Path(__file__).resolve().parents[1] / "src")]
	monkeypatch.setitem(sys.modules, prefix, package)
	rc = types.ModuleType(f"{prefix}.RenderContext")
	for name in ("Texture", "Mesh", "ComputePipeline"):
		setattr(rc, name, object)
	rc.RenderPipeline = RenderPipeline
	rc.GpuBuffer, rc.Shader = Buffer, Shader
	rc.BufferUsage = NS(STORAGE=1, COPY_DST=2, INDIRECT=4)
	rc.TextureUsage = NS()
	rc.BindGroup = BindGroup
	rc.RenderContext = NS(record_stat=Mock(), require_features=Mock())
	math = types.ModuleType(f"{prefix}.Math")
	math.extract_frustum_planes = lambda x: x
	monkeypatch.setitem(sys.modules, rc.__name__, rc)
	monkeypatch.setitem(sys.modules, math.__name__, math)
	loaded = {}
	for name in ("utils", "DrawList", "draw_layout", "draw_compiler", "draw_resources", "draw_culling", "HZB", "DrawBatches"):
		module_name = f"{prefix}.{name}"
		module = importlib.import_module(module_name)
		# Undo imported modules along with the resource stubs after the test.
		del sys.modules[module_name]
		monkeypatch.setitem(sys.modules, module_name, module)
		loaded[name] = module
	return NS(_DrawResources=loaded["draw_resources"]._DrawResources,
		_CullingState=loaded["draw_culling"]._CullingState,
		public=loaded["DrawBatches"], hzb=loaded["HZB"])


@pytest.fixture
def d(m):
	d = m._DrawResources(np.dtype([("a", "u4"), ("b", "u4")]))
	for i in (1, 2):
		mesh = NS(index_count=i * 3, index_range=(i * 5, 10), vertex_range=(i * 7, 10))
		d.register_mesh(i, mesh, bounds=([0, 0, 0], [1, 1, 1]))
	d.register_lod_group(10, (1, None), (20,))
	d.register_lod_group(20, (2,))
	d.register_shader(0)
	d.register_shader(1, prepass=False)
	return d


@pytest.fixture
def synced(d):
	transforms = NS(membership_version=0)
	refs = Refs([1, 1, 2, 3], [20, 10, 10, 20], [1, 0, 0, 1])
	assert d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	return d, transforms, refs


def test_multiple_components_and_compact_layout(synced):
	d, _, _ = synced
	check(d)
	np.testing.assert_array_equal(d.entities, [1, 2, 1, 3])
	np.testing.assert_array_equal(d.mesh_field("value"), [1, 2, 0, 3])
	d.write_instance_fields(a=d.mesh_field("value"), b=42)
	np.testing.assert_array_equal(d.buffers["instances"].gpu["a"][:4], [1, 2, 0, 3])
	assert d.buffers["instances"].uploads[-1] == (0, 4)


def test_field_writes_preserve_omitted_fields_and_aliases(synced):
	d, _, _ = synced
	d.write_instance_fields(a=d.mesh_field("value"), b=42)
	buffer = d.buffers["instances"]
	d.write_instance_fields(a=7)
	np.testing.assert_array_equal(buffer.gpu["b"][:4], [42] * 4)
	d.write_instance_fields(a=buffer.content["b"][:4], b=buffer.content["a"][:4])
	np.testing.assert_array_equal(buffer.gpu["a"][:4], [42] * 4)
	np.testing.assert_array_equal(buffer.gpu["b"][:4], [7] * 4)
	before = buffer.content.copy()
	uploads = len(buffer.uploads)
	with pytest.raises(ValueError):
		d.write_instance_fields(a=5, b=[1, 2, 3])
	np.testing.assert_array_equal(before, buffer.content)
	assert len(buffer.uploads) == uploads


def test_record_writes_validate_count_and_handle_aliases(synced):
	d, _, _ = synced
	values = np.zeros(4, dtype=d.instance_dtype)
	values["a"] = np.arange(4)
	d.write_instances(values)
	buffer = d.buffers["instances"]
	d.write_instances(buffer.content[:4][::-1])
	np.testing.assert_array_equal(buffer.gpu[:4], values[::-1])
	with pytest.raises(ValueError):
		d.write_instances(values[:3])


def test_unchanged_sync_does_not_read_or_upload(synced):
	d, transforms, refs = synced
	reads = refs.reads
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items() if hasattr(buffer, "uploads")}
	version = d.instance_order_version
	assert not d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	assert refs.reads == reads
	assert d.instance_order_version == version
	assert uploads == {name: len(buffer.uploads) for name, buffer in d.buffers.items() if hasattr(buffer, "uploads")}


def test_registry_updates_preserve_instances_and_selection(synced, m):
	d, _, refs = synced
	d.write_instance_fields(a=d.mesh_field("value"), b=42)
	before = d.buffers["instances"].content.copy()
	reads = refs.reads
	version = d.instance_order_version
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items() if hasattr(buffer, "uploads")}
	d.register_lod_group(10, (1, None), (30,))
	d.sync_resources()
	changed = {name for name, buffer in d.buffers.items() if hasattr(buffer, "uploads") and len(buffer.uploads) != uploads[name]}
	assert changed == {"lod_distances"}
	row = d.group_rows[10]
	offset = int(d.buffers["mesh_metadata"].content[row]["lod_offset"])
	assert d.buffers["lod_distances"].gpu[offset + 1] == 30
	d.register_shader(1)
	d.sync_resources()
	check(d)
	assert np.all(d.buffers["batches"].content[:len(d.source_batches)]["prepass"] == 1)
	assert refs.reads == reads and d.instance_order_version == version
	np.testing.assert_array_equal(before, d.buffers["instances"].content)


def test_same_owners_with_reordered_components_invalidate_inputs(synced):
	d, transforms, refs = synced
	owners = d.entities.copy()
	version = d.instance_order_version
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items() if hasattr(buffer, "uploads")}
	refs.fields["lod_id"][:2] = [10, 20]
	refs.fields["shader_id"][:2] = [0, 1]
	refs.v += 1
	assert d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	assert d.instance_order_version == version + 1
	np.testing.assert_array_equal(d.entities, owners)
	np.testing.assert_array_equal(d.mesh_field("value"), [0, 2, 1, 3])
	assert uploads == {name: len(buffer.uploads) for name, buffer in d.buffers.items() if hasattr(buffer, "uploads")}


def test_accessor_replacement_with_equal_versions_invalidates_inputs(synced):
	d, transforms, refs = synced
	replacement = Refs([1, 2, 3], [10, 10, 20], [0, 0, 1])
	assert replacement.v == refs.v
	assert d.sync_batches(np.array([1, 2, 3]), transforms, replacement)
	assert replacement.reads == 1
	assert len(d.entities) == 3
	check(d)


def test_real_component_selection_is_refreshed_after_layout_only_change(d):
	@component
	class Transform:
		x: float

	@component(multiples=True)
	class MeshRef:
		lod_id: int
		shader_id: int
		value: int

	world = ECS()
	transforms, refs = world.register(Transform, MeshRef, capacity=8)
	ids = world.create(3)
	world.add(ids, Transform(0))
	world.add([ids[0], ids[0], ids[1], ids[2]], MeshRef,
		(np.array([20, 10, 10, 20]), np.array([1, 0, 0, 1]), np.array([0, 1, 2, 3])))
	world.remove(ids[1], MeshRef)  # Keep a hole that can be compacted later.
	entities = world.where(Transform, MeshRef)
	assert d.sync_batches(entities, transforms, refs)
	check(d)
	expected = d.mesh_field("value").copy()
	owners = d.entities.copy()
	version = d.instance_order_version
	membership = refs.membership_version
	fields = refs.version("lod_id", "shader_id")

	refs._store._defragment(refs._store._capacity)
	assert refs.membership_version == membership
	assert refs.version("lod_id", "shader_id") == fields
	with pytest.raises(RuntimeError, match="stale"):
		d.mesh_field("value")
	assert d.sync_batches(entities, transforms, refs)
	assert d.instance_order_version == version + 1
	np.testing.assert_array_equal(d.entities, owners)
	np.testing.assert_array_equal(d.mesh_field("value"), expected)
	check(d)

	world.add(ids[0], MeshRef(10, 0, 9))
	assert d.sync_batches(entities, transforms, refs)
	assert sorted(d.mesh_field("value").tolist()) == [0, 1, 3, 9]
	assert not d.sync_batches(entities, transforms, refs)
	check(d)


def test_growth_shrink_empty_and_selection_changes(d):
	transforms = NS(membership_version=0)
	rng = np.random.default_rng(19)
	capacity = 1
	for n in (0, 1, 2, 63, 64, 65, 128, 129, 0, 300, 7):
		refs = Refs(np.arange(n) // 2, rng.choice([10, 20], n), rng.integers(0, 2, n))
		assert d.sync_batches(np.unique(refs.owners), transforms, refs)
		check(d)
		current = d.buffers["instances"].content.size
		assert current >= capacity and current & (current - 1) == 0
		capacity = current
		d.write_instance_fields(a=d.mesh_field("value"), b=9)
		np.testing.assert_array_equal(d.buffers["instances"].gpu["a"][:n], d.mesh_field("value"))
		if n > 2:
			assert d.sync_batches(np.unique(refs.owners)[::2], transforms, refs)
			check(d)


def test_workgroups_reused_within_dispatch_boundary(d):
	transforms = NS(membership_version=0)
	for n in (63, 64, 65):
		refs = Refs(np.arange(n), [10] * n, [0] * n)
		previous = len(d.buffers["workgroups"].uploads)
		assert d.sync_batches(refs.owners, transforms, refs)
		assert len(d.buffers["workgroups"].uploads) == previous + (n != 64)
		assert d.workgroup_count == (n + 63) // 64
		check(d)


def test_explicit_invalidation(synced):
	d, transforms, refs = synced
	version = d.instance_order_version
	d.invalidate_batches()
	assert d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	assert d.instance_order_version == version + 1


def test_mutating_selection_requires_regrouping(d):
	transforms = NS(membership_version=0)
	refs = Refs([1, 2, 3], [10, 10, 20], [0, 0, 1])
	selection = np.array([1, 2], dtype=Entity)
	assert d.sync_batches(selection, transforms, refs)
	selection[1] = 3
	assert d.sync_batches(selection, transforms, refs)
	np.testing.assert_array_equal(d.entities, [1, 3])


def test_register_shader_rejects_compute_pipeline(m):
	public = m.public
	draws = public.DrawBatches(np.dtype("u4"))
	shader = public.RenderShader({"main": public.ShaderPass(RenderPipeline())})
	draws.register_shader(0, shader)
	with pytest.raises(TypeError, match="RenderPipeline"):
		draws.register_shader(0, public.RenderShader({"main": public.ShaderPass(object())}))
	assert draws.shaders[0] is shader
	assert draws._resources._shader_prepass == {0: True}


def test_draw_replaces_material_bindings(synced, m):
	d, _, _ = synced
	public = m.public
	draws = public.DrawBatches(d.instance_dtype)
	draws._resources = d
	extra = NS(group=2)
	draws.register_shader(0, public.RenderShader({"main": public.ShaderPass(RenderPipeline(), (extra,))}))
	draws.register_shader(1, public.RenderShader({"main": public.ShaderPass(RenderPipeline())}))
	for info in d.meshes.values():
		info.mesh.vertex_buffer = object()
		info.mesh.index_buffer = object()
		info.mesh.index_format = "uint32"
	visibility = draws.create_visibility().all_visible()
	rp = Mock()
	draws.draw(rp, visibility)
	sets = [call.args for call in rp.set_bind_groups.call_args_list]
	assert [binding.group for binding in sets[0]] == [1, 2]
	assert [binding.group for binding in sets[-1]] == [1]
	assert rp.draw_indexed_indirect.call_count == 2
	registered = draws.shaders[0]["main"]
	override = public.ShaderPass(RenderPipeline(), (NS(group=3),))
	rp.reset_mock()
	draws.draw(rp, visibility, pass_override=override)
	assert draws.shaders[0]["main"] is registered
	assert all(call.args == (override.pipeline,) for call in rp.set_pipeline.call_args_list)
	assert all([binding.group for binding in call.args] == [1, 3] for call in rp.set_bind_groups.call_args_list)
	assert rp.draw_indexed_indirect.call_count == 2
	rp.reset_mock()
	draws.draw(rp, visibility, "prepass")
	rp.draw_indexed_indirect.assert_not_called()
	# An explicit override can add a pass without changing registered materials.
	draws.draw(rp, visibility, "guide", pass_override=override)
	assert rp.draw_indexed_indirect.call_count == 2


@pytest.mark.parametrize("position,matrix", [
	([1, 2], np.eye(4)), ([1, 2, np.nan], np.eye(4)),
	([1, 2, 3], np.eye(3)), ([1, 2, 3], np.full((4, 4), np.inf)),
])
def test_bad_camera_input_leaves_uniforms_unchanged(m, position, matrix):
	d = m._CullingState(Shader(), {})
	buffer = d.camera_params_buffer
	before = buffer.content.copy()
	d._camera_params_dirty = False
	with pytest.raises(ValueError):
		d.update_cull_camera(position, matrix)
	np.testing.assert_array_equal(buffer.content, before)
	assert not d._camera_params_dirty


@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1), (7, 5)])
def test_hzb_mips_dispatch_and_resize_rebinding(d, m, monkeypatch, size):
	class Texture:
		def __init__(self, size, **kwargs):
			self.size = size
			self.default_view = object()
		def view(self, **kwargs):
			return object() if kwargs else self.default_view

	monkeypatch.setattr(m.hzb, "Texture", Texture)
	monkeypatch.setattr(m.hzb, "TextureUsage", NS(RENDER_ATTACHMENT=1, TEXTURE_BINDING=2, STORAGE_BINDING=4))
	pipelines = (object(), object())
	hzb = m.hzb.HZB(shader=Shader(), pipelines=pipelines)
	assert hzb.resize(size)
	width, height = max(1, size[0] // 2), max(1, size[1] // 2)
	expected = []
	for _ in range(max(width, height).bit_length()):
		expected.append((width, height))
		width, height = max(1, width // 2), max(1, height // 2)
	assert [(w, h) for _, _, w, h in hzb.passes] == expected
	passes = []
	@contextmanager
	def compute_pass(**kwargs):
		cp = Mock()
		passes.append(cp)
		yield cp
	hzb.build(NS(compute_pass=compute_pass))
	for cp, (w, h) in zip(passes, expected):
		cp.dispatch.assert_called_once_with((w + 15) // 16, (h + 15) // 16)
	draws = m._CullingState(Shader(), d.buffers)
	draws.refresh_bindings(hzb.view)
	old_binding, old_view = draws.bindings["hiz"], hzb.view
	assert not hzb.resize(size)
	draws.refresh_bindings(hzb.view)
	assert draws.bindings["hiz"] is old_binding
	assert hzb.resize((size[0] + 2, size[1] + 2))
	draws.refresh_bindings(hzb.view)
	assert hzb.view is not old_view
	assert draws.bindings["hiz"].resources["hzb_texture"] is hzb.view


@pytest.mark.parametrize("dtype", [np.dtype(object), np.dtype("V0"), np.dtype("u1"), np.dtype(("f4", (3,)))])
def test_invalid_instance_layout(m, dtype):
	with pytest.raises(ValueError):
		m._DrawResources(dtype)
