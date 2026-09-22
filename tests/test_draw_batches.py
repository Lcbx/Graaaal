"""CPU regression tests for DrawBatches
This checks CPU layout and requested uploads, not WGSL execution or actual GPU behavior.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


from ECS import Entity, EntityArray, EntityLike, FieldArray, as_entities
from RenderContext import higher_pow2, BindGroup


class Buffer:

	def __init__(self, content, usage=0, upload=True, label=None):
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

class Shader:

	def __init__(self, **kwargs):
		pass

	def UniformBuffer(self, name):
		return Buffer(np.zeros(1, dtype=[('workgroup_count', 'u4'), ('batch_count', 'u4'), ('command_count', 'u4')]))

	def bind_group(self, *args, **kwargs):
		return NS(resources=kwargs)

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
		return self.owners[rows].copy()

	def version(self, *args):
		return self.v

def check(d):
	n = len(d.entities)
	assert [b.offset for b in d.source_batches] == list(np.cumsum([0] + [b.count for b in d.source_batches[:-1]])) if d.source_batches else n == 0
	assert sum((b.count for b in d.source_batches)) == n
	for batch, params in zip(d.source_batches, d.buffers['batches'].gpu):
		assert params['instance_offset'] == batch.offset and params['instance_count'] == batch.count
	offset = 0
	for b, c in zip(d.draw_batches, d.buffers['main_draw_cmd'].gpu):
		assert b.offset == offset and c['first_instance'] == offset
		offset += b.count
	assert d.workgroup_count == sum(((b.count + 63) // 64 for b in d.source_batches))
	assert d.buffers['instances'].content.size >= n

@pytest.fixture
def m(monkeypatch):
	ecs = types.ModuleType("ECS")
	ecs.ComponentAccessor = object
	ecs.Entity = np.uint64
	ecs.EntityArray = EntityArray
	ecs.FieldArray = FieldArray
	ecs.EntityLike = EntityLike
	ecs.as_entities = as_entities
	rc = types.ModuleType("RenderContext")
	for name in ("Texture", "Mesh", "ComputePipeline", "RenderPipeline"):
		setattr(rc, name, object)
	rc.GpuBuffer, rc.Shader = Buffer, Shader
	rc.BufferUsage = NS(STORAGE=1, COPY_DST=2, INDIRECT=4)
	rc.TextureUsage = NS()
	rc.higher_pow2 = higher_pow2
	rc.BindGroup = BindGroup
	utils = types.ModuleType("Utils")
	utils.extract_frustum_planes = lambda x: x
	for name, module in (("ECS", ecs), ("RenderContext", rc), ("Utils", utils)):
		monkeypatch.setitem(sys.modules, name, module)
	root = Path(__file__).resolve().parent
	candidates = (root.parent / "src" / "DrawBatches.py", root / "src" / "DrawBatches.py", root / "DrawBatches.py")
	path = next((path for path in candidates if path.is_file()), None)
	if path is None:
		pytest.fail("Place this test in tests/ beside src/DrawBatches.py, or beside DrawBatches.py")
	spec = importlib.util.spec_from_file_location("_compact_draw_batches_under_test", path)
	module = importlib.util.module_from_spec(spec)
	monkeypatch.setitem(sys.modules, spec.name, module)
	spec.loader.exec_module(module)
	return module


@pytest.fixture
def d(m):
	d = m.DrawBatches(np.dtype([("a", "u4"), ("b", "u4")]))
	for i in (1, 2):
		mesh = NS(index_count=i * 3, index_range=(i * 5, 10), vertex_range=(i * 7, 10))
		d.register_mesh(i, mesh, bounds=([0, 0, 0], [1, 1, 1]))
	d.register_lod_group(10, (1, None), (20,))
	d.register_lod_group(20, (2,))
	pipeline = NS(shader=Shader())
	d.register_shader(0, m.RenderShader(m.ShaderPass(pipeline), m.ShaderPass(pipeline)))
	d.register_shader(1, m.RenderShader(m.ShaderPass(pipeline), None))
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
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items()}
	version = d.instance_order_version
	assert not d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	assert refs.reads == reads
	assert d.instance_order_version == version
	assert uploads == {name: len(buffer.uploads) for name, buffer in d.buffers.items()}


def test_registry_updates_preserve_instances_and_selection(synced, m):
	d, _, refs = synced
	d.write_instance_fields(a=d.mesh_field("value"), b=42)
	before = d.buffers["instances"].content.copy()
	reads = refs.reads
	version = d.instance_order_version
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items()}
	d.register_lod_group(10, (1, None), (30,))
	d.sync_resources()
	changed = {name for name, buffer in d.buffers.items() if len(buffer.uploads) != uploads[name]}
	assert changed == {"lod_distances"}
	row = d.group_rows[10]
	offset = int(d.buffers["mesh_metadata"].content[row]["lod_offset"])
	assert d.buffers["lod_distances"].gpu[offset + 1] == 30
	pipeline = NS(shader=Shader())
	d.register_shader(1, m.RenderShader(m.ShaderPass(pipeline), m.ShaderPass(pipeline)))
	d.sync_resources()
	check(d)
	assert np.all(d.buffers["batches"].gpu[:len(d.source_batches)]["prepass"] == 1)
	assert refs.reads == reads and d.instance_order_version == version
	np.testing.assert_array_equal(before, d.buffers["instances"].content)


def test_same_owners_with_reordered_components_invalidate_inputs(synced):
	d, transforms, refs = synced
	owners = d.entities.copy()
	version = d.instance_order_version
	uploads = {name: len(buffer.uploads) for name, buffer in d.buffers.items()}
	refs.fields["lod_id"][:2] = [10, 20]
	refs.fields["shader_id"][:2] = [0, 1]
	refs.v += 1
	assert d.sync_batches(np.array([1, 2, 3]), transforms, refs)
	assert d.instance_order_version == version + 1
	np.testing.assert_array_equal(d.entities, owners)
	np.testing.assert_array_equal(d.mesh_field("value"), [0, 2, 1, 3])
	assert uploads == {name: len(buffer.uploads) for name, buffer in d.buffers.items()}


def test_accessor_replacement_with_equal_versions_invalidates_inputs(synced):
	d, transforms, refs = synced
	replacement = Refs([1, 2, 3], [10, 10, 20], [0, 0, 1])
	assert replacement.v == refs.v
	assert d.sync_batches(np.array([1, 2, 3]), transforms, replacement)
	assert replacement.reads == 1
	assert len(d.entities) == 3
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


@pytest.mark.parametrize("dtype", [np.dtype(object), np.dtype("V0"), np.dtype("u1"), np.dtype(("f4", (3,)))])
def test_invalid_instance_layout(m, dtype):
	with pytest.raises(ValueError):
		m.DrawBatches(dtype)
