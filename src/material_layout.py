"""Explicit layouts derived from the material shaders' reflected resource ABI."""
import re

from .RenderContext import BindGroupLayout, PipelineLayout, wgpu


def _key(value):
	if isinstance(value, dict): return tuple((name, _key(item)) for name, item in sorted(value.items()))
	if isinstance(value, (tuple, list)): return tuple(map(_key, value))
	return value


def _entry(shader, info, visibility, unfilterable=()):
	entry = dict(binding=info.binding, visibility=visibility)
	if info.storage == "uniform":
		entry["buffer"] = dict(type="uniform", min_binding_size=shader.info.uniforms[info.name].dtype.itemsize)
	elif info.storage == "storage":
		entry["buffer"] = dict(type="read-only-storage" if info.access == "read" else "storage")
	elif info.type_name == "sampler":
		entry["sampler"] = dict(type="filtering")
	elif info.type_name == "sampler_comparison":
		entry["sampler"] = dict(type="comparison")
	else:
		match = re.fullmatch(r"texture_(storage_)?(depth_)?(multisampled_)?(1d|2d_array|2d|3d|cube_array|cube)(?:<([^>]+)>)?", info.type_name)
		if match is None: raise ValueError(f"Unsupported material resource type: {info.type_name}")
		storage, depth, multisampled, dimension, arguments = match.groups()
		dimension = dimension.replace("_", "-")
		if storage:
			format, access = map(str.strip, arguments.split(","))
			entry["storage_texture"] = dict(format=format, access={"write": "write-only", "read": "read-only", "read_write": "read-write"}[access], view_dimension=dimension)
		else:
			sample_type = "depth" if depth else {"u32": "uint", "i32": "sint", "f32": "float"}[arguments]
			if sample_type == "float" and (multisampled or info.name in ("blue_noise", "noise_hierarchy") or info.name in unfilterable):
				sample_type = "unfilterable-float"
			entry["texture"] = dict(sample_type=sample_type, view_dimension=dimension, multisampled=bool(multisampled))
	return entry


def material_pipeline_layout(shader, entries, cache, unfilterable=()):
	"""Share stable groups; limit pass texture bindings to the chosen entry points."""
	active = shader.info.used_bindings(entries)
	stable = shader.info.used_bindings(tuple(shader.info.entry_points))
	bindings = {name: info for name, info in shader.info.bindings.items()
		if info.group == 0 or (name in stable if info.group in (1, 2) else name in active)}
	stages = wgpu.ShaderStage
	visibility = stages.COMPUTE if any(shader.info.entry_points[name].stage == "compute" for name in entries if name) else stages.VERTEX | stages.FRAGMENT
	groups = []
	for group in range(max((info.group for info in bindings.values()), default=-1)+1):
		descriptors = [_entry(shader, info, stages.VERTEX | stages.FRAGMENT | stages.COMPUTE if group == 0 else visibility, unfilterable)
			for info in sorted(bindings.values(), key=lambda info: info.binding) if info.group == group]
		key = _key(descriptors)
		if key not in cache: cache[key] = BindGroupLayout(descriptors)
		groups.append(cache[key])
	return PipelineLayout(groups)
