"""Dictionary materials with shader-defined layouts and explicit instance overrides.

CPU preparation is independent of ECS and GPU setup. MaterialLibrary packs
headers/parameters and links functions declared by a rendering adapter interface.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import re

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .ShaderShorthand import ShaderShorthand, MISSING


SHADERS = Path(__file__).with_name("shaders")
KINDS = {"f32": 1, "vec2f": 2, "vec3f": 3, "vec4f": 4, "u32": 1, "bool": 1}


@dataclass(frozen=True)
class Parameter:
	kind: str
	default: object
	scope: str = "material"

	def __post_init__(self):
		if self.kind not in KINDS or self.scope not in ("material", "instance", "frame"):
			raise ValueError("Expected a supported parameter type and material/instance/frame scope")
		self.pack(self.default)

	def pack(self, value):
		words = np.zeros(4, dtype=np.uint32)
		if self.kind == "bool":
			if not isinstance(value, (bool, np.bool_)): raise ValueError("Expected boolean parameter")
			words[0] = value
		elif self.kind == "u32":
			if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 0xffffffff:
				raise ValueError("Expected uint32 parameter")
			words[0] = value
		else:
			values = np.asarray(value, dtype=np.float32)
			expected = () if self.kind == "f32" else (KINDS[self.kind],)
			if values.shape != expected or not np.isfinite(values).all():
				raise ValueError(f"Expected finite {self.kind} parameter")
			words[:values.size] = values.reshape(-1).view(np.uint32)
		return words


class ParameterLayout:
	"""Pack fields into vec4u lanes; CPU packing and WGSL use the same offsets."""
	def __init__(self, parameters, scope):
		fields = [(name, p) for name, p in parameters.items() if p.scope == scope]
		lanes, offsets = [], {}
		for name, p in sorted(fields, key=lambda item: -KINDS[item[1].kind]):
			width = KINDS[p.kind]
			slot = next((i for i, used in enumerate(lanes) if used+width <= 4), len(lanes))
			if slot == len(lanes): lanes.append(0)
			offsets[name] = (slot, lanes[slot])
			lanes[slot] += width
		self.offsets = MappingProxyType(offsets)
		self.slot_count = len(lanes)
		self.fields = tuple(fields)

	def pack(self, values):
		data = np.zeros((self.slot_count, 4), dtype=np.uint32)
		for name, p in self.fields:
			slot, lane = self.offsets[name]
			data[slot, lane:lane+KINDS[p.kind]] = p.pack(values[name])[:KINDS[p.kind]]
		return data


@dataclass(frozen=True)
class ShaderHook:
	name: str
	result: str
	arguments: str
	expression: str
	fallback: str
	optional_function: str | None = None


@dataclass(frozen=True)
class FrameParameter:
	name: str
	kind: str
	expression: str


@dataclass(frozen=True)
class MaterialInterface:
	"""Rendering adapter contract; expressions address an instance's material data."""
	address: str
	hooks: tuple[ShaderHook, ...]
	frame_parameters: tuple[FrameParameter, ...] = ()


@dataclass(eq=False)
class MaterialShader:
	interface: MaterialInterface
	template: str | Path | None = None
	default_material: dict = field(default_factory=dict)
	transparent: bool = False
	source: str | None = field(default=None, kw_only=True)
	filepath: str | Path | None = field(default=None, kw_only=True)
	_parameters: dict = field(init=False, repr=False)

	@property
	def parameters(self):
		return self._parameters

	def __post_init__(self):
		if type(self.transparent) is not bool: raise ValueError("transparent must be boolean")
		if sum(value is not None for value in (self.template, self.source, self.filepath)) != 1:
			raise ValueError("Supply exactly one of template, source or filepath")
		path = None if self.source is not None else Path(self.template or self.filepath).resolve()
		env = Environment(loader=FileSystemLoader(([str(path.parent)] if path else [])+[str(SHADERS)]),
			undefined=StrictUndefined, autoescape=False, line_statement_prefix="#", extensions=[ShaderShorthand])
		self._template = env.from_string(self.source) if self.source is not None else env.get_template(path.name)
		self._render_cache = {}
		_, declarations = self._render(0)
		self._parameters = MappingProxyType(declarations)
		frame_parameters = {p.name: p for p in self.interface.frame_parameters}
		for name, p in declarations.items():
			if p.scope == "frame" and (name not in frame_parameters or frame_parameters[name].kind != p.kind):
				raise ValueError(f"Unsupported frame parameter: {name} ({p.kind})")
		self.validate(self.default_material)

	def _render(self, prefix, hooks=None):
		hooks = frozenset(h.name for h in self.interface.hooks) if hooks is None else frozenset(hooks)
		key = (prefix, self.transparent, hooks)
		if key in self._render_cache: return self._render_cache[key]
		declarations = {}
		def parameter(name, kind, default, scope="material"):
			if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
				raise ValueError("Parameters require WGSL-compatible names")
			value = Parameter(kind, default, scope)
			if name in declarations and declarations[name].kind != kind:
				raise ValueError(f"Inherited parameter {name} requires {declarations[name].kind}")
			declarations[name] = value
			return ""
		def field(struct, name, kind, *, default=MISSING, **attributes):
			scopes = {"MaterialParameters": "material", "InstanceParameters": "instance", "FrameParameters": "frame"}
			if struct not in scopes or attributes: raise ValueError(f"Invalid material field declaration: {struct}.{name}")
			if default is MISSING: raise ValueError(f"Material parameter {name} requires a default")
			return parameter(name, kind, default, scopes[struct])
		code = self._template.render(PREFIX=prefix, TRANSPARENT=self.transparent, field=field, HOOKS=hooks)
		self._render_cache[key] = code, declarations
		return code, declarations

	def render(self, prefix, hooks=None):
		code, declarations = self._render(prefix, hooks)
		if declarations != self.parameters:
			raise ValueError("Shader declarations changed; create a new MaterialShader")
		return code

	def validate(self, values, *, overrides=False):
		for name, value in values.items():
			if name not in self.parameters: raise ValueError(f"Unknown material parameter: {name}")
			parameter = self.parameters[name]
			if parameter.scope == "frame": raise ValueError(f"{name} is a renderer-owned frame parameter")
			if overrides and parameter.scope != "instance":
				raise ValueError(f"{name} is shared; declare it with instance scope to override it")
			try: parameter.pack(value)
			except (ValueError, TypeError) as error:
				raise ValueError(f"Invalid material parameter {name}: {error}") from error

	def validate_resolved(self, values):
		"""Optional shader-specific semantic validation after dictionary resolution."""


@dataclass
class MaterialBinding:
	shader: MaterialShader
	material: dict | None = None
	overrides: dict = field(default_factory=dict)


class MaterialLibrary:
	"""Generic schema packing and shader linking. No surface or lighting policy.

	pack() produces headers, shared slots, instance slots, material IDs and instance
	offsets. A header contains shader ID and shared offset; no material parameters
	are interpreted here. Shader layouts are fixed; dictionary values remain live.
	"""
	def __init__(self, shaders):
		self.shaders = tuple(dict.fromkeys(shaders))
		if not self.shaders: raise ValueError("Supply at least one material shader")
		self.interface = self.shaders[0].interface
		if any(s.interface != self.interface for s in self.shaders):
			raise ValueError("Linked shaders must implement the same rendering interface")
		self.shader_ids = {shader: index for index, shader in enumerate(self.shaders)}
		self.parameter_layouts = {shader: {scope: ParameterLayout(shader.parameters, scope)
			for scope in ("material", "instance")} for shader in self.shaders}
		self._records = {}
		self._layouts = self._layout_keys()
		self.source = self._source()

	def _layout_keys(self):
		return tuple((s.transparent, str(s.template), s.source, str(s.filepath), s.interface, tuple((name, p.kind, p.scope)
			for name, p in s.parameters.items())) for s in self.shaders)

	def pack(self, bindings, *, dirty_materials=None):
		if self._layouts != self._layout_keys():
			raise ValueError("Shader layout, source or routing changed; create a new MaterialLibrary")
		headers, shared, instances, ids, offsets, seen = [], [], [], [], [], {}
		defaults, records = {}, {}
		dirty = None if dirty_materials is None else {id(value) for value in dirty_materials}
		for binding in bindings:
			shader = binding.shader
			if shader not in self.shader_ids: raise ValueError("Material shader is not registered in this library")
			if shader not in defaults:
				shader.validate(shader.default_material)
				defaults[shader] = {name: p.default for name, p in shader.parameters.items()} | shader.default_material
			base = shader.default_material if binding.material is None else binding.material
			key = (shader, id(base))
			if key not in seen:
				cached = self._records.get(key)
				if (dirty is not None and id(base) not in dirty and id(shader.default_material) not in dirty
					and cached is not None and cached[0] is base and cached[1] is shader.default_material):
					_, _, values, data = cached
				else:
					shader.validate(base)
					values = defaults[shader] | base
					data = self.parameter_layouts[shader]["material"].pack(values)
				records[key] = (base, shader.default_material, values, data)
				seen[key] = (len(headers), values)
				headers.append((self.shader_ids[shader], len(shared), 0, 0))
				shared.extend(data)
			material_id, values = seen[key]
			shader.validate(binding.overrides, overrides=True)
			resolved = values | binding.overrides
			shader.validate_resolved(resolved)
			ids.append(material_id)
			offsets.append(len(instances))
			instances.extend(self.parameter_layouts[shader]["instance"].pack(resolved))
		def slots(rows):
			return np.array(rows or [(0, 0, 0, 0)], dtype=np.uint32).reshape(-1, 4)
		self._records = records
		return slots(headers), slots(shared), slots(instances), np.array(ids, dtype=np.uint32), np.array(offsets, dtype=np.uint32)

	def source_for(self, hooks):
		selected = frozenset(hooks)
		unknown = selected - {hook.name for hook in self.interface.hooks}
		if unknown: raise ValueError(f"Unknown material hooks: {sorted(unknown)}")
		return self._source(selected)

	def _source(self, hooks=None):
		if hooks is None: hooks = frozenset(hook.name for hook in self.interface.hooks)
		parts = ["struct Materials { values: array<vec4u>, }",
			"struct InstanceOverrides { values: array<vec4u>, }",
			"struct MaterialHeaders { values: array<vec4u>, }",
			"fn library_material_count() -> u32 { return arrayLength(&material_headers.values); }"]
		functions = []
		frame_parameters = {p.name: p for p in self.interface.frame_parameters}
		for index, shader in enumerate(self.shaders):
			fields, assignments = [], []
			for name, p in shader.parameters.items():
				fields.append(f"{name}: {p.kind},")
				if p.scope == "frame":
					assignments.append(f"p.{name} = {frame_parameters[name].expression};")
					continue
				base = "material_offset" if p.scope == "material" else "instance_offset"
				buffer = "material_parameters" if p.scope == "material" else "instance_parameters"
				slot, lane = self.parameter_layouts[shader][p.scope].offsets[name]
				value = f"{buffer}.values[{base}+{slot}u].{'xyzw'[lane:lane+KINDS[p.kind]]}"
				if p.kind == "bool": value += " != 0u"
				elif p.kind != "u32": value = f"bitcast<{p.kind}>({value})"
				assignments.append(f"p.{name} = {value};")
			parts.append(f"struct Params{index} {{ {' '.join(fields) if fields else '_unused: u32,'} }}")
			header = "let material_offset = material_headers.values[material_id].y;" if self.parameter_layouts[shader]["material"].slot_count else ""
			parts.append(f"fn parameters_{index}(material_id: u32, instance_offset: u32) -> Params{index} {{ {header} var p: Params{index}; {' '.join(assignments)} return p; }}")
			code = shader.render(index, hooks)
			parts.append(code)
			from .RenderContext import ShaderSource
			functions.append(ShaderSource(code).functions if any(h.optional_function for h in self.interface.hooks) else {})
		for hook in self.interface.hooks:
			if hook.name not in hooks: continue
			parts.append(f"fn {hook.name}(id: u32{', ' if hook.arguments else ''}{hook.arguments}) -> {hook.result} {{ let key = {self.interface.address}; switch material_headers.values[key.x].x {{")
			for index in range(len(self.shaders)):
				expression = hook.expression.format(i=index)
				if hook.optional_function and hook.optional_function.format(i=index) not in functions[index]:
					expression = hook.fallback
				parts.append(f"case {index}u: {{ return {expression}; }}")
			parts.append(f"default: {{ return {hook.fallback}; }} }} }}")
		return "\n".join(parts)
