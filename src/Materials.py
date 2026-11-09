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


SHADERS = Path(__file__).with_name("shaders")
KINDS = {"f32": 1, "vec2f": 2, "vec3f": 3, "vec4f": 4, "u32": 1, "bool": 1}


@dataclass(frozen=True)
class Parameter:
	kind: str
	default: object
	scope: str = "material"

	def __post_init__(self):
		if self.kind not in KINDS or self.scope not in ("material", "instance"):
			raise ValueError("Expected a supported parameter type and material/instance scope")
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


@dataclass(frozen=True)
class ShaderHook:
	name: str
	result: str
	arguments: str
	expression: str
	fallback: str


@dataclass(frozen=True)
class MaterialInterface:
	"""Rendering adapter contract; expressions address an instance's material data."""
	address: str
	hooks: tuple[ShaderHook, ...]


@dataclass(eq=False)
class MaterialShader:
	interface: MaterialInterface
	template: str | Path
	default_material: dict = field(default_factory=dict)
	transparent: bool = False
	_parameters: dict = field(init=False, repr=False)

	@property
	def parameters(self):
		return self._parameters

	def __post_init__(self):
		if type(self.transparent) is not bool: raise ValueError("transparent must be boolean")
		path = Path(self.template).resolve()
		env = Environment(loader=FileSystemLoader([str(path.parent), str(SHADERS)]),
			undefined=StrictUndefined, autoescape=False, line_statement_prefix="#")
		self._template = env.get_template(path.name)
		_, declarations = self._render(0)
		self._parameters = MappingProxyType(declarations)
		self.validate(self.default_material)

	def _render(self, prefix):
		declarations = {}
		def parameter(name, kind, default, scope="material"):
			if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
				raise ValueError("Parameters require WGSL-compatible names")
			value = Parameter(kind, default, scope)
			if name in declarations and declarations[name].kind != kind:
				raise ValueError(f"Inherited parameter {name} requires {declarations[name].kind}")
			declarations[name] = value
			return ""
		code = self._template.render(PREFIX=prefix, TRANSPARENT=self.transparent, parameter=parameter)
		return code, declarations

	def render(self, prefix):
		code, declarations = self._render(prefix)
		if declarations != self.parameters:
			raise ValueError("Shader declarations changed; create a new MaterialShader")
		return code

	def validate(self, values, *, overrides=False):
		for name, value in values.items():
			if name not in self.parameters: raise ValueError(f"Unknown material parameter: {name}")
			parameter = self.parameters[name]
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
		self._layouts = self._layout_keys()
		self.source = self._source()

	def _layout_keys(self):
		return tuple((s.transparent, str(s.template), s.interface, tuple((name, p.kind, p.scope)
			for name, p in s.parameters.items())) for s in self.shaders)

	def pack(self, bindings):
		if self._layouts != self._layout_keys():
			raise ValueError("Shader layout, template or routing changed; create a new MaterialLibrary")
		headers, shared, instances, ids, offsets, seen = [], [], [], [], [], {}
		defaults = {}
		for binding in bindings:
			shader = binding.shader
			if shader not in self.shader_ids: raise ValueError("Material shader is not registered in this library")
			if shader not in defaults:
				shader.validate(shader.default_material)
				defaults[shader] = {name: p.default for name, p in shader.parameters.items()} | shader.default_material
			base = shader.default_material if binding.material is None else binding.material
			key = (shader, id(base))
			if key not in seen:
				shader.validate(base)
				values = defaults[shader] | base
				seen[key] = (len(headers), values)
				headers.append((self.shader_ids[shader], len(shared), 0, 0))
				shared.extend(p.pack(values[name]) for name, p in shader.parameters.items() if p.scope == "material")
			material_id, values = seen[key]
			shader.validate(binding.overrides, overrides=True)
			resolved = values | binding.overrides
			shader.validate_resolved(resolved)
			ids.append(material_id)
			offsets.append(len(instances))
			instances.extend(p.pack(resolved[name]) for name, p in shader.parameters.items() if p.scope == "instance")
		def slots(rows):
			return np.array(rows or [(0, 0, 0, 0)], dtype=np.uint32).reshape(-1, 4)
		return slots(headers), slots(shared), slots(instances), np.array(ids, dtype=np.uint32), np.array(offsets, dtype=np.uint32)

	def _source(self):
		parts = ["@group(2) @binding(6) var<storage, read> material_parameters: array<vec4u>;",
			"@group(2) @binding(7) var<storage, read> instance_parameters: array<vec4u>;",
			"@group(2) @binding(8) var<storage, read> material_headers: array<vec4u>;",
			"fn library_material_count() -> u32 { return arrayLength(&material_headers); }"]
		for index, shader in enumerate(self.shaders):
			fields, assignments = [], []
			counts = dict(material=0, instance=0)
			for name, p in shader.parameters.items():
				fields.append(f"{name}: {p.kind},")
				base = "material_headers[material_id].y" if p.scope == "material" else "instance_offset"
				buffer = "material_parameters" if p.scope == "material" else "instance_parameters"
				value = f"{buffer}[{base}+{counts[p.scope]}u]"
				counts[p.scope] += 1
				if p.kind == "bool": value += ".x != 0u"
				elif p.kind == "u32": value += ".x"
				else: value = f"bitcast<{p.kind}>({value}.{('x', 'xy', 'xyz', 'xyzw')[KINDS[p.kind]-1]})"
				assignments.append(f"p.{name} = {value};")
			parts.append(f"struct Params{index} {{ {' '.join(fields) if fields else '_unused: u32,'} }}")
			parts.append(f"fn parameters_{index}(material_id: u32, instance_offset: u32) -> Params{index} {{ var p: Params{index}; {' '.join(assignments)} return p; }}")
			parts.append(shader.render(index))
		for hook in self.interface.hooks:
			parts.append(f"fn {hook.name}(id: u32{', ' if hook.arguments else ''}{hook.arguments}) -> {hook.result} {{ let key = {self.interface.address}; switch material_headers[key.x].x {{")
			for index in range(len(self.shaders)):
				parts.append(f"case {index}u: {{ return {hook.expression.format(i=index)}; }}")
			parts.append(f"default: {{ return {hook.fallback}; }} }} }}")
		return "\n".join(parts)
