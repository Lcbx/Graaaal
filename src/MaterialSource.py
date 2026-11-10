"""Plain WGSL material linking using the renderer's existing shader reflection."""
import re

from .RenderContext import ShaderSource, _normalise_wgsl_type


def _literal(expression, kind):
	"""Defaults are scalar literals or float-vector constructors, never Python eval."""
	expression = expression.strip()
	if kind.startswith("vec"):
		width = int(kind[3])
		match = re.fullmatch(rf"(?:{kind}|vec{width}\s*<\s*f32\s*>)\s*\((.*)\)", expression, re.S)
		if match is None: raise ValueError("Expected a float-vector literal constructor")
		parts = [part.strip() for part in match[1].rstrip().rstrip(",").split(",")]
		values = tuple(_literal(part, "f32") for part in parts)
		if len(values) == 1: values *= width
		if len(values) != width: raise ValueError(f"Expected {width} vector components")
		return values
	if kind == "bool":
		if expression not in ("true", "false"): raise ValueError("Expected true or false")
		return expression == "true"
	if kind == "u32":
		if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)u?", expression):
			raise ValueError("Expected an unsigned integer literal")
		value = expression.removesuffix("u")
		return int(value, 16 if value.lower().startswith("0x") else 10)
	if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?f?", expression):
		raise ValueError("Expected a float literal")
	return float(expression.removesuffix("f"))


class MaterialSource:
	"""Reflect declarations and namespace functions before the usual Shader compile.

	MaterialParameters/InstanceParameters declare dictionary fields. FrameParameters
	opt into renderer-owned values, without allocating dictionary buffer slots. Optional
	const default_NAME declarations supply literal defaults in the WGSL itself.
	The rendering adapter supplies Params, frame data and the material functions.
	"""
	def __init__(self, source):
		self.info = ShaderSource(source)
		self.code = self.info.source
		if "{{" in self.code or re.search(r"^\s*#", self.code, re.M):
			raise ValueError("source requires plain WGSL; use template for Jinja shaders")
		self.globals = self.info.declarations
		if self.info.bindings or self.info.entry_points:
			raise ValueError("Material source uses renderer-provided bindings and entry points; define material functions instead")
		if "Params" in self.globals or "parameters" in self.globals:
			raise ValueError("Params and parameters are provided by the material renderer")
		if any(name.startswith("default_") for name in self.info.functions):
			raise ValueError("Function names starting with default_ are reserved for default helpers")

	def declarations(self, inherited):
		result = dict(inherited)
		seen = set()
		from .Materials import Parameter
		for struct_name, scope in (("MaterialParameters", "material"), ("InstanceParameters", "instance"), ("FrameParameters", "frame")):
			struct = self.info.structs.get(struct_name)
			for field in struct.fields if struct else ():
				name = field.name
				if name in seen or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
					raise ValueError(f"Invalid or duplicate material parameter: {name}")
				seen.add(name)
				kind = _normalise_wgsl_type(field.type_name)
				kind = {f"vec{n}<f32>": f"vec{n}f" for n in (2, 3, 4)}.get(kind, kind)
				previous = result.get(name)
				if previous is not None and previous.kind != kind:
					raise ValueError(f"Inherited parameter {name} requires {previous.kind}")
				default = previous.default if previous else False if kind == "bool" else 0 if kind == "u32" else 0.0
				if kind in ("vec2f", "vec3f", "vec4f") and previous is None: default = (0.0,)*int(kind[3])
				declaration = self.globals.get("default_"+name)
				if declaration is not None and declaration.kind == "const":
					if declaration.initializer is None: raise ValueError(f"Invalid default_{name} declaration")
					try: default = _literal(declaration.initializer, kind)
					except ValueError as error: raise ValueError(f"Invalid default_{name}: {error}") from error
				result[name] = Parameter(kind, default, scope)
		return result

	def link(self, prefix, base):
		base_info = ShaderSource(base)
		functions = set(base_info.functions)
		base_renames = {}
		names = {name: f"{name}_{prefix}" for name in self.globals}
		names["Params"] = f"Params{prefix}"
		for name, function in self.info.functions.items():
			original = base_info.functions.get(f"{name}_{prefix}")
			if original is None: continue  # user helper
			def signature(info):
				return (tuple(_normalise_wgsl_type(field.type_name).replace(f"Params{prefix}", "Params") for field in info.parameters),
					_normalise_wgsl_type(info.return_type or ""))
			if signature(function) != signature(original):
				raise ValueError(f"Material function {name} must match the default signature")
			if function.return_attributes or any(field.attributes for field in function.parameters):
				raise ValueError(f"Material function {name} must not declare stage IO attributes")
			functions.remove(f"{name}_{prefix}")
			base_renames[f"{name}_{prefix}"] = f"default_{name}_{prefix}"
			functions.add(f"default_{name}_{prefix}")
		# Stable helper names refer to the existing implementation for this family.
		for name in functions:
			if name.endswith(f"_{prefix}"):
				stable = name[:-(len(str(prefix))+1)]
				if stable in self.globals: raise ValueError(f"Reserved default helper: {stable}")
				names[stable] = name
				if not stable.startswith("default_"): names.setdefault("default_"+stable, name)
		return base_info.rename_symbols(base_renames)+"\n"+self.info.rename_symbols(names)
