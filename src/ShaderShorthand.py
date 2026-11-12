"""Small WGSL declaration and texture shorthand expanded before Jinja2 parses templates."""
import re

from jinja2.ext import Extension


MISSING = object()
_IDENTIFIER = r"[A-Za-z_][A-Za-z_0-9]*"
_FIELD = re.compile(
	rf"^(?P<indent>[ \t]*)<(?P<struct>{_IDENTIFIER})>[ \t]+"
	rf"(?P<attrs>(?:@{_IDENTIFIER}\([^()\n]*\)[ \t]*)*)"
	rf"(?P<name>{_IDENTIFIER})[ \t]*:[ \t]*(?P<type>[^;=\n]+?)"
	r"(?:[ \t]*=[ \t]*(?P<default>[^;\n]+))?[ \t]*;[ \t]*(?P<comment>//[^\n]*)?$"
)
_BINDING = re.compile(
	rf"^(?P<indent>[ \t]*)(?P<kind>uniform|storage|texture|sample_texture)<[ \t]*"
	r"(?P<group>\d+)[ \t]*,[ \t]*(?P<slot>\d+)"
	r"(?:[ \t]*,[ \t]*(?P<access>\w+))?[ \t]*>[ \t]+"
	rf"(?P<name>{_IDENTIFIER})[ \t]*:[ \t]*(?P<type>[^;\n]+?)[ \t]*;[ \t]*(?P<comment>//[^\n]*)?$"
)
_ATTRIBUTE = re.compile(r"@(?P<name>location|builtin|interpolate|align|size)\((?P<value>[^()]*)\)[ \t]*")
_PROTECTED = re.compile(r'//[^\n]*|/\*.*?\*/|\{\{.*?\}\}|\{%.*?%\}|\{\#.*?\#\}|^[ \t]*\#[^\n]*|"(?:\\.|[^"\\])*"', re.DOTALL | re.MULTILINE)
_SAMPLE_LOAD = re.compile(_PROTECTED.pattern+r"|\bsample_texture_load\s*\(", re.DOTALL | re.MULTILINE)
_TEMPLATE_START = re.compile(r"\b(?:array|atomic|ptr|vec[234]|mat[234]x[234]|bitcast|texture_\w+)\s*<")


def _template_argument(environment, argument):
	parts, expression = [], []
	for _, kind, value in environment.lex(argument):
		if kind == "data": parts.append(repr(value))
		elif kind == "variable_begin": expression = []
		elif kind == "variable_end": parts.append("("+"".join(expression)+")")
		else: expression.append(value)
	return "("+" ~ ".join(parts)+")"


def _sample_texture_loads(source, environment):
	result, start, position = [], 0, 0
	while match := _SAMPLE_LOAD.search(source, position):
		position = match.end()
		if not match.group().startswith("sample_texture_load"): continue
		arguments, stack, argument_start = [], [")"], position
		while stack and position < len(source):
			protected = _PROTECTED.match(source, position)
			if protected:
				position = protected.end()
				continue
			template = _TEMPLATE_START.match(source, position)
			if template:
				stack.append(">")
				position = template.end()
				continue
			char = source[position]
			if char in "([{":
				stack.append({"(": ")", "[": "]", "{": "}"}[char])
			elif char in ")]}":
				if char != stack.pop(): raise ValueError("Unbalanced sample_texture_load arguments")
				if not stack: arguments.append(source[argument_start:position].strip())
			elif char == ">" and stack[-1] == ">":
				stack.pop()
			elif char == "," and len(stack) == 1:
				arguments.append(source[argument_start:position].strip())
				argument_start = position+1
			position += 1
		if stack: raise ValueError("Unclosed sample_texture_load arguments")
		if len(arguments) not in (3, 4) or not all(arguments):
			raise ValueError("sample_texture_load requires texture, pixel, sample and optional layered=true/false")
		options = []
		if len(arguments) == 4:
			layered = re.fullmatch(r"(?:layered\s*=\s*)?(true|false)", arguments[3])
			if layered is None: raise ValueError("sample_texture_load layered must be true or false")
			options.append("layered="+layered[1])
		args = [_template_argument(environment, argument) for argument in arguments[:3]]+options
		result.append(source[start:match.start()]+"{{ sample_texture_load("+", ".join(args)+") }}")
		start = position
	result.append(source[start:])
	return "".join(result), start != 0


class ShaderShorthand(Extension):
	def preprocess(self, source, name, filename=None):
		source, sample_textures = _sample_texture_loads(source, self.environment)
		result = []
		for line in source.splitlines(keepends=True):
			body = line.rstrip("\r\n")
			ending = line[len(body):]
			match = _FIELD.fullmatch(body)
			if match:
				options = []
				for attribute in _ATTRIBUTE.finditer(match["attrs"]):
					key, value = attribute["name"], attribute["value"].strip()
					if key in ("align", "size", "location"):
						if not value.isdecimal(): raise ValueError(f"Invalid @{key}: {value}")
						options.append(f"{key}={value}")
					else:
						options.append(f"{key}={value!r}")
				if "".join(item.group() for item in _ATTRIBUTE.finditer(match["attrs"])) != match["attrs"]:
					raise ValueError(f"Invalid WGSL shorthand attributes: {match['attrs']}")
				if match["default"] is not None: options.append(f"default={match['default'].strip()}")
				args = [repr(match[key]) for key in ("struct", "name", "type")]
				result.append(f"{match['indent']}{{{{ field({', '.join(args + options)}) }}}}"+(" "+match["comment"] if match["comment"] else "")+ending)
				continue
			match = _BINDING.fullmatch(body)
			if match:
				kind, access = match["kind"], match["access"]
				if kind == "uniform" and access not in (None, "read"):
					raise ValueError("Uniform bindings must be read-only")
				if kind == "storage" and access not in (None, "read", "read_write"):
					raise ValueError("Storage access must be read or read_write")
				options = []
				type_name = repr(match["type"])
				if kind in ("uniform", "storage"):
					options.append("storage="+repr(kind))
					if kind == "storage": options.append("access="+repr(access or "read"))
				elif kind == "sample_texture":
					if access not in (None, "layered"): raise ValueError("sample_texture layout must be layered or omitted")
					if match["type"] not in ("f32", "i32", "u32", "depth"): raise ValueError("Invalid sample_texture type")
					type_name = f"sample_texture_type({match['type']!r}, layered={'true' if access else 'false'})"
					sample_textures = True
				elif access is not None:
					raise ValueError("Texture bindings do not take an access argument; specify it in the texture type")
				args = [repr(match["name"]), type_name, match["group"], match["slot"]]
				result.append(f"{match['indent']}{{{{ binding({', '.join(args + options)}) }}}}"+(" "+match["comment"] if match["comment"] else "")+ending)
				continue
			if re.match(rf"^[ \t]*(?:<{_IDENTIFIER}>[ \t]+|(?:uniform|storage|texture|sample_texture)<)", body):
				raise ValueError(f"Invalid WGSL shorthand declaration: {body.strip()}")
			result.append(line)
		if sample_textures:
			result.insert(0, '#from "utils_texture.shaderlib" import sample_texture_type, sample_texture_load with context\n')
		return "".join(result)
