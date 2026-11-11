"""Small WGSL declaration shorthand expanded before Jinja2 parses templates."""
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
	rf"^(?P<indent>[ \t]*)(?P<kind>uniform|storage)<[ \t]*"
	r"(?P<group>\d+)[ \t]*,[ \t]*(?P<slot>\d+)"
	r"(?:[ \t]*,[ \t]*(?P<access>\w+))?[ \t]*>[ \t]+"
	rf"(?P<name>{_IDENTIFIER})[ \t]*:[ \t]*(?P<type>[^;\n]+?)[ \t]*;[ \t]*(?P<comment>//[^\n]*)?$"
)
_ATTRIBUTE = re.compile(r"@(?P<name>location|builtin|interpolate|align|size)\((?P<value>[^()]*)\)[ \t]*")


class ShaderShorthand(Extension):
	def preprocess(self, source, name, filename=None):
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
				options = ["storage="+repr(kind)]
				if kind == "storage": options.append("access="+repr(access or "read"))
				args = [repr(match[key]) for key in ("name", "type")]+[match["group"], match["slot"]]
				result.append(f"{match['indent']}{{{{ binding({', '.join(args + options)}) }}}}"+(" "+match["comment"] if match["comment"] else "")+ending)
				continue
			if re.match(rf"^[ \t]*(?:<{_IDENTIFIER}>[ \t]+|(?:uniform|storage)<)", body):
				raise ValueError(f"Invalid WGSL shorthand declaration: {body.strip()}")
			result.append(line)
		return "".join(result)
