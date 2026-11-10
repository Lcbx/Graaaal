"""Shared WGSL source spans and lexical rewriting; GPU validation stays in Shader."""
from dataclasses import dataclass
import re


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
_NUMBER = (r"(?:0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?[fhiu]?"
	r"|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[fhiu]?)")
_TOKEN = re.compile(rf"{_NUMBER}|[A-Za-z_]\w*|[^\s]")
_COMMENT = re.compile(r"//[^\r\n]*|/\*")
_BLOCK_COMMENT = re.compile(r"/\*|\*/")


def strip_comments(source):
	"""Mask comments without moving source positions, including nested WGSL blocks."""
	pieces, end = [], 0
	while match := _COMMENT.search(source, end):
		start, stop = match.span()
		if match[0] == "/*":
			depth = 1
			while depth:
				closing = _BLOCK_COMMENT.search(source, stop)
				if closing is None: raise ValueError("Unterminated WGSL block comment")
				depth += 1 if closing[0] == "/*" else -1
				stop = closing.end()
		pieces.extend((source[end:start], re.sub(r"[^\r\n]", " ", source[start:stop])))
		end = stop
	pieces.append(source[end:])
	return "".join(pieces)


@dataclass(frozen=True)
class SourceSpan:
	"""Half-open character offsets in ShaderSource.source."""
	start: int
	end: int


@dataclass(frozen=True)
class ShaderDeclaration:
	name: str
	kind: str
	span: SourceSpan
	name_span: SourceSpan
	initializer: str | None = None


class ShaderSyntax:
	"""Cached lexical data using the functions and members already reflected."""
	def __init__(self, source, functions, structs):
		self.source, self.functions, self.structs = source, functions, structs
		self.tokens = list(_TOKEN.finditer(source))
		self.positions = {token.start(): index for index, token in enumerate(self.tokens)}
		self.pairs, stack = {}, []
		for index, token in enumerate(self.tokens):
			if token[0] in ("(", "[", "{"): stack.append(index)
			elif token[0] in (")", "]", "}"):
				if not stack or self.tokens[stack[-1]][0] != {")": "(", "]": "[", "}": "{"}[token[0]]:
					raise ValueError("Unbalanced WGSL delimiters")
				self.pairs[stack.pop()] = index
		if stack: raise ValueError("Unbalanced WGSL delimiters")
		self.declarations = self._declarations()
		self._references = None

	def _declarations(self):
		tokens, result, index = self.tokens, {}, 0
		while index < len(tokens):
			kind = tokens[index][0]
			if kind not in ("fn", "struct", "alias", "const", "override", "var"):
				index += 1
				continue
			name_index = index+1
			if name_index < len(tokens) and tokens[name_index][0] == "<":
				while name_index < len(tokens) and tokens[name_index][0] != ">": name_index += 1
				name_index += 1
			if name_index >= len(tokens): raise ValueError("Missing WGSL declaration name")
			name = tokens[name_index][0]
			if not _IDENTIFIER.fullmatch(name): raise ValueError("Invalid WGSL declaration name")
			end, initializer = name_index+1, None
			if kind in ("fn", "struct"):
				while end < len(tokens) and tokens[end][0] != "{": end += 1
				if end not in self.pairs: raise ValueError(f"Missing body for {name}")
				end = self.pairs[end]
			else:
				while end < len(tokens) and tokens[end][0] != ";": end += 1
				if end == len(tokens): raise ValueError(f"Missing semicolon after {name}")
				equals = next((i for i in range(name_index+1, end) if tokens[i][0] == "="), None)
				if equals is not None: initializer = self.source[tokens[equals].end():tokens[end].start()].strip()
			if name in result: raise ValueError(f"Duplicate shader declaration: {name}")
			result[name] = ShaderDeclaration(name, kind, SourceSpan(tokens[index].start(), tokens[end].end()),
				SourceSpan(*tokens[name_index].span()), initializer)
			index = end+1
		return result

	def function_references(self):
		"""Conservative identifiers/calls for the existing binding reachability walk."""
		if self._references is None:
			self._references = {}
			for name, declaration in self.declarations.items():
				if declaration.kind != "fn": continue
				opening = self.positions[declaration.name_span.start]+1
				while self.tokens[opening][0] != "{": opening += 1
				body = self.tokens[opening+1:self.pairs[opening]]
				words = {token[0] for token in body if _IDENTIFIER.fullmatch(token[0])}
				calls = {token[0] for index, token in enumerate(body[:-1])
					if _IDENTIFIER.fullmatch(token[0]) and body[index+1][0] == "("}
				self._references[name] = words, calls
		return self._references

	def rename_symbols(self, names):
		"""Rename module/free identifiers without capturing locals or changing members."""
		if any(not _IDENTIFIER.fullmatch(name) for pair in names.items() for name in pair):
			raise ValueError("Shader symbol names must be identifiers")
		targets = [names.get(name, name) for name in self.declarations]
		if len(targets) != len(set(targets)): raise ValueError("Renamed shader declarations would collide")
		tokens = self.tokens
		generated = set(names.values())
		used = {token[0] for token in tokens} | generated
		serial = 0
		def local_name(name):
			nonlocal serial
			if name not in generated: return name
			while True:
				candidate = f"shader_local_{serial}"
				serial += 1
				if candidate not in used:
					used.add(candidate)
					return candidate
		function_scopes, replacements = {}, {}
		for function in self.functions.values():
			bindings = {}
			for parameter in function.parameters:
				name = parameter.name
				if name not in bindings: bindings[name] = local_name(name)
				replacements[parameter.name_span.start] = bindings[name]
			function_scopes[self.positions[function.body_start]] = bindings
		for_ends = set()
		for index, token in enumerate(tokens[:-1]):
			if token[0] == "for" and tokens[index+1][0] == "(":
				body = self.pairs[index+1]+1
				if body in self.pairs: for_ends.add(self.pairs[body])
		fields = {field.name_span.start for struct in self.structs.values() for field in struct.fields}
		# Attribute names and enum arguments are not module references. Numeric
		# attributes such as @size(N) still refer to a module constant N.
		attributes = set()
		for index, token in enumerate(tokens[:-1]):
			if token[0] == "@":
				attributes.add(index+1)
				if tokens[index+1][0] in ("builtin", "interpolate", "diagnostic"):
					opening = index+2
					if opening in self.pairs: attributes.update(range(opening+1, self.pairs[opening]))
		scopes, pending = [], None
		for index, token in enumerate(tokens):
			word, start = token[0], token.start()
			if word == "for": scopes.append({})
			elif word == "{": scopes.append(function_scopes.get(index, {}))
			elif word == "}":
				scopes.pop()
				if index in for_ends: scopes.pop()
			if scopes and word in ("let", "var", "const"):
				name_index = index+1
				if tokens[name_index][0] == "<":
					while tokens[name_index][0] != ">": name_index += 1
					name_index += 1
				name = tokens[name_index][0]
				target = scopes[-1][name] if name in scopes[-1] else local_name(name)
				replacements[tokens[name_index].start()] = target
				pending = (name, target)
			elif word == ";" and pending:
				# The initializer resolves outer bindings before this local is in scope.
				scopes[-1][pending[0]] = pending[1]
				pending = None
			if start in replacements or start in fields or index in attributes or not _IDENTIFIER.fullmatch(word): continue
			if index and tokens[index-1][0] == ".": continue
			for scope in reversed(scopes):
				if word in scope:
					replacements[start] = scope[word]
					break
			else:
				if word in names: replacements[start] = names[word]
		pieces, end = [], 0
		for token in tokens:
			if token.start() in replacements:
				pieces.extend((self.source[end:token.start()], replacements[token.start()]))
				end = token.end()
		pieces.append(self.source[end:])
		return "".join(pieces)
