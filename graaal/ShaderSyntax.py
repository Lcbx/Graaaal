"""WGSL token spans for function dependency checks; declarations come from Jinja2."""
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


class ShaderSyntax:
	"""Find global bindings reachable from shader entry points through function calls."""
	def __init__(self, source, functions):
		self.source, self.functions = source, functions
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

	def function_references(self):
		"""Conservative function references, excluding struct member access."""
		result = {}
		for name, function in self.functions.items():
			opening = self.positions[function.body_start]
			body = self.tokens[opening+1:self.pairs[opening]]
			words = {token[0] for index, token in enumerate(body)
				if _IDENTIFIER.fullmatch(token[0]) and (index == 0 or body[index-1][0] != ".")}
			calls = {token[0] for index, token in enumerate(body[:-1])
				if _IDENTIFIER.fullmatch(token[0]) and body[index+1][0] == "("
				and (index == 0 or body[index-1][0] != ".")}
			result[name] = words, calls
		return result
