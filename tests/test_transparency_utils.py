from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from graaal.transparency_utils import record_transparency


@pytest.mark.parametrize("mode,nearest,kernel,expected", [
	("stochastic", False, 1, ["transmittance", "select0", "shade0", "select1", "shade1"]),
	("stochastic", True, 3, ["nearest", "shade_nearest", "guide", "transmittance", "select0", "shade0", "select1", "shade1"]),
	("banded", True, 3, ["nearest", "shade_nearest", "transmittance", "tail_statistics", "bands", "shade"]),
	("min_key", True, 3, ["nearest", "shade_nearest", "transmittance", "select0", "select1", "shade"]),
])
def test_record_transparency_sequence(mode, nearest, kernel, expected):
	events = []
	geometry = object()
	commands = object()
	@contextmanager
	def render_pass(*args, **kwargs):
		assert args == (commands,)
		yield object()
	class Pass:
		exact_nearest = nearest
		banded_tail = mode != "stochastic"
		keyed_tail = mode == "min_key"
		mixed_tail = mode == "banded"
		winner_count = 2
		bilateral_filter = SimpleNamespace(kernel_size=kernel)
		nearest_pass = guide_pass = transmittance_pass = selection_pass = tail_statistics_pass = banded_pass = staticmethod(render_pass)
		def selection_name(self, stream): return f"select{stream}"
		def shade_nearest(self, command, supplied):
			assert command is commands and supplied is geometry
			events.append("shade_nearest")
		def shade_winner(self, command, supplied, *, stream=None):
			assert command is commands and supplied is geometry
			events.append("shade" if stream is None else f"shade{stream}")
	record_transparency(commands, Pass(), geometry, lambda render, name: events.append(name))
	assert events == expected
