"""Helpers for recording transparency passes."""


def record_transparency(commands, transparency, geometry, draw):
	"""Record transparent geometry passes using draw(render, pass_name)."""
	if transparency.exact_nearest:
		with transparency.nearest_pass(commands) as render:
			draw(render, "nearest")
		transparency.shade_nearest(commands, geometry)
	# The point estimator never samples a reconstruction guide.
	if not transparency.banded_tail and transparency.bilateral_filter.kernel_size != 1:
		with transparency.guide_pass(commands) as render:
			draw(render, "guide")
	with transparency.transmittance_pass(commands) as render:
		draw(render, "transmittance")
	if transparency.keyed_tail:
		for stream in (0, 1):
			with transparency.selection_pass(commands, stream=stream) as render:
				draw(render, transparency.selection_name(stream))
		transparency.shade_winner(commands, geometry)
	elif transparency.banded_tail:
		if transparency.mixed_tail:
			with transparency.tail_statistics_pass(commands) as render:
				draw(render, "tail_statistics")
		with transparency.banded_pass(commands) as render:
			draw(render, "bands")
		transparency.shade_winner(commands, geometry)
	for stream in range(0 if transparency.banded_tail else transparency.winner_count):
		with transparency.selection_pass(commands, stream=stream) as render:
			draw(render, transparency.selection_name(stream))
		transparency.shade_winner(commands, geometry, stream=stream)
