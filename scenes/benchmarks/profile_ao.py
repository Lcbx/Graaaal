"""Compare AO inside composition with the separate AO postprocess pass."""
import argparse
import json
from pathlib import Path

import numpy as np

from graaal.GpuProfiler import GpuProfiler
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from scenes import transparency_example as scene


MODES = ("ao_compose", "ao_pass")


def run(size=(1280, 720), frames=30, warmup=5, rounds=3, samples=(1, 4)):
	RenderContext.setup_graphics_backend(True)
	if "timestamp-query" not in RenderContext.adapter.features:
		raise RuntimeError("AO performance comparison requires timestamp-query support")
	RenderContext.setup_graphics_backend(True, required_features=("indirect-first-instance", "timestamp-query"))
	RenderContext.window_dimensions = size
	RenderContext.presentation_format = "rgba8unorm"
	output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	original_screen = RenderContext.screen
	RenderContext.screen = lambda **kwargs: output.color_attachment()
	profiler = GpuProfiler(max_passes=128)
	try:
		scene.init_scene(sample_count=1, tail_allocation="stochastic")
		measurements = {(count, mode): [] for count in samples for mode in MODES}
		baseline = {count: [] for count in samples}
		order = [(count, mode) for count in samples for mode in ("off", *MODES)]
		for trial in range(rounds):
			for sample_count, mode in order if trial % 2 == 0 else reversed(order):
				scene.set_msaa_enabled(sample_count == 4)
				scene.set_ao_method("compose" if mode == "ao_compose" else "pass")
				scene.set_ao_enabled(mode != "off")
				for frame in range(warmup+frames):
					scene.render_frame(profiler=profiler)
					passes = profiler.read()
					if frame < warmup:
						continue
					row = dict(frame_ms=profiler.last_frame_ms,
						compose_ms=passes.get("5_reconstruct_compose", 0.0),
						ao_pass_ms=passes.get("6_ambient_occlusion", 0.0))
					if mode == "off": baseline[sample_count].append(row)
					else: measurements[(sample_count, mode)].append(row)
		rows = []
		for (sample_count, mode), values in measurements.items():
			rows.append(dict(sample_count=sample_count, mode=mode,
				gpu_frame_ms=float(np.median([value["frame_ms"] for value in values])),
				compose_ms=float(np.median([value["compose_ms"] for value in values])),
				ao_pass_ms=float(np.median([value["ao_pass_ms"] for value in values])),
				combined_ao_ms=float(np.median([value["compose_ms"]+value["ao_pass_ms"] for value in values]))))
		return dict(adapter=dict(RenderContext.adapter.info), size=size, sample_counts=samples, frames_per_round=frames,
			warmup_per_mode=warmup, rounds=rounds,
			off_gpu_frame_ms={str(count): float(np.median([value["frame_ms"] for value in values])) for count, values in baseline.items()}, rows=rows,
			timestamp_scope="Pass-boundary GPU timestamp queries; each frame is synchronized before the next.")
	finally:
		RenderContext.screen = original_screen
		profiler.close()
		RenderContext.cleanup()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--size", default="1280x720")
	parser.add_argument("--frames", type=int, default=30)
	parser.add_argument("--warmup", type=int, default=5)
	parser.add_argument("--rounds", type=int, default=3)
	parser.add_argument("--samples", nargs="+", type=int, choices=(1, 4), default=[1, 4])
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	try:
		size = tuple(map(int, args.size.lower().split("x")))
	except ValueError:
		parser.error("size must be WIDTHxHEIGHT")
	if len(size) != 2 or min(size) < 64 or args.frames < 1 or args.warmup < 0 or args.rounds < 1:
		parser.error("Expected size>=64, positive frames/rounds and nonnegative warmup")
	result = run(size, args.frames, args.warmup, args.rounds, tuple(dict.fromkeys(args.samples)))
	if args.output is not None:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
	print("off", result["off_gpu_frame_ms"], "ms/frame")
	for row in result["rows"]:
		print(f"{row['sample_count']}x", row["mode"], "frame", row["gpu_frame_ms"], "ms; compose", row["compose_ms"],
			"ms; AO pass", row["ao_pass_ms"], "ms; compose+AO", row["combined_ao_ms"], "ms")


if __name__ == "__main__":
	main()
