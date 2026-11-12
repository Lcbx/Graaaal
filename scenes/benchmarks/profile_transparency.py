"""Offscreen tree pass timings, sample classification, and compute/fragment parity.

Uses native GPU timestamps inside each pass. Classification/readback happen after
timing. Static images establish parity, not motion quality or statistical accuracy.
"""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np

from numpyecs.GpuProfiler import GpuProfiler
from numpyecs.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, Shader, Texture, TextureUsage
from numpyecs.TransparencyPass import SHADERS
from scenes import transparency_example as scene


def classification(oit):
	shader = Shader(filepath=str(SHADERS / "stochastic_profile.shader"), params={
		"sample_count": oit.sample_count, "force_single_contributor": oit.force_single_contributor,
		"winner_count": oit.winner_count, "winner_mix": oit.winner_mix, "compute_reconstruction": True,
		"guide_available": oit.bilateral_filter.kernel_size != 1})
	pipeline = ComputePipeline(shader, entry="classify", layout=oit._pipeline_layout(shader, "classify"))
	counts = GpuBuffer(np.zeros(10, np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_SRC)
	inputs = shader.bind_group(1, **{k: v for k, v in oit.compose_resources.resources.items() if k in shader.info.bindings})
	width, height = oit.size
	cmd = RenderContext.commands("untimed_sample_classification")
	with cmd.compute_pass() as compute:
		compute.set_pipeline(pipeline)
		for bindings in (oit._frame_group(shader), inputs, shader.bind_group(2, counters=counts)):
			compute.set_bind_group(bindings)
		compute.dispatch((width+7)//8, (height+7)//8, oit.sample_count)
	RenderContext.submit(cmd.finish())
	values = np.frombuffer(RenderContext.device.queue.read_buffer(counts.handle), np.uint32).copy()
	counts.close()
	keys = ("total_samples", "empty_samples", "forced_samples", "certain_unforced_samples", "stochastic_samples",
		"winner_present", "missing_active_winner", "first_stream_empty_active", "filter_samples", "filter_tiles")
	result = dict(zip(keys, map(int, values)))
	result["total_tiles"] = ((width+7)//8)*((height+7)//8)*oit.sample_count
	return result


def capture(output):
	width, height, _ = output.size
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": width*4}, output.size)
	return np.frombuffer(data, np.uint8).reshape(height, width, 4).copy()


def configurations(stencil=False):
	"""Matched force on/off cases at each MSAA count; Hi-Z policy is unchanged."""
	result = []
	for samples in (1, 4):
		for force in (False, True):
			prefix = "single" if samples == 1 else "msaa"
			suffix = "_force" if force else ""
			for kernel in (3, 5, 9):
				result.append((f"{prefix}{suffix}_{kernel}", 1, samples, kernel, force, "average", False, 1))
			for mix in ("average", "fill"):
				name = ("" if samples == 1 else "msaa_")+mix+suffix+"_3"
				result.append((name, 2, samples, 3, force, mix, False, 1))
				if stencil and mix == "fill": result.append((name+"_stencil", 2, samples, 3, force, mix, True, 1))
			for winners, kernel, passes in ((4, 3, 1), (8, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (4, 3, 2), (2, 5, 1), (2, 9, 1)):
				name = ("" if samples == 1 else "msaa_")+f"fill{winners}"+suffix+f"_{kernel}x{passes}"
				result.append((name, winners, samples, kernel, force, "fill", False, passes))
	return result


def distribution(values):
	if not values: return None
	return {"median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
		"p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
		"iqr": float(np.percentile(values, 75)-np.percentile(values, 25))}


def run(size=(1280, 720), frames=30, warmup=5, configs=None, zoom=1.0, rounds=4,
	implementations=("fragment", "compute"), seed=11012, stencil=False):
	RenderContext.setup_graphics_backend(True)
	features = ["indirect-first-instance"]
	timestamps = "timestamp-query" in RenderContext.adapter.features
	if timestamps: features.append("timestamp-query")
	if stencil:
		if "depth32float-stencil8" not in RenderContext.adapter.features:
			raise ValueError("Stencil experiment requires depth32float-stencil8")
		features.append("depth32float-stencil8")
	RenderContext.setup_graphics_backend(True, required_features=features)
	RenderContext.window_dimensions = size
	RenderContext.presentation_format = "rgba8unorm"
	output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	original_screen = RenderContext.screen
	RenderContext.screen = lambda **kwargs: output.color_attachment()
	profiler = GpuProfiler(max_passes=128) if timestamps else None
	try:
		scene.init_scene()
		target = np.asarray(scene.camera.target)
		scene.camera.position = scene.Vec3(target+(np.asarray(scene.camera.position)-target)/zoom)
		oit = scene.transparency
		oit.set_noise_mode("anchored_blue")
		settings = [c for c in configurations(stencil) if configs is None or c[0] in configs]
		unknown = set(configs or ())-{c[0] for c in settings}
		if unknown: raise ValueError(f"Unknown configurations: {sorted(unknown)}")
		measurements, images, classes, schedule = {}, {}, {}, []
		rng = np.random.default_rng(seed)
		for trial in range(rounds):
			count = frames//rounds+int(trial < frames%rounds)
			if count == 0: continue
			for config_index in rng.permutation(len(settings)):
				name, winners, samples, kernel, force, mix, early, passes = settings[config_index]
				if getattr(oit, "early_b_rejection", False): oit.set_early_b_rejection(False)
				if oit.winner_count > 2: oit.set_winner_count(2)
				oit.set_winner_mix("average")
				oit.set_sample_count(samples)
				oit.set_force_single_contributor(force)
				oit.set_winner_mix(mix)
				oit.set_winner_count(winners)
				oit.set_reconstruction_passes(passes)
				if early: oit.set_early_b_rejection(True)
				oit.set_bilateral_filter(replace(oit.bilateral_filter, kernel_size=kernel))
				order = implementations if trial%2 == 0 else implementations[::-1]
				for mode in order:
					schedule.append(dict(round=trial, config=name, reconstruction=mode))
					oit.set_reconstruction(mode)
					key = (name, mode)
					records = measurements.setdefault(key, [])
					for frame in range(warmup+count):
						start = perf_counter()
						scene.render_frame(profiler=profiler)
						if profiler is not None: gpu = profiler.read()
						else:
							RenderContext.device.queue.on_submitted_work_done_sync()
							gpu = None
						if frame >= warmup:
							records.append(dict(round=trial, gpu_pass_ms=gpu,
								gpu_frame_ms=profiler.last_frame_ms if profiler else None,
								wall_ms=(perf_counter()-start)*1000))
					if key not in images:
						images[key] = capture(output)
						classes[key] = classification(oit)
		rows = []
		for name, winners, samples, kernel, force, mix, early, passes in settings:
			for mode in implementations:
				records = measurements[(name, mode)]
				gpu = [r["gpu_pass_ms"] for r in records if r["gpu_pass_ms"] is not None]
				pass_medians = {k: float(np.median([m.get(k, 0) for m in gpu])) for k in sorted(set().union(*(m.keys() for m in gpu)))} if gpu else None
				image = images[(name, mode)]
				baseline = images.get((name, "fragment"), image)
				stats = distribution([r["gpu_frame_ms"] for r in records if r["gpu_frame_ms"] is not None])
				rows.append(dict(config=name, reconstruction=mode, winners=winners, msaa=samples, kernel=kernel,
					reconstruction_passes=passes,
					winner_coupling=oit.winner_coupling, same_estimator_as_fragment=mode in ("fragment", "compute"),
					force=force, winner_mix=mix, early_b_rejection=early, gpu_frame_ms=stats,
					gpu_frame_ms_median=stats["median"] if stats else None, gpu_pass_ms_median=pass_medians,
					gpu_pass_sum_ms_median=float(np.median([sum(m.values()) for m in gpu])) if gpu else None,
					synchronized_wall_ms=distribution([r["wall_ms"] for r in records]),
					classification=classes[(name, mode)], measurements=records,
					output_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
					max_output_byte_delta=int(np.max(np.abs(image.astype(int)-baseline))),
					mean_output_byte_delta=float(np.mean(np.abs(image.astype(int)-baseline)))))
		return dict(adapter=dict(RenderContext.adapter.info), size=size, zoom=zoom, frames=frames, warmup_per_block=warmup,
			rounds=rounds, seed=seed, schedule=schedule, timestamps=timestamps, rows=rows)
	finally:
		RenderContext.screen = original_screen
		if profiler is not None: profiler.close()
		RenderContext.cleanup()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--size", default="1280x720")
	parser.add_argument("--frames", type=int, default=30)
	parser.add_argument("--warmup", type=int, default=5)
	parser.add_argument("--zoom", type=float, default=1.0, help="Divide the initial camera distance by this factor")
	parser.add_argument("--configs", nargs="+")
	parser.add_argument("--rounds", type=int, default=4)
	parser.add_argument("--implementations", nargs="+", choices=("fragment", "compute", "push_pull", "atrous"), default=["fragment", "compute"])
	parser.add_argument("--seed", type=int, default=11012)
	parser.add_argument("--stencil", action="store_true", help="Include optional fill B stencil-gating comparisons")
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	try: size = tuple(map(int, args.size.lower().split("x")))
	except ValueError: parser.error("size must be WIDTHxHEIGHT")
	if len(size) != 2 or min(size) < 1 or args.frames < 1 or args.warmup < 0 or args.rounds < 1 or not np.isfinite(args.zoom) or args.zoom <= 0:
		parser.error("Positive size/frames/zoom and nonnegative warmup required")
	result = run(size, args.frames, args.warmup, args.configs, args.zoom, args.rounds, tuple(dict.fromkeys(args.implementations)), args.seed, args.stencil)
	args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
	for row in result["rows"]:
		print(row["config"], row["reconstruction"], "frame", row["gpu_frame_ms_median"],
			"ms; pass sum", row["gpu_pass_sum_ms_median"], "ms; max byte delta", row["max_output_byte_delta"])


if __name__ == "__main__": main()
