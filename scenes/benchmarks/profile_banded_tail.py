"""Timestamp banded-tail stages and compare staggered quads with sorted over."""
import argparse
import json
from pathlib import Path

import numpy as np

from graaal.DefaultMaterial import DefaultMaterialShader
from graaal.GpuProfiler import GpuProfiler
from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.TransparencyData import VERTEX_DTYPE
from graaal import TransparencyPass as transparency_module


VARIANTS = {
	"baseline": dict(proxy=False, subset=False, tail_msaa=True),
	"proxy": dict(proxy=True, subset=False, tail_msaa=True),
	"subset": dict(proxy=True, subset=True, tail_msaa=True),
	"single": dict(proxy=True, subset=True, tail_msaa=False),
	"single_full": dict(proxy=True, subset=False, tail_msaa=False),
}


def make_scene(size):
	"""Return three emissive quads with pixel-aligned edges and analytic references."""
	width, height = size
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block lighting
	return p.emission;
#endblock
''')
	colors = np.eye(3, dtype=np.float32)
	alphas = (0.15, 0.25, 0.15)
	rects = ((-.9, -.85, .9, .85), (-.4, -.6, .25, .6), (-.8, -.75, .8, .75))
	items, masks, bounds = [], [], []
	y, x = np.indices((height, width))
	for index, (rect, color, alpha) in enumerate(zip(rects, colors, alphas)):
		left, bottom, right, top = np.rint((np.asarray(rect)+1)*np.array([width, height, width, height])/2).astype(int)
		bounds.append((left, bottom, right, top))
		masks.append((x >= left) & (x < right) & (y >= height-top) & (y < height-bottom))
		ndc = np.array([(left, bottom), (right, bottom), (right, top), (left, bottom), (right, top), (left, top)])*2/np.array(size)-1
		vertices = np.zeros(6, dtype=VERTEX_DTYPE)
		vertices["position"][:, :2] = ndc*(3+index*.3)*np.tan(np.deg2rad(55)/2)*np.array([width/height, 1])
		vertices["normal"] = (0, 0, 1)
		items.append(DrawInstance(MaterialMesh(vertices, shader), material={"base_color": (*color, alpha), "emission": tuple(color)}, position=(0, 0, -index*.3)))
	a = np.array(masks)*np.array(alphas)[:, None, None]
	reference = np.zeros((height, width, 3))
	t = np.ones((height, width))
	for opacity, color in zip(a, colors):
		reference += (t*opacity)[..., None]*color
		t *= 1-opacity
	s = a[1:].sum(axis=0)
	q = np.einsum("lhw,lc->hwc", a[1:], colors[1:])
	tail = (1-np.prod(1-a[1:], axis=0))[..., None]*q/np.maximum(s[..., None], 1e-8)
	banded = a[0, ..., None]*colors[0]+(1-a[0, ..., None])*tail
	return items, Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=.01, far=100), reference, banded, a[0], bounds[1], 1-np.prod(1-a[1:], axis=0)


def stages(values):
	"""Group timestamp labels without charging nearest/clear work to tail shading."""
	return dict(transmittance=values.get("2_transmittance", 0), allocation=values.get("3_banded_tail", 0),
		statistics=values.get("2_tail_statistics", 0),
		shading=values.get("4_shade_winner_0", 0), blur=sum(value for key, value in values.items() if key in ("5_kawase_down", "5_kawase_up")),
		resolve=values.get("5_kawase_finish", 0),
		composite=values.get("5_reconstruct_compose", 0), clear=values.get("2_clear_bands", 0),
		nearest=sum(value for key, value in values.items() if key in ("2_nearest_layer", "2_shade_nearest")))


def read_color(texture):
	width, height, layers = texture.size
	data = RenderContext.device.queue.read_texture({"texture": texture.handle}, {"bytes_per_row": width*8, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, np.float16).reshape(layers, height, width, 4).astype(np.float32)


def metrics(actual, reference, banded, bounds, pairs):
	height, width = actual.shape[:2]
	left, bottom, right, top = bounds
	rows = slice(max(height-top+8, height//3), min(height-bottom-8, height*2//3))
	radius = 2**pairs
	columns = np.arange(width)
	edge = np.minimum(abs(columns-left), abs(columns-right)) <= radius
	error = actual-reference
	band_error = actual-banded
	mean_error = error[rows].mean(axis=0)
	mean_band_error = band_error[rows].mean(axis=0)
	return dict(halo_sorted=float(np.abs(mean_error[edge]).max()), halo_banded=float(np.abs(mean_band_error[edge]).max()),
		noise_std=float((error[rows]-mean_error).std(axis=(0, 1)).max()), rmse_sorted=float(np.sqrt(np.mean(error**2))), edge_radius=radius)


def run(size=(1280, 720), frames=24, warmup=5, pairs=(1, 2, 3), samples=(1, 4), rounds=3, seed=20261112, variants=("baseline", "proxy", "subset", "single")):
	RenderContext.setup_graphics_backend(True)
	if "timestamp-query" not in RenderContext.adapter.features: raise RuntimeError("GPU timestamp-query support is required")
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance", "timestamp-query"])
	profiler = GpuProfiler(max_passes=64)
	original_texture = transparency_module.Texture
	def readable(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1: usage |= TextureUsage.COPY_SRC
		return original_texture(*args, usage=usage, **kwargs)
	transparency_module.Texture = readable
	try:
		items, camera, reference, banded, a1, bounds, tail_coverage = make_scene(size)
		renderer = MaterialRenderer(items, target_format="rgba8unorm", tail_allocation="banded", banded_proxy=False)
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
		configs = [(v, s, p) for v in variants for s in samples for p in pairs]
		measurements, quality, schedule = {}, {}, []
		rng = np.random.default_rng(seed)
		for trial in range(rounds):
			count = frames//rounds+int(trial < frames%rounds)
			if count == 0: continue
			for index in rng.permutation(len(configs)):
				variant, sample_count, blur_pairs = configs[index]
				renderer.set_banded_options(**VARIANTS[variant])
				renderer.set_sample_count(sample_count)
				renderer.set_kawase_passes(blur_pairs)
				key = (variant, sample_count, blur_pairs)
				schedule.append(dict(round=trial, config=variant, msaa=sample_count, pairs=blur_pairs))
				for frame in range(warmup+count):
					commands = profiler.wrap(RenderContext.commands("banded_profile"))
					renderer.record(commands, size, camera, output.color_attachment())
					RenderContext.submit(commands.finish())
					raw = profiler.read()
					if frame >= warmup: measurements.setdefault(key, []).append(dict(frame_ms=profiler.last_frame_ms, passes=raw, stages=stages(raw)))
				tail = read_color(renderer.transparency.reconstruction_color)[..., :3].mean(axis=0)
				if renderer.transparency.mixed_tail: tail *= tail_coverage[..., None]
				actual = (1-a1[..., None])*tail
				actual[..., 0] += a1
				quality[key] = metrics(actual, reference, banded, bounds, blur_pairs)
				counters = np.frombuffer(RenderContext.device.queue.read_buffer(renderer.transparency.band_counts.handle), np.uint32).reshape(size[1], size[0], renderer.transparency.tail_sample_count, 2)
				active = np.any(counters[..., 0] > 0, axis=-1)
				padded = np.pad(active, ((0, (-size[1])%8), (0, (-size[0])%8)))
				tiles = padded.reshape((size[1]+7)//8, 8, (size[0]+7)//8, 8).any(axis=(1, 3))
				quality[key].update(active_pixel_fraction=float(active.mean()), empty_tile_fraction=float(1-tiles.mean()))
		rows = []
		for (variant, sample_count, blur_pairs), values in measurements.items():
			rows.append(dict(config=variant, options=VARIANTS[variant], msaa=sample_count, pairs=blur_pairs, frame_ms=float(np.median([v["frame_ms"] for v in values])),
				stage_ms={name: float(np.median([v["stages"][name] for v in values])) for name in values[0]["stages"]}, quality=quality[(variant, sample_count, blur_pairs)], measurements=values))
		return dict(adapter=dict(RenderContext.adapter.info), size=size, frames=frames, warmup=warmup, rounds=rounds, seed=seed, schedule=schedule,
			rows=rows, timestamp_scope="Pass-boundary GPU queries, synchronized after each frame. Quality readback is untimed.")
	finally:
		transparency_module.Texture = original_texture
		profiler.close()
		RenderContext.cleanup()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--size", default="1280x720")
	parser.add_argument("--frames", type=int, default=24)
	parser.add_argument("--warmup", type=int, default=5)
	parser.add_argument("--rounds", type=int, default=3)
	parser.add_argument("--pairs", nargs="+", type=int, default=[1, 2, 3])
	parser.add_argument("--samples", nargs="+", type=int, choices=(1, 4), default=[1, 4])
	parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["baseline", "proxy", "subset", "single"])
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	size = tuple(map(int, args.size.lower().split("x")))
	if len(size) != 2 or min(size) < 64 or args.frames < 1 or args.rounds < 1 or args.warmup < 0 or any(p < 0 or p > 8 for p in args.pairs): parser.error("Expected size>=64, positive frames/rounds, nonnegative warmup and pairs 0-8")
	result = run(size, args.frames, args.warmup, args.pairs, args.samples, args.rounds, variants=args.variants)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
	for row in result["rows"]: print(row["config"], row["msaa"], row["pairs"], row["frame_ms"], row["stage_ms"], row["quality"])


if __name__ == "__main__": main()
