"""Compare complemented noise and confidence-filter error against sorted layers.

Run as a module. Writes raw metrics and an RMSE plot, with no temporal history.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from numpyecs.Assets import DEFAULT_BLUE_NOISE_PATH, load_blue_noise
from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
from numpyecs.TransparencyData import VERTEX_DTYPE
from numpyecs.TransparencyPass import BilateralFilter, ConfidenceFilter
from scenes.benchmarks.diagnose_anchored_noise import NoiseProbe


def err_stats(a, p, b):
	"""Variance and low-frequency power of a two-threshold acceptance estimator."""
	error = 0.5*((a < p).astype(float)+(b < p).astype(float))-p
	spectrum = np.abs(np.fft.fft2(error))**2/error.size**2
	fy, fx = np.meshgrid(np.fft.fftfreq(error.shape[0]), np.fft.fftfreq(error.shape[1]), indexing="ij")
	band = (fx*fx+fy*fy > 0) & (fx*fx+fy*fy <= 0.125**2)
	return dict(mean_error=float(error.mean()), variance=float(error.var()), low_frequency_power=float(spectrum[band].sum()))


def noise_comparison(size):
	probe = NoiseProbe()
	data = np.ascontiguousarray(load_blue_noise(DEFAULT_BLUE_NOISE_PATH.with_name("64x64_second.npy")).T)
	probe.noise_second = Texture(data.shape[::-1], format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": probe.noise_second.handle}, data, {"bytes_per_row": data.shape[1]*4}, (*data.shape[::-1], 1))
	y, x = np.indices((size, size))
	pixel = np.stack((x+0.5, y+0.5), axis=-1).reshape(-1, 2)
	local = np.column_stack((pixel/64, np.full(len(pixel), 0.17)))
	rows = []
	for mode in (1, 7):
		a = probe.evaluate(local, pixel, (1/64, 0, 0), (0, 1/64, 0), mode=mode, complement=True).reshape(size, size)
		# Use the same A phase when comparing the transposed independent texture.
		original = probe.noise
		probe.noise = probe.noise_second
		b = probe.evaluate(local, pixel, (1/64, 0, 0), (0, 1/64, 0), mode=mode, complement=True).reshape(size, size)
		probe.noise = original
		for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
			rows.append(dict(mode=mode, alpha=p, complement=err_stats(a, p, 1-a), transposed_second=err_stats(a, p, b)))
	return rows


def read(texture, dtype=np.float16):
	width, height, layers = texture.size
	data = RenderContext.device.queue.read_texture({"texture": texture.handle}, {"bytes_per_row": width*4*np.dtype(dtype).itemsize, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype).reshape(layers, height, width, 4).astype(np.float32)


def layer_comparison(size, targets, reconstruction="atrous", support_seeds=0, exact_nearest=False):
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block lighting
	return p.emission;
#endblock
''')
	vertices = np.zeros(6, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
	vertices["normal"] = (0, 0, 1)
	colors = np.array([(1, 0, 0), (0, 1, 1), (0, 1, 0), (1, 0, 1), (0, 0, 1), (1, 1, 0), (1, 0.3, 0), (0.1, 0.2, 1)], np.float32)
	mesh = MaterialMesh(vertices, shader)
	items = [DrawInstance(mesh, material={"base_color": (1, 1, 1, 0.1), "emission": tuple(color)}, position=(0, 0, -0.03*i), surface_id=17+i) for i, color in enumerate(colors)]
	renderer = MaterialRenderer(items, target_format="rgba8unorm", bilateral_filter=BilateralFilter(kernel_size=3), reconstruction=reconstruction, exact_nearest=exact_nearest)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=0.01, far=100)
	output = Texture((size, size), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	rows = []
	for layers in (2, 4, 8):
		renderer.instances = items[:layers]
		for alpha in (0.05, 0.1, 0.15):
			for item in items: item.material["base_color"] = (1, 1, 1, alpha)
			reference = np.sum((alpha*(1-alpha)**np.arange(layers))[:, None]*colors[:layers], axis=0)
			for coupling in ("complement", "independent"):
				renderer.set_winner_coupling(coupling)
				for target in targets:
					renderer.set_confidence_filter(ConfidenceFilter(target_winners=target))
					commands = RenderContext.commands()
					renderer.record(commands, (size, size), camera, output.color_attachment())
					RenderContext.submit(commands.finish())
					actual = read(renderer.transparency.reconstruction_color)[0, ..., :3]
					if exact_nearest:
						a1 = read(renderer.transparency.nearest_alpha)[0, ..., :3]
						c1 = read(renderer.transparency.nearest_color)[0, ..., :3]
						actual = a1*c1+(1-a1)*actual
					error = actual-reference
					rows.append(dict(layers=layers, alpha=alpha, coupling=coupling, reconstruction=reconstruction, exact_nearest=exact_nearest, noise_mode=renderer.transparency.noise_mode, footprint_balance=renderer.transparency.noise_footprint_balance, target_winners=target, rmse=float(np.sqrt(np.mean(error**2))), mean_error=error.mean(axis=(0, 1)).tolist(), reference=reference.tolist()))
	support = None
	if support_seeds:
		# Force the final footprint, so its measured variance and D describe the
		# same kernel rather than a mixture of output levels. No temporal history.
		renderer.instances = items
		for item in items: item.material["base_color"] = (1, 1, 1, 0.1)
		renderer.set_winner_coupling("complement")
		renderer.set_confidence_filter(ConfidenceFilter(target_winners=1e6))
		raw, filtered = [], []
		rng = np.random.default_rng(771)
		for seed in range(support_seeds):
			base = int(rng.integers(1, 0x7fffff00))
			for index, item in enumerate(items): item.surface_id = base+index
			commands = RenderContext.commands()
			renderer.record(commands, (size, size), camera, output.color_attachment())
			RenderContext.submit(commands.finish())
			raw.append(read(renderer.transparency.winner_color)[0, ..., :3])
			filtered.append(read(renderer.transparency.reconstruction_color)[0, ..., :3])
		acceptance = 1-read(renderer.transparency.transmittance)[0, ..., 3]
		variance_raw = np.var(raw, axis=0, ddof=1).sum(axis=-1)
		variance_filtered = np.var(filtered, axis=0, ddof=1).sum(axis=-1)
		empirical = acceptance*variance_raw/np.maximum(variance_filtered, 1e-12)
		pair = renderer.transparency.pyramid[1]["pair"]
		count = read(pair, np.float32)[0, ..., 3]*256
		# Exclude image truncation; report a statistical diagnostic, not an
		# independence guarantee for correlated spatial blue noise.
		interior = (slice(16, -16), slice(16, -16))
		ratio = empirical[interior]/np.maximum(count[interior], 1e-12)
		support = dict(seeds=support_seeds, layers=8, alpha=0.1, final_footprint_only=True,
			empirical_to_D_percentiles=np.percentile(ratio, (10, 50, 90)).tolist(), fraction_empirical_below_D=float(np.mean(ratio < 1)),
			mean_raw_variance=float(variance_raw[interior].mean()), mean_filtered_variance=float(variance_filtered[interior].mean()))
	return rows, support


def plot(rows, path):
	import matplotlib
	matplotlib.use("Agg")
	from matplotlib import pyplot as plt
	figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
	for axis, layers in zip(axes, (2, 4, 8)):
		for alpha, color in zip((0.05, 0.1, 0.15), ("tab:blue", "tab:orange", "tab:green")):
			for coupling, style in (("complement", "-"), ("independent", "--")):
				series = [r for r in rows if r["layers"] == layers and r["alpha"] == alpha and r["coupling"] == coupling]
				axis.plot([r["target_winners"] for r in series], [r["rmse"] for r in series], style, marker="o", color=color, label=f"alpha={alpha:g}, {coupling}")
		axis.set(title=f"{layers} sorted layers", xlabel="Support budget D" if rows[0].get("reconstruction") == "atrous" else "Target expected winners")
		axis.grid(alpha=0.3)
	axes[0].set_ylabel("Linear premultiplied RGB RMSE")
	axes[-1].legend(fontsize=7)
	figure.tight_layout()
	figure.savefig(path, dpi=160)
	plt.close(figure)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--targets", nargs="+", type=float, default=(1, 2, 4, 8, 16))
	parser.add_argument("--reconstruction", choices=("atrous", "push_pull"), default="atrous")
	parser.add_argument("--exact-nearest", action="store_true", help="Shade the nearest fragment deterministically and filter only the tail")
	parser.add_argument("--support-seeds", type=int, default=0, help="Optional final-footprint variance audit; atrous only, at least two seeds")
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	if args.support_seeds and (args.support_seeds < 2 or args.size <= 32 or args.reconstruction != "atrous"): parser.error("Support audit requires atrous, size > 32 and at least two seeds")
	if args.size < 8 or not all(np.isfinite(t) and t > 0 for t in args.targets): parser.error("Positive targets and size >= 8 required")
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	try:
		# Readback is confined to this diagnostic.
		from numpyecs import TransparencyPass as transparency
		original = transparency.Texture
		transparency.Texture = lambda *a, usage, **kw: original(*a, usage=usage | (TextureUsage.COPY_SRC if kw.get("sample_count", 1) == 1 else 0), **kw)
		noise = noise_comparison(args.size)
		rows, support = layer_comparison(args.size, args.targets, args.reconstruction, args.support_seeds, args.exact_nearest)
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(dict(size=args.size, noise=noise, layers=rows, support=support), indent=2)+"\n", encoding="utf-8")
		plot(rows, args.output.with_suffix(".png"))
		print(f"Saved {len(rows)} layer cases and {len(noise)} spectral comparisons to {args.output}")
	finally: RenderContext.cleanup()


if __name__ == "__main__": main()
