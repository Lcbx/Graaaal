"""Run the banded-tail guide's offline decision gate with the packaged noise tile."""

import argparse
import json
from pathlib import Path

import numpy as np

from graaal.Assets import load_blue_noise


def uniform_tile(tile):
	"""Map distinct tile values to uniform rank midpoints without moving texels."""
	tile = np.asarray(tile)
	if tile.ndim != 2 or not np.isfinite(tile).all(): raise ValueError("Expected a finite 2D noise tile")
	_, ranks, counts = np.unique(tile, return_inverse=True, return_counts=True)
	if np.any(counts != 1): raise ValueError("Expected distinct noise ranks")
	return (ranks.reshape(tile.shape)+0.5)/tile.size


def select(noise, probabilities, *, banded, shifts=None):
	"""Return one fragment index per threshold; -1 represents an empty selection."""
	p = np.asarray(probabilities, dtype=np.float64)
	if p.ndim != 1 or not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
		raise ValueError("Expected nonempty probabilities in [0, 1]")
	if banded:
		edges = np.concatenate(([0.0], np.cumsum(p)))
		winner = np.searchsorted(edges, noise*max(1.0, edges[-1]), side="right")-1
		return np.where(winner < len(p), winner, -1)
	if shifts is None or np.shape(shifts) != (len(p), 2): raise ValueError("Expected one integer tile shift per fragment")
	winner = np.full(noise.shape, -1, dtype=np.int32)
	for i in range(len(p)-1, -1, -1):
		winner[np.roll(noise, shifts[i], axis=(0, 1)) < p[i]] = i
	return winner


def filter3(image, spatial_sigma=None):
	"""Apply a wrapped 3x3 box or spatial Gaussian to a constant-guide image."""
	weights = np.ones(3) if spatial_sigma is None else np.exp(-np.arange(-1, 2)**2/(2*spatial_sigma**2))
	weights /= weights.sum()
	return sum(weights[i+1]*weights[j+1]*np.roll(image, (i, j), axis=(0, 1)) for i in (-1, 0, 1) for j in (-1, 0, 1))


def estimate(noise, probabilities, colors, *, banded, shifts=None, spatial_sigma=None):
	"""Return filtered complementary-pair tail radiance using deterministic coverage."""
	p = np.asarray(probabilities, dtype=np.float64)
	colors = np.asarray(colors, dtype=np.float64)
	numerator = np.zeros((*noise.shape, 3))
	for threshold in (noise, 1-noise):
		winner = select(threshold, p, banded=banded, shifts=shifts)
		numerator += 0.5*np.where((winner >= 0)[..., None], colors[np.maximum(winner, 0)], 0)
	coverage = 1-np.prod(1-p)
	denominator = min(1.0, p.sum()) if banded else coverage
	return filter3(numerator, spatial_sigma)*coverage/max(denominator, 1e-12)


def error_stats(image, reference):
	"""Measure maximum-channel bias, spatial noise, RMSE and low-frequency power."""
	error = image-reference
	spectrum = np.abs(np.fft.fft2(error, axes=(0, 1)))**2/np.prod(error.shape[:2])**2
	fy, fx = np.meshgrid(np.fft.fftfreq(error.shape[0]), np.fft.fftfreq(error.shape[1]), indexing="ij")
	band = (fx*fx+fy*fy > 0) & (fx*fx+fy*fy <= 0.125**2)
	return dict(bias=float(np.abs(error.mean(axis=(0, 1))).max()), noise=float(error.std(axis=(0, 1)).max()),
		rmse=float(np.sqrt(np.mean(error**2))), low_frequency_power=float(spectrum[band].sum(axis=0).max()))


def sweep(trials=64, seed=2026):
	"""Compare 2-6 equal-opacity layers, random colors and shifts, with both filters."""
	if type(trials) is not int or trials < 1: raise ValueError("trials must be positive")
	noise = uniform_tile(load_blue_noise())
	rng = np.random.default_rng(seed)
	rows = []
	for layers in range(2, 7):
		for alpha in (0.03, 0.05, 0.1, 0.15, 0.2):
			p = np.full(layers, alpha)
			coverage = 1-np.prod(1-p)
			weights = p*np.cumprod(np.concatenate(([1.0], 1-p[:-1])))
			measurements = {name: {mode: [] for mode in ("nearest", "banded")} for name in ("box", "gaussian")}
			predicted_bias = []
			for _ in range(trials):
				colors = rng.random((layers, 3))
				shifts = rng.integers(0, noise.shape, size=(layers, 2))
				reference = weights@colors
				predicted_bias.append(float(np.abs(coverage*(p@colors)/p.sum()-reference).max()))
				for name, sigma in (("box", None), ("gaussian", 0.9)):
					for mode in ("nearest", "banded"):
						image = estimate(noise, p, colors, banded=mode == "banded", shifts=shifts, spatial_sigma=sigma)
						measurements[name][mode].append(error_stats(image, reference))
			for name, modes in measurements.items():
				means = {mode: {key: float(np.mean([m[key] for m in stats])) for key in stats[0]} for mode, stats in modes.items()}
				ratio = means["nearest"]["noise"]/max(means["banded"]["noise"], 1e-12)
				rows.append(dict(layers=layers, alpha=alpha, filter=name, **means, noise_reduction=ratio,
					predicted_band_bias=float(np.mean(predicted_bias)), passes_noise_gate=ratio >= 2))
	flips = []
	for front, late in ((0.03, 0.03), (0.1, 0.05), (0.1, 0.1), (0.2, 0.1)):
		before = noise < late
		after = (noise >= front) & (noise < front+late)
		flips.append(dict(entering_alpha=front, existing_alpha=late, fraction=float(np.mean(before != after)), expected=2*min(front, late)))
	return dict(seed=seed, trials=trials, tile_shape=list(noise.shape), histogram="uniform rank midpoints", pair="complement",
		noise_gate=2.0, rows=rows, layer_entry_flips=flips,
		limitations=["Constant flat guides, screen-aligned tile shifts; no surface CDF or footprint distortion.",
			"Box matches the guide; Gaussian matches spatial_sigma=0.9 with constant guides.",
			"Offline results exclude WebGPU atomic scheduling, motion, MSAA and GPU performance."])


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--trials", type=int, default=64)
	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--output", type=Path, default=Path("scenes/benchmarks/results/banded_tail_gate.json"))
	args = parser.parse_args()
	result = sweep(args.trials, args.seed)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
	for name in ("box", "gaussian"):
		rows = [r for r in result["rows"] if r["filter"] == name]
		faint = [r["noise_reduction"] for r in rows if r["alpha"] <= 0.1]
		print(f"{name}: faint-tail noise reduction {min(faint):.2f}-{max(faint):.2f}x; {sum(r['passes_noise_gate'] for r in rows)}/{len(rows)} cases reach 2x")
	print(f"Saved {args.output}")


if __name__ == "__main__":
	main()
