"""Pre-implementation gate for conditional B selection using production fields.

Random rest/UV points, constant raw footprints, native-style sample positions and
the production MSAA seed rule. This is a threshold-field audit, not rasterization.
Shared-ID layers deliberately diagnose known within-stream overlap correlation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from graaal.RenderContext import RenderContext
from graaal.TransparencyPass import NOISE_MODES
from scenes.benchmarks.diagnose_anchored_noise import NoiseProbe


def audit(count=1048576, attempts=(2,), modes=NOISE_MODES):
	probe = NoiseProbe()
	rng = np.random.default_rng(11011)
	local = rng.uniform(-100, 100, (count, 3))
	positions = np.array([[0.375, 0.125], [0.875, 0.375], [0.125, 0.625], [0.625, 0.875]])
	for samples in (1, 4):
		sample = np.arange(count, dtype=np.uint32) % samples if samples == 4 else None
		pixel = np.floor(local[:, :2]*24)+(positions[sample] if sample is not None else 0.5)
		for shared in (False, True):
			for mode, name in enumerate(NOISE_MODES):
				if name not in modes: continue
				# Evaluate four layers once, then reuse their fields across opacities.
				fields = np.empty((max(attempts), 4, count), np.float32)
				for stream in range(max(attempts)):
					for layer in range(4):
						point = local.copy()
						point[:, 2] += layer*0.1
						fields[stream, layer] = probe.evaluate(point, pixel, (1/24, 0, 0), (0, 1/24, 0),
							mode=mode, stream=stream, seed=17 if shared else 17+layer, sample=sample, pixels=1/samples**0.5)
				for layers, p in ((1, 0.3), (2, 0.5), (3, 0.1), (4, 0.3)):
					winners = []
					for stream in range(max(attempts)):
						accepted = fields[stream, :layers] < p
						winners.append(np.where(accepted.any(axis=0), accepted.argmax(axis=0), layers))
					expected = np.array([p*(1-p)**i for i in range(layers)]+[(1-p)**layers])
					t = expected[-1]
					filled = np.full(count, layers)
					observed_streams = []
					for attempt, winner in enumerate(winners, 1):
						filled = np.where(filled < layers, filled, winner)
						observed_streams.append(np.bincount(winner, minlength=layers+1)/count)
						if attempt not in attempts: continue
						fill_expected = np.r_[expected[:-1]*sum(t**i for i in range(attempt)), t**attempt]
						observed = np.bincount(filled, minlength=layers+1)/count
						z = [np.abs(actual-expected)/np.sqrt(np.maximum(expected*(1-expected), 1/count)/count) for actual in observed_streams]
						z.append(np.abs(observed-fill_expected)/np.sqrt(np.maximum(fill_expected*(1-fill_expected), 1/count)/count))
						yield dict(mode=name, msaa=samples, shared_id=shared, layers=layers, p=p, count=count, attempts=attempt,
							single_expected=expected.tolist(), fill_expected=fill_expected.tolist(),
							streams=[v.tolist() for v in observed_streams], fill=observed.tolist(),
							max_z=float(np.max(z)), passes_5sigma=bool(np.max(z) <= 5))


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--samples", type=int, default=1048576)
	parser.add_argument("--attempts", nargs="+", type=int, default=[2])
	parser.add_argument("--modes", nargs="+", choices=NOISE_MODES, default=NOISE_MODES)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	if args.samples < 1 or min(args.attempts) < 1: parser.error("Positive sample and attempt counts required")
	RenderContext.setup_graphics_backend(True)
	try:
		rows = list(audit(args.samples, args.attempts, args.modes))
		passed = [name for name in args.modes if all(r["passes_5sigma"] for r in rows if r["mode"] == name and not r["shared_id"])]
		args.output.write_text(json.dumps(dict(samples=args.samples, passing_distinct_id_modes=passed, rows=rows), indent=2)+"\n", encoding="utf-8")
		print("Passing distinct-ID modes:", passed)
	finally: RenderContext.cleanup()


if __name__ == "__main__": main()
