"""Joint validity, error spectra, same-ID overlap and GPU cost of dual winners.

Offscreen diagnostics; no TAA, history, or projection offsets. Timing uses GPU
timestamps when available. The synthetic scene is not a motion-quality verdict.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.Math import Camera
from numpyecs.RenderContext import RenderContext, Texture, TextureUsage, wgpu
from numpyecs.TransparencyData import VERTEX_DTYPE
from numpyecs.TransparencyPass import BilateralFilter, NOISE_MODES
from scenes.benchmarks.diagnose_anchored_noise import NoiseProbe, pixel_samples


def low_power(error):
	"""Mean-square power in 0 < spatial frequency <= 1/8 cycles/pixel."""
	n = error.shape[0]
	f = np.fft.fftfreq(n)
	radius = np.hypot(f[:, None], f[None, :])
	band = (radius > 0) & (radius <= 0.125)
	return float((np.abs(np.fft.fft2(error))**2)[band].sum()/error.size**2)


def stream_audit(probe, size=256):
	y, x = np.indices((size, size))
	pixel = np.stack((x+0.5, y+0.5), axis=-1).reshape(-1, 2)
	local = np.column_stack((pixel/64, np.full(len(pixel), 0.17)))
	for mode, name in enumerate(NOISE_MODES):
		a, b = [probe.evaluate(local, pixel, (1/64, 0, 0), (0, 1/64, 0), mode=mode, stream=s) for s in (0, 1)]
		for p in (0.1, 0.5, 0.9):
			aa, bb = a < p, b < p
			pa, pb = [low_power((v.astype(float)-p).reshape(size, size)) for v in (aa, bb)]
			mixed = (aa.astype(float)+bb)*0.5
			pm = low_power((mixed-p).reshape(size, size))
			yield dict(mode=name, p=p, marginal_a=float(aa.mean()), marginal_b=float(bb.mean()),
				neither=float(np.mean(~aa & ~bb)), exactly_one=float(np.mean(aa ^ bb)), both=float(np.mean(aa & bb)),
				expected=[(1-p)**2, 2*p*(1-p), p*p], low_power_a=pa, low_power_b=pb, low_power_mix=pm,
				low_power_ratio_to_independent=4*pm/(pa+pb) if pa+pb > 1e-24 else None)


def same_id_audit(probe, count=32768):
	local = np.random.default_rng(815).uniform(-30, 30, (count, 3))
	pixel = np.floor(local[:, :2]*24)+0.5
	local[:, :2] = pixel/24
	for separation in (0, 0.1, 0.25, 0.5, 1, 2):
		back = local.copy()
		back[:, 2] += separation
		for stream in (0, 1):
			a, b = [probe.evaluate(v, pixel, (1/24, 0, 0), (0, 1/24, 0), stream=stream) < 0.5 for v in (local, back)]
			yield dict(separation_cells=separation, stream=stream, rear_win=float(np.mean(~a & b)), independent_target=0.25)


def motion_audit(probe, count=4096, poses=16):
	"""Approximate surface point -> raster position -> anchor camera sweeps."""
	rng = np.random.default_rng(816)
	local = np.column_stack((rng.uniform(-0.7, 0.7, (count, 2)), np.zeros(count)))
	x, y = local[:, 0], local[:, 1]
	local[:, 2] = 0.15*np.sin(2.3*x)*np.cos(1.7*y)
	for motion in ("pan", "orbit", "dolly"):
		for mode, name in ((1, "uv_blue_lerp"), (7, "anchored_blue")):
			previous, delta = {}, {1: [], 2: []}
			for t in np.linspace(0, 1, poses):
				if motion == "pan": eye, target = (0.6*t, 0, 3), (0.6*t, 0, 0)
				elif motion == "orbit": eye, target = (3*np.sin(0.6*t), 0, 3*np.cos(0.6*t)), (0, 0, 0)
				else: eye, target = (0, 0, 2.5+1.5*t), (0, 0, 0)
				camera = Camera(eye, target, (0, 1, 0), 50, near=0.05, far=100)
				points, pixels, dx, dy = pixel_samples(local, camera)
				a, b = [probe.evaluate(points, pixels, dx, dy, mode=mode, stream=s) < 0.1 for s in (0, 1)]
				for winners, values in ((1, a.astype(float)), (2, (a.astype(float)+b)*0.5)):
					if winners in previous: delta[winners].append(float(np.mean((values-previous[winners])**2)))
					previous[winners] = values
			for winners in (1, 2):
				yield dict(motion=motion, mode=name, winners=winners, threshold_change_rms=float(np.sqrt(np.mean(delta[winners]))),
					probability=0.1, note="Unfiltered acceptance changes, not rendered motion quality")


def benchmark(size=256, frames=30, *, noise_mode="uv_blue_lerp", improvements=False):
	"""Two low-opacity colored sheets; includes all material rendering GPU passes."""
	shader = DefaultMaterialShader(transparent=True, source='''
        fn shade(material: Material, p: Params, uv: vec2f, position: vec3f, normal: vec3f, lod: f32) -> vec3f {
            return p.emission;
        }
    ''')
	v = np.zeros(6, dtype=VERTEX_DTYPE)
	v["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
	v["normal"] = (0, 0, 1)
	v["uv"] = [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]
	mesh = MaterialMesh(v, shader)
	items = [DrawInstance(mesh, material={"base_color": (1, 1, 1, 0.1), "emission": color},
		position=(0, 0, -0.3*i), surface_id=17+i) for i, color in enumerate(((1, 0, 0), (0, 0, 1)))]
	renderer = MaterialRenderer(items, target_format="rgba8unorm", noise_mode=noise_mode)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=0.01, far=100)
	output = Texture((size, size), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	device = RenderContext.device
	timestamps = "timestamp-query" in device.features
	queries = device.create_query_set(type="timestamp", count=4) if timestamps else None
	resolved = device.create_buffer(size=32, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC) if timestamps else None
	configs = [(w, s, k, "average", False, 2) for w, s, k in ((1, 1, 3), (2, 1, 3), (1, 1, 5), (1, 4, 3))]
	if improvements:
		configs += [(2, 1, 3, "fill", False, 2), (2, 1, 3, "fill", True, 2),
			(1, 1, 3, "average", False, 1), (1, 1, 3, "average", True, 1)]
	for winners, samples, kernel, mix, force, layers in configs:
		renderer.set_winner_count(winners)
		renderer.set_sample_count(samples)
		renderer.set_winner_mix(mix)
		renderer.set_force_single_contributor(force)
		renderer.instances = items[:layers]
		renderer.transparency.set_bilateral_filter(BilateralFilter(kernel_size=kernel))
		timings, wall = [], []
		for frame in range(frames+5):
			start = perf_counter()
			cmd = RenderContext.commands("multiple_winners_benchmark")
			if timestamps:
				cmd.handle.begin_compute_pass(timestamp_writes=dict(query_set=queries,
					beginning_of_pass_write_index=0, end_of_pass_write_index=1)).end()
			renderer.record(cmd, (size, size), camera, output.color_attachment(), time=0)
			if timestamps:
				cmd.handle.begin_compute_pass(timestamp_writes=dict(query_set=queries,
					beginning_of_pass_write_index=2, end_of_pass_write_index=3)).end()
				cmd.handle.resolve_query_set(queries, 0, 4, resolved, 0)
			RenderContext.submit(cmd.finish())
			if timestamps:
				values = np.frombuffer(device.queue.read_buffer(resolved), dtype=np.uint64)
				ms = float(values[2]-values[1])/1e6
			else:
				device.queue.on_submitted_work_done_sync()
				ms = None
			if frame >= 5:
				wall.append((perf_counter()-start)*1000)
				if ms is not None: timings.append(ms)
		data = device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size*4}, output.size)
		encoded = np.frombuffer(data, np.uint8).reshape(size, size, 4)[..., :3]/255
		linear = np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)
		hdr = np.array([0.1, 0, 0.09 if layers == 2 else 0])+0.9**layers*np.array([0.035, 0.05, 0.08])
		yield dict(winners=winners, msaa=samples, kernel=kernel, size=size, frames=frames,
			winner_mix=mix, force_single_contributor=force, layers=layers, noise_mode=noise_mode,
			gpu_ms_median=float(np.median(timings)) if timings else None, synchronized_wall_ms_median=float(np.median(wall)),
			linear_display_rmse=float(np.sqrt(np.mean((linear-hdr/(1+hdr))**2))))
	if queries is not None:
		queries.destroy()
		resolved.destroy()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--frames", type=int, default=30)
	parser.add_argument("--noise-mode", choices=NOISE_MODES)
	parser.add_argument("--improvements", action="store_true", help="Include force/fill comparisons; defaults to anchored blue")
	args = parser.parse_args()
	if args.size < 16 or args.frames < 1: parser.error("size >= 16 and frames >= 1 required")
	RenderContext.setup_graphics_backend(True)
	features = ["indirect-first-instance"]
	if "timestamp-query" in RenderContext.adapter.features: features.append("timestamp-query")
	RenderContext.setup_graphics_backend(True, required_features=features)
	try:
		probe = NoiseProbe()
		mode = args.noise_mode or ("anchored_blue" if args.improvements else "uv_blue_lerp")
		result = dict(adapter=RenderContext.adapter.info, streams=list(stream_audit(probe, args.size)),
			same_id_layers=list(same_id_audit(probe)), motion=list(motion_audit(probe)),
			benchmark=list(benchmark(args.size, args.frames, noise_mode=mode, improvements=args.improvements)))
		text = json.dumps(result, indent=2)
		if args.output: args.output.write_text(text+"\n", encoding="utf-8")
		else: print(text)
	finally: RenderContext.cleanup()


if __name__ == "__main__": main()
