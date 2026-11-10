"""GPU audits of same-ID layer wins and point -> pixel -> anchor motion.

Layer separation is in coarse rest-grid cells. Motion samples the nearest pixel
center and its first-order rest-space footprint on a curved deformed patch;
it is a diagnostic approximation, not a full rasterized visibility benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from numpyecs.Math import Camera
from numpyecs.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, Shader
from numpyecs.TextureProcessing import get_default_blue_noise


class NoiseProbe:
	def __init__(self):
		self.noise = get_default_blue_noise()
		self.shader = Shader(source='''
            #include "utils.shaderlib"
            #include "stochastic_noise.shaderlib"
            struct Probe { local: vec4f, dx: vec4f, dy: vec4f, pixel: vec4f, info: vec4u, }
            @group(0) @binding(0) var<storage, read> probes: array<Probe>;
            @group(0) @binding(1) var<storage, read_write> values: array<f32>;
            @group(0) @binding(2) var noise: texture_2d<f32>;
            @compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) id: vec3u) {
                if id.x >= arrayLength(&probes) { return; }
                let p = probes[id.x];
                let seed = hash_u32(p.info.x);
                if p.info.y == 7u {
                    values[id.x] = anchored_blue_threshold(noise, p.pixel.xy, p.local.xyz,
                        p.dx.xyz, p.dy.xyz, p.local.w, seed, p.dx.w, p.dy.w);
                } else {
                    values[id.x] = surface_noise_threshold(noise, noise, p.local.xyz,
                        p.dx.xyz, p.dy.xyz, vec3f(0.0, 0.0, 1.0), p.local.xy, p.dx.xy, p.dy.xy,
                        p.local.w, seed, p.info.y);
                }
            }
        ''')
		self.pipeline = ComputePipeline(self.shader)
		self.dtype = np.dtype([(name, "f4", 4) for name in ("local", "dx", "dy", "pixel")]+[("info", "u4", 4)])

	def evaluate(self, local, pixel, dx, dy, *, pixels=1.0, anchor_pixels=24.0, subdivisions=4, seed=17, mode=7):
		data = np.zeros(len(local), dtype=self.dtype)
		data["local"][:, :3], data["local"][:, 3] = local, pixels
		data["dx"][:, :3], data["dx"][:, 3] = dx, anchor_pixels
		data["dy"][:, :3], data["dy"][:, 3] = dy, subdivisions
		data["pixel"][:, :2] = pixel
		data["info"][:, :2] = (seed, mode)
		source = GpuBuffer(data, BufferUsage.STORAGE)
		output = GpuBuffer(None, BufferUsage.STORAGE | BufferUsage.COPY_SRC, dtype=np.float32, count=len(data))
		bindings = self.shader.bind_group(0, probes=source, values=output, noise=self.noise)
		commands = RenderContext.commands("anchored_noise_probe")
		with commands.compute_pass() as compute:
			compute.set_pipeline(self.pipeline)
			compute.set_bind_group(bindings)
			compute.dispatch((len(data)+63)//64)
		RenderContext.submit(commands.finish())
		result = np.frombuffer(RenderContext.device.queue.read_buffer(output.handle), dtype=np.float32).copy()
		source.close()
		output.close()
		return result


def layer_audit(probe, count=32768):
	local = np.random.default_rng(815).uniform(-30, 30, (count, 3))
	pixel = np.floor(local[:, :2]*24)+0.5
	local[:, :2] = pixel/24
	for subdivisions in (1, 4, 8, 16):
		front = probe.evaluate(local, pixel, (1/24, 0, 0), (0, 1/24, 0), subdivisions=subdivisions) < 0.5
		for separation in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
			back_local = local.copy()
			back_local[:, 2] += separation
			back = probe.evaluate(back_local, pixel, (1/24, 0, 0), (0, 1/24, 0), subdivisions=subdivisions) < 0.5
			yield dict(subdivisions=subdivisions, separation_cells=separation,
				front_win=float(front.mean()), rear_win=float(np.mean(~front & back)),
				union=float(np.mean(front | back)), target_rear=0.25)


def pixel_samples(local, camera, *, wind=0.0, size=512):
	"""Project material points, quantize to pixels, then evaluate the local footprint."""
	x, y = local[:, 0], local[:, 1]
	du = np.stack((np.ones(len(local)), np.zeros(len(local)), 0.345*np.cos(2.3*x)*np.cos(1.7*y)), axis=-1)
	dv = np.stack((np.zeros(len(local)), np.ones(len(local)), -0.255*np.sin(2.3*x)*np.sin(1.7*y)), axis=-1)
	world = local.copy()
	world[:, 2] += 0.12*np.sin(3*x+wind)
	world_du = du.copy()
	world_du[:, 2] += 0.36*np.cos(3*x+wind)
	vp = camera.view() @ camera.projection(1)
	def project(points):
		clip = np.column_stack((points, np.ones(len(points)))) @ vp
		return (clip[:, :2]/clip[:, 3:4]*[0.5, -0.5]+0.5)*size
	point_pixel = project(world)
	epsilon = 1e-4
	su = (project(world+epsilon*world_du)-point_pixel)/epsilon
	sv = (project(world+epsilon*dv)-point_pixel)/epsilon
	determinant = su[:, 0]*sv[:, 1]-sv[:, 0]*su[:, 1]
	dx = (du*sv[:, 1:2]-dv*su[:, 1:2])/determinant[:, None]
	dy = (-du*sv[:, 0:1]+dv*su[:, 0:1])/determinant[:, None]
	pixel = np.floor(point_pixel)+0.5
	offset = pixel-point_pixel
	return local+dx*offset[:, :1]+dy*offset[:, 1:2], pixel, dx, dy


def motion_audit(probe, count=4096, poses=24):
	xy = np.random.default_rng(816).uniform(-0.7, 0.7, (count, 2))
	local = np.column_stack((xy, 0.15*np.sin(2.3*xy[:, 0])*np.cos(1.7*xy[:, 1])))
	for motion in ("pan", "orbit", "dolly", "wind"):
		previous, changes = {}, {name: [] for name in ("object_hash", "uv_blue_lerp", "anchored_blue")}
		for t in np.linspace(0, 1, poses):
			if motion == "pan": position, target = (0.6*t, 0, 3), (0.6*t, 0, 0)
			elif motion == "orbit": position, target = (3*np.sin(0.6*t), 0, 3*np.cos(0.6*t)), (0, 0, 0)
			elif motion == "dolly": position, target = (0, 0, 2.5+1.5*t), (0, 0, 0)
			else: position, target = (0, 0, 3), (0, 0, 0)
			camera = Camera(position, target, (0, 1, 0), 50, near=0.05, far=100)
			points, pixels, dx, dy = pixel_samples(local, camera, wind=2*t if motion == "wind" else 0)
			for name, mode in (("object_hash", 0), ("uv_blue_lerp", 1), ("anchored_blue", 7)):
				accepted = probe.evaluate(points, pixels, dx, dy, mode=mode) < 0.5
				if name in previous: changes[name].append(float(np.mean(accepted != previous[name])))
				previous[name] = accepted
		for name, values in changes.items():
			yield dict(motion=motion, mode=name, mean_flip_percent=100*float(np.mean(values)), poses=poses, points=count)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--layer-samples", type=int, default=32768)
	parser.add_argument("--motion-points", type=int, default=4096)
	parser.add_argument("--poses", type=int, default=24)
	args = parser.parse_args()
	if min(args.layer_samples, args.motion_points) < 1 or args.poses < 2: parser.error("Positive sample counts and at least two poses required")
	RenderContext.setup_graphics_backend(True)
	try:
		probe = NoiseProbe()
		result = dict(layers=list(layer_audit(probe, args.layer_samples)), motion=list(motion_audit(probe, args.motion_points, args.poses)))
		text = json.dumps(result, indent=2)
		if args.output: args.output.write_text(text+"\n", encoding="utf-8")
		else: print(text)
	finally: RenderContext.cleanup()


if __name__ == "__main__": main()
