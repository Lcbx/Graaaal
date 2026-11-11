"""Measure screen-space threshold autocorrelation on an orthographic tilted plane.

No rasterization, fading, shading, reconstruction or temporal sampling is
involved. The anchored mode uses the production first-order anchor on this plane.
Oversampling measures subpixel spatial structure;
it is not extra MSAA. Run from the repository root with rendering dependencies.
"""
import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

from numpyecs.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, Shader
from numpyecs.TextureProcessing import get_default_blue_noise, build_correlated_noise_hierarchy
from numpyecs.TransparencyPass import NOISE_MODES


def autocorrelation_width(image, axis, oversample):
	"""Full width at half maximum of the mean axis autocorrelation, in pixels."""
	centered = image.astype(np.float64)-image.mean()
	spectrum = np.fft.rfft(centered, axis=axis)
	power = np.mean(np.abs(spectrum)**2, axis=1-axis)
	corr = np.fft.irfft(power, n=image.shape[axis])
	if corr[0] <= 0: raise ValueError("Constant threshold image has no autocorrelation width")
	corr /= corr[0]
	crossings = np.flatnonzero(corr[1:len(corr)//2] <= 0.5)+1
	if not len(crossings): return float("nan")
	i = int(crossings[0])
	lag = i-1+(corr[i-1]-0.5)/(corr[i-1]-corr[i])
	return 2*lag/oversample


def measure(*, mode="uv_blue_lerp", screen_size=128, oversample=8, pixels=1.0, samples=1, images=None,
	anchor_pixels=24.0, subdivisions=4):
	RenderContext.setup_graphics_backend(True)
	noise = get_default_blue_noise()
	hierarchy = build_correlated_noise_hierarchy(noise) if mode == "uv_blue_correlated" else noise
	shader = Shader(source='''
        #include "utils.shaderlib"
        #include "stochastic_noise.shaderlib"
        <Settings> shape: vec4u;
        <Settings> derivatives: vec4f;
        <Settings> values: vec4f;
        <Settings> anchor: vec4f;
        uniform<0, 0> settings: Settings;
        storage<0, 1, read_write> result: array<f32>;
        {{ binding('noise', 'texture_2d<f32>', 0, 2) }}
        {{ binding('hierarchy', 'texture_2d<f32>', 0, 3) }}
        @compute @workgroup_size(8, 8)
        fn evaluate(@builtin(global_invocation_id) id: vec3u) {
            if any(id.xy >= settings.shape.xy) { return; }
            let pixel = (vec2f(id.xy)+0.5)/settings.values.z;
            let dx = settings.derivatives.xy;
            let dy = settings.derivatives.zw;
            let uv = pixel.x*dx+pixel.y*dy+vec2f(0.137, 0.231);
            if settings.shape.w == 7u {
                result[id.y*settings.shape.x+id.x] = anchored_blue_threshold(noise, pixel, vec3f(uv, 0.0),
                    vec3f(dx, 0.0), vec3f(dy, 0.0), settings.values.x, hash_u32(settings.shape.z),
                    settings.anchor.x, settings.anchor.y);
                return;
            }
            result[id.y*settings.shape.x+id.x] = surface_noise_threshold_balanced(noise, hierarchy,
                vec3f(uv, 0.0), vec3f(dx, 0.0), vec3f(dy, 0.0), vec3f(0.0, 0.0, 1.0),
                uv, dx, dy, settings.values.x, hash_u32(settings.shape.z), settings.shape.w, settings.values.y);
        }
	''', features=("NOISE_FOOTPRINT_BALANCE",))
	settings = shader.uniform_buffer("settings")
	side = screen_size*oversample
	output = GpuBuffer(None, BufferUsage.STORAGE | BufferUsage.COPY_SRC, dtype=np.float32, count=side*side)
	bindings = shader.bind_group(0, settings=settings, result=output, noise=noise, hierarchy=hierarchy)
	pipeline = ComputePipeline(shader, entry="evaluate")
	settings.content["shape"] = (side, side, 17, NOISE_MODES.index(mode))
	settings.content["anchor"] = (anchor_pixels, subdivisions, 0, 0)
	for angle in (0, 60, 75, 83):
		cosine = np.cos(np.deg2rad(angle))
		settings.content["derivatives"] = (1/64, 0, 0, 1/(64*cosine))
		for balance in ((0.0,) if mode == "anchored_blue" else (0.0, 0.5, 1.0)):
			settings.content["values"] = (pixels/np.sqrt(samples), balance, oversample, 0)
			settings.upload_changed()
			commands = RenderContext.commands("noise_tilt_diagnostic")
			with commands.compute_pass() as compute:
				compute.set_pipeline(pipeline)
				compute.set_bind_group(bindings)
				compute.dispatch((side+7)//8, (side+7)//8)
			RenderContext.submit(commands.finish())
			image = np.frombuffer(RenderContext.device.queue.read_buffer(output.handle), dtype=np.float32).reshape(side, side)
			x, y = (autocorrelation_width(image, axis, oversample) for axis in (1, 0))
			if images is not None: images[f"tilt_{angle}_balance_{balance:g}"] = image.copy()
			yield dict(mode=mode, tilt=angle, balance=balance, samples=samples, effective_pixels=pixels/np.sqrt(samples),
				fwhm_x=x, fwhm_y=y, resolution_limited=min(x, y) < 2/oversample)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mode", choices=NOISE_MODES[1:5]+("anchored_blue",), default="uv_blue_lerp")
	parser.add_argument("--anchor-pixels", type=float, default=24.0)
	parser.add_argument("--subdivisions", type=int, default=4)
	parser.add_argument("--screen-size", type=int, default=128)
	parser.add_argument("--oversample", type=int, default=8)
	parser.add_argument("--pixels", type=float, default=1.0)
	parser.add_argument("--samples", type=int, choices=(1, 4), default=1)
	parser.add_argument("--output", type=Path, help="CSV output; defaults to stdout")
	parser.add_argument("--images", type=Path, help="Optional NPZ archive of threshold images")
	args = parser.parse_args()
	if args.screen_size < 8 or args.oversample < 1 or not np.isfinite(args.pixels) or args.pixels <= 0:
		parser.error("screen-size must be >=8; oversample and pixels must be positive")
	if not np.isfinite(args.anchor_pixels) or not 1 <= args.anchor_pixels <= 256 or not 1 <= args.subdivisions <= 32:
		parser.error("anchor-pixels must be in [1,256]; subdivisions in [1,32]")
	images = {} if args.images else None
	try:
		rows = list(measure(mode=args.mode, screen_size=args.screen_size, oversample=args.oversample,
			pixels=args.pixels, samples=args.samples, images=images, anchor_pixels=args.anchor_pixels, subdivisions=args.subdivisions))
		with args.output.open("w", newline="", encoding="utf-8") if args.output else nullcontext(sys.stdout) as stream:
			writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
			writer.writeheader()
			writer.writerows(rows)
		if args.images: np.savez_compressed(args.images, **images)
	finally:
		RenderContext.cleanup()


if __name__ == "__main__":
	main()
