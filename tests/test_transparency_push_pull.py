"""Complemented selections and deterministic-confidence reconstruction."""
from dataclasses import replace

import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render
from numpyecs.TransparencyPass import BilateralFilter, ConfidenceFilter, TransparencyPass
from numpyecs.RenderContext import BindingInfo, BufferUsage, GpuBuffer, RenderContext, Texture, TextureUsage
from numpyecs.Math import Camera
from numpyecs.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from numpyecs.DefaultMaterial import DefaultMaterialShader
from numpyecs.TransparencyData import VERTEX_DTYPE


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.75, 1.0])
def test_complemented_marginals_and_minimum_overlap(factory, samples, alpha):
	renderer = factory(sample_count=samples, alpha=alpha, kernel=1, winner_coupling="complement")
	render(renderer, (256, 256))
	oit = renderer.transparency
	a = read(oit.winner_first)[..., 3]
	mixed = read(oit.winner_color)[..., 3]
	b = 2*mixed-a
	assert abs(a.mean()-alpha) < 0.015
	assert abs(b.mean()-alpha) < 0.015
	assert abs(np.mean((a > 0) & (b > 0))-max(0, 2*alpha-1)) < 0.015
	assert oit.noise_second is None
	assert oit.winner_first_payload is not None
	if alpha == 0.5: np.testing.assert_array_equal(mixed, 0.5)
	if alpha == 1: np.testing.assert_array_equal(mixed, 1)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("noise_mode", ["uv_blue_lerp", "anchored_blue"])
@pytest.mark.parametrize("ids", [((1, 1), (2, 2)), ((1, 2), (2, 1))])
def test_complemented_surface_primitive_pairs_do_not_alias(factory, samples, noise_mode, ids):
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block lighting
	return p.emission;
#endblock
''')
	items = []
	for layer, (surface, primitive) in enumerate(ids):
		# Only the requested triangle covers the image. Orthographic projection
		# gives both layers identical local/UV coordinates and derivatives.
		vertices = np.zeros(9, dtype=VERTEX_DTYPE)
		vertices["position"] = (100, 100, 0)
		vertices["position"][3*primitive:3*primitive+3] = [(-10, -10, 0), (10, -10, 0), (0, 10, 0)]
		vertices["normal"] = (0, 0, 1)
		vertices["uv"] = vertices["position"][:, :2]
		items.append(DrawInstance(MaterialMesh(vertices, shader), material={"base_color": (1, 1, 1, 0.5), "emission": (1, 0, 0) if layer == 0 else (0, 0, 1)}, position=(0, 0, -0.3*layer), surface_id=surface))
	renderer = MaterialRenderer(items, sample_count=samples, noise_mode=noise_mode, target_format="rgba8unorm", reconstruction="atrous", bilateral_filter=BilateralFilter(kernel_size=1), tail_allocation="stochastic")
	output = Texture((256, 256), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	commands = RenderContext.commands()
	renderer.record(commands, (256, 256), Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 2, perspective=False), output.color_attachment())
	RenderContext.submit(commands.finish())
	first = read(renderer.transparency.winner_first)
	mixed = read(renderer.transparency.winner_color)
	for stream in (first, 2*mixed-first):
		# Each stream must retain front/rear/empty probabilities 1/2, 1/4, 1/4.
		np.testing.assert_allclose(stream.mean(axis=(0, 1, 2)), (0.5, 0, 0.25, 0.75), atol=0.02)
	np.testing.assert_array_equal(mixed[..., 0], 0.5)
	renderer.transparency.set_bilateral_filter(BilateralFilter())
	commands = RenderContext.commands()
	renderer.record(commands, (256, 256), Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 2, perspective=False), output.color_attachment())
	RenderContext.submit(commands.finish())
	color = read(renderer.transparency.reconstruction_color)
	np.testing.assert_allclose(color[..., :3].mean(axis=(0, 1, 2)), (0.5, 0, 0.25), atol=0.02)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("levels", [2, 3])
def test_push_pull_clears_resizes_and_preserves_constant_mean(factory, samples, levels):
	renderer = factory(sample_count=samples, alpha=0.5, winner_coupling="complement", reconstruction="push_pull", confidence_filter=ConfidenceFilter(levels=levels))
	image = render(renderer, (37, 29))
	oit = renderer.transparency
	assert len(oit.pyramid) == levels
	color = read(oit.reconstruction_color)
	np.testing.assert_allclose(color[..., 0], 0.5, atol=0.001)
	np.testing.assert_array_equal(color[..., 1:3], 0)
	assert np.isfinite(image).all()
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0)
	render(renderer, (37, 29))
	np.testing.assert_array_equal(read(oit.reconstruction_color), 0)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0.5)
	render(renderer, (19, 13))
	assert oit.reconstruction_color.size == (19, 13, samples)
	renderer.set_confidence_filter(ConfidenceFilter(levels=5-levels, target_winners=8))
	render(renderer, (19, 13))
	assert len(oit.pyramid) == 5-levels
	before = render(renderer, (19, 13))
	oit.set_bilateral_filter(replace(oit.bilateral_filter, kernel_size=1))
	assert render(renderer, (19, 13)).shape == before.shape


@pytest.mark.parametrize("samples", [1, 4])
def test_same_payload_is_shaded_only_once(factory, samples, monkeypatch):
	# This diagnostic storage binding is outside the material parameter ABI.
	original_layout = TransparencyPass._pipeline_layout
	def counter_layout(oit, shader, *entries):
		if "shade_winner" in entries or "shade_mix" in entries:
			shader.info.bindings["shade_calls"] = BindingInfo("shade_calls", 4, 0, "storage", "read_write", "array<atomic<u32>>")
		return original_layout(oit, shader, *entries)
	monkeypatch.setattr(TransparencyPass, "_pipeline_layout", counter_layout)
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block helpers
{{ super() }}
#if STAGE == "shading"
@group(4) @binding(0) var<storage, read_write> shade_calls: array<atomic<u32>>;
#endif
#endblock
#block lighting
#if STAGE == "shading"
	atomicAdd(&shade_calls[0], 1u);
#endif
	return p.emission;
#endblock
''')
	vertices = np.zeros(6, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
	vertices["normal"] = (0, 0, 1)
	renderer = MaterialRenderer([DrawInstance(MaterialMesh(vertices, shader), material={"base_color": (1, 1, 1, 0.75), "emission": (1, 0, 0)})], sample_count=samples, target_format="rgba8unorm", bilateral_filter=BilateralFilter(kernel_size=1), tail_allocation="stochastic")
	counter = GpuBuffer(np.zeros(1, np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_SRC)
	renderer.geometry[1].shading_bindings = (renderer.transparency.shading_shader.bind_group(4, shade_calls=counter),)
	render(renderer, (37, 29))
	actual = np.frombuffer(RenderContext.device.queue.read_buffer(counter.handle), np.uint32)[0]
	assert actual == 37*29*samples


@pytest.mark.parametrize("samples", [1, 4])
def test_push_pull_depth_edges_force_switches_and_transmittance(factory, samples):
	renderer = factory(sample_count=samples, alpha=0.08, layers=2, winner_coupling="complement", reconstruction="push_pull", force_single_contributor=True)
	renderer.instances[0].scale = (0.07, 0.04, 1)
	renderer.instances[0].rotation = (0, 0.3, 0, (1-0.3**2)**0.5)
	image = render(renderer, (43, 35))
	assert np.isfinite(image).all()
	renderer.transparency.set_debug_view("transmittance")
	before = render(renderer, (43, 35))
	pyramid_shader = renderer.transparency.pyramid_shader
	renderer.set_reconstruction("fragment")
	np.testing.assert_array_equal(render(renderer, (43, 35)), before)
	renderer.set_reconstruction("push_pull")
	assert renderer.transparency.pyramid_shader is pyramid_shader
	renderer.set_winner_coupling("independent")
	np.testing.assert_array_equal(render(renderer, (43, 35)), before)
	renderer.set_winner_mix("fill")
	renderer.set_winner_count(4)
	renderer.transparency.set_debug_view("final")
	assert np.isfinite(render(renderer, (31, 27))).all()
	assert renderer.transparency.winner_first_payload is None


def test_pyramid_uses_expected_acceptance_not_actual_winner_count(factory):
	renderer = factory(alpha=0.1, winner_count=1, reconstruction="push_pull")
	render(renderer, (17, 13))
	oit = renderer.transparency
	first = read(oit.pyramid[0]["pair"], np.float32)
	# Fully covered guide, constant A_s. Interior cells each sum four A_s values.
	t = read(oit.transmittance)[..., 3]
	np.testing.assert_allclose(first[0, :-1, :-1, 3], 4*(1-t[0, 0, 0]), atol=1e-6)
	assert np.unique(read(oit.winner_color)[..., 3]).size == 2


@pytest.mark.parametrize("samples", [1, 4])
def test_confidence_reconstruction_uses_animated_material_frame_fields(factory, samples):
	shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block vertex
	var v = MaterialVertex(position, normal, uv);
	v.position.x += 0.1*sin(frame.time+position.y);
	return v;
#endblock
#block lighting
	return vec3f(frame.time, 0.0, 0.0);
#endblock
''')
	vertices = np.zeros(3, dtype=VERTEX_DTYPE)
	vertices["position"] = [(-100, -100, 0), (100, -100, 0), (0, 100, 0)]
	vertices["normal"] = (0, 0, 1)
	renderer = MaterialRenderer([DrawInstance(MaterialMesh(vertices, shader), material={"base_color": (1, 1, 1, 0.5)})], sample_count=samples, reconstruction="atrous", target_format="rgba8unorm", frame_fields={"time": ("f32", 0.0)}, tail_allocation="stochastic")
	output = Texture((37, 29), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	for seconds in (0.25, 0.75):
		commands = RenderContext.commands()
		renderer.record(commands, (37, 29), Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55), output.color_attachment(), frame={"time": seconds})
		RenderContext.submit(commands.finish())
		color = read(renderer.transparency.reconstruction_color)
		np.testing.assert_allclose(color[..., 0], 0.5*seconds, atol=0.001)
		np.testing.assert_array_equal(color[..., 1:3], 0)


def confidence_reference(pair, sigma, target, levels):
	"""CPU push-pull reference for a constant guide, including image boundaries."""
	pyramid = [pair]
	for _ in range(levels):
		samples, height, width, _ = pyramid[-1].shape
		padded = np.pad(pyramid[-1], ((0, 0), (0, height%2), (0, width%2), (0, 0)))
		pyramid.append(padded.reshape(samples, (height+1)//2, 2, (width+1)//2, 2, 4).sum(axis=(2, 4)))
	parent = None
	for pair in reversed(pyramid):
		_, height, width, _ = pair.shape
		local = np.zeros_like(pair)
		for dy in (-1, 0, 1):
			for dx in (-1, 0, 1):
				p = (slice(None), slice(max(0, -dy), min(height, height-dy)), slice(max(0, -dx), min(width, width-dx)))
				q = (slice(None), slice(max(0, dy), min(height, height+dy)), slice(max(0, dx), min(width, width+dx)))
				local[p] += np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*pair[q]
		mu = local[..., :3]/np.maximum(local[..., 3:], 1e-6)
		if parent is not None:
			y, x = np.indices((height, width))
			px, py = (x+0.5)/2-0.5, (y+0.5)/2-0.5
			cx, cy = np.floor(px+0.5).astype(int), np.floor(py+0.5).astype(int)
			inherited = np.zeros_like(mu)
			weight_sum = np.zeros((height, width))
			for dy in (-1, 0, 1):
				for dx in (-1, 0, 1):
					qx, qy = cx+dx, cy+dy
					valid = (qx >= 0) & (qy >= 0) & (qx < parent.shape[2]) & (qy < parent.shape[1])
					weight = np.exp(-((qx-px)**2+(qy-py)**2)/(2*sigma*sigma))*valid
					inherited += weight[None, ..., None]*parent[:, np.clip(qy, 0, parent.shape[1]-1), np.clip(qx, 0, parent.shape[2]-1)]
					weight_sum += weight
			confidence = np.clip(local[..., 3:]/target, 0, 1)
			mu = (1-confidence)*inherited/np.maximum(weight_sum[None, ..., None], 1e-6)+confidence*mu
		parent = mu
	return parent


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("levels", [2, 3])
def test_confidence_gather_matches_cpu_reference(factory, samples, levels):
	renderer = factory(sample_count=samples, alpha=0.1, winner_count=1, reconstruction="push_pull", confidence_filter=ConfidenceFilter(levels=levels))
	render(renderer, (37, 29))
	oit = renderer.transparency
	pair = read(oit.winner_color)
	# The constant 0.9 transmission rounds down to this half-float value at 1x/4x.
	acceptance = np.full_like(pair[..., 3:], 1-float(np.float16(0.9)))
	pair[..., 3:] = acceptance
	expected = acceptance*confidence_reference(pair, oit.bilateral_filter.spatial_sigma, 4, levels)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], expected, atol=0.0002)
	textures = [level["pair"] for level in oit.pyramid]
	renderer.set_confidence_filter(ConfidenceFilter(levels=levels, target_winners=8))
	assert textures == [level["pair"] for level in oit.pyramid]
	render(renderer, (37, 29))
	expected = acceptance*confidence_reference(pair, oit.bilateral_filter.spatial_sigma, 8, levels)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], expected, atol=0.0002)


@pytest.mark.parametrize("samples", [1, 4])
def test_confidence_filter_stops_at_separated_depth_boundaries(factory, samples):
	renderer = factory(sample_count=samples, layers=2, alpha=0.5, winner_coupling="complement", reconstruction="push_pull", confidence_filter=ConfidenceFilter(target_winners=32))
	renderer.instances[0].position = (-0.8, 0, 0)
	renderer.instances[1].position = (0.8, 0, -0.8)
	for item in renderer.instances: item.scale = (0.075, 0.2, 1)
	render(renderer, (83, 61))
	color = read(renderer.transparency.reconstruction_color)
	left, right = color[:, :, :27], color[:, :, 56:]
	assert left[..., 0].max() > 0.1 and right[..., 2].max() > 0.1
	assert left[..., 2].max() < 0.0001
	assert right[..., 0].max() < 0.0001


@pytest.mark.parametrize("options", [{"target_winners": 0}, {"target_winners": float("nan")}, {"levels": 1}, {"levels": True}, {"levels": 4}])
def test_confidence_filter_rejects_invalid_settings(options):
	with pytest.raises(ValueError): ConfidenceFilter(**options)
