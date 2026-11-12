import numpy as np
import pytest

from test_transparency_multiple_winners import factory, read, render


def matching_proxy(renderer):
	for item in renderer.instances:
		item.material["base_color"] = (*item.material["emission"], item.material["base_color"][3])


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("subset", [False, True])
def test_exact_proxy_correction_survives_sparse_shading(factory, samples, subset):
	renderer = factory(sample_count=samples, layers=2, alpha=.5, tail_allocation="banded", winner_coupling="complement", banded_proxy=True, banded_subset=subset)
	matching_proxy(renderer)
	render(renderer, (43, 29))
	oit = renderer.transparency
	pair = read(oit.winner_color)
	np.testing.assert_array_equal(pair[..., :3], 0)
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], np.broadcast_to((0, 0, .5), (samples, 29, 43, 3)), atol=.001)
	if subset:
		assert oit.band_payloads.size[:2] == (22, 15)
		assert .2 < np.mean(pair[..., 3] > 0) < .3
		assert np.all(pair[pair[..., 3] == 0] == 0)
	else: np.testing.assert_array_equal(pair[..., 3], .5)
	oit.set_debug_view("winner")
	image = render(renderer, (43, 29))
	np.testing.assert_allclose(image[..., :3], 188, atol=1)  # linear 0.5 means zero residual


def test_signed_residual_is_preserved_through_blur(factory):
	renderer = factory(layers=2, alpha=.1, tail_allocation="banded", winner_coupling="complement", banded_proxy=True)
	matching_proxy(renderer)
	render(renderer, (128, 128))
	residual = read(renderer.transparency.winner_color)[..., 2]
	assert residual.min() < -.09 and residual.max() > .39
	assert abs(residual.mean()) < .001
	assert abs(read(renderer.transparency.reconstruction_color)[..., 2].mean()-.1) < .003


@pytest.mark.parametrize("proxy", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_single_sample_tail_keeps_native_nearest_and_transmission(factory, proxy, subset):
	renderer = factory(sample_count=4, layers=2, alpha=.5, tail_allocation="banded", winner_coupling="complement", banded_proxy=proxy, banded_subset=subset, banded_tail_msaa=False)
	matching_proxy(renderer)
	image = render(renderer, (43, 29))
	oit = renderer.transparency
	assert oit.tail_sample_count == 1 and oit.mixed_tail
	assert oit.nearest_alpha.sample_count == 4 and oit.transmittance.sample_count == 4
	assert oit.band_payloads.size[2] == 2 and oit.winner_color.size[2] == 1
	assert oit.band_proxy.sample_count == 1
	# Mixed mode stores mean color; physical per-sample transmission is applied
	# in composition, rather than resolved before multiplication.
	np.testing.assert_allclose(read(oit.reconstruction_color)[..., :3], np.broadcast_to((0, 0, 1), (1, 29, 43, 3)), atol=.001)
	renderer.set_banded_options(tail_msaa=True)
	np.testing.assert_allclose(render(renderer, (43, 29)), image, atol=1)
	renderer.set_banded_options(tail_msaa=False)
	assert np.isfinite(render(renderer, (31, 19))).all()


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("tail_msaa", [False, True])
def test_thick_tail_uses_sum_normalization_and_sparse_confidence(factory, samples, tail_msaa):
	renderer = factory(sample_count=samples, layers=4, alpha=.8, tail_allocation="banded", winner_coupling="complement", banded_proxy=True, banded_subset=True, banded_tail_msaa=tail_msaa)
	matching_proxy(renderer)
	render(renderer, (32, 32))
	oit = renderer.transparency
	pair = read(oit.winner_color)
	assert np.count_nonzero(pair[..., 3]) == 32*32*oit.tail_sample_count//4
	np.testing.assert_array_equal(pair[pair[..., 3] == 0], 0)
	blue = read(oit.reconstruction_color)[..., 2]
	np.testing.assert_allclose(blue, 1 if oit.mixed_tail else .992, atol=.002)
	assert np.all(np.isfinite(blue))


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("tail_msaa", [False, True])
def test_proxy_point_cancellation_with_animated_custom_lighting(samples, tail_msaa):
	from numpyecs.DefaultMaterial import DefaultMaterialShader
	from numpyecs.MaterialRenderer import MaterialRenderer
	from numpyecs.RenderContext import RenderContext, Texture, TextureUsage
	from scenes.benchmarks.profile_banded_tail import make_scene
	from test_material_animation import ANIMATED_SOURCE
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	items, camera, *_ = make_scene((96, 64))
	items = [items[0], items[2]]
	shader = DefaultMaterialShader(transparent=True, source=ANIMATED_SOURCE)
	for item in items: item.mesh.shader = shader
	renderer = MaterialRenderer(items, tail_allocation="banded", kawase_passes=0, sample_count=samples,
		banded_proxy=False, banded_tail_msaa=tail_msaa, target_format="rgba8unorm", frame_fields={"time": ("f32", 0.0)})
	output = Texture((96, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	for seconds in (0., 1.):
		images = []
		for proxy in (False, True):
			renderer.set_banded_options(proxy=proxy)
			commands = RenderContext.commands()
			renderer.record(commands, (96, 64), camera, output.color_attachment(), frame={"time": seconds})
			RenderContext.submit(commands.finish())
			images.append(read(output, np.uint8))
		np.testing.assert_allclose(images[0], images[1], atol=1)
