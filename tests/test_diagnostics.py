"""CPU-only render diagnostics tests."""
import sys
from types import SimpleNamespace as NS

import pytest

from graaal.Diagnostics import get_draw_counts, get_shader_sources


@pytest.fixture
def renderer_types(monkeypatch):
	class DrawBatches: pass
	class MaterialRenderer: pass
	monkeypatch.setitem(sys.modules, "graaal.DrawBatches", NS(DrawBatches=DrawBatches))
	monkeypatch.setitem(sys.modules, "graaal.MaterialRenderer", NS(MaterialRenderer=MaterialRenderer))
	return DrawBatches, MaterialRenderer


def shader(source):
	return NS(info=NS(source=source))


def pass_(source):
	return NS(pipeline=NS(shader=shader(source)))


def draws(renderer_types):
	DrawBatches, _ = renderer_types
	renderer = DrawBatches()
	renderer.draw_batches = [NS(mesh_id=10, shader_id=0), NS(mesh_id=11, shader_id=0),
		NS(mesh_id=None, shader_id=0), NS(mesh_id=12, shader_id=1)]
	renderer.shaders = {
		0: NS(passes={"main": pass_("opaque"), "prepass": pass_("opaque")}),
		1: NS(passes={"main": pass_("transparent"), "transmittance": pass_("transparent"),
			"guide": pass_("transparent"), "second": pass_("transparent")}),
	}
	renderer._cull_shader = shader("cull")
	return renderer


def material_renderer(renderer_types, **options):
	_, MaterialRenderer = renderer_types
	settings = dict(winner_count=2, sample_count=4, reconstruction_passes=3, reconstruction="fragment",
		debug_view="final", bilateral_filter=NS(kernel_size=5), shading_shader=shader("shade"),
		compose_shader=shader("compose"), pair_shader=shader("filter"), reconstruct_shader=None)
	settings.update(options)
	renderer = MaterialRenderer()
	renderer.draws = draws(renderer_types)
	renderer.draws.shaders[0].passes = {"opaque": pass_("opaque")}
	renderer.transparency = NS(**settings)
	return renderer


def test_draw_batches_counts(renderer_types):
	renderer = draws(renderer_types)
	assert get_draw_counts(renderer) == (3, 3)
	assert get_draw_counts(renderer, passes=("prepass", "main")) == (3, 5)
	assert get_draw_counts(renderer, passes="guide") == (3, 1)
	assert get_draw_counts(renderer, passes=()) == (3, 0)


def test_material_renderer_counts(renderer_types):
	renderer = material_renderer(renderer_types)
	# Two opaque, three transparent passes for one transparent batch,
	# one composition draw, and two filtering stages with four samples each.
	assert get_draw_counts(renderer) == (3, 15)
	assert get_draw_counts(renderer, passes="opaque") == (3, 2)
	assert get_draw_counts(renderer, passes="transmittance") == (3, 1)
	assert get_draw_counts(material_renderer(renderer_types, bilateral_filter=NS(kernel_size=1))) == (3, 6)
	assert get_draw_counts(material_renderer(renderer_types, reconstruction="compute")) == (3, 7)
	assert get_draw_counts(material_renderer(renderer_types, debug_view="transmittance")) == (3, 7)


def test_sources_deduplicated_and_active(renderer_types):
	assert get_shader_sources(draws(renderer_types)) == ["cull", "opaque", "transparent"]
	assert get_shader_sources(material_renderer(renderer_types)) == ["cull", "opaque", "transparent", "shade", "compose", "filter"]
	assert get_shader_sources(material_renderer(renderer_types, reconstruct_shader=shader("reconstruct")))[-1] == "reconstruct"


def test_unsupported_input(renderer_types):
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_draw_counts(object())
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_shader_sources(object())
	# Attribute-compatible objects are not accepted unless they have the right type.
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_draw_counts(NS(draw_batches=[], shaders={}))
