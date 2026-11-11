"""CPU-only shader generation checks; GPU compilation is exercised by rendering tests."""
from pathlib import Path

import pytest

from numpyecs.Materials import MaterialLibrary
from numpyecs.RenderContext import Shader, ShaderSource
from scenes.material_example import make_instances


SHADERS = Path(__file__).resolve().parents[1] / "src" / "shaders"
GEOMETRY_HOOKS = ("library_vertex", "library_material", "library_opacity", "library_grazing", "library_shade")
SHADING_HOOKS = ("library_vertex", "library_material", "library_shade")


def load_shader(name, **params):
	loader = Shader.__new__(Shader)
	source, declarations = loader._render_source(None, str(SHADERS / name), None, None, params)
	return ShaderSource(source, declarations=declarations)


def test_material_library_only_emits_stage_hooks():
	instances, _, _ = make_instances()
	library = MaterialLibrary([instance.mesh.shader for instance in instances])
	geometry = library.source_for(GEOMETRY_HOOKS)
	shading = library.source_for(SHADING_HOOKS)
	geometry_info = load_shader("stochastic_geometry.shader", material_library=geometry)
	shading_info = load_shader("stochastic_shade.shader", material_library=shading)
	assert "opaque" in geometry_info.entry_points
	assert "library_shade" in geometry_info.functions
	assert "shade_material" in geometry_info.functions
	assert "library_grazing" not in shading_info.functions
	assert "library_opacity" not in shading_info.functions
	assert "shade_0" in geometry_info.functions
	assert "grazing_0" not in shading_info.functions
	assert "opacity_0" not in shading_info.functions
	assert "shade_0" in shading_info.functions
	assert "grazing_0" in geometry_info.functions
	assert "opacity_0" in geometry_info.functions
	assert "default_lighting_3" in geometry_info.functions
	assert "default_lighting_3" in shading_info.functions
	assert "shade_3" in geometry_info.functions
	assert "library_shade" in library.source  # full source remains available
	assert library.source_for(hook.name for hook in library.interface.hooks) == library.source
	with pytest.raises(ValueError, match="Unknown material hooks"):
		library.source_for(("nonexistent",))


@pytest.mark.parametrize("name,expected,missing", [
	("stochastic_compose.shader", "reconstruct_transparency", "reconstruct_pair"),
	("stochastic_filter.shader", "reconstruct_pair", "reconstruct_transparency"),
	("stochastic_profile.shader", "forced_at", "reduce_neighborhood"),
])
def test_reconstruction_helpers_are_stage_specific(name, expected, missing):
	params = {"srgb_surface": False, "sample_count": 1, "winner_count": 2,
		"winner_mix": "average", "compute_reconstruction": False, "filter_input": False}
	info = load_shader(name, **params)
	assert expected in info.functions
	assert missing not in info.functions


@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("pairs", (False, True))
def test_compute_reconstruction_selects_requested_result(count, pairs):
	info = load_shader("stochastic_reconstruct.shader", sample_count=count, compute_reconstruction=True,
		filter_pairs=pairs, tiled_reconstruction=True)
	assert ("reconstruct_pair" in info.functions) is pairs
	assert ("reconstruct_transparency" in info.functions) is not pairs


def test_utility_compatibility_include_still_exports_macros():
	path = SHADERS.parents[1] / "scenes" / "shaders" / "simple.shader"
	loader = Shader.__new__(Shader)
	source, declarations = loader._render_source(None, str(path), None, None, {})
	info = ShaderSource(source, declarations=declarations)
	assert "unpack_instance_transform" in info.functions
	assert "unpack_rgba8_srgb" in info.functions
	assert "hash_u32" in info.functions  # legacy include still provides all utilities


@pytest.mark.parametrize("sample_count,winner_count,winner_mix,force", [
	(1, 1, "average", False),
	(4, 2, "average", True),
	(1, 3, "fill", False),
	(4, 2, "fill", True),
])
def test_material_shader_variants_expand(sample_count, winner_count, winner_mix, force):
	instances, _, _ = make_instances()
	library = MaterialLibrary([instance.mesh.shader for instance in instances])
	base = {"sample_count": sample_count, "winner_count": min(winner_count, 2),
		"winner_mix": winner_mix, "force_single_contributor": force}
	geometry = load_shader("stochastic_geometry.shader", material_library=library.source_for(GEOMETRY_HOOKS), **base)
	shading = load_shader("stochastic_shade.shader", material_library=library.source_for(SHADING_HOOKS), **base)
	assert "select_winner" in geometry.entry_points
	assert "shade_winner" in shading.entry_points
	assert "library_shade" in geometry.functions
	assert "shade_material" in geometry.functions
	assert "library_opacity" not in shading.functions
	compose = load_shader("stochastic_compose.shader", srgb_surface=True, compute_reconstruction=False, **base)
	assert "compose" in compose.entry_points
	assert "linear_to_srgb" not in compose.functions
