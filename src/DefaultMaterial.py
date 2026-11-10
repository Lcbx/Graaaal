"""Default lit material policy and its opaque/stochastic rendering contract.

MaterialLibrary does not know these parameters or their meaning. A different
shader may implement the same interface with an entirely different schema.
"""
from .Materials import MaterialInterface, MaterialShader, ShaderHook
import numpy as np


TRANSPARENCY_INTERFACE = MaterialInterface(
	address="metadata[id].yw",
	hooks=(
		ShaderHook("library_material", "Material", "", "surface_{i}(parameters_{i}(key.x, key.y))", "Material()"),
		ShaderHook("library_shade", "vec3f", "material: Material, uv: vec2f, position: vec3f, normal: vec3f, lod: f32",
			"shade_{i}(material, parameters_{i}(key.x, key.y), uv, position, normal, lod)", "vec3f(0.0)"),
		ShaderHook("library_opacity", "vec3f", "material: Material, alpha: vec3f, uv: vec2f, position: vec3f, lod: f32",
			"opacity_{i}(material, parameters_{i}(key.x, key.y), alpha, uv, position, lod)", "alpha"),
		ShaderHook("library_grazing", "vec3f", "", "grazing_{i}(parameters_{i}(key.x, key.y))", "vec3f(0.0)"),
	),
)



class DefaultMaterialShader(MaterialShader):
	def __init__(self, *, transparent=False, template=None, source=None, filepath=None, default_material=None):
		super().__init__(interface=TRANSPARENCY_INTERFACE, transparent=transparent,
			template=template if any(value is not None for value in (template, source, filepath)) else "default_material.shader",
			source=source, filepath=filepath,
			default_material={} if default_material is None else default_material)

	def source_template(self):
		return "default_material.shader"

	def validate_resolved(self, values):
		if not 0 <= np.float32(values["grazing_start"]) < np.float32(values["grazing_end"]) <= 90:
			raise ValueError("Grazing angles must satisfy 0 <= start < end <= 90")
		if values["alpha_mode"] not in (0, 1, 2) or (not self.transparent and values["alpha_mode"] == 2):
			raise ValueError("Expected alpha_mode 0/1/2; BLEND requires transparency")
