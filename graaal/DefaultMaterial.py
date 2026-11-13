"""Default lit material policy and its opaque/stochastic rendering contract.

MaterialLibrary does not know these parameters or their meaning. A different
shader may implement the same interface with an entirely different schema.
"""
from .Materials import MaterialShader
import numpy as np


class DefaultMaterialShader(MaterialShader):
	def __init__(self, *, transparent=False, template=None, source=None, filepath=None, default_material=None):
		super().__init__(transparent=transparent,
			template=template if any(value is not None for value in (template, source, filepath)) else "default_material.shader",
			source=source, filepath=filepath,
			default_material={} if default_material is None else default_material)

	def is_blended(self, material=None, overrides=None):
		"""Route alpha_mode=2 to transparency; OPAQUE and MASK write opaque depth."""
		mode = self.default_material.get("alpha_mode", self.parameters["alpha_mode"].default)
		if material is not None: mode = material.get("alpha_mode", mode)
		if overrides is not None: mode = overrides.get("alpha_mode", mode)
		return mode == 2

	def validate_resolved(self, values):
		if not 0 <= np.float32(values["grazing_start"]) < np.float32(values["grazing_end"]) <= 90:
			raise ValueError("Grazing angles must satisfy 0 <= start < end <= 90")
		if values["alpha_mode"] not in (0, 1, 2) or (not self.transparent and values["alpha_mode"] == 2):
			raise ValueError("Expected alpha_mode 0/1/2; BLEND requires transparency")
