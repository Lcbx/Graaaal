"""Optional ECS-to-array drawing adapter; application schemas remain caller-owned."""

import numpy as np


class EcsDrawAdapter:
    def __init__(self, draws, mesh_refs, *, lod_field="lod_id", shader_field="shader_id"):
        self.draws, self.mesh_refs = draws, mesh_refs
        self.lod_field, self.shader_field = lod_field, shader_field
        self._token, self._entities, self._selection = None, None, None
        self._order = np.empty(0, dtype=np.intp)

    def sync(self, entities):
        entities = np.asarray(entities)
        token = self.mesh_refs.layout_version, self.mesh_refs.version(self.lod_field, self.shader_field)
        changed = token != self._token or not np.array_equal(entities, self._entities)
        if changed:
            selection = self.mesh_refs[entities]
            labels = self.mesh_refs.owner_ids(selection)
            self.draws.sync(selection[self.lod_field], selection[self.shader_field], labels=labels)
            # A retained checked selection also expires after layout-only changes.
            self._selection, self._order = selection, self.draws.instance_order.copy()
            self._entities, self._token = entities.copy(), token
            self.draws.invalidate_instances()
        self.draws.sync_resources()
        return changed

    def mesh_field(self, name):
        if self._selection is None: raise RuntimeError("Synchronize the ECS adapter first")
        return self._selection[name][self._order]
