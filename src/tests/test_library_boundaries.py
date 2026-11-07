import subprocess
import sys
from pathlib import Path


def test_cpu_library_imports_need_no_optional_dependencies_or_path_mutation(tmp_path):
    result = subprocess.run([sys.executable, "-c", '''
import sys
import importlib.abc
import numpy
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'wgpu', 'glfw', 'jinja2', 'pyrr', 'pygltflib', 'PIL', 'imgui_bundle', 'mypy_extensions'}:
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, BlockOptional())
before = list(sys.path)
import numpyecs
from numpyecs import ECS, Math, Assets, TransparencyData, DrawList, InstanceData
from numpyecs import draw_compiler, draw_layout, utils
from types import SimpleNamespace as NS
from numpyecs.__main__ import simulate
from numpyecs.Diagnostics import WatchTimer
with WatchTimer("cpu"): pass
assert simulate().shape == (2, 3)
assert sys.path == before
assert Math.Camera((0,0,3), (0,0,0), (0,1,0), 60).view().shape == (4,4)
grouping = DrawList.DrawList.build([10, 10], [3, 3])
geometry = NS(index_count=3, index_range=(4,3), vertex_range=(8,3))
meshes = {1: NS(mesh=geometry, box_center=numpy.zeros(3), box_extents=numpy.ones(3))}
layout = draw_compiler.compile_destinations(2, grouping.batches,
    {10: draw_layout.LodGroup((1,), ())}, meshes, {3: True})
assert layout.arrays['main_draw_cmd']['first_index'].tolist() == [4]
assert layout.main_count == 2 and layout.prepass_count == 2
'''], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_scene_imports_do_not_initialize_glfw_or_gpu():
    result = subprocess.run([sys.executable, "-c", '''
import glfw, wgpu
def forbidden(*args, **kwargs): raise AssertionError('graphics initialization during import')
glfw.init = forbidden
wgpu.gpu.request_adapter_sync = forbidden
from scenes import transparency_example, draw_batches_example, direct_draw_example, prototype
assert all(callable(scene.run_scene) for scene in (transparency_example, draw_batches_example, direct_draw_example, prototype))
'''], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
