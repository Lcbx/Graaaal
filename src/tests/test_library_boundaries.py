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
from numpyecs.__main__ import simulate
from numpyecs.Diagnostics import WatchTimer
with WatchTimer("cpu"): pass
assert simulate().shape == (2, 3)
assert sys.path == before
assert Math.Camera((0,0,3), (0,0,0), (0,1,0), 60).view().shape == (4,4)
'''], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_scene_imports_do_not_initialize_glfw_or_gpu():
    result = subprocess.run([sys.executable, "-c", '''
import glfw, wgpu
def forbidden(*args, **kwargs): raise AssertionError('graphics initialization during import')
glfw.init = forbidden
wgpu.gpu.request_adapter_sync = forbidden
from scenes import transparency_example, drawbatches_example, direct_draw_example, prototype
assert all(callable(scene.run_scene) for scene in (transparency_example, drawbatches_example, direct_draw_example, prototype))
'''], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
