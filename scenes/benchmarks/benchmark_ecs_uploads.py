"""Measure direct-draw CPU packing and buffer upload submission in a bounded scene.

Run from the repository root. Upload durations are CPU wall time around upload(),
not GPU completion time. Includes scene initialization and every buffer type;
the instance-buffer record isolates the movement upload workload.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('frames must be positive')

    from RenderContext import RenderContext, GpuBuffer
    from scenes.direct_draw_example import CubeInstances, run_scene

    original_loop, original_upload, original_update = RenderContext.window_loop, GpuBuffer.upload, CubeInstances.update
    records = {}
    packing = {'calls': 0, 'changed': 0, 'cpu_ms': 0.0}
    frame = 0

    def window_loop():
        nonlocal frame
        if not original_loop() or frame >= args.frames:
            return False
        frame += 1
        return True

    def upload(buffer, *positional, **keywords):
        start = time.perf_counter_ns()
        result = original_upload(buffer, *positional, **keywords)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        label = str(getattr(buffer, 'label', 'unlabelled'))
        data = getattr(buffer, 'content', None)
        # Group by schema so an unlabelled instance buffer stays distinguishable.
        schema = str(data.dtype) if isinstance(data, np.ndarray) else 'no CPU array'
        key = label + ' | ' + schema
        record = records.setdefault(key, {'calls': 0, 'array_bytes_submitted': 0, 'cpu_ms': 0.0})
        record['calls'] += 1
        record['cpu_ms'] += elapsed
        record['array_bytes_submitted'] += data.nbytes if isinstance(data, np.ndarray) else 0
        return result

    def update(cache):
        start = time.perf_counter_ns()
        changed = original_update(cache)
        packing['calls'] += 1
        packing['changed'] += int(changed)
        packing['cpu_ms'] += (time.perf_counter_ns() - start) / 1e6
        return changed

    RenderContext.window_loop, GpuBuffer.upload, CubeInstances.update = window_loop, upload, update
    try:
        run_scene()
    finally:
        RenderContext.window_loop, GpuBuffer.upload, CubeInstances.update = original_loop, original_upload, original_update
    args.output.write_text(json.dumps({
        'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'frames': frame, 'cpu_instance_packing': packing, 'uploads': records,
        'metric': 'CPU submission time, not GPU completion; byte counts use full CPU arrays.',
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Completed {frame} frames; packing {packing}; upload groups: {len(records)}')


if __name__ == '__main__':
    main()
