"""Alternate cached-world query scans to investigate between-process timing noise."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
sys.path.insert(1, str(root))
from scenes.benchmarks import benchmark_ecs as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_ecs_pair_baseline', args.implementation)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    current = benchmark.ECS
    records = {}
    for count in (100000, 1000, 100):
        benchmark.ECS = baseline.ECS
        old = benchmark.make_world(100000, count)[0].query(benchmark.Position, benchmark.Rare)
        benchmark.ECS = current
        new = benchmark.make_world(100000, count)[0].query(benchmark.Position, benchmark.Rare)
        np.testing.assert_array_equal(old.entities(), new.entities())
        for _ in range(20):
            old._scan_entities()
            new._scan_entities()
        samples = {'baseline': [], 'current': []}
        for sample in range(25):
            actions = [('baseline', old), ('current', new)]
            if sample % 2:
                actions.reverse()
            for name, query in actions:
                start = time.perf_counter_ns()
                for _ in range(200):
                    query._scan_entities()
                samples[name].append((time.perf_counter_ns() - start) / 200 / 1e6)
        records[str(count)] = {name: {'median_ms': statistics.median(values), 'samples_ms': values}
                               for name, values in samples.items()}
    args.output.write_text(json.dumps({'entities': 100000, 'repeats': 25, 'loops': 200,
        'baseline_sha256': hashlib.sha256(args.implementation.read_bytes()).hexdigest(),
        'queries': records}, indent=2) + '\n', encoding='utf-8')
    print({count: {name: record['median_ms'] for name, record in values.items()} for count, values in records.items()})


if __name__ == '__main__':
    main()
