"""Compare equivalent cached movement operations; setup is excluded."""
import json
import statistics
import time
import tracemalloc
import numpy as np
from graaal.ECS import ECS, component, columns, array_field


@component
class Position:
    value: np.ndarray = array_field(np.float32, shape=(3,))


def benchmark(count=100000, samples=7, iterations=30):
    world = ECS()
    positions, = world.register(Position)
    ids = world.create(count)
    world.add(ids, columns(Position, value=np.zeros((count, 3), dtype=np.float32)))
    selection = positions[ids]
    velocity = np.full((count, 3), 0.1, dtype=np.float32)
    out, scratch = np.empty_like(velocity), np.empty_like(velocity)

    def copied(): selection.assign(value=selection.value + velocity / 60)

    def reused():
        selection.read("value", out=out)
        np.divide(velocity, 60, out=scratch)
        np.add(out, scratch, out=out)
        selection.assign(value=out)

    def dense():
        with positions.edit_dense("value") as fields:
            np.divide(velocity, 60, out=scratch)
            np.add(fields["value"], scratch, out=fields["value"])

    results = {}
    for name, operation in (("copied", copied), ("reused", reused), ("dense", dense)):
        operation()  # Warm caches, including the gather index conversion.
        durations = []
        for _ in range(samples):
            start = time.perf_counter()
            for _ in range(iterations): operation()
            durations.append((time.perf_counter() - start) * 1000 / iterations)
        tracemalloc.start()
        operation()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        results[name] = {"median_ms": statistics.median(durations), "peak_traced_bytes": peak}
    return {"entities": count, "dtype": "float32[3]", "samples": samples, "iterations": iterations, "results": results}


if __name__ == "__main__": print(json.dumps(benchmark(), indent=2))
