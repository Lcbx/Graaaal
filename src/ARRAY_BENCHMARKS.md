# NumPy array access measurements

Measured locally on 2026-11-07 with 100,000 entities, one `float32[3]` position
field, seven samples and 30 movement steps per sample. Setup, reusable scratch
allocation, and warm-up are excluded. CPU timings are medians per step;
tracemalloc peaks measure one warmed step, excluding retained buffers.

| Cached movement path | Median ms/step | Peak traced bytes/step |
| --- | ---: | ---: |
| Copy selected values, compute, assign | 2.0284 | 3,600,288 |
| `read(out=...)`, NumPy `out` operations, assign | 1.1621 | 69,424 |
| `edit_dense()`, NumPy `out` operations | 0.0447 | 945 |

The reusable gather path retains two 1.2 MB work arrays and a lazily cached
0.8 MB index conversion for this selection. Dense editing retains one 1.2 MB
scratch array. These costs are excluded from the per-step peaks above.

Dense editing is appropriate when a system operates in that component store's
row order. It does not make independently ordered stores align or eliminate
gathers required by general joins. These measurements compare access paths in
the same implementation and dtype, not complete game frame times or historical
ECS revisions. No dirty-row tracking or storage algorithm changes were introduced.

Reproduce after installing the local package:

```powershell
python src/tests/benchmark_array_access.py
```
