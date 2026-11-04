# ECS CPU measurements and optimization decisions

Phase 6 measurements on Windows 11, AMD64 Family 25 Model 97, Python 3.14.6,
NumPy 2.4.4. These installed versions differ from `requirements.txt`; results are
local observations rather than guarantees for the pinned environment or CI.
All 198 ECS/query/tag/scheduler/optimization/draw-batch regression tests pass.

## Reproduction and method

Run the standalone CPU benchmark without the rendering engine:

```powershell
python scenes/benchmarks/benchmark_ecs.py --entities 100000 --repeat 11 --loops 50 --output scenes/benchmarks/ecs_optimized.json
```

For comparisons, `--implementation path/to/saved/ECS.py` loads a compatible saved
implementation. The recorded baseline used the staged pre-Phase-6 source, exported
to a temporary file before making changes. JSON records include implementation
SHA-256, interpreter/NumPy/platform, entity counts, repeat count, and loops per sample.

- [Benchmark script](../scenes/benchmarks/benchmark_ecs.py)
- [Baseline observations](../scenes/benchmarks/ecs_baseline.json)
- [Optimized observations](../scenes/benchmarks/ecs_optimized.json)

Each action has an untimed warmup, then 11 timing samples; medians are shown below.
Repeated read/update actions run 50 times per sample. Mutating removal and append
workloads get fresh worlds and one action per sample. Setup is excluded from timing.
The membership-churn action adds a tag to 100 owners, fetches the affected query,
removes the tag, then fetches again. Append timings include complete Python ECS API
calls, normalization, validation, grouping, and capacity management.

Peak allocations come from a separate single-action tracemalloc pass after setup.
They include NumPy allocations where the installed NumPy allocator reports them,
exclude already allocated world storage, and are **not process peak RSS**. Persistent
storage measurements use array `nbytes`. Python allocator state, CPU frequency, and
other host activity affect elapsed time; small changes are not treated as wins.

## Measurements

All workloads use an entity index space of 100,000 unless noted. Times are ms;
memory columns are action-only traced peak bytes.

| Operation | Before ms | After ms | Before bytes | After bytes |
| --- | ---: | ---: | ---: | ---: |
| Required query: 100,000 matches | 0.0954 | 0.0868 | 1,000,736 | 1,000,784 |
| Required query: 1,000 matches | 0.0677 | 0.0116 | 1,000,736 | 31,384 |
| Required query: 100 matches | 0.0645 | 0.00594 | 1,000,736 | 6,184 |
| Query after deleting all but 100 entities | 0.0563 | 0.00650 | 1,000,736 | 6,184 |
| Remove 1 component row | 0.0694 | 0.0422 | 104,424 | 4,312 |
| Remove 4 component rows | 0.0521 | 0.0539 | 104,528 | 4,952 |
| Remove 8 component rows | 0.0770 | 0.0526 | 104,688 | 5,112 |
| Dense movement: uncached joins and gather/scatter | 5.078 | 5.007 | 4,001,658 | 4,001,482 |
| Dense movement: cached joins and gather/scatter | 2.228 | 2.216 | 2,400,760 | 2,400,760 |
| 100-owner membership churn, two query fetches | 1.865 | 1.784 | 7,126,427 | 7,126,427 |
| 1,024 individual appends to one tail owner | 114.12 | 70.41 | 73,506 | 49,727 |
| 1,024 alternating appends to two owners | 130.89 | 128.59 | 58,830 | 59,646 |
| 1,024 rows appended in one batch | 0.1465 | 0.1497 | 108,658 | 108,658 |

The sparse-query gains are about 5.8x at 1,000 matches, 10.9x at 100 matches,
and 8.7x in the heavily deleted world. Tail-owner appends improve about 1.6x.
Removing one row uses about 24x less temporary memory. Four-row removal illustrates
the memory tradeoff: roughly 21x less allocation with no elapsed-time improvement.

Dense queries, dense movement, membership churn, and alternating-owner appends
retain their existing algorithms; their small timing differences are not claimed
as improvements. Larger-removal timings vary between runs; their batch path is
also retained. Cached query fetch alone stays around 0.0006 ms in these runs, but
movement still needs field gathers, arithmetic, and scatter writes. Measuring only
query lookup substantially understates a system's allocation and execution costs.

## Changes adopted

1. **Adaptive required-component candidates.** `where()` and reusable query refresh
   share the same matching helper. When the smallest required store's row span is
   at most 1/16 of the entity index space, gather candidates from that store, sort
   by sparse index, then apply masks. Multicomponent candidates ignore holes and
   deduplicate owners. Other cases retain the full scan. Using the row span is
   conservative for fragmented multicomponent pools. Unconstrained, OR-only, and
   exclusion-only queries retain full scans.

   Prototype comparisons include sorting costs: all-entity candidate sorting is
   much slower than scanning, while the 1% and 0.1% cases clearly favor candidates.
   At 10% the difference is smaller, so the selected threshold keeps that case on
   the existing scan. Density policies can be revisited with application workloads.

2. **Small single-component removals.** A single removal compares against the tail
   directly. Batches up to eight rows in pools of at least 4,096 rows use a tiny
   tail-versus-removal comparison matrix. This avoids an allocation proportional
   to the entire pool. Other batches retain the full removal mask. Object cleanup,
   sparse-map repairs, field alignment, and version invalidation are unchanged.

3. **Tail-owner multicomponent extension.** Appending to an existing block at the
   end writes only the new rows, without copying the owner's existing fields.
   Capacity management still grows or compacts as needed, and the head is resolved
   again after compaction. Non-tail owners and multi-owner batches retain block
   rebuilding. Count limits and layout invalidation remain in force.

## Decisions deferred after measurement

- **General multicomponent block allocation:** alternating owner updates still
  rebuild blocks. Spare capacity/chunks would change grouping, offsets, and memory
  management. The tail optimization helps a common case without introducing that
  redesign. Batch additions remain dramatically cheaper than individual API calls;
  callers should batch when they can. No claim is made that alternating-owner
  append complexity has improved.
- **Paged sparse maps:** one tag on owner 99,999 allocates 524,288 bytes of sparse
  mapping and 1,024 bytes of dense owner storage; world state separately uses
  2,097,152 bytes. A hypothetical 256-entry sparse page needs 1,024 bytes before
  page tables/metadata. This identifies a memory opportunity, but is a byte estimate,
  not an implemented paging or lookup benchmark. Retain flat maps until lookup
  measurements and real component/index distributions justify a storage change.
- **Dirty entity/chunk tracking:** for 100 changed entities out of 100,000, three
  float64 position fields contain 2.4 MB in total versus 2.4 KB of changed values.
  Those are byte estimates; gathering, dirty bookkeeping, upload ranges, and GPU
  layout differ. No GPU upload benchmark was run. Retain field-wide tokens until
  downstream workloads demonstrate that incremental tracking pays for itself.

The stored artifacts allow future comparisons without turning noisy timing results
into correctness tests. Archetype storage, paged maps, general block allocation,
and dirty-row tracking remain separate proposals.
