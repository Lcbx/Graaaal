# ECS implementation status and backlog

Phases 1-10 are complete. This file records delivered work and deferred decisions;
the application-facing contracts and examples live in [ECS_API.md](ECS_API.md).
Measurement methodology and observations live in [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md).

## Completed work

| Phase | Delivered behavior |
| --- | --- |
| 1 | Single-component insertion invariants, explicit replacement, whole-request preflight, multicomponent count limits, reserved metadata isolation |
| 2 | Checked selections/rows with store identity and layout expiry; defensive row exports |
| 3 | Reusable structural joins, exclusions/OR/optional terms, aligned multicomponent offsets, live-entity inspection |
| 4 | Zero-field tags and validated synchronous named-field predicates |
| 5 | FIFO deferred edits, creation tokens, failure recovery, sequential scheduling and explicit flush boundaries |
| 6 | CPU benchmarks, sparse query candidate filtering, small-removal allocation reduction, tail-owner append optimization |
| 7 | Shaped vector round trips, integer quantity/handle validation, atomic field/vector writes, nested object cleanup, synchronous system validation |
| 8 | Checked row editing, positional selection subsets, named column payloads, atomic partial field assignment, representative caller migration |
| 9 | Shared schemas and write commits, named prepared mutations, defensive mask snapshots, extracted command/scheduler/type modules, mypyc compatibility |
| 10 | Storage/world diagnostics, invariant auditing, reserve/compact controls, CPU comparisons and bounded rendering upload measurements |

## Design constraints retained

- Independent sparse component pools with NumPy structure-of-arrays fields.
- Generational entity handles, at most 64 registered types, and at most 65,535
  component instances per owner/type.
- Insert-only single-component `add()`; existing membership is updated with `set()`.
- Validation before expected user errors can change values or version counters.
  Allocation failures and arbitrary user-code effects are outside this guarantee.
- Numeric reads copy; assignments write back. Object reads retain references and
  require `touch()` after otherwise untracked in-place edits.
- Structural edits conservatively expire selections from the affected store.
- Ordinary synchronous systems, explicit deferred-command visibility, and sequential
  execution. Version tokens provide cache invalidation, not thread synchronization.

## Validation record

The implementation checkpoint passed 266 cases across the nine ECS/follow-up/
scene/draw-batch suites, including 56 follow-up cases and seeded reference models.
The mypyc entry built, and 240 cases passed against a temporary compiled extension;
query-suite monkeypatch instrumentation was exercised in the source-runtime run.
The migrated direct-draw scene completed a measured 120-frame run. The API guide's
20 Python examples were subsequently executed successfully in document order.
These are recorded checks, not a promise that later edits have been tested.

From the repository root:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_ecs_main.py tests/test_ecs_queries.py tests/test_ecs_tags.py tests/test_ecs_schedule.py tests/test_ecs_optimizations.py tests/test_ecs_followup.py tests/test_scenes.py tests/test_scene_caches.py tests/test_draw_batches.py -q -p no:cacheprovider
```

For future changes, start with the affected suite and expand to integrations when
selection/query/version behavior or scene callers change. Add regressions for
actual failures and extend the reference model for new mutation contracts.
Historical benchmark series remain separate under
[`scenes/benchmarks/results`](../scenes/benchmarks/results/).

## Deferred work and adoption criteria

| Feature | Reason to revisit |
| --- | --- |
| Dirty-row/chunk tracking and partial uploads | A sparsely changing render workload demonstrates a benefit after bookkeeping/range-upload costs |
| Paged sparse maps | High-index rare components cause meaningful memory pressure and lookup benchmarks support paging |
| General owner-block spare capacity/chunks | Non-tail append workloads remain expensive after callers batch operations |
| Strict stale-handle debug mode | A caller needs diagnostics beyond the current ignore-stale behavior |
| Stable multicomponent instance handles and replacement | Callers need identities surviving structural relocation |
| Explicit upsert | A concrete caller needs combined insertion/replacement semantics |
| Relationships/hierarchies/deletion policies | Application traversal and ownership requirements establish the contract |
| Prefabs/serialization/schema migration | Persistence requirements establish entity-reference remapping semantics |
| Resources/events/lifecycle observers | Systems need shared resource injection or event routing |
| More than 64 types, parallel scheduling, alternate backends | Real workload measurements justify changing representation or execution rules |

The current CPU/rendering measurements and reasons for retaining existing storage
algorithms are recorded in the benchmark report. There are no unfinished mandatory
phases in this backlog; deferred items require a caller or measured benefit.
