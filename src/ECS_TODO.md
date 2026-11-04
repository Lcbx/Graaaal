# ECS implementation plan

Scope: strengthen `src/ECS.py`, then extend its query and update APIs while
preserving its NumPy column storage and batched operations. This document plans
the work and tracks implementation progress.

Review baseline: all 40 tests in `tests/ecs_tests.py` passed. Additional review
probes reproduced duplicate single-component rows, stale selection writes,
multicomponent count overflow, partially applied additions, and metadata field
collisions. Convert those probes into permanent regression tests first.

## Implementation status

Phases 1-6 are complete. Phase 6 added reproducible CPU measurements and three
targeted optimizations; larger storage and dirty-tracking changes remain deferred
with recorded reasons in [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md).
Component selections reject access after structural changes to their store;
reusable queries refresh their cached membership and row mappings as needed.
Legacy raw row arrays remain available and require caller-managed lifetimes.

Validation: 210 ECS, query, tag/predicate, scheduler, optimization, scene, and draw-batch tests passed, including two seeded reference-model
runs of 250 mixed mutations each. Regressions cover rejected duplicate insertions,
whole-request preflight, replacement versions, object/shaped payload validation,
metadata sentinels, registration rollback, and multicomponent count boundaries.
Lifetime regressions cover reads and writes after deletion/ID reuse, row removal,
growth, multicomponent block rebuilding, compaction, foreign stores, empty selections,
and defensive row copies. A real-accessor renderer integration test checks refresh
after layout-only compaction. The randomized tests now use checked predicate rows.
Query regressions include two further seeded reference-model runs of 140 mutations,
optional and multicomponent alignment, exclusion/OR filters, lifecycle changes and
ID reuse, cache reuse without field-value caching, and defensive result exports.
The draw-batch scene now reuses movement joins and render membership queries.
Tag regressions cover both insertion forms, compact owner-only storage, queries,
removal, deletion/ID reuse, and atomic rejection of multiple-instance tags. Predicate
tests cover functions, partials, callable objects, bound methods, default arguments,
and explicit errors for unsupported signatures, including on empty storage.
Deferred-edit tests cover FIFO visibility, copied targets, stale generations, partial
failure recovery, creation token ownership/resolution, cancellation, and reentrancy.
Scheduler tests cover fixed ticks, phase boundaries, explicit flush points, stable
dependency ordering, cycles/unknown dependencies, and failure before render prep.
The draw-batch scene queues cube creation/deletion during update; edits flush before
instance preparation. Movement uses the previous entity set for that update.
Optimization regressions cover sparse owner ordering/generation reuse, duplicate
multicomponent candidates and holes, small/batch removals with object cleanup,
and tail appends through capacity growth and compaction. Benchmark artifacts record
the environment, implementation hashes, timing, and temporary/persistent memory.

Scene follow-up: direct drawing now reuses movement/render queries. The combat
prototype uses the current ECS and native renderer; its missing `shader_util`
dependency and obsolete ECS methods were removed. The launcher supports explicit
`run_scene()` entry points and bounded `--frames` checks. Prototype/direct drawing
completed 120 frames each; draw batching completed 1,500 frames including a deferred
membership update. These are GPU runtime checks; window screenshot capture was
blocked by Windows policy. See [scene instructions](../scenes/README.md).

Further scene adoption: the legacy Raylib ECS benchmark now reuses queries, direct
drawing caches packed static instance fields and refreshes render membership, and
the prototype rebuilds/uploads panels only when displayed fields change. Added
CPU regressions for packing calls, membership/ID reuse, and benchmark callbacks;
direct drawing then completed 180 GPU frames and the prototype 120 more.

Compatibility: `add()` now rejects existing single components and duplicate live
owners. Replace existing values explicitly:

```python
positions, = world.register(Position)
ids = world.create(2)
world.add(ids, Position(0, 0))
world.set(ids, Position(1, 2))  # Existing rows, field versions advanced.
positions[ids]["x"] = 3       # Bracket syntax also supports method-name collisions.
```

`set()` rejects missing components and multicomponent storage. Repeated payloads
for a multicomponent type in `add()` are combined and checked against the per-owner
limit before any store changes. Stale handles remain ignored with their matching
payload entries. Validation preflight does not provide recovery from allocation
failure or arbitrary exceptions in user code.

### Checked row APIs and selection lifetimes

Each store exposes a separate `layout_version`. Nonempty additions/removals and
actual dense growth/compaction advance it; a single edit may advance it more than
once. Every selection from that store expires, even if its particular rows did not
move. Field assignments, `set()`, and `touch()` keep the layout version unchanged.
Structural edits to other stores do not expire the selection. These checks do not
provide thread synchronization.

```python
selected = positions[ids]
matching = positions.query_rows(lambda x: x > 0)
all_rows = positions.rows()
subset = selected.checked_rows()[[1, 0]]  # Checked slicing/reordering.
owners = positions.owner_ids(subset)      # Checks store identity and lifetime.
positions.remove(matching)               # Checks before mutation; expires old rows.
# selected.x, len(all_rows), or reuse of matching now raises RuntimeError.
```

`ComponentRows` represents checked rows; `ComponentSelection` adds field access.
Both are accepted by `owner_ids()` and `remove()`. A different store/world raises
`ValueError`; stale rows raise `RuntimeError` before touching fields or counters.
`query_rows()` is a snapshot of which rows matched: later field writes do not
refresh its predicate result. Predicates must not structurally edit their own store;
the checked query rejects a layout change during evaluation.

For NumPy interoperability, `get_rows()` exports a copy. Accessor `get_rows()` and
`query()` still return raw arrays for compatibility. Raw indices have no storage
identity or expiry checks; reacquire them after structural edits. `DrawBatches`
resolves owners using its checked selection and refreshes on layout changes.

Numeric field reads copy data: `selected.x += 1` writes back through the setter,
whereas `selected.x[:] = 1` edits a temporary. Object arrays copy references,
so mutating `selected.data[0]` in place can change stored objects without advancing
field versions; call the corresponding accessor's `touch("data")` afterward.

### Reusable queries and world inspection

`world.is_alive(ids)` and `world.has(ids, Component)` return a one-dimensional
boolean array with one entry per input, preserving duplicates and input order.
Scalars produce a one-element array. Stale/out-of-range handles return false;
`has()` raises for an unregistered component type, even with empty input.
`world.entities()` returns every live handle, including entities with no components.
`entity_version` advances once per successful nonempty create/delete call; empty,
invalid, and stale-only requests do not advance it.

Create queries once and fetch results in update functions:

```python
movement = world.query(Position, Velocity)

def move(dt):
    matched = movement.fetch()
    p = matched[Position].selection
    v = matched[Velocity].selection
    p.x += v.x * dt

targets = world.query(
    Position,
    any_of=(Velocity, Health),
    exclude=Sleeping,
    optional=(Scale, MeshRef),
)
matched = targets.fetch()
scale = matched[Scale]
scale_owners = matched.entities[scale.present]
scale_values = scale.selection  # Packed in the same order as scale_owners.

meshes = matched[MeshRef]
counts_per_owner = np.diff(meshes.offsets)
flat_mesh_values = meshes.selection
# Owner i has flat rows offsets[i]:offsets[i + 1], including empty groups.
```

Components in these examples must already be registered. `Sleeping` can be a
zero-field marker component as shown below.

Required terms use AND, nonempty `any_of` uses OR, and `exclude` rejects any matching
term. Optional terms do not filter owners. Keyword terms accept a component type,
a sequence of types, or `None`; duplicates are normalized. Empty `any_of` adds no
constraint. Contradictory required/excluded terms produce an empty result.

`QueryResult[Component]` returns a `QueryComponent` for each required, OR, or
optional term. Exclusion-only terms are not returned. Each term exposes a checked
`selection`, a per-entity `present` mask, and int64 `offsets` of length N+1.
Required single components have one row per result entity. Optional single
components align with `matched.entities[term.present]`. Multicomponents have packed,
owner-grouped rows; independent multicomponent terms never form a Cartesian product
or silently zip different row counts. All exported metadata arrays are copies.

`query.entities()` refreshes membership without building row selections;
`query.fetch()` also caches those selections and returns the same result while its
structural tokens remain valid. Both order entities by ascending sparse index,
including after generation changes; they do not sort by the full numeric handle.

Membership tokens depend on the required/excluded/OR stores and on entity lifecycle
when there are no required terms. Optional-store changes and returned-store layout
changes refresh rows without rescanning membership. Field writes, `set()`, and
`touch()` keep membership/row caches; subsequent selection reads see current values.
Value predicates remain separate, uncached `accessor.query_rows()` evaluations.

Retained results and term metadata raise `RuntimeError` after relevant membership
or layout changes; call `fetch()` again. Already extracted component selections
retain their ordinary store-specific lifetime checks, rather than becoming live
query views. Cache tracking assumes structural mutations use the ECS APIs and
does not provide thread synchronization.

`world.query()` matches all live entities; exclusion-only and optional-only queries
are also supported. Legacy `where()` keeps its existing behavior, including an
empty result when no required components are supplied.

### Marker components and predicate signatures

```python
@component
class Sleeping:
    pass

@component
class Selected:
    pass

sleeping, selected = world.register(Sleeping, Selected)
world.add(ids[:1], Sleeping)     # Bare types need no payload for tags.
world.add(ids[1:], Selected())   # Empty instances are equivalent.
awake = world.query(Position, exclude=Sleeping, optional=Selected)
matched = awake.fetch()
selected_owners = matched.entities[matched[Selected].present]
token = sleeping.version()      # Membership-only token; no field counters.
world.remove(ids[:1], Sleeping)
assert sleeping.version() != token
# Refetch awake after the membership edit; retained query results have expired.
```

Tags keep the ordinary sparse owner mapping and dense owner IDs, without dummy
field arrays. They support inclusion, exclusion, OR terms, optional presence masks,
checked rows, and generation checks. `multiples=True` and the registration-level
multiple-instance option reject tags before any types in the batch are registered.
Duplicate tag insertion follows the insert-only contract. `set(ids, Tag)` only
validates existing membership; it changes no values or counters. Tag vectors have
shape `(number_of_rows, 0)`.

Predicate parameters are inspected explicitly. Named parameters matching fields
(including `_entity`) receive those arrays, even when they have defaults. Non-field
parameters with defaults retain their configured values, allowing bound thresholds:

```python
from functools import partial

def above(x, *, threshold=0):
    return x > threshold

rows = positions.query_rows(partial(above, threshold=10))
owners = positions.owner_ids(rows)
finite_rows = positions.query_rows(lambda x: np.isfinite(x))
```

Functions, bound methods, callable objects, and inspectable partials are supported.
Predicates must be synchronous, accept named parameters, and return either a scalar
boolean or one boolean per row. Positional-only parameters, variadic signatures,
coroutine/generator functions, and callables without inspectable signatures raise
`TypeError`; wrap a NumPy ufunc in a function naming its component field. Required
non-field parameters also raise; defaulted non-field parameters remain untouched.
Empty queries validate signatures and field names without invoking the predicate.
Predicate results remain uncached.

### Deferred edits and scheduling

`world.commands()` creates an independent FIFO `CommandBuffer`. It supports
`create`, `add`, `set`, `remove`, and `delete`; enqueueing does not edit storage or
expire selections. `flush()` applies the commands present at its entry and returns
the number completed, including no-ops. Commands enqueued during execution wait
for the next flush. Recursive flushes and clearing a flushing buffer are errors.

Handle arrays are copied at enqueue time. Component payloads are retained by
reference until execution, so copy mutable values explicitly when enqueue-time
snapshots are needed. Execution uses the normal ECS APIs: each command preflights
independently and checks full generational handles. Stale targets remain ignored,
including payload entries filtered in lockstep; recycled entities are never targeted
through an old handle.

```python
commands = world.commands()
created = commands.create(2)      # DeferredEntities token; no live IDs yet.
commands.add(created, Position(0, 0), Selected)
commands.flush()
new_ids = created.resolve()       # Copy of full generational handles.
commands.remove(new_ids, Selected)
commands.delete(new_ids[:1])
commands.flush()
```

Tokens can be targets only in their originating buffer. They cannot be sliced or
mixed into handle arrays. Tokens inside component payloads are not auto-resolved.
To pass IDs to another buffer or component field, resolve explicitly after the
creation command has executed. Resolution is not a liveness check: later deletion
does not rewrite the saved IDs. Failed or cancelled creations never resolve;
successful creation tokens remain usable even if a later command fails.

On failure, earlier successful commands remain applied. The failed command is
consumed, subsequent commands remain pending, and `CommandError` reports the
buffer-local command index and operation, with the original error in `__cause__`.
Recover deliberately, then flush the remaining tail, or call `clear()` to discard
it. `clear()` cancels pending creation tokens without undoing applied work. This is
per-command preflight, not a transaction over the whole buffer.

```python
schedule = Scheduler(world)
movement = world.query(Position, Velocity)

def move(world, commands, dt):
    matched = movement.fetch()
    matched[Position].selection.x += matched[Velocity].selection.x * dt
    # commands.add/remove/delete(...) can be queued without expiring matched.

def prepare_instances(world, commands, dt):
    render_entities = world.query(Position).entities()
    # Consume the state after the update-phase flush.

schedule.add("movement", move, phase="update")
schedule.add("instances", prepare_instances, phase="render_prep", after="movement")
schedule.compile()               # Optional early validation of the entire graph.
schedule.run(frame_dt, fixed_steps=2, fixed_dt=1 / 60)
```

Callbacks receive `(world, commands, dt)`. Phases run in `fixed_update`, `update`,
`render_prep` order; the caller supplies the fixed tick count (default zero), with
one flush after each fixed tick. Pending commands also flush before the first
system and at every phase end, even for an empty phase. `before`/`after` accept
system names or sequences, including forward references. Dependencies order
callbacks without adding flushes; use `flush_after=True` on a system when a later
system in the same phase must see its queued edits. Refetch selections across flushes.

The scheduler validates all dependencies before flushing or invoking systems.
Unknown names, cycles, and dependencies contradicting phase order are errors.
Ready systems in the same phase keep registration order. Adding a system invalidates
the compiled order; changing or recursively running an active schedule is rejected.
Time steps must be finite and nonnegative, fixed_dt positive, and fixed_steps a
nonnegative integer. Parallel execution and automatic fixed-time accumulation remain
outside this scheduler.

System exceptions stop the run without flushing that system's pending commands;
direct writes already performed remain. A command failure at a boundary also stops
later systems. The buffer remains available for explicit recovery or clearing.

## Design decisions

- Keep independent sparse-set component pools and structure-of-arrays fields.
- Keep generational entity handles and the existing 64-component limit initially.
- Make `add()` insert-only for single-instance components: reject existing
  membership and duplicate live owners in one request. Multicomponent additions
  continue to append. Add explicit replacement APIs separately.
- Validate expected user errors before mutating state. Do not promise recovery
  from arbitrary allocation failures or exceptions in user-provided code.
- Keep numeric selection reads as copies with explicit assignment to write back.
- Invalidate selections when their storage layout changes; fail explicitly on
  subsequent use instead of silently targeting another component.
- Keep systems as ordinary functions. Add scheduling and deferred edits in small,
  separate layers after the storage and query contracts are stable.

## Phase 1: restore storage invariants (highest priority)

### 1.1 Single-component insertion and replacement

- [x] Reject additions for owners that already have the single component.
- [x] Reject duplicate live owners within a single-component addition batch.
- [x] Reject repeated single-component types in one `add()` call before commit;
      this includes instance and class-plus-values forms.
- [x] Preserve filtering of stale entity IDs and their corresponding field values.
- [x] Add `set(entities, *components)` for single-instance components, requiring
      existing membership and writing existing rows without changing row order.
- [x] Reject `set()` for multicomponent stores until an instance identity API exists.
- [x] Defer `upsert()` until there is a concrete caller; do not overload `add()`.

Acceptance: one dense row per single-component owner; every sparse entry resolves
to that owner's row; removal and deletion leave no orphan rows. Rejected requests
leave values, membership, and version counters unchanged. Successful `set()` calls
advance the affected field versions without advancing membership/layout versions.

### 1.2 Validate complete requests before commit

- [x] Split `add()` into parse, normalize, validate, and commit steps.
- [x] Normalize every payload and validate cross-payload constraints before any
      storage, entity mask, or version counter is changed.
- [x] Apply the same preflight rule to `set()` and multi-type `remove()` calls.
- [x] Validate all types in a batch registration before installing any stores.
- [x] Cover a valid first component followed by an invalid second payload.

Acceptance: shape, dtype conversion, registration, duplicate, and capacity-limit
validation errors leave the world unchanged, including counters and masks.

### 1.3 Multicomponent count limits

- [x] Compute incoming and final owner counts in a wider integer type.
- [x] Check the `ComponentPerEntityCount` maximum before narrowing or clearing
      any old owner blocks. Initially retain `uint16` and raise `OverflowError`.
- [x] Validate both a large first insertion and an append to an existing owner.
- [x] Test 65,535 rows, a rejected 65,536th row, and mixed-owner batches containing
      one overflowing owner; rejected batches must preserve every owner's data.

Acceptance: no count wraps, lost blocks, or disagreement between membership masks
and selectable component rows.

### 1.4 Schema and metadata isolation

- [x] Reject `_entity` and `_count` as user fields during registration.
- [x] Document bracket access as the reliable API for fields whose names collide
      with selection methods, such as `vector` or `get_rows`.
- [x] Fix initialization that currently replaces the filled owner array with an
      uninitialized array in `ComponentStorage.__init__()`.
- [x] Correct the repeated `_ENTITY_STR` branch in `_defragment()` so unused count
      metadata is cleared through `_COUNT_STR`.

Acceptance: invalid schemas fail before registration; inactive owner/count
metadata has consistent sentinel values after initialization, growth, and compaction.

## Phase 2: make selection and row lifetimes explicit

Depends on Phase 1.

- [x] Add a per-store `layout_version`, separate from membership and field versions.
- [x] Advance it whenever rows are relocated, removed, or reused. A conservative
      invalidation on every nonempty structural edit is acceptable initially.
- [x] Capture the version in `ComponentSelection`; validate it on reads, writes,
      vector operations, and `get_rows()`. Raise a clear `RuntimeError` when stale.
- [x] Add a versioned row-selection wrapper for safe reuse of predicate results;
      check storage identity and layout version in row-based removal/owner lookup.
- [x] Keep legacy raw row arrays as explicitly ephemeral compatibility inputs:
      they cannot detect staleness. Migrate internal callers to checked selections
      before deciding whether to deprecate those inputs.
- [x] Ensure arrays returned by `get_rows()` cannot mutate a selection's internal
      row mapping accidentally.
- [x] Document that `selection.x += value` writes back, while
      `selection.x[:] = value` edits a temporary numeric array.
- [x] Document object aliasing and when `touch()` is required for in-place edits.

Acceptance: selecting A, deleting A, then writing through the saved selection raises
instead of modifying a moved entity. Cover ordinary swap removal, multicomponent
block rebuilding, defragmentation, growth, and checked rows from the wrong store.

## Phase 3: reusable queries and basic world inspection

Depends on Phases 1 and 2.

- [x] Add public `is_alive(entities)` and `has(entities, Component)` returning one
      boolean per supplied entity, with stale handles returning false.
- [x] Add `entities()` to enumerate live handles, including component-free entities.
- [x] Preserve existing `where()` behavior initially; provide a separate query API
      for exclusion-only and unconstrained queries rather than silently changing it.
- [x] Introduce reusable queries with required, excluded, and `any_of` component sets.
- [x] Return aligned per-component row selections for matching entities so systems
      can reuse joins across several field operations.
- [x] Add optional components with explicit presence masks; never align them by
      independently dropping missing owners.
- [x] For multicomponent terms, initially return owner-grouped rows and offsets;
      do not silently zip unequal component counts or form Cartesian products.
- [x] Cache membership using relevant membership versions. Include a world entity
      lifecycle version for queries matching entities with no required components.
- [x] Invalidate cached row mappings using layout versions; ordinary field writes
      must not invalidate a membership-only query.
- [x] Keep value-predicate results uncached initially, or explicitly key them on
      the field versions they read; never reuse them on membership tokens alone.

Acceptance: results match an uncached reference after create/delete/add/remove/set,
including recycled IDs, exclusions, empty worlds, optional data, and multiple rows
per owner. Preserve existing ascending-index `where()` ordering; document ordering
for new query APIs.

## Phase 4: marker components and API usability

Depends on Phase 3 for query integration.

- [x] Allow zero-field `@component` classes as tags with owner membership storage
      and no dummy field arrays. Reject `multiples=True` for tags initially.
- [x] Define `add(ids, Tag)` and `add(ids, Tag())` as equivalent tag insertion forms;
      update argument parsing without weakening missing-payload errors for data types.
- [x] Cover tag inclusion, exclusion, removal, deletion, and recycled entities.
- [x] Replace direct `func.__code__` inspection with explicit callable signature
      validation for field predicates. Support inspectable functions/partials and
      give clear errors for unsupported signatures or callables.
- [x] Add concise examples for batch insertion, replacement, query joins, tags,
      selection expiry, and field version tokens.

Acceptance: `Sleeping` and `Selected` can be represented as empty components;
existing dataclass, shaped-field, object-field, and IntFlag examples still work.

## Phase 5: deferred structural edits and update ordering

Depends on stable mutation and query APIs from Phases 1-4.

- [x] Add a command buffer for queued additions, removals, and deletions of existing
      entity handles. Apply commands in enqueue order at explicit flush points.
- [x] Validate handles when commands execute so recycled IDs cannot be targeted by
      commands recorded for an older generation.
- [x] Specify failure behavior: each command preflights independently; a failure
      stops flushing, preserves earlier successful commands, and reports the failed
      command. Do not imply whole-buffer transactionality.
- [x] Add deferred creation only with a documented temporary-handle resolution API.
- [x] Add a small sequential scheduler with fixed-update, update, and render-prep
      phases plus explicit `before`/`after` dependencies and cycle detection.
- [x] Define flush boundaries so later systems know when structural edits become visible.
- [x] Migrate one scene to demonstrate query reuse and deferred edits.

Acceptance: systems can queue edits while processing selections; structural changes
occur at documented boundaries, and dependency order is deterministic. Parallel
scheduling remains deferred until access rules and workload measurements justify it.

## Phase 6: measure and optimize

Establish baselines before changing storage algorithms. Depends on the preceding
correctness contracts; individual optimizations can be delivered independently.

- [x] Benchmark dense movement joins, rare-component queries, heavily deleted
      worlds, repeated cached queries, and frequent membership changes.
- [x] Compare full mask scans with smallest-required-pool candidate filtering;
      preserve query ordering or account explicitly for sorting cost.
- [x] Measure gather/scatter allocations, not just query lookup time.
- [x] Add a small-removal path that avoids allocating an entire-pool removal mask;
      retain the batch path when it performs better.
- [x] Measure repeated multicomponent appends; consider owner-block spare capacity
      or chunked blocks if rebuild costs dominate.
- [x] Measure sparse-array memory for rare components on high-index owners; evaluate
      paged sparse maps only if savings justify lookup overhead.
- [x] Consider dirty entity sets or dirty chunks when field-wide version tokens
      cause excessive work in transforms, bounds updates, or rendering uploads.

Acceptance: record entity/component counts, operation mixes, elapsed time, peak
memory, and environment. Adopt optimizations only with measured benefits and
unchanged correctness. An archetype backend is a separate future proposal.

Results, limitations, and decisions: [ECS_BENCHMARKS.md](ECS_BENCHMARKS.md).
Paging and dirty tracking received byte-cost estimates, not lookup/upload benchmarks;
they are deferred. Candidate selection, small removals, and tail-owner appends were
implemented and measured. Other paths retain their existing algorithms.

## Validation and delivery

- [x] Deliver each numbered correctness item as a focused change with regressions.
- [x] Add a seeded randomized reference-model test covering create, delete,
      single-component insert/set/remove, multicomponent edits, and ID recycling.
- [x] Check storage invariants after each modeled operation: owner/row consistency,
      sparse mappings, live counts, masks, and clearing of removed object references.
- [x] Run `tests/test_ecs_main.py` after each storage/API change, then
      `tests/test_draw_batches.py` for selection/version/query integration changes.
      Include `tests/test_ecs_queries.py` for reusable-query changes.
      Include `tests/test_ecs_tags.py` for marker and predicate changes.
      Include `tests/test_ecs_schedule.py` for deferred-edit and scheduler changes.
      Include `tests/test_ecs_optimizations.py` for optimized storage/query paths.
      Include `tests/test_scenes.py` for scene models and launcher behavior.
      Include `tests/test_scene_caches.py` for instance/panel cache behavior.
- [x] Extend the repository's existing test invocation conventions; for PowerShell,
      set `$env:PYTHONPATH='src'` and run
      `python -m pytest tests/test_ecs_main.py tests/test_ecs_queries.py tests/test_ecs_tags.py tests/test_ecs_schedule.py tests/test_ecs_optimizations.py tests/test_scenes.py tests/test_scene_caches.py tests/test_draw_batches.py -q -p no:cacheprovider`.
- [x] Update affected scene examples and describe intentional compatibility changes
      alongside each API change.

## Deferred until a caller needs them

- General relationship traversal, parent-child hierarchies, and deletion policies.
- Prefabs, serialization, schema migration, and entity-reference remapping.
- Resource injection, lifecycle observers, and event routing.
- More than 64 component types and configurable mask representations.
- Automatic parallel scheduling and alternate storage backends.

These are useful ECS extensions, but they do not block repairing the current core.

## Reference designs

- [EnTT: sparse pools, views, tags, and storage tradeoffs](https://github.com/skypjack/entt/wiki/Entity-Component-System)
- [Flecs: query features and caching](https://www.flecs.dev/flecs/Queries.html)
- [Flecs: systems and scheduling](https://www.flecs.dev/flecs/Systems.html)
- [Bevy: deferred structural commands](https://docs.rs/bevy_ecs/latest/bevy_ecs/system/struct.Commands.html)
