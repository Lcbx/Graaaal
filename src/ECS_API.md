# ECS batch and selection APIs

Import the public API from `ECS`. Schema conversion lives in `ecs_schema`, input
types/validation in `ecs_types`, deferred edits in `ecs_commands`, scheduling in
`ecs_schedule`, and inspection in `ecs_diagnostics`. Storage and joins remain
together in `ECS.py` because they share row lifetime and cache contracts.

## Named batches and explicit writes

```python
import numpy as np
from ECS import ECS, component, columns

@component
class Position:
    x: float
    y: float

world = ECS()
positions, = world.register(Position)
ids = world.create(3)
world.add(ids, columns(Position, x=np.arange(3), y=0))
world.set(ids, columns(Position, x=5, y=6))
positions[ids].assign(x=7)  # Partial update; all requested fields preflight together.

commands = world.commands()
created = commands.create(2)
commands.add(created, columns(Position, x=[1, 2], y=0))
commands.flush()
```

Named batches require exactly the declared fields, including for replacement.
`columns(Tag)` represents a zero-field tag. Component instances still broadcast
one value to all owners; existing positional payload forms remain supported.
Multi-type requests preflight before committing. Stale entity handles and their
matching payload entries are ignored. Multicomponent batches append instances;
`set()` continues to require existing single-instance components.

Field payloads are retained by reference, including in deferred commands. Copy
mutable arrays before enqueueing if execution must see their current contents.

## Entities and checked component rows

`positions[ids]` accepts entity handles. Predicate queries return component rows:

```python
rows = positions.query_rows(lambda x: x > 0)
selected = positions.select_rows(rows)
selected.assign(x=selected.x * 2)
selected.take([1, 0]).assign(y=4)
```

`select_rows()` accepts checked rows/selections from the same store. For a
multicomponent, it selects exactly the matching instances without expanding every
instance belonging to their owners. `take()` accepts positional slices, integer
indices, and boolean masks; bracket indexing selects field names. Order and
duplicates are preserved. Repeated rows follow NumPy assignment semantics:
`selection.take([0, 0]).x += 1` increments row zero once.

Selections expire on structural changes to their store. Raw `get_rows()`/`query()`
arrays remain ephemeral compatibility inputs. Required/excluded/optional joins
use reusable `world.query(...)`; `query()` without terms matches all live entities.
Legacy `where()` without required terms still returns no entities.

Numeric reads copy: `selected.x += 1` writes back; `selected.x[:] = 1` edits the
temporary. Use `assign()` or bracket assignment for explicit writes. Object reads
retain object references; notify in-place object edits with `positions.touch()`.

## Vector layout and validation

`vector(*fields)` packs numeric/boolean scalar and shaped fields into a 2-D array,
flattening each field's trailing dimensions in C order. `set_vector()` requires
exactly the same row/column shape and reverses that layout. Tags use `(rows, 0)`.
Duplicate field names and nonnumeric fields are rejected. NumPy promotes mixed
dtypes; combining large integer IDs with floats can lose precision, so select
compatible fields or assign fields separately when exact integer values matter.

Vector, attribute, bracket, and `assign()` writes validate before changing values
or counters. Successful nonempty writes increment each affected field token once;
they preserve layout/membership. `ECS.set()` uses the same commit semantics.
Allocation failure and arbitrary exceptions in user code are outside the preflight
guarantee. Structured/shaped object-reference leaves are cleared when rows become
inactive, including during compaction and block rebuilding.

Creation/reservation counts accept Python/NumPy indexable integers and reject
booleans, floats, and strings. Negative counts raise `ValueError`. Entity and row
inputs require nonnegative integers representable by their uint64/uint32 dtype:
invalid types raise `TypeError`, negative values `ValueError`, overflow
`OverflowError`. Representable stale handles keep the existing liveness behavior.
Systems must execute synchronously; coroutine/generator callbacks are rejected at
registration. Command failure, flush visibility, and dependency ordering retain
their established contracts.

## Inspection and allocation control

```python
world.reserve(10000)                       # Entity slots; creates no entities.
positions.reserve(1000, sparse_capacity=10000)
snapshot = world.inspect()                # Counts, schema, fragmentation, bytes.
position_stats = positions.inspect()
world.check_invariants()                   # Explicit debug audit; raises on failure.
positions.compact()                        # Packs multicomponent holes; retains capacity.
```

Dense reservation that reallocates expires selections. Sparse-only/world
reservation leaves component selections and version tokens valid. Actual
compaction relocates rows and expires selections; compaction of an already packed
store is a no-op. None of these operations change field values or membership.
Capacity grows geometrically, so reservation is a minimum, not an exact size.

Inspection returns independent metadata snapshots. `entity_masks` also returns a
defensive copy; membership edits must use the ECS APIs. Byte counts cover NumPy
arrays and exclude Python overhead and referenced object graphs. Invariant checks
are explicitly invoked and are not part of frame updates; run Python without `-O`
when using these assertion-based diagnostics.
