# ECS usage and API reference

This guide covers the application-facing API. Python examples can be executed in
document order; later examples use the reusable setup below and reset their world
explicitly. Run after installing the package with `pip install -e .` from the repository root.

Related documents: [implementation status](ECS_TODO.md),
[benchmark report](ECS_BENCHMARKS.md), and [source guide](README.md).

- [Quick start: named batches and writes](#named-batches-and-explicit-writes)
- [Entity and row addressing](#entities-and-checked-component-rows)
- [Vectors](#vector-layout-and-validation)
- [Inspection and allocation](#inspection-and-allocation-control)
- [Reusable example setup](#reusable-example-setup)
- [Component schemas and registration](#component-schemas-and-registration)
- [Entity lifecycle](#entity-lifecycle)
- [Every insertion and replacement form](#insertion-and-replacement-forms)
- [Selections and row lifetimes](#selections-and-row-lifetimes)
- [Multicomponent instances](#multicomponent-instances)
- [Predicates](#value-predicates)
- [Structural joins and optional data](#structural-queries-and-joins)
- [Version tokens and caches](#version-tokens-and-caches)
- [Deferred commands and recovery](#deferred-commands-and-recovery)
- [Scheduling](#scheduling)
- [Diagnostics reference](#diagnostics-reference)
- [API inventory and errors](#api-inventory-and-error-reference)

Import the public API from `ECS`. Schema conversion lives in `ecs_schema`, input
types/validation in `ecs_types`, deferred edits in `ecs_commands`, scheduling in
`ecs_schedule`, and inspection in `ecs_diagnostics`. Storage and joins remain
together in `ECS.py` because they share row lifetime and cache contracts.

## Named batches and explicit writes

```python
import numpy as np
from numpyecs.ECS import ECS, component, columns

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

## Reusable example setup

These definitions are shared by the remaining examples. Every call to
`example_world()` returns a new world with three moving entities.

```python
from numpyecs.ECS import (
    ECS, Scheduler, CommandBuffer, CommandError, component, columns,
    Entity, Index, entity_index, entity_generation, as_entities, as_rows,
)

@component
class Velocity:
    x: float
    y: float

@component
class Health:
    value: float

@component
class Sleeping:
    pass

@component(multiples=True)
class Item:
    value: float

def example_world():
    world = ECS()
    positions, velocities, health, sleeping, items = world.register(
        Position, Velocity, Health, Sleeping, Item, capacity=8,
    )
    ids = world.create(3)
    world.add(ids, columns(Position, x=[0, 1, 2], y=0), Velocity(2, 0))
    return world, ids, positions, velocities, health, sleeping, items
```

## Component schemas and registration

`@component` applies `dataclass` and accepts `multiples=False` by default. Instances
are convenient payloads; the world stores field arrays rather than retaining the
dataclass instance itself. Defaults apply when constructing an instance. Named
column payloads still require all fields, including fields with dataclass defaults.
Annotated ordinary classes and ordinary dataclasses can also be registered.
`ClassVar` fields are excluded. Type hints must resolve when registration occurs.

| Annotation | Storage |
| --- | --- |
| `float`, `int`, `bool`, NumPy scalar types | NumPy dtype resolved from the annotation |
| `str`, `bytes`, `object`, unsupported annotations | Python object references |
| `IntFlag` subclass | `int64`; selection reads return integer arrays |
| `np.dtype((base, shape))` | One shaped value per row |
| Class with `storage_dtype()` | Dtype returned by that class method |
| Structured NumPy dtype | Record array, including nested object leaves |

Prefer explicit NumPy types such as `np.float32` and `np.uint32` when field width
matters. Strings are object references, not fixed-width NumPy strings. No general
runtime validation of arbitrary Python object types is performed.

```python
from enum import IntFlag
from typing import ClassVar

class Kind(IntFlag):
    FRIEND = 1
    FLYING = 2

@component
class Details:
    category: ClassVar[str] = 'actor'
    name: str = 'unnamed'
    flags: Kind = Kind.FRIEND

class Vec3:
    @classmethod
    def storage_dtype(cls):
        return np.dtype((np.float32, (3,)))

@component
class Transform:
    position: Vec3
    matrix: np.dtype((np.float32, (2, 2)))

schema_world = ECS()
details, transforms = schema_world.register(Details, Transform, capacity=4)
owners = schema_world.create(2)
schema_world.add(owners, Details())
schema_world.add(owners, columns(
    Transform, position=[1, 2, 3], matrix=np.eye(2, dtype=np.float32),
))
assert transforms[owners].position.shape == (2, 3)
assert details[owners].name.tolist() == ['unnamed', 'unnamed']
assert len(details.query_rows(lambda flags: (flags & int(Kind.FRIEND)) != 0)) == 2
```

Shaped fields round-trip through vectors using flattened column spans:

```python
transform_selection = transforms[owners]
packed = transform_selection.vector('matrix', 'position')
assert packed.shape == (2, 7)           # Four matrix entries, then three position entries.
transform_selection.set_vector(packed, 'matrix', 'position')
assert transform_selection.matrix.shape == (2, 2, 2)
assert transform_selection.position.tolist() == [[1, 2, 3], [1, 2, 3]]
```

Structured records remain one field in the component schema:

```python
record_dtype = np.dtype([('weight', np.float32), ('payload', object)])

@component
class Record:
    value: record_dtype

records, = schema_world.register(Record)
values = np.empty(2, dtype=record_dtype)
values['weight'] = [1, 2]
values['payload'] = [{'label': 'a'}, {'label': 'b'}]
schema_world.add(owners, columns(Record, value=values))
assert records[owners].value['weight'].tolist() == [1, 2]
schema_world.remove(owners, Record)      # Clears reference leaves in inactive rows.
```

`register(*types, capacity=128, allow_same_type_components_per_entity=False)`
returns a tuple of accessors in input order, including for a single type:
`positions, = world.register(Position)`. `world.get(Position)` retrieves another
accessor for the same store. Registration is explicit, limited to 64 types, and
rejects duplicate types and the reserved field names `_entity`/`_count`. Batch
registration validates all schemas before installing any store. The registration
flag enables multicomponents for every type in that call; the decorator enables
them for that component wherever it is registered. Multiple-instance tags are
rejected. There is no unregister operation.

Tags support both `world.add(ids, Sleeping)` and `Sleeping()` payloads, plus
`columns(Sleeping)`. They allocate owner metadata without user field arrays.
Setting an existing tag only checks membership and changes no counters.

## Entity lifecycle

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
single = world.create()                 # Shape (1,), not a scalar.
entity = single[0]                      # np.uint64 handle.
assert world.is_alive(entity).tolist() == [True]
assert world.has(ids, Position).tolist() == [True, True, True]
assert len(world) == 4                  # Includes the component-free entity.
before = world.entity_version
world.delete(single)
replacement = world.create()[0]        # Reuses an index with a new generation.
assert entity_index(replacement) == entity_index(entity)
assert entity_generation(replacement) != entity_generation(entity)
assert not world.is_alive(entity)[0]
assert world.entity_version == before + 2
assert len(world.create(0)) == 0
world.delete([entity, entity])          # Stale handles are ignored.
all_live = world.entities()            # Includes component-free replacement.
```

Handles pack a uint32 index and uint32 generation in uint64. They are local to a
world: do not use a handle from another world, even if its number happens to match.
Generations protect against recycled indices; they do not encode world identity.
`delete()` removes all owned components. Duplicate live handles are deleted once.
`entities()` orders by ascending entity index, not the full numeric handle.
Creation order, especially after reuse, need not match that order.

`is_alive()`/`has()` preserve input order and duplicates and always return a 1-D
boolean array; scalar input yields one entry. `has()` rejects unregistered types
even for empty inputs. `entity_version` advances once per successful nonempty
create/delete call. Empty and stale-only calls do not advance it.

`as_entities()` and `as_rows()` normalize scalars/sequences/1-D arrays to uint64 and
uint32 arrays. They may return the original correctly typed array; copy it if
independent ownership is needed. Nested arrays are rejected. Use complete handles,
not `entity_index(handle)`, when calling entity APIs after recycling.

## Insertion and replacement forms

The following alternatives insert the same two-field component. Each uses fresh
owners, because single-component `add()` is insert-only.

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
forms = [
    (columns(Position, x=[1, 3], y=[2, 4]),),  # Named columns.
    (Position(1, 2),),                       # One instance broadcast to both owners.
    (Position, ([1, 3], [2, 4])),             # Tuple of field columns.
    (Position, np.array([[1, 2], [3, 4]])),   # Packed rows: shape (owners, fields).
    (Position, np.array([1, 2])),            # One packed row broadcast.
]
for payload in forms:
    pair = world.create(2)
    world.add(pair, *payload)
    assert len(positions[pair]) == 2

world.add(ids, columns(Health, value=[10, 20, 30]), Sleeping())
world.set(ids, columns(Position, x=5, y=6), Health(100))
world.remove(ids[:1], Sleeping, Health)    # Missing membership is harmless.
assert not world.has(ids[:1], Sleeping)[0]
```

For a one-field component, raw scalar/1-D payloads are the field values, e.g.
`world.add(ids, Health, [10, 20, 30])`. A tuple with one element explicitly wraps
its field column. For several fields, a tuple always means fields; an ordinary
matrix means rows. Shaped fields require named columns or a tuple of fields,
rather than the packed-matrix payload form.

For insertion/replacement, a numeric field of shape `S` accepts a single value of
shape `S`, `(1,) + S`, or a batch `(owner_count,) + S`. Scalars broadcast for scalar
fields. A vector field needs its full vector shape: scalar `0` is not an insertion
payload for a Vec3 field. Selection field assignment follows NumPy broadcasting,
so `selection.position = 0` can fill an existing vector field.

`set()` requires every live owner to already have each single component; it
preserves rows. `add()` rejects existing single components, duplicate live owners,
and repeated single types in one call. Multiple types preflight together. Repeated
multicomponent payloads append and are combined for limit validation. `set()` does
not support multicomponents; edit selected instances instead. No implicit upsert
or automatic registration is performed.

Stale handles are filtered with their payload entries in lockstep. Payload shapes
refer to the original request length, including stale entries. Empty requests,
requests with no component payloads, and stale-only requests can return before
payload validation; do not use them to validate a schema or payload.

## Selections and row lifetimes

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
selection = positions[ids[[2, 0, 2]]]
assert selection.x.tolist() == [2, 0, 2]
selection.take(slice(0, 2)).assign(y=9)
selection['x'] = [30, 10, 40]           # Repeated target: final occurrence wins.
assert positions[ids].x.tolist() == [10, 1, 40]
rows = selection.checked_rows()
assert positions.owner_ids(rows).tolist() == ids[[2, 0, 2]].tolist()
assert len(rows[:1]) == 1
raw = rows.get_rows()                  # Independent raw row-index copy.
assert raw.dtype == np.uint32
world.remove(ids[:1], Position)        # Expires every selection from this store.
try:
    selection.x
except RuntimeError:
    selection = positions[ids]         # Missing/dead owners are silently dropped.
assert len(selection) == 2
```

Accessor indexing preserves entity input order and duplicates while dropping
stale handles and owners without that component. Consequently, independently
indexing different stores can yield differently sized arrays. Use structural
joins with explicit optional presence masks when alignment matters.

`ComponentRows` supports positional indexing; `ComponentSelection` supports
`take(key)` for positions and `['field']` for fields. Both expose `len()` and
`get_rows()`. `ComponentSelection` additionally exposes `checked_rows()`, field
attributes/brackets, `assign()`, `vector()`, and `set_vector()`. Construct them via
accessors or queries, rather than manually using storage objects.

Use brackets for user field names such as `vector`, `assign`, or `get_rows` that
collide with selection methods. Internal metadata is reserved; use `owner_ids()`
for ownership. Raw row numbers are store-local and change after structural edits.
`positions.remove(rows)` removes component instances; `world.remove(ids, Position)`
removes components by entity. `owner_ids()` and accessor `remove()` accept checked
rows/selections or legacy raw rows; only checked inputs verify ownership/lifetime.
Accessor removal deduplicates rows and ignores inactive/out-of-span raw rows.

| Operation | Existing selections from that store |
| --- | --- |
| Field writes, `set()`, `touch()` | Stay valid; reads see current values |
| Nonempty addition/removal, owner deletion | Expire conservatively, even if their own rows did not move |
| Dense reallocation or actual compaction | Expire |
| Sparse-only reservation, unrelated-store edits | Stay valid |
| No-op/empty edit, compaction of an already packed pool | Stay valid |

Saved numeric arrays are detached snapshots and remain readable after selection
expiry. Saved object arrays retain references to their objects.

## Multicomponent instances

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
world.add(ids[[0, 0, 1]], columns(Item, value=[10, 20, 30]))
world.add(ids[0], Item(40))
assert items[ids].value.tolist() == [10, 20, 40, 30]
chosen = items.select_rows(items.query_rows(lambda value: value == 20))
chosen.assign(value=25)                 # Exactly one instance.
items.remove(items.query_rows(lambda value: value >= 30))
assert items[ids].value.tolist() == [10, 25]
assert world.has(ids, Item).tolist() == [True, False, False]
world.remove(ids[0], Item)              # Removes all remaining instances for owner 0.
```

Multicomponents keep instances grouped by owner. Incoming instances retain stable
order within each owner; appends follow existing values. Different owners' physical
blocks may move. Per-owner counts are limited to 65,535; an overflowing append
raises before changing any owner in the request. Stable instance IDs and retained
instance references across structural changes are not provided. Use checked rows
within a structural lifetime and reacquire after edits. Batch appends are much
cheaper than repeated appends, especially when alternating owners.

## Value predicates

```python
from functools import partial

world, ids, positions, velocities, health, sleeping, items = example_world()
def above(x, *, threshold=0):
    return x > threshold

rows = positions.query_rows(partial(above, threshold=0.5), entities=ids[:2])
assert positions.owner_ids(rows).tolist() == [ids[1]]
positions.select_rows(rows).assign(y=7)
assert len(positions.query_rows(lambda: True)) == 3
assert len(positions.query_rows(lambda _entity: _entity == ids[0])) == 1
legacy_rows = positions.query(lambda x, y: (x >= 1) & (y >= 0))
assert legacy_rows.dtype == np.uint32
```

Parameters name component fields. Matching fields are passed even if the parameter
has a default. Non-field parameters need defaults (or must be bound away with a
partial). Named ordinary/keyword-only parameters, functions, bound methods,
callable objects, and inspectable partials are supported. Positional-only and
variadic signatures, uninspectable ufuncs, and async/generator callables are rejected.
Wrap a ufunc as `lambda x: np.isfinite(x)`.

Return a scalar boolean or a 1-D result with one entry per selected component row;
results are coerced to bool. Reduce shaped-field comparisons yourself, e.g.
`lambda position: np.all(position >= 0, axis=1)`. Empty stores validate signatures
without invoking predicates. Entity subsets are deduplicated/sorted before row
lookup. Unrestricted queries follow active storage-row order, not entity order.

`query_rows()` captures matching rows and rejects structural changes to its store
during callback evaluation. It does not undo callback side effects. Predicates
should be read-only. Results are snapshots of membership, not live filters; field
edits keep those rows valid without reevaluating the condition. `query()` is the
legacy unchecked raw-row equivalent. Value predicates are never cached.

## Structural queries and joins

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
world.add(ids[1:], columns(Health, value=[20, 30]))
world.add(ids[2], Sleeping)
query = world.query(Position, Velocity, exclude=Sleeping, optional=Health)
result = query.fetch()
assert result.entities.tolist() == ids[:2].tolist()
p = result[Position].selection
v = result[Velocity].selection
p.assign(x=p.x + v.x * 0.5)             # Aligned required single components.
hp = result[Health]
assert hp.present.tolist() == [False, True]
health_owners = result.entities[hp.present]
assert hp.selection.value.tolist() == [20]
assert hp.offsets.tolist() == [0, 0, 1]

assert len(world.query().entities()) == 3
assert world.query(exclude=Sleeping).entities().tolist() == ids[:2].tolist()
assert len(world.query(optional=Health).entities()) == 3
assert len(world.query(any_of=(Health, Sleeping)).entities()) == 2
assert len(world.query(Position, exclude=Position).entities()) == 0
assert world.where(Position, exclude=Sleeping).tolist() == ids[:2].tolist()
assert len(world.where()) == 0
```

`query(*required, exclude=None, any_of=None, optional=None)` constructs a reusable
query. Required terms are ANDed; a nonempty `any_of` requires at least one match;
`exclude` rejects any matching term. Optional terms do not filter. Keyword terms
accept a type, a sequence, or `None`; duplicates normalize, an empty `any_of` adds
no constraint, and contradictory terms produce no matches. All types must already
be registered. Returned terms are required + any_of + optional, in declaration
order with duplicates removed; exclusion-only terms cannot be indexed in results.

`query.entities()` returns a copy of matching handles without constructing component
row mappings. `fetch()` returns a cached `QueryResult`; `len(result)` is the owner
count, and `result[Type]` yields a `QueryComponent`. Matching owners are ordered by
ascending sparse index. Dense storage order is never used to align different terms.
`where(*required, exclude=...)` is an uncached convenience with legacy empty-term
semantics and no optional/OR terms.

Each returned term exposes `selection`, `present`, and `offsets`. Missing optional
owners are omitted from the packed selection, while their presence bits and offsets
stay aligned with the full result. Multiple multicomponent terms do not zip unequal
counts or form a Cartesian product:

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
world.add(ids[[0, 0, 2]], columns(Item, value=[5, 6, 7]))
result = world.query(Position, optional=Item).fetch()
term = result[Item]
assert term.present.tolist() == [True, False, True]
assert term.offsets.tolist() == [0, 2, 2, 3]
groups = []
for i, owner in enumerate(result.entities):
    start, stop = term.offsets[i:i+2]
    groups.append(term.selection.take(slice(start, stop)).value.tolist())
assert groups == [[5, 6], [], [7]]
```

Exported entity/presence/offset arrays are copies. Field writes reuse cached
membership and rows. Changes to filtering stores' membership refresh membership;
optional-store layout changes refresh row mappings without rescanning membership.
Queries without required terms also track entity lifecycle. Relevant structural
changes expire retained result metadata: refetch before accessing it. An already
extracted selection keeps its own store lifetime rules, even if another query term
changes. Cached results contain current row mappings, not cached field values.

## Version tokens and caches

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
token = positions.version('x', 'y')
layout = positions.layout_version
positions[ids].assign(x=positions[ids].x + 1)
assert positions.version('x', 'y') != token
assert positions.layout_version == layout
assert positions.membership_version == token[0]
token = positions.version('x')
positions[ids].x = positions[ids].x     # Same values still count as a write.
assert positions.version('x') != token
```

`version(*fields)` returns `(membership_version, field_version_1, ...)`; no names
means all schema fields in order. Tags therefore return a one-element membership
token. Compare tokens only for the same store and ordered field list. Membership
is a conservative structural counter, including multicomponent count changes.
`layout_version` tracks row lifetimes and can advance more than once per edit.
Tokens are counters, not timestamps or thread synchronization primitives.

Object mutation requires explicit notification. Broadcast objects alias:

```python
@component
class Payload:
    data: object

object_world = ECS()
payloads, = object_world.register(Payload)
owners = object_world.create(2)
shared = {'score': 0}
object_world.add(owners, Payload(shared))
token = payloads.version('data')
payloads[owners].data[0]['score'] += 1
assert payloads[owners].data[1]['score'] == 1
assert payloads.version('data') == token
payloads.touch('data')
assert payloads.version('data') != token

separate = np.empty(2, dtype=object)
separate[:] = [{'score': 10}, {'score': 20}]
object_world.set(owners, columns(Payload, data=separate))
```

`touch(*fields)` increments named counters without writing values or changing
membership/layout; no names means all fields. Unknown names raise `KeyError` before
any counter changes. Direct writes through internal dense buffers also require
notification, but application code should use selections. Object-field batches
must be 1-D; use an explicitly allocated object array for sequences whose elements
would otherwise become an extra array dimension.

## Deferred commands and recovery

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
commands = world.commands()             # Equivalent to CommandBuffer(world).
created = commands.create(2)
commands.add(created, columns(Position, x=[8, 9], y=0))
commands.set(ids, Position(1, 2))
commands.remove(ids, Velocity)
commands.delete(ids[-1:])
assert len(commands) == 5
assert commands.flush() == 5
new_ids = created.resolve()             # Fresh copy of the created handles.
assert positions[new_ids].x.tolist() == [8, 9]
assert len(commands) == 0
```

Buffers are independent. Targets are copied when enqueued; component payloads are
retained by reference until execution. Commands apply FIFO using ordinary APIs and
full generation checks. A stale command cannot target a replacement entity. Each
flush processes only commands present at entry; commands enqueued during execution
wait until the next flush. The returned count includes successful no-ops.

Deferred tokens can target commands only in their originating buffer. They cannot
be sliced, combined into handle arrays, or automatically resolved inside component
fields. To store entity references, resolve first and enqueue a later payload.
Resolution before creation executes raises `RuntimeError`; failed/cancelled creation
tokens never resolve. Resolving a successful token is not a liveness check.

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
commands = world.commands()
commands.add(ids, Sleeping)             # Succeeds.
commands.add(ids, Position(9, 9))        # Fails: existing single components.
commands.delete(ids[:1])                # Remains pending after failure.
try:
    commands.flush()
except CommandError as error:
    assert error.command_index == 1     # Monotonically increasing, buffer-local.
    assert error.operation == 'add'
    assert isinstance(error.__cause__, ValueError)
assert len(commands) == 1
assert world.has(ids, Sleeping).all()
assert commands.flush() == 1            # Explicitly execute the remaining tail.
cancelled = commands.create()
commands.clear()                       # Discards pending work, cancels this token.
try:
    cancelled.resolve()
except RuntimeError:
    pass
```

The failed command is consumed; successful earlier edits remain. `clear()` does
not undo completed work, reset command numbering, or invalidate already successful
creation tokens. Recursive flush and clearing during a flush raise `RuntimeError`.
Whole-buffer transactionality is not provided.

## Scheduling

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
movement = world.query(Position, Velocity)
schedule = Scheduler(world)

def fixed(world, commands, dt):
    # Fixed tick count/accumulation is controlled by the caller.
    pass

def spawn(world, commands, dt):
    commands.add(commands.create(), Position(0, 0), Velocity(1, 0))

def move(world, commands, dt):
    matched = movement.fetch()
    p, v = matched[Position].selection, matched[Velocity].selection
    p.assign(x=p.x + v.x * dt, y=p.y + v.y * dt)

seen = []
def prepare(world, commands, dt):
    seen.append(len(movement.fetch()))

schedule.add('physics', fixed, phase='fixed_update')
schedule.add('spawn', spawn, before='move', flush_after=True)
schedule.add('move', move, after='spawn')
schedule.add('prepare', prepare, phase='render_prep', after='move')
assert schedule.compile() == ('physics', 'spawn', 'move', 'prepare')
schedule.run(1 / 60, fixed_steps=2, fixed_dt=1 / 120)
assert seen == [4]
```

Callbacks receive `(world, commands, dt)` and execute sequentially. Names must be
unique/nonempty. Dependencies accept names or sequences, including forward
references. Ties within a phase keep registration order. `compile()` validates the
whole graph before execution and returns ordered names; `run()` compiles lazily.
Cycles, unknown dependencies, and dependencies contradicting phase order raise
`ValueError` before commands flush or callbacks execute.

| Boundary | Command visibility |
| --- | --- |
| Before the run | `schedule.commands` is flushed |
| After each fixed tick | Queued edits become visible to the next tick/update |
| After update / render-prep phases | Queued edits become visible, even if phase is empty |
| After `flush_after=True` system | Edits become visible to the next system in that phase |
| Dependency alone | Orders callbacks; does not create a flush |

Refetch selections after a structural flush. Without `flush_after=True` above,
movement would process the previous membership set; the spawn would appear at the
update-phase boundary. Systems can access `schedule.commands` before running.

`dt` must be finite/nonnegative, `fixed_dt` finite/positive, and `fixed_steps` a
nonnegative indexable integer (default zero). No automatic wall-clock accumulator
or parallel scheduling is provided. Adding systems invalidates the compiled order;
editing or recursively running an active schedule raises `RuntimeError`.
There is no remove-system API. A system exception stops the run and leaves its
commands pending; direct writes already performed remain. A boundary flush failure
also stops later systems and exposes the buffer for recovery or clearing.

## Diagnostics reference

```python
world, ids, positions, velocities, health, sleeping, items = example_world()
world.check_invariants()
info = world.inspect()
assert info['entities'] == len(world)
assert info['array_bytes'] >= info['entity_state_bytes']
stats = positions.inspect()
assert stats['component'] is Position
assert stats['owners'] == stats['rows'] == 3
assert stats['fields']['x']['shape'] == ()
mask_snapshot = world.entity_masks
mask_snapshot[:] = 0                   # Does not mutate membership.
assert world.has(ids, Position).all()
```

| Snapshot | Keys |
| --- | --- |
| World | `entities`, `entity_capacity`, `entity_state_bytes`, `components`, `array_bytes` |
| Component | `component`, `fields`, `owners`, `rows`, `span`, `capacity`, `holes`, `dense_bytes`, `sparse_bytes` |
| Field metadata | `dtype` (string), `shape` (tuple), `hasobject` (boolean) |

World `components` is a tuple of component snapshots. `owners` counts distinct
owners; `rows` counts active instances; `span` includes multicomponent holes;
`holes = span - rows`. Capacity includes unused tail space. `component` is the class
itself, so snapshots are not directly JSON-serializable without adapting that key.
Invariant checking verifies live handles, recycled indices, ownership, masks,
sparse mappings, contiguous multicomponent blocks/counts, and inactive reference
cleanup. Its scans are for debugging and are disabled by Python's `-O` assertions.

## API inventory and error reference

The tables summarize application entry points; the examples above demonstrate
their return values and lifetime rules. Mutators return `None` unless specified.

| Object | Public entry points |
| --- | --- |
| Definitions | `component(cls=None, *, multiples=False)`, `columns(Type, /, **fields)` |
| World setup | `ECS()`, `register(*types, allow_same_type_components_per_entity=False, capacity=128) -> tuple`, `get(Type) -> ComponentAccessor` |
| Entity state | `create(quantity=1) -> EntityArray`, `delete(ids)`, `len(world)`, `entities()`, `is_alive(ids)`, `has(ids, Type)`, `entity_version`, `entity_masks` |
| World mutation | `add(ids, *payloads)`, `set(ids, *payloads)`, `remove(ids, *types)` |
| World queries | `where(*required, exclude=None) -> EntityArray`, `query(*required, exclude=None, any_of=None, optional=None) -> Query` |
| Accessor selection | `[ids] -> ComponentSelection`, `rows() -> ComponentRows`, `select_rows(checked) -> ComponentSelection`, `owner_ids(rows) -> EntityArray` |
| Accessor predicates | `query(condition, entities=None) -> IndexArray`, `query_rows(condition, entities=None) -> ComponentRows` |
| Accessor mutation/tracking | `remove(rows)`, `touch(*fields)`, `version(*fields) -> tuple`, `membership_version`, `layout_version` |
| Checked rows | `len(rows)`, `rows[key] -> ComponentRows`, `get_rows() -> IndexArray` |
| Selection | `len(selection)`, `['field']`, `.field`, `assign(**fields)`, `take(key)`, `checked_rows()`, `get_rows()`, `vector(*fields)`, `set_vector(value, *fields)` |
| Query/result | `Query.entities()`, `Query.fetch()`, `len(result)`, `result.entities`, `result[Type]`; term `.selection`, `.present`, `.offsets` |
| Allocation/inspection | World `reserve(capacity)`, `inspect()`, `check_invariants()`; accessor `reserve(capacity, sparse_capacity=None)`, `compact()`, `inspect()` |
| Deferred edits | `world.commands()` / `CommandBuffer(world)`, `len(buffer)`, `create`, `add`, `set`, `remove`, `delete`, `flush() -> int`, `clear()`, token `resolve() -> EntityArray` |
| Scheduling | `Scheduler(world)`, `.commands`, `add(name, callback, phase='update', before=(), after=(), flush_after=False)`, `compile() -> tuple[str, ...]`, `run(dt, fixed_steps=0, fixed_dt=1/60)` |

`exclude`, `any_of`, `optional`, scheduler configuration after the callback, run
options after `dt`, and `sparse_capacity` are keyword-only arguments.
Accessor `get_rows()` is also available as an unchecked export of all active rows.

Exported numeric types are `Entity=np.uint64`, `Index=np.uint32`,
`Generation=np.uint32`, `Mask=np.uint64`, `ComponentPerEntityCount=np.uint16`.
Array aliases include `EntityArray`, `IndexArray`, `GenerationArray`, `MaskArray`,
`CountArray`, `BoolArray`, and `FieldArray`; input aliases are `EntityLike`,
`IndexLike`, and `RowLike`. `NONE` is the missing row sentinel, `NO_ENTITY` the
missing owner sentinel; neither is a usable live identity. `entity_index()` and
`entity_generation()` accept a scalar or array and preserve scalar/array form.
`higher_pow2(n)` returns the power of two strictly above nonnegative `n` (zero
returns one), as used by growth policies. `arg_names(predicate)` returns validated
predicate parameter names.

The facade also exports `ComponentColumns`, `ComponentSchema`, `ComponentStorage`,
and `MultiComponentStorage`. Schema/storage construction is implementation-facing;
application code should register types and use inspection, selections, and queries.
Methods beginning with `_` and dense/sparse buffers are internal.

| Error | Typical cause |
| --- | --- |
| `KeyError` | Unregistered type, unknown field, or indexing a query term not requested |
| `ValueError` | Invalid shape, missing/existing single-component membership, duplicates, reserved schema fields, negative ID/count, invalid schedule/dependencies |
| `TypeError` | Noninteger/boolean handle or creation quantity, unsupported callback, wrong row wrapper, unsupported vector dtype, multicomponent `set()` or tag registration |
| `OverflowError` | Handle/row representation overflow, exhausted index space, owner instance-count overflow |
| `RuntimeError` | Stale selection/result, unresolved/cancelled token, reentrant flush/run, editing an active schedule |
| `CommandError` | Failed command; inspect `.command_index`, `.operation`, and `.__cause__` |
| `AssertionError` | Explicit invariant check found inconsistent storage |

Checked rows from a different store/world raise `ValueError`. Raw out-of-bounds
owner lookup can raise NumPy `IndexError`; use checked rows for retained work.
Expected validation failures preserve values and version counters within the API
request. They do not roll back arbitrary callback effects, allocation failures,
earlier successful buffered commands, or earlier systems' work. The ECS provides
no thread synchronization. Relationships, serialization, resource injection,
automatic parallel scheduling, and more than 64 component types are outside this API.


## Explicit arrays and reusable field reads

Use `array_field(np.float32, shape=(3,))` as a dataclass field declaration when
the annotation alone does not describe the storage shape. `object` explicitly
selects Python-reference storage; unsupported annotations raise `TypeError`.

`selection.read(name, out=array)` gathers into an exact dtype/shape output and
retains the same output object. Storage aliases and read-only outputs are rejected.
The selection caches its index conversion after the first reusable read.

`with accessor.edit_dense(*fields) as columns:` borrows numeric columns in storage
order. Structural edits to the store, nested writable scopes, and multi-store
mutations affecting the borrowed store are rejected before applying changes.
Fields are touched on exit even after exceptions; writes are not rolled back.
Exported views must not escape the scope. Compact multicomponent holes first.
Independent component stores still require joins/gathers to align their rows.

See [the procedural guide](PROCEDURAL_API.md) for a complete shaped-field example
and [array measurements](ARRAY_BENCHMARKS.md) for scratch-memory tradeoffs.
