# mojo-watchdog

`mojo-watchdog` is a Mojo implementation of filesystem event coalescing with a
Python API shaped like [watchdog](https://github.com/gorakhargosh/watchdog).
It removes repeated adjacent events before a consumer has to dispatch them.

The covered queue and event APIs match the upstream version pinned in
`pixi.lock`. For large event bursts, `coalesce_events` compares adjacent Python
event values directly in one pass. Backends that already represent events as
integer fields can use `coalesce_indices` for a SIMD Mojo scan.

## Covered subset

- All watchdog filesystem event value classes: file and directory create,
  modify, delete, move, open, and close events
- `FileSystemEventHandler` dispatch
- `watchdog.utils.bricks.SkipRepeatsQueue`
- `watchdog.observers.api.EventQueue` and `ObservedWatch`
- Stateful and stateless batch coalescing through `EventCoalescer` and
  `coalesce_events`
- Dense integer-key coalescing through `coalesce_indices`

The package name is `mojo_watchdog`, so switching the covered upstream API
requires changing the import and not the class names or call signatures:

```python
from mojo_watchdog.observers.api import EventQueue
from mojo_watchdog.events import FileModifiedEvent
```

It is not a filesystem observer. Native inotify, FSEvents, kqueue,
ReadDirectoryChangesW, polling observers, emitters, pattern handlers,
`watchmedo`, and event debouncing are not included. The intended integration
point is an observer or producer that already has a burst of events to reduce.

## Install

```bash
pixi install
pixi run build
```

The repository pins its Mojo nightly toolchain. The shared library is written
to `dist/libmojo-watchdog.so`.

Run examples from the repository through Pixi so that the package and compiled
library are both on the expected paths.

## Usage

```python
from mojo_watchdog import EventCoalescer, FileCreatedEvent, FileModifiedEvent

events = [
    FileModifiedEvent("src/app.py"),
    FileModifiedEvent("src/app.py"),
    FileCreatedEvent("src/new.py"),
]

coalescer = EventCoalescer()
assert coalescer.coalesce(events) == [events[0], events[2]]

# The final event from one batch is remembered at the next batch boundary.
assert coalescer.coalesce([FileCreatedEvent("src/new.py")]) == []
```

For a native producer with integer event kind and path IDs:

```python
import numpy as np
from mojo_watchdog import coalesce_indices

keys = np.array([
    [3, 10, 0, 0],
    [3, 10, 0, 0],
    [1, 11, 0, 0],
], dtype=np.int64)
assert coalesce_indices(keys).tolist() == [0, 2]
```

`coalesce_indices` accepts a rectangular NumPy-compatible input with an
integer dtype that can be represented safely as `int64`. It copies strided
inputs into C-contiguous storage for the duration of the native call. Floating
point and narrowing integer conversions are rejected.

## Correctness

Parity tests run against the real watchdog package pinned in `pixi.lock`. They compare event
constructor signatures, fields, representations, hashes, queue results and
capacity behavior, `ObservedWatch`, handler dispatch, randomized batches, and
cross-batch state. The numeric kernel is checked against a NumPy reference over
multiple shapes, including single-vector rows, SIMD remainders, and large
inputs.

```bash
pixi run test
```

## Benchmarks

Fresh output from `pixi run bench` on this machine (Intel Xeon E5-2697 v4 at
2.30 GHz, Linux x86-64):

| case | mojo-watchdog | reference | speedup |
| --- | ---: | ---: | ---: |
| dense int64 keys, 5M x 6 | 37.43 ms | 133.07 ms (NumPy adjacent-row scan) | 3.55x faster |
| event objects, 250k | 113.95 ms | 242.10 ms (`watchdog.EventQueue`) | 2.12x faster |

The object path avoids temporary dense-key matrices and dictionary interning,
so it is faster in this run than feeding the same burst through
`watchdog.EventQueue`. The numeric scan performs at most one integer comparison
per 16 bytes read, far below the roughly 2-flop/byte threshold for useful GPU
offload, so there is no GPU path. A parallel CPU prototype was also slower at
both 5M and 10M rows because its required compaction pass adds memory traffic.

The benchmark uses repeated wall-time measurements after loading the library
and asserts that both implementations produce identical indices or event
counts. It prints the Markdown table above:

```bash
pixi run bench
```

## How it works

The Mojo kernel receives a C-contiguous `int64` matrix with one event per row
and one equality field per column. It compares each row with its immediate
predecessor and writes the starting index of each distinct run into a
caller-owned `int64` output buffer. Rows at least one native vector wide use
native-width SIMD loads and a scalar remainder; narrower rows retain scalar
early exit.

Python event objects use a direct adjacent-value scan, avoiding the former
temporary NumPy matrix, six dictionary lookups per event, and result-index
array. Order and object identity are preserved.

The numeric API calls one exported C-ABI function through `ctypes`. Buffers
cross the boundary as integer addresses and are reconstructed as
`UnsafePointer[Int64, AnyOrigin[mut=True]]` inside Mojo. Mojo allocates no
memory; NumPy owns both the row-major input and output buffers.

## License

MIT.
