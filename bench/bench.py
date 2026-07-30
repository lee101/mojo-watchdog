from __future__ import annotations

import math
import os
import platform
import queue
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojo_watchdog import FileModifiedEvent, coalesce_events, coalesce_indices  # noqa: E402
from watchdog.events import FileModifiedEvent as UpstreamFileModifiedEvent  # noqa: E402
from watchdog.observers.api import EventQueue as UpstreamEventQueue  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def numpy_coalesce(keys):
    return np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])


def upstream_queue_coalesce(events):
    event_queue = UpstreamEventQueue()
    for event in events:
        event_queue.put(event)
    result = []
    while True:
        try:
            result.append(event_queue.get_nowait())
        except queue.Empty:
            return result


def main():
    rng = np.random.default_rng(0)
    rows = 5_000_000
    keys = rng.integers(0, 100_000, size=(rows, 6), dtype=np.int64)
    keys[1::4] = keys[0::4][: len(keys[1::4])]
    keys[2::4] = keys[0::4][: len(keys[2::4])]

    coalesce_indices(keys[:10])
    mojo_numeric, mojo_indices = timeit(lambda: coalesce_indices(keys))
    numpy_numeric, numpy_indices = timeit(lambda: numpy_coalesce(keys))
    assert np.array_equal(mojo_indices, numpy_indices)

    event_count = 250_000
    mojo_events = []
    upstream_events = []
    for index in range(event_count):
        path = f"/workspace/file-{index // 5 % 20_000}"
        mojo_events.append(FileModifiedEvent(path))
        upstream_events.append(UpstreamFileModifiedEvent(path))

    mojo_objects, mojo_result = timeit(lambda: coalesce_events(mojo_events))
    upstream_objects, upstream_result = timeit(
        lambda: upstream_queue_coalesce(upstream_events)
    )
    assert len(mojo_result) == len(upstream_result)

    rows_out = [
        (
            "dense int64 keys, 5M x 6",
            mojo_numeric,
            numpy_numeric,
            "NumPy adjacent-row scan",
        ),
        (
            "event objects, 250k",
            mojo_objects,
            upstream_objects,
            "watchdog.EventQueue",
        ),
    ]

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-watchdog | reference | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, reference_time, reference_name in rows_out:
        ratio = reference_time / mojo_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{reference_time * 1000:.2f} ms ({reference_name}) | "
            f"{ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
