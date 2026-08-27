from __future__ import annotations

import inspect
import queue
import threading

import numpy as np
import pytest
import watchdog.events as upstream_events
from watchdog.observers.api import EventQueue as UpstreamEventQueue
from watchdog.observers.api import ObservedWatch as UpstreamObservedWatch
from watchdog.utils.bricks import SkipRepeatsQueue as UpstreamSkipRepeatsQueue

import mojo_watchdog.events as events
import mojo_watchdog._lib as native
from mojo_watchdog import EventCoalescer, coalesce_events, coalesce_indices
from mojo_watchdog.observers.api import EventQueue, ObservedWatch
from mojo_watchdog.utils.bricks import SkipRepeatsQueue

EVENT_CLASSES = [
    "FileSystemEvent",
    "FileSystemMovedEvent",
    "FileDeletedEvent",
    "FileModifiedEvent",
    "FileCreatedEvent",
    "FileMovedEvent",
    "FileClosedEvent",
    "FileClosedNoWriteEvent",
    "FileOpenedEvent",
    "DirDeletedEvent",
    "DirModifiedEvent",
    "DirCreatedEvent",
    "DirMovedEvent",
]


def event_state(event):
    return (
        type(event).__name__,
        event.src_path,
        event.dest_path,
        event.event_type,
        event.is_directory,
        event.is_synthetic,
    )


def drain(q):
    values = []
    while True:
        try:
            values.append(q.get_nowait())
        except queue.Empty:
            return values


@pytest.mark.parametrize("name", EVENT_CLASSES)
def test_event_classes_match_upstream(name):
    ours_cls = getattr(events, name)
    upstream_cls = getattr(upstream_events, name)
    arguments = (b"old", b"new", True) if "Moved" in name else (b"path", "", True)
    ours = ours_cls(*arguments)
    upstream = upstream_cls(*arguments)
    assert inspect.signature(ours_cls) == inspect.signature(upstream_cls)
    assert event_state(ours) == event_state(upstream)
    assert repr(ours).replace("mojo_watchdog.", "watchdog.") == repr(upstream)
    assert hash(ours) == hash(upstream)


@pytest.mark.parametrize("queue_classes", [(SkipRepeatsQueue, UpstreamSkipRepeatsQueue), (EventQueue, UpstreamEventQueue)])
def test_queue_sequence_parity(queue_classes):
    ours_cls, upstream_cls = queue_classes
    ours = ours_cls()
    upstream = upstream_cls()
    ours_values = [
        events.FileModifiedEvent("/a"),
        events.FileModifiedEvent("/a"),
        events.FileCreatedEvent("/a"),
        events.FileCreatedEvent("/a"),
        events.FileModifiedEvent("/a"),
        events.FileModifiedEvent("/b"),
    ]
    upstream_values = [
        getattr(upstream_events, type(item).__name__)(
            item.src_path, item.dest_path, item.is_synthetic
        )
        for item in ours_values
    ]
    for value in ours_values:
        ours.put(value)
    for value in upstream_values:
        upstream.put(value)
    assert [event_state(value) for value in drain(ours)] == [
        event_state(value) for value in drain(upstream)
    ]


def test_queue_boundary_and_full_behavior_match_upstream():
    for queue_cls in (SkipRepeatsQueue, UpstreamSkipRepeatsQueue):
        q = queue_cls(maxsize=1)
        first = events.FileCreatedEvent("x")
        if queue_cls is UpstreamSkipRepeatsQueue:
            first = upstream_events.FileCreatedEvent("x")
        q.put(first)
        q.put(type(first)("x"), block=False)
        assert q.qsize() == 1
        with pytest.raises(queue.Full):
            q.put(type(first)("y"), block=False)
        assert q.get_nowait() == first
        q.put(type(first)("x"), block=False)
        assert q.qsize() == 1


def test_queue_thread_safety_and_task_accounting():
    q = EventQueue()
    count = 2000

    def producer(prefix):
        for index in range(count):
            event = events.FileModifiedEvent(f"{prefix}/{index}")
            q.put(event)
            q.put(event)

    threads = [threading.Thread(target=producer, args=(prefix,)) for prefix in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    values = drain(q)
    assert count * 2 <= len(values) <= count * 4
    assert all(left != right for left, right in zip(values, values[1:]))
    assert len(set(values)) == count * 2
    assert q.unfinished_tasks == len(values)
    for _ in values:
        q.task_done()
    q.join()


def test_observed_watch_parity():
    ours = ObservedWatch(
        "/tmp", recursive=True, event_filter=[events.FileCreatedEvent, events.FileMovedEvent]
    )
    upstream = UpstreamObservedWatch(
        "/tmp",
        recursive=True,
        event_filter=[upstream_events.FileCreatedEvent, upstream_events.FileMovedEvent],
    )
    assert inspect.signature(ObservedWatch) == inspect.signature(UpstreamObservedWatch)
    assert ours.path == upstream.path
    assert ours.is_recursive == upstream.is_recursive
    assert {cls.__name__ for cls in ours.event_filter} == {
        cls.__name__ for cls in upstream.event_filter
    }
    assert repr(ours) == repr(upstream)
    assert ours == ObservedWatch(
        "/tmp", recursive=True, event_filter=[events.FileMovedEvent, events.FileCreatedEvent]
    )
    assert hash(ours) == hash(
        ObservedWatch(
            "/tmp",
            recursive=True,
            event_filter=[events.FileCreatedEvent, events.FileMovedEvent],
        )
    )


@pytest.mark.parametrize("rows,columns", [(1, 1), (100, 1), (1000, 6), (4097, 9)])
def test_numeric_kernel_matches_numpy(rows, columns):
    rng = np.random.default_rng(rows + columns)
    keys = rng.integers(0, 8, size=(rows, columns), dtype=np.int64)
    keys[1::3] = keys[::3][: len(keys[1::3])]
    expected = np.flatnonzero(
        np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
    )
    assert np.array_equal(coalesce_indices(keys), expected)


def test_numeric_kernel_empty_and_validation():
    assert coalesce_indices(np.empty((0, 4), dtype=np.int64)).size == 0
    with pytest.raises(ValueError, match="two-dimensional"):
        coalesce_indices(np.arange(5))
    with pytest.raises(ValueError, match="at least one"):
        coalesce_indices(np.empty((3, 0), dtype=np.int64))
    with pytest.raises(TypeError, match="integer dtype"):
        coalesce_indices(np.ones((3, 2), dtype=np.float64))
    with pytest.raises(TypeError, match="represented safely"):
        coalesce_indices(np.ones((3, 2), dtype=np.uint64))


def test_numeric_kernel_accepts_safe_integer_conversion_and_strides():
    base = np.array([[1, 9, 2, 9], [1, 8, 2, 8], [3, 7, 4, 7]], dtype=np.int32)
    keys = base[:, ::2]
    assert not keys.flags.c_contiguous
    assert coalesce_indices(keys).tolist() == [0, 2]


def test_numeric_kernel_rejects_invalid_native_count(monkeypatch):
    class InvalidLibrary:
        @staticmethod
        def mwd_coalesce_indices(*_args):
            return 3

    monkeypatch.setattr(native, "lib", lambda: InvalidLibrary())
    with pytest.raises(RuntimeError, match="invalid result count"):
        native.coalesce_indices(np.ones((2, 1), dtype=np.int64))


def test_numeric_kernel_simd_tail():
    keys = np.zeros((19, 17), dtype=np.int64)
    keys[3, 16] = 1
    keys[4, 16] = 1
    keys[10, 8] = 2
    assert coalesce_indices(keys).tolist() == [0, 3, 5, 10, 11]


def test_numeric_kernel_single_simd_vector_and_tail():
    keys = np.zeros((7, 6), dtype=np.int64)
    keys[2:6, 0] = 1
    keys[4:6, 5] = 2
    assert coalesce_indices(keys).tolist() == [0, 2, 4, 6]


@pytest.mark.parametrize("rows", [999_999, 1_000_003])
def test_numeric_kernel_large_input(rows):
    keys = np.zeros((rows, 6), dtype=np.int64)
    keys[500_001:500_003] = 1
    keys[-1] = 2
    assert coalesce_indices(keys).tolist() == [0, 500_001, 500_003, rows - 1]


def test_batch_coalescing_matches_upstream_event_queue():
    rng = np.random.default_rng(4)
    classes = [
        events.FileCreatedEvent,
        events.FileModifiedEvent,
        events.FileDeletedEvent,
        events.FileMovedEvent,
        events.DirModifiedEvent,
    ]
    values = []
    for index in range(2000):
        cls = classes[int(rng.integers(len(classes)))]
        src = f"/tree/{int(rng.integers(20))}"
        dest = f"/tree/{int(rng.integers(20))}" if cls is events.FileMovedEvent else ""
        event = cls(src, dest, bool(index % 7 == 0))
        values.extend([event] * int(rng.integers(1, 5)))

    upstream = UpstreamEventQueue()
    for event in values:
        cls = getattr(upstream_events, type(event).__name__)
        upstream.put(cls(event.src_path, event.dest_path, event.is_synthetic))

    assert [event_state(value) for value in coalesce_events(values)] == [
        event_state(value) for value in drain(upstream)
    ]


def test_batch_preserves_instances_and_distinguishes_all_fields():
    first = events.FileModifiedEvent("/x")
    same = events.FileModifiedEvent("/x")
    synthetic = events.FileModifiedEvent("/x", is_synthetic=True)
    directory = events.DirModifiedEvent("/x")
    byte_path = events.FileModifiedEvent(b"/x")
    result = coalesce_events([first, same, synthetic, directory, byte_path])
    assert result == [first, synthetic, directory, byte_path]
    assert result[0] is first


def test_batch_validates_events_and_previous():
    event = events.FileModifiedEvent("/x")
    with pytest.raises(TypeError, match="FileSystemEvent"):
        coalesce_events([object()])
    with pytest.raises(TypeError, match="FileSystemEvent"):
        coalesce_events([event, object()])
    with pytest.raises(TypeError, match="FileSystemEvent"):
        coalesce_events([event], previous=object())
    with pytest.raises(TypeError, match="FileSystemEvent"):
        coalesce_events([], previous=object())


def test_stateful_coalescer_handles_batch_edges_and_reset():
    a1 = events.FileModifiedEvent("/a")
    a2 = events.FileModifiedEvent("/a")
    b = events.FileModifiedEvent("/b")
    coalescer = EventCoalescer()
    assert coalescer.coalesce([a1, a2]) == [a1]
    assert coalescer.coalesce([a2, b, b]) == [b]
    assert coalescer.last_event is b
    assert coalescer.coalesce([]) == []
    coalescer.reset()
    assert coalescer.last_event is None
    assert coalescer.coalesce([a2]) == [a2]


def test_handler_dispatch_matches_upstream_order():
    calls = []

    class Handler(events.FileSystemEventHandler):
        def on_any_event(self, event):
            calls.append(("any", event))

        def on_created(self, event):
            calls.append(("created", event))

    event = events.FileCreatedEvent("/new")
    Handler().dispatch(event)
    assert calls == [("any", event), ("created", event)]


@pytest.mark.parametrize(
    "event,method",
    [
        (events.FileMovedEvent("a", "b"), "moved"),
        (events.FileCreatedEvent("a"), "created"),
        (events.FileDeletedEvent("a"), "deleted"),
        (events.FileModifiedEvent("a"), "modified"),
        (events.FileClosedEvent("a"), "closed"),
        (events.FileClosedNoWriteEvent("a"), "closed_no_write"),
        (events.FileOpenedEvent("a"), "opened"),
    ],
)
def test_handler_dispatch_supports_every_event_type(event, method):
    calls = []

    class Handler(events.FileSystemEventHandler):
        def on_any_event(self, dispatched):
            calls.append(("any", dispatched))

        def __getattribute__(self, name):
            if name == f"on_{method}":
                return lambda dispatched: calls.append((method, dispatched))
            return super().__getattribute__(name)

    Handler().dispatch(event)
    assert calls == [("any", event), (method, event)]
