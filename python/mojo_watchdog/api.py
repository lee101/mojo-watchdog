from __future__ import annotations

from pathlib import Path

from .bricks import SkipRepeatsQueue
from .events import FileSystemEvent


class EventQueue(SkipRepeatsQueue):
    pass


class ObservedWatch:
    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool,
        event_filter: list[type[FileSystemEvent]] | None = None,
    ):
        self._path = str(path) if isinstance(path, Path) else path
        self._is_recursive = recursive
        self._event_filter = (
            frozenset(event_filter) if event_filter is not None else None
        )

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_recursive(self) -> bool:
        return self._is_recursive

    @property
    def event_filter(self) -> frozenset[type[FileSystemEvent]] | None:
        return self._event_filter

    @property
    def key(self) -> tuple[str, bool, frozenset[type[FileSystemEvent]] | None]:
        return self.path, self.is_recursive, self.event_filter

    def __eq__(self, watch: object) -> bool:
        if not isinstance(watch, ObservedWatch):
            return NotImplemented
        return self.key == watch.key

    def __ne__(self, watch: object) -> bool:
        if not isinstance(watch, ObservedWatch):
            return NotImplemented
        return self.key != watch.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        if self.event_filter is not None:
            event_filter_str = "|".join(
                sorted(cls.__name__ for cls in self.event_filter)
            )
            event_filter_str = f", event_filter={event_filter_str}"
        else:
            event_filter_str = ""
        return (
            f"<{type(self).__name__}: path={self.path!r}, "
            f"is_recursive={self.is_recursive}{event_filter_str}>"
        )


def coalesce_events(
    events: list[FileSystemEvent] | tuple[FileSystemEvent, ...],
    *,
    previous: FileSystemEvent | None = None,
) -> list[FileSystemEvent]:
    """Drop consecutive equal events, preserving order and object identity."""
    if previous is not None and not isinstance(previous, FileSystemEvent):
        raise TypeError("coalesce_events accepts FileSystemEvent instances")
    if not events:
        return []
    result = []
    last = previous
    for event in events:
        if not isinstance(event, FileSystemEvent):
            raise TypeError("coalesce_events accepts FileSystemEvent instances")
        if last is None or event != last:
            result.append(event)
        last = event
    return result


class EventCoalescer:
    """Stateful batch coalescer that also removes repeats across batch edges."""

    def __init__(self) -> None:
        self._last_event: FileSystemEvent | None = None

    @property
    def last_event(self) -> FileSystemEvent | None:
        return self._last_event

    def coalesce(
        self, events: list[FileSystemEvent] | tuple[FileSystemEvent, ...]
    ) -> list[FileSystemEvent]:
        result = coalesce_events(events, previous=self._last_event)
        if events:
            self._last_event = events[-1]
        return result

    def reset(self) -> None:
        self._last_event = None
