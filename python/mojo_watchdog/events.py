from __future__ import annotations

from dataclasses import dataclass, field

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"
EVENT_TYPE_CLOSED = "closed"
EVENT_TYPE_CLOSED_NO_WRITE = "closed_no_write"
EVENT_TYPE_OPENED = "opened"


@dataclass(unsafe_hash=True)
class FileSystemEvent:
    src_path: bytes | str
    dest_path: bytes | str = ""
    event_type: str = field(default="", init=False)
    is_directory: bool = field(default=False, init=False)
    is_synthetic: bool = field(default=False)


class FileSystemMovedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_MOVED


class FileDeletedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_DELETED


class FileModifiedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_MODIFIED


class FileCreatedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_CREATED


class FileMovedEvent(FileSystemMovedEvent):
    pass


class FileClosedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_CLOSED


class FileClosedNoWriteEvent(FileSystemEvent):
    event_type = EVENT_TYPE_CLOSED_NO_WRITE


class FileOpenedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_OPENED


class DirDeletedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_DELETED
    is_directory = True


class DirModifiedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_MODIFIED
    is_directory = True


class DirCreatedEvent(FileSystemEvent):
    event_type = EVENT_TYPE_CREATED
    is_directory = True


class DirMovedEvent(FileSystemMovedEvent):
    is_directory = True


class FileSystemEventHandler:
    def dispatch(self, event: FileSystemEvent) -> None:
        self.on_any_event(event)
        getattr(self, f"on_{event.event_type}")(event)

    def on_any_event(self, event: FileSystemEvent) -> None:
        pass

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        pass

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        pass

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        pass

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        pass

    def on_closed(self, event: FileClosedEvent) -> None:
        pass

    def on_closed_no_write(self, event: FileClosedNoWriteEvent) -> None:
        pass

    def on_opened(self, event: FileOpenedEvent) -> None:
        pass
