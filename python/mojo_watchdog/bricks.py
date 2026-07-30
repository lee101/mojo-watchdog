from __future__ import annotations

import queue
from typing import Any


class SkipRepeatsQueue(queue.Queue):
    def _init(self, maxsize: int) -> None:
        super()._init(maxsize)
        self._last_item = None

    def put(
        self, item: Any, block: bool = True, timeout: float | None = None
    ) -> None:
        if self._last_item is None or item != self._last_item:
            super().put(item, block, timeout)

    def _put(self, item: Any) -> None:
        super()._put(item)
        self._last_item = item

    def _get(self) -> Any:
        item = super()._get()
        if item is self._last_item:
            self._last_item = None
        return item
