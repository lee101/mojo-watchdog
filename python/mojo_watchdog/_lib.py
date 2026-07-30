from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_WATCHDOG_LIB") or os.path.join(
    ROOT, "dist", "libmojo-watchdog.so"
)

I64 = ctypes.c_int64
_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "coalesce.mojo")
    if not force and os.path.exists(LIB) and (
        not os.path.exists(source) or os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        raise BuildError(
            f"compiled library not found at {LIB}; set MOJO_WATCHDOG_LIB to its path"
        )
    result = subprocess.run(
        ["bash", script], cwd=ROOT, text=True, capture_output=True, timeout=1800
    )
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        fn = _library.mwd_coalesce_indices
        fn.argtypes = [I64, I64, I64, I64]
        fn.restype = I64
    return _library


def coalesce_indices(keys: np.ndarray) -> np.ndarray:
    """Return the starting row of every run of equal adjacent integer keys."""
    array = np.asarray(keys)
    if array.ndim != 2:
        raise ValueError("keys must be a two-dimensional array")
    if array.shape[1] == 0:
        raise ValueError("keys must have at least one column")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("keys must have an integer dtype")
    if not np.can_cast(array.dtype, np.dtype(np.int64), casting="safe"):
        raise TypeError("keys dtype cannot be represented safely as int64")
    array = np.ascontiguousarray(array, dtype=np.int64)
    if not len(array):
        return np.empty(0, dtype=np.int64)
    indices = np.empty(len(array), dtype=np.int64)
    count = lib().mwd_coalesce_indices(
        array.ctypes.data,
        len(array),
        array.shape[1],
        indices.ctypes.data,
    )
    if count < 1 or count > len(array):
        raise RuntimeError(f"native kernel returned invalid result count {count}")
    return indices[:count]
