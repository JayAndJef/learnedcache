"""Binary log loading for cache trace data in workload/iter directory layout.

Copied verbatim from ``learnedcache.binary_loading`` (kept independent so the
ranker package is untouched). Access records are 88-byte structs, eviction
records are 8-byte ``ts`` values; both are memory-mapped via numpy structured
dtypes -- no CSV conversion needed.
"""

from __future__ import annotations

import glob
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Binary format: 88-byte access records, 8-byte eviction records.
# See cache_ext_lc/policies/read_binary_logs.py and the kernel tracepoints.
_ACCESS_DTYPE = np.dtype(
    [
        ("ts", "<u8"),
        ("pd", "<u8"),
        ("p2", "<u8"),
        ("id", "<u8"),
        ("i2", "<u8"),
        ("dm", "<u4"),
        ("dn", "<u4"),
        ("in", "<u8"),
        ("of", "<u8"),
        ("sd", "<u4"),
        ("_pad", "<u4"),
        ("sz", "<u8"),
        ("fq", "<u4"),
        ("ie", "<u4"),
    ]
)

_EVICTION_DTYPE = np.dtype([("ts", "<u8")])

# 32-byte cache_insertion event: ts, major, minor, inode, page index.
_INSERTION_DTYPE = np.dtype(
    [
        ("ts", "<u8"),
        ("dm", "<u4"),
        ("dn", "<u4"),
        ("in", "<u8"),
        ("ix", "<u8"),
    ]
)

# All meaningful access-log fields (excludes the alignment padding).
_ACCESS_FIELDS: tuple[str, ...] = tuple(
    name for name in _ACCESS_DTYPE.names if name != "_pad"
)


def _mmap_or_warn(filepath: Path, dtype: np.dtype, label: str) -> np.ndarray:
    """Memory-map *filepath* as a structured array of *dtype*.

    Validates the file size is a multiple of the record size, warns on
    truncation, and returns an empty array for zero-length files.
    """
    file_size = filepath.stat().st_size
    record_size = dtype.itemsize
    expected_count = file_size // record_size
    if expected_count == 0:
        return np.empty(0, dtype=dtype)
    remainder = file_size % record_size
    if remainder:
        warnings.warn(
            f"{filepath}: {label} has trailing {remainder} bytes "
            f"(expected a multiple of {record_size}); truncating."
        )
    return np.memmap(filepath, dtype=dtype, mode="r")


def read_binary_access_log(filepath: str | Path) -> np.ndarray:
    """Memory-map a binary cache-access log file (dtype ``_ACCESS_DTYPE``)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Access log not found: {filepath}")
    return _mmap_or_warn(filepath, _ACCESS_DTYPE, "access log")


def read_binary_eviction_log(filepath: str | Path) -> np.ndarray:
    """Memory-map a binary cache-eviction log file (single field ``ts``)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Eviction log not found: {filepath}")
    return _mmap_or_warn(filepath, _EVICTION_DTYPE, "eviction log")


def read_binary_insertion_log(filepath: str | Path) -> np.ndarray:
    """Memory-map a binary cache-insertion log file (dtype ``_INSERTION_DTYPE``)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Insertion log not found: {filepath}")
    return _mmap_or_warn(filepath, _INSERTION_DTYPE, "insertion log")


_ITER_PATTERN = re.compile(r"^iter_(\d+)$")


def discover_workloads_and_iters(
    base_dir: str | Path,
    workloads: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Scan *base_dir* for workload subdirectories containing ``iter_*`` dirs.

    Returns a dict mapping ``workload_name -> sorted list of iter dir paths``.
    Raises ``ValueError`` if no workloads are found or a requested one is missing.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise ValueError(f"Data directory not found: {base_dir}")

    available: dict[str, Path] = {}
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            available[entry.name] = entry

    if workloads is not None:
        missing = [w for w in workloads if w not in available]
        if missing:
            raise ValueError(
                f"Workload(s) not found: {missing}. Available: {sorted(available)}"
            )
        selected = {w: available[w] for w in workloads}
    else:
        selected = available

    if not selected:
        raise ValueError(f"No workload directories found in {base_dir}")

    result: dict[str, list[Path]] = {}
    for name, path in sorted(selected.items()):
        iter_dirs: list[tuple[int, Path]] = []
        for entry in sorted(path.iterdir()):
            m = _ITER_PATTERN.match(entry.name)
            if m and entry.is_dir():
                iter_dirs.append((int(m.group(1)), entry))
        iter_dirs.sort(key=lambda x: x[0])
        if not iter_dirs:
            raise ValueError(f"No iter_* directories found in workload '{name}': {path}")
        result[name] = [d for _, d in iter_dirs]

    return result


def build_pairs_from_binary(
    iter_dirs: list[Path],
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Yield ``(trial_id, access, eviction, insertion)`` for a workload's iters.

    Each iter dir must contain exactly one ``mglru_lc_access_*.bin`` and one
    ``mglru_lc_eviction_*.bin``; the insertion log (``mglru_lc_insertion_*.bin``)
    is optional and yielded as ``None`` when absent (the sampler then skips the
    re-insertion TSA-anchor correction). Arrays are memory-mapped and yielded
    one trial at a time so only one trial's data is resident at once.
    """
    for trial_id, iter_dir in enumerate(iter_dirs):
        access_files = sorted(glob.glob(str(iter_dir / "mglru_lc_access_*.bin")))
        eviction_files = sorted(glob.glob(str(iter_dir / "mglru_lc_eviction_*.bin")))
        insertion_files = sorted(glob.glob(str(iter_dir / "mglru_lc_insertion_*.bin")))

        if len(access_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 access file in {iter_dir}, found {len(access_files)}"
            )
        if len(eviction_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 eviction file in {iter_dir}, found {len(eviction_files)}"
            )
        if len(insertion_files) > 1:
            raise FileNotFoundError(
                f"Expected at most 1 insertion file in {iter_dir}, found {len(insertion_files)}"
            )
        if not insertion_files:
            warnings.warn(
                f"{iter_dir}: no insertion log; TSA-anchor correction disabled."
            )

        access_arr = read_binary_access_log(access_files[0])
        eviction_arr = read_binary_eviction_log(eviction_files[0])
        insertion_arr = (
            read_binary_insertion_log(insertion_files[0]) if insertion_files else None
        )
        yield trial_id, access_arr, eviction_arr, insertion_arr
        del access_arr, eviction_arr, insertion_arr
