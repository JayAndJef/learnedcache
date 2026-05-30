"""Binary log loading for cache trace data in workload/iter directory layout."""

from __future__ import annotations

import glob
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Binary format: 88-byte access records, 8-byte eviction records.
# See read_binary_logs.py and the kernel tracepoint definitions.
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
            f"{filepath}: expected {expected_count} records, got {expected_count} "
            f"(trailing {remainder} bytes truncated)"
        )
    return np.memmap(filepath, dtype=dtype, mode="r")


def read_binary_access_log(filepath: str | Path) -> np.ndarray:
    """Memory-map a binary cache-access log file.

    Returns a numpy structured array with dtype ``_ACCESS_DTYPE``.  The file
    is memory-mapped so pages are loaded on demand -- no heap allocation
    occurs until the caller accesses fields or sorts.

    Use ``_ACCESS_FIELDS`` to iterate over the meaningful column names
    (``_pad`` is excluded).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Access log not found: {filepath}")
    return _mmap_or_warn(filepath, _ACCESS_DTYPE, "access log")


def read_binary_eviction_log(filepath: str | Path) -> np.ndarray:
    """Memory-map a binary cache-eviction log file.

    Returns a numpy structured array with a single field ``ts``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Eviction log not found: {filepath}")
    return _mmap_or_warn(filepath, _EVICTION_DTYPE, "eviction log")


_ITER_PATTERN = re.compile(r"^iter_(\d+)$")


def discover_workloads_and_iters(
    base_dir: str | Path,
    workloads: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Scan base_dir for workload subdirectories containing iter_* dirs.

    Args:
        base_dir: Root directory containing workload subdirectories.
        workloads: Optional list of workload names to process. If None, all
            workloads in base_dir are used.

    Returns:
        Dict mapping workload_name -> sorted list of iter directory paths.

    Raises:
        ValueError: If no workloads are found, or a requested workload doesn't exist.
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
        iter_dirs: list[Path] = []
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
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield access/eviction trial pairs from a workload's sorted iter directories.

    For each iter directory, finds the access and eviction binary files
    (mglru_lc_access_*.bin and mglru_lc_eviction_*.bin), memory-maps them
    into numpy structured arrays, and assigns trial_id = iter_index.

    Yields one (trial_id, access_arr, eviction_arr) at a time so that only
    one trial's arrays are held in memory at any point.  The caller should
    iterate (not list()) to preserve the memory benefit.

    No ``pd.to_numeric`` is needed: the binary fields are already typed
    as unsigned integers by the structured dtype.

    Yields:
        Tuples of (trial_id, access, eviction) where *access* is a structured
        array with dtype ``_ACCESS_DTYPE`` and *eviction* is a structured
        array with dtype ``_EVICTION_DTYPE``.
    """
    for trial_id, iter_dir in enumerate(iter_dirs):
        access_files = sorted(glob.glob(str(iter_dir / "mglru_lc_access_*.bin")))
        eviction_files = sorted(glob.glob(str(iter_dir / "mglru_lc_eviction_*.bin")))

        if len(access_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 access file in {iter_dir}, found {len(access_files)}: {access_files}"
            )
        if len(eviction_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 eviction file in {iter_dir}, found {len(eviction_files)}: {eviction_files}"
            )

        access_arr = read_binary_access_log(access_files[0])
        eviction_arr = read_binary_eviction_log(eviction_files[0])
        yield trial_id, access_arr, eviction_arr
        del access_arr, eviction_arr
