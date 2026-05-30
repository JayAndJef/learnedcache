from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import csv
import glob

import pandas as pd

def _parse_kv_line(line: str) -> dict[str, str]:
    """Parse a whitespace-separated key=value line into a dictionary."""
    parsed: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed

def parse_log_to_csv(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """Parse a single key=value log file line-by-line into a CSV file."""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    with input_path.open("r", encoding="utf-8") as infile:
        first_line = infile.readline()
        if not first_line:
            return

        initial_data = _parse_kv_line(first_line)
        if not initial_data:
            raise ValueError(f"First line in {input_path} did not contain key=value fields.")

        fieldnames = list(initial_data.keys())
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(initial_data)

            for line in infile:
                if not line.strip():
                    continue
                row_data = _parse_kv_line(line)
                if row_data:
                    writer.writerow(row_data)

def transform_logs_to_csvs(input_pattern: str) -> None:
    """Parse all log files matching pattern and write corresponding .csv files."""
    filepaths = sorted(glob.glob(input_pattern))
    for filepath in filepaths:
        parse_log_to_csv(filepath, Path(filepath).with_suffix(".csv"))

def read_csvs_to_dataframe(file_pattern: str) -> pd.DataFrame:
    """
    Read multiple CSV files and concatenate into one dataframe with trial_id.
    """
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        raise ValueError(f"No CSV files matched pattern: {file_pattern}")

    dataframes: list[pd.DataFrame] = []
    for trial_id, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df["trial_id"] = trial_id
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

def _trial_token_from_path(path: str) -> str:
    """
    Derive a pairing token from a file path.

    Supported naming patterns include:
      <token>_access.csv
      <token>_eviction.csv
      <prefix>_access_<id>_access.csv
      <prefix>_eviction_<id>_eviction.csv
    """
    stem = Path(path).stem
    if stem.endswith("_access"):
        stem = stem[:-7]
    elif stem.endswith("_eviction"):
        stem = stem[:-9]

    stem = stem.replace("_access_", "_")
    stem = stem.replace("_eviction_", "_")
    return stem

def _index_files_by_token(filepaths: list[str], file_kind: str) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for path in filepaths:
        token = _trial_token_from_path(path)
        if token in indexed:
            raise ValueError(
                f"Duplicate {file_kind} token '{token}' from files: {indexed[token]} and {path}"
            )
        indexed[token] = path
    return indexed

def read_access_eviction_trial_pairs(
    access_pattern: str,
    eviction_pattern: str,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Lazy-load paired access/eviction CSV files by filename token.

    Yields the CSVs for one trial at a time, keeping at most one trial's
    DataFrames in memory at any point.  Validation (token matching, duplicate
    detection, empty patterns) runs on the first call to ``next()``.

    Expected naming convention:
      <token>_access.csv
      <token>_eviction.csv

    Yields:
      (trial_id, access_df, eviction_df) tuples, ordered by token.
    """
    access_files = sorted(glob.glob(access_pattern))
    eviction_files = sorted(glob.glob(eviction_pattern))

    if not access_files:
        raise ValueError(f"No access files matched pattern: {access_pattern}")
    if not eviction_files:
        raise ValueError(f"No eviction files matched pattern: {eviction_pattern}")

    access_by_token = _index_files_by_token(access_files, "access")
    eviction_by_token = _index_files_by_token(eviction_files, "eviction")

    access_tokens = set(access_by_token)
    eviction_tokens = set(eviction_by_token)

    common_tokens = sorted(access_tokens & eviction_tokens)
    if not common_tokens:
        raise ValueError(
            "No matching access/eviction file pairs found by token. "
            f"access_pattern={access_pattern}, eviction_pattern={eviction_pattern}"
        )

    access_only_tokens = sorted(access_tokens - eviction_tokens)
    eviction_only_tokens = sorted(eviction_tokens - access_tokens)
    if access_only_tokens or eviction_only_tokens:
        raise ValueError(
            "Access/eviction trial token sets must match exactly (no cross-trial boundaries). "
            f"Access-only tokens: {access_only_tokens}; "
            f"Eviction-only tokens: {eviction_only_tokens}"
        )

    for trial_id, token in enumerate(common_tokens):
        access_df = pd.read_csv(access_by_token[token])
        eviction_df = pd.read_csv(eviction_by_token[token])
        yield trial_id, access_df, eviction_df


def count_access_eviction_trial_pairs(
    access_pattern: str,
    eviction_pattern: str,
) -> int:
    """Count access/eviction trial pairs without reading any CSV data.

    Uses the same token-matching logic as
    :func:`read_access_eviction_trial_pairs` but only counts matching
    tokens — no CSV files are opened or parsed.

    Returns:
        The number of matching access/eviction file pairs.

    Raises:
        ValueError: If the access and eviction token sets do not match
            exactly (same validation as ``read_access_eviction_trial_pairs``).
    """
    access_files = sorted(glob.glob(access_pattern))
    eviction_files = sorted(glob.glob(eviction_pattern))
    access_by_token = _index_files_by_token(access_files, "access")
    eviction_by_token = _index_files_by_token(eviction_files, "eviction")
    common_tokens = sorted(set(access_by_token) & set(eviction_by_token))

    access_only = sorted(set(access_by_token) - set(eviction_by_token))
    eviction_only = sorted(set(eviction_by_token) - set(access_by_token))
    if access_only or eviction_only:
        raise ValueError(
            "Access/eviction trial token sets must match exactly. "
            f"Access-only tokens: {access_only}; "
            f"Eviction-only tokens: {eviction_only}"
        )
    return len(common_tokens)