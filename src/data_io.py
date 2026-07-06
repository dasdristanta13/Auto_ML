"""Shared, format/encoding-tolerant dataset loading.

Every pipeline stage that touches the dataset file (profiling, EDA, feature
application, training dispatch, LLM data tools) loads through here so the
whole pipeline agrees on what "readable" means. Real-world CSVs are routinely
not UTF-8 (Excel exports are cp1252/latin-1) and are routinely not
comma-delimited (European locales export semicolons); both used to crash the
run at whichever node happened to read the file first.
"""

from __future__ import annotations

import pandas as pd

_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_ALT_DELIMITERS = (";", "\t", "|")


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"could not decode '{path}' with any supported encoding") from last_error


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset by extension (.parquet/.json/anything-else-is-CSV),
    trying common encodings and sniffing non-comma delimiters. Raises
    ValueError with an actionable message for empty/undecodable files rather
    than letting a bare pandas error surface mid-pipeline."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".json"):
        return pd.read_json(path)

    try:
        df = _read_csv_any_encoding(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"dataset file '{path}' is empty") from exc

    # A single-column result whose header embeds a delimiter almost always
    # means the file uses ; \t or | instead of commas — re-read with sniffing.
    if df.shape[1] == 1 and any(d in str(df.columns[0]) for d in _ALT_DELIMITERS):
        for encoding in _CSV_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                break
    return df
