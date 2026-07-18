"""Explicit, deterministic data helpers.

The default series is synthetic. Tailguard performs no live network fetches.
Requested local inputs are validated and rejected on failure rather than
replaced with synthetic values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import warnings
from decimal import Decimal, InvalidOperation
from numbers import Number
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input series is missing or invalid."""


def _stable_seed(label: str) -> int:
    try:
        encoded = label.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataValidationError("series_id must be valid UTF-8 text") from exc
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def synthetic_series(
    series_id: str = "GSCPI_PROXY", *, start: str = "2017-01-01", end: str = "2026-04-01"
) -> pd.DataFrame:
    """Return a deterministic monthly synthetic proxy series.

    The data are deliberately illustrative and are not NY Fed observations.
    ``series_id`` changes a stable random seed, so results are repeatable for a
    fixed NumPy and Python runtime stack.
    """

    if not isinstance(series_id, str) or not series_id.strip():
        raise DataValidationError("series_id must be a non-empty string")
    start_ts = _timestamp_or_error(start, "start")
    end_ts = _timestamp_or_error(end, "end")
    if start_ts > end_ts:
        raise DataValidationError("start must not be after end")
    dates = pd.date_range("2017-01-01", "2026-04-01", freq="MS")
    rng = np.random.default_rng(_stable_seed(series_id))
    values = np.zeros(len(dates), dtype=float)
    for index in range(1, len(values)):
        values[index] = 0.85 * values[index - 1] + rng.normal(0.0, 0.4)
    covid = (dates >= "2020-03-01") & (dates <= "2021-12-01")
    values[covid] += np.linspace(1.5, 3.5, int(covid.sum())) * rng.uniform(0.7, 1.3, int(covid.sum()))
    normalization = (dates >= "2022-01-01") & (dates <= "2022-10-01")
    values[normalization] += np.linspace(3.0, -0.5, int(normalization.sum()))
    frame = pd.DataFrame({"date": dates, "value": np.round(values, 4)})
    result = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].reset_index(drop=True)
    if result.empty:
        raise DataValidationError("requested interval has no synthetic observations")
    return result


def _timestamp_or_error(value: object, name: str) -> pd.Timestamp:
    if isinstance(value, (Number, np.bool_, np.ndarray)) or getattr(value, "ndim", 0) != 0:
        raise DataValidationError(
            f"{name} must be an explicit date string or timestamp, not a number"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError, Warning):
        raise DataValidationError(f"{name} must be a valid date or timestamp") from None
    if pd.isna(timestamp):
        raise DataValidationError(f"{name} must be a valid date or timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _raw_csv_header(text: str) -> list[str]:
    """Validate record widths and return the header before pandas can mangle it."""

    header: Optional[list[str]] = None
    expected_fields: Optional[int] = None
    try:
        for row in csv.reader(io.StringIO(text, newline=""), strict=True):
            if not row or not any(field.strip() for field in row):
                continue
            if header is None:
                header = row
                expected_fields = len(row)
            elif len(row) != expected_fields:
                raise DataValidationError(
                    "series CSV rows must contain the same number of fields as the header"
                )
    except csv.Error as exc:
        raise DataValidationError("could not read series CSV") from exc
    if header is None:
        raise DataValidationError("series CSV is empty")
    return header


def _series_from_text(text: str, *, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    raw_header = _raw_csv_header(text)
    normalized_header = [name.strip().lower() for name in raw_header]
    if len(normalized_header) != len(set(normalized_header)):
        raise DataValidationError("series CSV must not contain duplicate column names")
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataValidationError("could not read series CSV") from exc
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise DataValidationError("series CSV must not contain duplicate column names")
    if not {"date", "value"}.issubset(frame.columns):
        raise DataValidationError("series CSV must contain date and value columns")
    result = frame[["date", "value"]].copy()
    source_values = result["value"].copy()
    exact_source_values: list[Decimal | None] = []
    for source_value in source_values:
        try:
            exact_source_values.append(Decimal(source_value.strip()))
        except InvalidOperation:
            exact_source_values.append(None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parsed_dates = pd.to_datetime(
                result["date"], format="ISO8601", errors="coerce", utc=True
            )
    except (TypeError, ValueError, OverflowError, Warning):
        raise DataValidationError("series CSV contains ambiguous or invalid dates") from None
    if parsed_dates.isna().any():
        raise DataValidationError("series CSV contains ambiguous or invalid dates")
    result["date"] = parsed_dates.dt.tz_convert(None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result["value"] = pd.to_numeric(result["value"], errors="coerce")
    except (TypeError, ValueError, OverflowError, Warning):
        raise DataValidationError("series CSV contains invalid numeric values") from None
    # Pandas treats exponentiated zero literals such as ``0e999`` as NaN even
    # though they are exactly zero and Python can represent their sign. Restore
    # only those exact-zero spellings before the ordinary invalid-row check.
    for index, (source_value, exact_value) in enumerate(
        zip(source_values, exact_source_values)
    ):
        if exact_value is not None and exact_value.is_finite() and exact_value == 0:
            result.iat[index, result.columns.get_loc("value")] = float(source_value)
    underflowed_values = 0
    for exact_value, converted_value in zip(exact_source_values, result["value"]):
        if exact_value is None:
            continue
        if exact_value.is_finite() and exact_value != 0 and converted_value == 0:
            underflowed_values += 1
    if underflowed_values:
        raise DataValidationError(
            "series CSV contains "
            f"{underflowed_values} non-zero numeric value(s) too small for floating-point representation"
        )
    invalid_rows = result["date"].isna() | result["value"].isna() | ~np.isfinite(result["value"].to_numpy(dtype=float))
    if invalid_rows.any():
        raise DataValidationError(
            f"series CSV contains {int(invalid_rows.sum())} invalid date or value row(s)"
        )
    if result["date"].duplicated().any():
        duplicates = result.loc[result["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d")
        examples = ", ".join(duplicates.drop_duplicates().head(3))
        raise DataValidationError(f"series CSV contains duplicate dates (for example: {examples})")
    result = result.sort_values("date", kind="mergesort")
    start_ts = _timestamp_or_error(start, "start") if start is not None else None
    end_ts = _timestamp_or_error(end, "end") if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise DataValidationError("start must not be after end")
    if start_ts is not None:
        result = result[result["date"] >= start_ts]
    if end_ts is not None:
        result = result[result["date"] <= end_ts]
    result = result.reset_index(drop=True)
    if len(result) < 4:
        raise DataValidationError("series needs at least four finite dated observations for CIR proxy fitting")
    return result


def load_series_csv_bytes(
    content: Union[bytes, bytearray, memoryview],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load an in-memory UTF-8 ``date,value`` CSV without creating a file."""

    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError("series CSV must be UTF-8 encoded") from exc
    except (TypeError, ValueError) as exc:
        raise DataValidationError("could not read series CSV bytes") from exc
    if "\x00" in text:
        raise DataValidationError("series CSV must not contain NUL characters")
    return _series_from_text(text, start=start, end=end)


def load_series_csv(
    path: Union[str, Path], *, start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    """Load a local ``date,value`` CSV with explicit provenance chosen by the user."""

    try:
        content = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise DataValidationError("series CSV was not found") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise DataValidationError("could not read series CSV") from exc
    return load_series_csv_bytes(content, start=start, end=end)
