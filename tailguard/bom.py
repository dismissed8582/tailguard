"""Validated flat sourcing-option input for Tailguard.

The optimizer intentionally models a *flat* bill of materials: exactly one
supplier option is chosen for each component.  It does not model a BOM DAG,
capacities, quantities, currencies, or a geographic definition of
``domestic``.  Input costs therefore have to be normalized to one finished
unit before they are supplied to this module.
"""

from __future__ import annotations

import csv
import io
import math
import warnings
from collections.abc import Set
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

REQUIRED_BOM_COLUMNS = frozenset({"component", "supplier", "base_cost", "kappa", "lead_time", "type"})
VALID_SOURCE_TYPES = frozenset({"offshore", "domestic"})


class BOMValidationError(ValueError):
    """Raised when a flat sourcing-option table cannot be optimized safely."""


@dataclass(frozen=True)
class SupplierOption:
    """A sourcing option for one component.

    ``source_type`` is a user-provided abstract source-class label. Tailguard
    does not infer geography from it and does not assume that a particular
    country is domestic for every user.
    """

    component: str
    supplier: str
    base_cost: float
    kappa: float
    lead_time: float
    source_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise BOMValidationError("component must be a non-empty string")
        if "\x00" in self.component:
            raise BOMValidationError("component must not contain NUL characters")
        try:
            self.component.encode("utf-8")
        except UnicodeEncodeError:
            raise BOMValidationError("component must be valid UTF-8 text") from None
        if not isinstance(self.supplier, str) or not self.supplier.strip():
            raise BOMValidationError("supplier must be a non-empty string")
        if "\x00" in self.supplier:
            raise BOMValidationError("supplier must not contain NUL characters")
        try:
            self.supplier.encode("utf-8")
        except UnicodeEncodeError:
            raise BOMValidationError("supplier must be valid UTF-8 text") from None
        if not isinstance(self.source_type, str) or self.source_type not in VALID_SOURCE_TYPES:
            valid = ", ".join(sorted(VALID_SOURCE_TYPES))
            raise BOMValidationError(f"source_type must be one of {valid}")
        for field_name, value in (
            ("base_cost", self.base_cost),
            ("kappa", self.kappa),
            ("lead_time", self.lead_time),
        ):
            if (
                getattr(value, "ndim", 0) != 0
                or isinstance(value, (bool, np.bool_, complex, np.complexfloating))
                or not isinstance(value, Real)
            ):
                raise BOMValidationError(f"{field_name} must be a finite non-negative number")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    numeric_value = float(value)
            except (TypeError, ValueError, OverflowError, Warning) as exc:
                raise BOMValidationError(f"{field_name} must be a finite non-negative number") from exc
            if value < 0 or not math.isfinite(numeric_value) or numeric_value < 0:
                raise BOMValidationError(f"{field_name} must be a finite non-negative number")
            if numeric_value == 0.0 and value != 0:
                raise BOMValidationError(
                    f"{field_name} is non-zero but too small for floating-point representation"
                )

    def deterministic_cost(self, lead_time_cost_per_day: float = 0.0) -> float:
        """Return the cost before stochastic shock events.

        A lead-time cost is deliberately explicit.  A zero value (the
        default) preserves the supplied financial cost instead of silently
        inventing an economic conversion from days to currency.
        """

        if (
            getattr(lead_time_cost_per_day, "ndim", 0) != 0
            or isinstance(
                lead_time_cost_per_day,
                (bool, np.bool_, complex, np.complexfloating),
            )
            or not isinstance(lead_time_cost_per_day, Real)
        ):
            raise BOMValidationError("lead_time_cost_per_day must be a finite non-negative number")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                rate = float(lead_time_cost_per_day)
        except (TypeError, ValueError, OverflowError, Warning) as exc:
            raise BOMValidationError("lead_time_cost_per_day must be a finite non-negative number") from exc
        if (
            lead_time_cost_per_day < 0
            or not math.isfinite(rate)
            or rate < 0
        ):
            raise BOMValidationError("lead_time_cost_per_day must be a finite non-negative number")
        if rate == 0.0 and lead_time_cost_per_day != 0:
            raise BOMValidationError(
                "lead_time_cost_per_day is non-zero but too small for floating-point representation"
            )
        exact_result = Fraction.from_float(float(self.base_cost)) + (
            Fraction.from_float(rate) * Fraction.from_float(float(self.lead_time))
        )
        try:
            result = float(exact_result)
        except OverflowError as exc:
            raise BOMValidationError(
                "deterministic cost overflows floating-point arithmetic"
            ) from exc
        if not math.isfinite(result):
            raise BOMValidationError("deterministic cost overflows floating-point arithmetic")
        if result == 0.0 and exact_result != 0:
            raise BOMValidationError(
                "deterministic cost is non-zero but too small for floating-point representation"
            )
        return result


BOM = Mapping[str, Sequence[SupplierOption]]


def _require_nonempty_text(value: object, field: str, row_number: int) -> str:
    if not isinstance(value, str):
        raise BOMValidationError(f"row {row_number}: {field} must be text")
    text = value.strip()
    if not text:
        raise BOMValidationError(f"row {row_number}: {field} must be non-empty text")
    if "\x00" in text:
        raise BOMValidationError(f"row {row_number}: {field} must not contain NUL characters")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise BOMValidationError(f"row {row_number}: {field} must be valid UTF-8 text") from None
    return text


def _require_nonnegative_number(value: object, field: str, row_number: int) -> float:
    if isinstance(
        value, (bool, np.bool_, complex, np.complexfloating, np.ndarray)
    ) or getattr(
        value, "ndim", 0
    ) != 0 or not isinstance(value, (str, Decimal, Real)):
        raise BOMValidationError(f"row {row_number}: {field} must be numeric")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            number = float(value)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise BOMValidationError(f"row {row_number}: {field} must be numeric") from exc
    exact_decimal: Union[Decimal, None] = value if isinstance(value, Decimal) else None
    if isinstance(value, str):
        try:
            exact_decimal = Decimal(value.strip())
        except InvalidOperation:
            exact_decimal = None
    if exact_decimal is not None and exact_decimal.is_finite() and exact_decimal < 0:
        raise BOMValidationError(f"row {row_number}: {field} must be a finite non-negative number")
    if isinstance(value, Real) and value < 0:
        raise BOMValidationError(f"row {row_number}: {field} must be a finite non-negative number")
    if not math.isfinite(number) or number < 0:
        raise BOMValidationError(f"row {row_number}: {field} must be a finite non-negative number")
    source_nonzero = (
        exact_decimal != 0
        if exact_decimal is not None and exact_decimal.is_finite()
        else isinstance(value, Real) and value != 0
    )
    if number == 0.0 and source_nonzero:
        raise BOMValidationError(
            f"row {row_number}: {field} is non-zero but too small for floating-point representation"
        )
    return number


def bom_from_dataframe(frame: pd.DataFrame) -> dict[str, tuple[SupplierOption, ...]]:
    """Validate a DataFrame and return a component mapping with immutable option sequences.

    The caller's DataFrame is never mutated.  Both source types are accepted
    but neither is required for every component: the optimization model only
    requires at least one feasible option per component.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    data = frame.copy(deep=True)
    normalized_columns = [str(column).strip().lower() for column in data.columns]
    duplicate_columns = pd.Index(normalized_columns)[pd.Index(normalized_columns).duplicated()].tolist()
    if duplicate_columns:
        raise BOMValidationError(
            "duplicate columns after normalization: " + ", ".join(sorted(set(duplicate_columns)))
        )
    data.columns = normalized_columns
    missing = REQUIRED_BOM_COLUMNS - set(data.columns)
    if missing:
        raise BOMValidationError("missing required columns: " + ", ".join(sorted(missing)))
    if data.empty:
        raise BOMValidationError("BOM must contain at least one supplier option")

    components: dict[str, list[SupplierOption]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row_number, (_, row) in enumerate(data.iterrows(), start=2):
        component = _require_nonempty_text(row["component"], "component", row_number)
        supplier = _require_nonempty_text(row["supplier"], "supplier", row_number)
        source_type = _require_nonempty_text(row["type"], "type", row_number).lower()
        if source_type not in VALID_SOURCE_TYPES:
            valid = ", ".join(sorted(VALID_SOURCE_TYPES))
            raise BOMValidationError(f"row {row_number}: type must be one of {valid}; got {source_type!r}")

        key = (component, supplier)
        if key in seen_pairs:
            raise BOMValidationError(f"row {row_number}: duplicate supplier {supplier!r} for component {component!r}")
        seen_pairs.add(key)

        option = SupplierOption(
            component=component,
            supplier=supplier,
            base_cost=_require_nonnegative_number(row["base_cost"], "base_cost", row_number),
            kappa=_require_nonnegative_number(row["kappa"], "kappa", row_number),
            lead_time=_require_nonnegative_number(row["lead_time"], "lead_time", row_number),
            source_type=source_type,
        )
        components.setdefault(component, []).append(option)

    return {component: tuple(options) for component, options in components.items()}


def _raw_bom_header(text: str) -> list[str]:
    """Read the source header before pandas can mangle duplicate field names."""

    try:
        for row in csv.reader(io.StringIO(text, newline="")):
            if not row or not any(field.strip() for field in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            return row
    except csv.Error as exc:
        raise BOMValidationError("could not read BOM CSV") from exc
    raise BOMValidationError("BOM CSV is empty")


def _without_documentation_lines(text: str) -> str:
    """Remove leading CSV records whose first field begins with ``#``.

    Filtering parsed records, rather than physical lines, preserves a line
    beginning with ``#`` when it is part of a quoted multiline field. Once the
    header is seen, every record is data; a component label may therefore begin
    with ``#`` without being discarded.
    """

    output = io.StringIO(newline="")
    header_seen = False
    expected_fields: Union[int, None] = None
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        writer = csv.writer(output)
        for row in reader:
            if not header_seen:
                if not row or not any(field.strip() for field in row):
                    continue
                if row[0].lstrip().startswith("#"):
                    continue
                header_seen = True
                expected_fields = len(row)
            elif row and len(row) != expected_fields:
                raise BOMValidationError(
                    "BOM CSV rows must contain the same number of fields as the header"
                )
            writer.writerow(row)
    except csv.Error as exc:
        raise BOMValidationError("could not read BOM CSV") from exc
    return output.getvalue()


def load_bom_csv_bytes(content: Union[bytes, bytearray, memoryview]) -> dict[str, tuple[SupplierOption, ...]]:
    """Load an in-memory UTF-8 BOM CSV without creating a local file.

    This is useful for notebook and web upload controls: the caller can
    validate uploaded bytes without persisting or displaying a filename.
    Leading ``#`` documentation comments are allowed.
    """

    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BOMValidationError("BOM CSV must be UTF-8 encoded") from exc
    except (TypeError, ValueError) as exc:
        raise BOMValidationError("could not read BOM CSV bytes") from exc
    if "\x00" in text:
        raise BOMValidationError("BOM CSV must not contain NUL characters")

    raw_header = _raw_bom_header(text)
    normalized_header = [field.strip().lower() for field in raw_header]
    duplicate_header = pd.Index(normalized_header)[pd.Index(normalized_header).duplicated()].tolist()
    if duplicate_header:
        raise BOMValidationError("duplicate columns after normalization: " + ", ".join(sorted(set(duplicate_header))))
    try:
        frame = pd.read_csv(
            io.StringIO(_without_documentation_lines(text)),
            dtype=str,
            keep_default_na=False,
        )
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise BOMValidationError("could not read BOM CSV") from exc
    return bom_from_dataframe(frame)


def load_bom_csv(path: Union[str, Path]) -> dict[str, tuple[SupplierOption, ...]]:
    """Load a UTF-8 BOM CSV, allowing leading ``#`` documentation comments."""

    try:
        content = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise BOMValidationError("BOM CSV was not found") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise BOMValidationError("could not read BOM CSV") from exc
    return load_bom_csv_bytes(content)


def flatten_bom(bom: BOM) -> tuple[SupplierOption, ...]:
    """Return options in deterministic component insertion order.

    This function validates hand-built mappings too, so solver callers cannot
    accidentally pass a malformed or empty component.
    """

    if not isinstance(bom, Mapping):
        raise BOMValidationError("BOM must be a component-to-options mapping")
    if not bom:
        raise BOMValidationError("BOM must contain at least one component")

    flattened: list[SupplierOption] = []
    seen_components: set[str] = set()
    for component, options in bom.items():
        if not isinstance(component, str) or not component.strip():
            raise BOMValidationError("component keys must be non-empty strings")
        if component in seen_components:
            raise BOMValidationError(f"duplicate component key {component!r}")
        seen_components.add(component)
        if isinstance(options, (str, bytes, bytearray, memoryview, Mapping, Set)) or getattr(
            options, "ndim", 1
        ) != 1:
            raise TypeError(
                f"options for component {component!r} must be an ordered one-dimensional iterable"
            )
        try:
            option_sequence = tuple(options)
        except TypeError as exc:
            raise TypeError(f"options for component {component!r} must be iterable") from exc
        if not option_sequence:
            raise BOMValidationError(f"component {component!r} has no supplier options")
        seen_suppliers: set[str] = set()
        for option in option_sequence:
            if not isinstance(option, SupplierOption):
                raise TypeError("BOM options must be SupplierOption instances")
            if option.component != component:
                raise BOMValidationError(
                    f"option {option.supplier!r} belongs to {option.component!r}, not {component!r}"
                )
            if option.supplier in seen_suppliers:
                raise BOMValidationError(
                    f"duplicate supplier {option.supplier!r} for component {component!r}"
                )
            seen_suppliers.add(option.supplier)
            flattened.append(option)
    return tuple(flattened)


def bom_to_dataframe(bom: BOM) -> pd.DataFrame:
    """Serialize a validated BOM without losing decimals, type, or lead time."""

    rows = [
        {
            "component": option.component,
            "supplier": option.supplier,
            "base_cost": option.base_cost,
            "kappa": option.kappa,
            "lead_time": option.lead_time,
            "type": option.source_type,
        }
        for option in flatten_bom(bom)
    ]
    return pd.DataFrame(
        rows,
        columns=["component", "supplier", "base_cost", "kappa", "lead_time", "type"],
    )


def count_by_source_type(options: Iterable[SupplierOption]) -> dict[str, int]:
    """Count the explicit input labels; do not infer type from lead time."""

    counts = {"offshore": 0, "domestic": 0}
    for option in options:
        counts[option.source_type] += 1
    return counts
