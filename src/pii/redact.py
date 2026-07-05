"""PII detection + redaction.

Runs BEFORE any profiling output is constructed (CLAUDE.md rule #5). Nothing in
src/profiling or src/agents should ever see a raw value from a column flagged
here without it having passed through `redact_value` first.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_NAME_HINTS: dict[str, str] = {
    "email": "email",
    "e_mail": "email",
    "phone": "phone_number",
    "mobile": "phone_number",
    "ssn": "ssn",
    "social_security": "ssn",
    "national_id": "national_id",
    "passport": "passport",
    "credit_card": "credit_card",
    "card_number": "credit_card",
    "cc_number": "credit_card",
    "address": "address",
    "street": "address",
    "zip": "postal_code",
    "postal": "postal_code",
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
    "first_name": "name",
    "last_name": "name",
    "full_name": "name",
    "surname": "name",
    "ip_address": "ip_address",
}

_VALUE_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "phone_number": re.compile(r"^\+?[\d][\d\-\s()]{7,}\d$"),
    "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "credit_card": re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
    "ip_address": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
}

REDACTED = "[REDACTED]"


def _name_hint(column: str) -> str | None:
    normalized = column.strip().lower().replace(" ", "_").replace("-", "_")
    for hint, pii_type in _NAME_HINTS.items():
        if hint in normalized:
            return pii_type
    return None


def _mostly_dates(sample: pd.Series) -> bool:
    """Digit-and-hyphen date strings ("2022-01-01") satisfy the loose phone
    pattern; a column that overwhelmingly parses as datetimes is a date
    column, not a phone list."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "could not infer format" chatter on mixed strings
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() > 0.8)


def _pattern_hint(series: pd.Series, sample_size: int = 200) -> str | None:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)
    for pii_type, pattern in _VALUE_PATTERNS.items():
        matches = sample.str.match(pattern)
        if matches.mean() > 0.8:
            if pii_type == "phone_number" and _mostly_dates(sample):
                continue
            return pii_type
    return None


def detect_pii_columns(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Returns {column: {"pii_type": ..., "detection_method": "name" | "pattern"}}"""
    detected: dict[str, dict[str, str]] = {}
    for col in df.columns:
        name_hit = _name_hint(str(col))
        if name_hit:
            detected[col] = {"pii_type": name_hit, "detection_method": "name"}
            continue
        # pandas >= 3.0 defaults plain string columns to a "str" dtype
        # rather than "object" — check both so pattern detection still runs.
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            pattern_hit = _pattern_hint(df[col])
            if pattern_hit:
                detected[col] = {"pii_type": pattern_hit, "detection_method": "pattern"}
    return detected


def redact_value(_value: Any) -> str:
    return REDACTED


def redact_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Returns (df_with_pii_columns_redacted, pii_report).

    The redacted frame is safe to sample from for any LLM-facing artifact.
    The original `df` is untouched and remains available for training, which
    never routes through the LLM.
    """
    pii_columns = detect_pii_columns(df)
    redacted = df.copy()
    for col in pii_columns:
        redacted[col] = REDACTED

    report = {
        "pii_columns_detected": len(pii_columns),
        "columns": pii_columns,
    }
    return redacted, report
