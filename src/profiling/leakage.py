"""Target leakage detection heuristics.

Best-effort only, NOT guaranteed (see CLAUDE.md "Open Questions" and PRD 6.1).
Every report that surfaces these flags must say so explicitly — do not present
this as a completeness guarantee.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

NEAR_PERFECT_CORRELATION = 0.95
CATEGORY_PURITY_THRESHOLD = 0.98
SUSPICIOUS_NAME_HINTS = ("future_", "post_", "outcome_", "result_", "_after", "leak")


def detect_target_leakage(df: pd.DataFrame, target_column: str) -> list[dict[str, Any]]:
    if target_column not in df.columns:
        return []

    flags: list[dict[str, Any]] = []
    target = df[target_column]
    is_numeric_target = pd.api.types.is_numeric_dtype(target)

    for col in df.columns:
        if col == target_column:
            continue

        normalized = col.strip().lower()
        if any(hint in normalized for hint in SUSPICIOUS_NAME_HINTS):
            flags.append(
                {
                    "column": col,
                    "reason": "column name suggests it may be derived from or occur after the target event",
                    "severity": "medium",
                }
            )

        if is_numeric_target and pd.api.types.is_numeric_dtype(df[col]):
            corr = df[[col, target_column]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= NEAR_PERFECT_CORRELATION:
                flags.append(
                    {
                        "column": col,
                        "reason": f"near-perfect correlation with target ({corr:.3f})",
                        "severity": "high",
                    }
                )
            continue

        if not is_numeric_target and df[col].nunique(dropna=True) > 1:
            # categorical purity: does each category map ~1:1 to a single target value?
            grouped = df.groupby(col)[target_column]
            purity = grouped.apply(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else 0.0)
            weights = grouped.size() / len(df)
            weighted_purity = float((purity * weights).sum())
            if weighted_purity >= CATEGORY_PURITY_THRESHOLD and df[col].nunique() < len(df):
                flags.append(
                    {
                        "column": col,
                        "reason": f"category values map almost 1:1 to a single target value (purity={weighted_purity:.3f})",
                        "severity": "high",
                    }
                )

    return flags
