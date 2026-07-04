"""Generates the synthetic fixture datasets required by CLAUDE.md's Testing
Expectations: imbalanced classification, high-cardinality categoricals,
time-series (leakage-prone), wide datasets (500+ columns), PII-injected data,
and ambiguous/missing-target data. Run: python -m tests.fixtures.generate_fixtures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent
RNG = np.random.default_rng(42)


def make_imbalanced_classification(n: int = 2000) -> pd.DataFrame:
    n_pos = int(n * 0.03)  # 3% positive class — heavily imbalanced
    n_neg = n - n_pos
    tenure = np.concatenate([RNG.normal(6, 3, n_pos), RNG.normal(24, 10, n_neg)]).clip(0)
    monthly_spend = np.concatenate([RNG.normal(80, 20, n_pos), RNG.normal(50, 15, n_neg)]).clip(0)
    support_tickets = np.concatenate([RNG.poisson(4, n_pos), RNG.poisson(1, n_neg)])
    churned = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "churned": churned.astype(int),
        }
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def make_high_cardinality_categorical(n: int = 3000) -> pd.DataFrame:
    merchant_ids = [f"merchant_{i}" for i in range(1500)]  # high cardinality vs n rows
    df = pd.DataFrame(
        {
            "transaction_id": range(1, n + 1),
            "merchant_id": RNG.choice(merchant_ids, n),
            "amount": RNG.exponential(50, n).round(2),
            "hour_of_day": RNG.integers(0, 24, n),
            "is_fraud": RNG.choice([0, 1], n, p=[0.98, 0.02]),
        }
    )
    return df


def make_time_series(n: int = 1500) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    trend = np.linspace(100, 300, n)
    seasonal = 20 * np.sin(np.arange(n) * 2 * np.pi / 365)
    noise = RNG.normal(0, 10, n)
    sales = trend + seasonal + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            # leakage-prone: computed FROM the target's own future window — a
            # leakage detector / feature-engineering agent should flag/avoid this.
            "sales_7day_future_avg": pd.Series(sales).rolling(7).mean().shift(-7),
            "promo_flag": RNG.choice([0, 1], n, p=[0.9, 0.1]),
        }
    )
    return df


def make_wide_dataset(n_rows: int = 500, n_cols: int = 520) -> pd.DataFrame:
    data = {f"feature_{i}": RNG.normal(0, 1, n_rows) for i in range(n_cols)}
    data["target"] = (data["feature_0"] + data["feature_1"] * 0.5 + RNG.normal(0, 1, n_rows) > 0).astype(int)
    return pd.DataFrame(data)


def make_pii_dataset(n: int = 500) -> pd.DataFrame:
    first_names = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey"]
    last_names = ["Smith", "Lee", "Patel", "Garcia", "Kim", "Nguyen"]
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "full_name": [f"{RNG.choice(first_names)} {RNG.choice(last_names)}" for _ in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],
            "phone": [f"+1-555-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}" for _ in range(n)],
            "ssn": [f"{RNG.integers(100,999)}-{RNG.integers(10,99)}-{RNG.integers(1000,9999)}" for _ in range(n)],
            "annual_income": RNG.normal(60000, 15000, n).round(0),
            "default_risk": RNG.choice([0, 1], n, p=[0.85, 0.15]),
        }
    )
    return df


def make_ambiguous_target(n: int = 500) -> pd.DataFrame:
    """Multiple plausible target columns and no obvious single label — should
    trigger the human-in-loop checkpoint (PRD FR-9), never a silent guess."""
    df = pd.DataFrame(
        {
            "record_id": range(1, n + 1),
            "revenue": RNG.normal(1000, 200, n),
            "profit": RNG.normal(200, 80, n),
            "customer_satisfaction_score": RNG.integers(1, 10, n),
            "will_renew": RNG.choice([0, 1], n),
            "notes": ["" for _ in range(n)],
        }
    )
    return df


def main() -> None:
    make_imbalanced_classification().to_csv(FIXTURES_DIR / "imbalanced_classification.csv", index=False)
    make_high_cardinality_categorical().to_csv(FIXTURES_DIR / "high_cardinality_categorical.csv", index=False)
    make_time_series().to_csv(FIXTURES_DIR / "time_series.csv", index=False)
    make_wide_dataset().to_csv(FIXTURES_DIR / "wide_dataset.csv", index=False)
    make_pii_dataset().to_csv(FIXTURES_DIR / "pii_dataset.csv", index=False)
    make_ambiguous_target().to_csv(FIXTURES_DIR / "ambiguous_target.csv", index=False)
    print(f"Fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
