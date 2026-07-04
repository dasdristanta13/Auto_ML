import pandas as pd

from src.pii.redact import detect_pii_columns, redact_dataframe


def test_detects_pii_by_column_name():
    df = pd.DataFrame({"full_name": ["Alex Smith"], "email": ["a@b.com"], "amount": [10.0]})
    detected = detect_pii_columns(df)
    assert "full_name" in detected
    assert "email" in detected
    assert "amount" not in detected


def test_detects_pii_by_value_pattern_even_without_name_hint():
    df = pd.DataFrame({"contact": [f"user{i}@example.com" for i in range(50)]})
    detected = detect_pii_columns(df)
    assert "contact" in detected
    assert detected["contact"]["pii_type"] == "email"


def test_redact_dataframe_masks_pii_columns_only():
    df = pd.DataFrame({"ssn": ["123-45-6789"], "amount": [10.0]})
    redacted, report = redact_dataframe(df)
    assert (redacted["ssn"] == "[REDACTED]").all()
    assert (redacted["amount"] == df["amount"]).all()
    assert report["pii_columns_detected"] == 1
