"""CLAUDE.md: any change to sandbox validation requires a test with
intentionally malicious/broken LLM-generated code (disallowed imports,
infinite loops, wrong schema output) confirming it's caught before execution."""

import pandas as pd
import pytest

from src.sandbox.execute import SandboxExecutionError, SandboxTimeoutError, SandboxValidationError, dry_run
from src.sandbox.validate import validate_code


def test_disallowed_import_is_rejected():
    code = "import os\ndef transform(df):\n    os.system('echo hi')\n    return df\n"
    result = validate_code(code)
    assert not result.valid
    assert any("disallowed import" in e for e in result.errors)


def test_disallowed_builtin_call_is_rejected():
    code = "def transform(df):\n    eval('1+1')\n    return df\n"
    result = validate_code(code)
    assert not result.valid
    assert any("disallowed call" in e for e in result.errors)


def test_missing_transform_function_is_rejected():
    code = "x = 1\n"
    result = validate_code(code)
    assert not result.valid
    assert any("transform" in e for e in result.errors)


def test_syntax_error_is_rejected():
    code = "def transform(df):\n    return df +\n"
    result = validate_code(code)
    assert not result.valid


def test_infinite_loop_is_caught_by_timeout():
    code = (
        "def transform(df):\n"
        "    while True:\n"
        "        pass\n"
        "    return df\n"
    )
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(SandboxTimeoutError):
        dry_run(code, df)


def test_wrong_return_type_is_rejected_at_execution():
    code = "def transform(df):\n    return 'not a dataframe'\n"
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(SandboxExecutionError):
        dry_run(code, df)


def test_valid_transform_runs_successfully():
    code = "def transform(df):\n    df = df.copy()\n    df['a_squared'] = df['a'] ** 2\n    return df\n"
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = dry_run(code, df)
    assert list(result["a_squared"]) == [1, 4, 9]
