"""Isolated dry-run + execution of validated LLM-generated code.

CLAUDE.md rule #6 / PRD FR-27/FR-28: every custom_code step is dry-run on a
small slice in isolation, resource-capped, before it ever touches the full
dataset.

NOTE (local dev limitation): PRD FR-28 specifies containerized/microVM
isolation (Docker/gVisor) with no network egress. This module is a
process-level stand-in for local testing — it enforces a wall-clock timeout
and a restricted builtins/import surface (via src/sandbox/validate.py), but
does NOT provide true OS-level isolation or memory capping. Do not use this
as-is against untrusted code in production; wire in the real container
sandbox described in the architecture doc before deploying.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.sandbox.validate import validate_code

_RUNTIME_CONFIG_PATH = "config/runtime.yaml"


class SandboxValidationError(RuntimeError):
    pass


class SandboxExecutionError(RuntimeError):
    pass


class SandboxTimeoutError(RuntimeError):
    pass


def _sandbox_config() -> dict[str, Any]:
    with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["sandbox"]


_SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


def _run_transform(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Executed in a worker process. No filesystem/network access is imported
    into the namespace, and `validate_code` has already rejected any attempt
    to reach os/sys/subprocess/socket/etc."""
    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "math": math,
        "re": re,
        "datetime": datetime,
    }
    local_ns: dict[str, Any] = {}
    exec(compile(code, "<sandboxed_transform>", "exec"), safe_globals, local_ns)  # noqa: S102 - validated above
    transform_fn = local_ns["transform"]
    result = transform_fn(df)
    if not isinstance(result, pd.DataFrame):
        raise SandboxExecutionError(f"transform() must return a DataFrame, got {type(result)}")
    return result


def dry_run(code: str, df_sample: pd.DataFrame) -> pd.DataFrame:
    """Validate then execute `code` against a small slice, isolated in a
    separate process with a wall-clock timeout. Raises SandboxValidationError,
    SandboxTimeoutError, or SandboxExecutionError on failure."""
    result = validate_code(code)
    if not result.valid:
        raise SandboxValidationError("; ".join(result.errors))

    cfg = _sandbox_config()
    sample = df_sample.head(cfg["dry_run_sample_rows"])

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_transform, code, sample)
        try:
            return future.result(timeout=cfg["timeout_seconds"])
        except concurrent.futures.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"sandboxed code exceeded {cfg['timeout_seconds']}s timeout"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised with sandbox context
            raise SandboxExecutionError(str(exc)) from exc


def run_on_full_dataset(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Only call this AFTER `dry_run` has already succeeded on a sample of
    the same code (CLAUDE.md rule #6: dry-run before full-dataset execution)."""
    result = validate_code(code)
    if not result.valid:
        raise SandboxValidationError("; ".join(result.errors))

    cfg = _sandbox_config()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_transform, code, df)
        try:
            return future.result(timeout=cfg["timeout_seconds"] * 10)
        except concurrent.futures.TimeoutError as exc:
            raise SandboxTimeoutError("sandboxed code exceeded full-dataset timeout") from exc
        except Exception as exc:  # noqa: BLE001
            raise SandboxExecutionError(str(exc)) from exc
