"""Logging and lightweight timing helpers used across the project."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from loguru import logger

app_logger = logger.bind(channel="app")
timing_logger = logger.bind(channel="timing")
_LOGGING_CONFIGURED = False


def setup_logging(
    *,
    log_dir: str | Path = "outputs/logs",
    app_level: str = "INFO",
    timing_level: str = "INFO",
) -> None:
    """Configure file and console log handlers once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    out = Path(log_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=app_level,
        filter=lambda r: r["extra"].get("channel") == "app",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    logger.add(
        out / "app.log",
        level=app_level,
        rotation="10 MB",
        retention=10,
        filter=lambda r: r["extra"].get("channel") == "app",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    logger.add(
        out / "timing.log",
        level=timing_level,
        rotation="10 MB",
        retention=10,
        filter=lambda r: r["extra"].get("channel") == "timing",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    _LOGGING_CONFIGURED = True


@contextmanager
def timed_calc(name: str, **fields: Any) -> Iterator[None]:
    """Measure and log elapsed time for a calculation block."""
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        if fields:
            timing_logger.info(
                "{name} | elapsed_ms={elapsed_ms:.3f} | fields={fields}",
                name=name,
                elapsed_ms=elapsed_ms,
                fields=fields,
            )
        else:
            timing_logger.info(
                "{name} | elapsed_ms={elapsed_ms:.3f}",
                name=name,
                elapsed_ms=elapsed_ms,
            )


from orbit_wars.observatory.decisions import (
    clear_decisions,
    enemy_fleets_snapshot,
    record_decision,
    record_enemy_fleet,
)
from orbit_wars.observatory.tracing import export_run_artifacts

__all__ = [
    "app_logger",
    "clear_decisions",
    "enemy_fleets_snapshot",
    "export_run_artifacts",
    "record_decision",
    "record_enemy_fleet",
    "setup_logging",
    "timed_calc",
    "timing_logger",
]
