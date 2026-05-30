"""
Shared Python utilities for the Market Microstructure Engine.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict

import yaml


def setup_logging(name: str = "MarketMicrostructure") -> logging.Logger:
    """Configures and returns a logger with ISO-8601 timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    """YAML config loader."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def timestamp_to_microseconds(dt: datetime.datetime) -> int:
    """Convert datetime to microsecond timestamp (int)."""
    return int(dt.timestamp() * 1_000_000)


def microseconds_to_timestamp(us: int) -> datetime.datetime:
    """Convert microsecond timestamp (int) to datetime."""
    return datetime.datetime.fromtimestamp(us / 1_000_000, tz=datetime.timezone.utc)


def format_indian_number(num: float | int) -> str:
    """Formats a number in Indian numbering system (lakh, crore)."""
    s = str(int(num))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.append(remaining)
    parts.reverse()
    return ",".join(parts) + "," + last_three
