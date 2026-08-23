"""Portable raw-artifact paths for the 4 um Au inverse-design campaign."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_RAW_ROOT = Path(
    "/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch"
)


def raw_root() -> Path:
    configured = os.environ.get("AU_DUALPOL_RAW_ROOT", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_RAW_ROOT


def raw_path(*parts: str) -> Path:
    return raw_root().joinpath(*parts)
