# -*- coding: utf-8 -*-
"""Compatibility wrapper for running scripts/run_team.py from the repo root."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "run_team.py"), run_name="__main__")
