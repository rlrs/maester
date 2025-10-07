"""Utilities for one-off architecture upgrade scripts.

This package intentionally stays lightweight.  Each helper mirrors a paper or
blog recipe so our standalone scripts (e.g. ``scripts/convert_to_gqa.py``)
can import the heavy lifting without pulling in the full training stack.
"""

from __future__ import annotations

from . import gqa

__all__ = [
    "gqa",
]
