"""Compatibility helpers for the bundled runtime."""
from __future__ import annotations

from pathlib import Path

def add_checkout(name: str, env_var: str) -> Path:
    """Return the bundled source root without discovering or importing external checkouts.

    Kept for existing callers. Environment overrides deliberately cannot shadow the
    implementation shipped in this standalone distribution.
    """
    return Path(__file__).resolve().parents[1]
