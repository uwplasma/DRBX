"""Literal Hermes engine scaffold.

This package will own the strict-parity engine once the guard-aware component
stack is fully translated. It is intentionally not wired into the CLI yet.
"""

from __future__ import annotations


def build_system(*args, **kwargs):
    raise NotImplementedError(
        "The hermes_literal engine scaffold exists, but the full component stack is not wired yet."
    )
