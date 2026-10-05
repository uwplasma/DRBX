"""FCI model-state containers for the fci_braginskii backend.

Every definition here was identical to (and closure-equivalent with)
``drbx.native.fci_model``; this module re-exports those objects so the
existing ``drbx.fci_braginskii.native.fci_model`` import path keeps working.
"""

from __future__ import annotations

from ...native.fci_model import (
    FciFieldBundle,
    FciFieldBundleT,
    FciModelState,
    FciModelStateT,
    assert_matching_field_names,
    inject_owned_field_to_halo,
    inject_owned_state_to_halo,
    inject_owned_vector_field_to_halo,
)

__all__ = [
    "FciFieldBundle",
    "FciModelState",
    "assert_matching_field_names",
    "inject_owned_field_to_halo",
    "inject_owned_vector_field_to_halo",
    "inject_owned_state_to_halo",
]
