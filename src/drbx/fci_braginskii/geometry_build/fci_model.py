from __future__ import annotations

from typing import TypeVar



# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ...native.fci_model import (  # noqa: F401
    update_halo_owned_slice,
    inject_owned_field_to_halo,
    inject_owned_vector_field_to_halo,
    extract_owned_field_from_halo,
)
from ...native.fci_model import (
    FciFieldBundle,
    FciModelState,
    assert_matching_field_names,
    extract_owned_state_from_halo,
    inject_owned_state_to_halo,
    update_state_halo_owned_slices,
)


FciFieldBundleT = TypeVar("FciFieldBundleT", bound="FciFieldBundle")
FciModelStateT = TypeVar("FciModelStateT", bound="FciModelState")


__all__ = [
    "FciFieldBundle",
    "FciModelState",
    "assert_matching_field_names",
    "extract_owned_field_from_halo",
    "extract_owned_state_from_halo",
    "inject_owned_field_to_halo",
    "inject_owned_vector_field_to_halo",
    "inject_owned_state_to_halo",
    "update_halo_owned_slice",
    "update_state_halo_owned_slices",
]
