from __future__ import annotations

from typing import Any

import equinox as eqx

from .params import DRBSystemParams, update_params_from_dict

_ALIASES: dict[str, tuple[str, ...]] = {
    "eta_par": ("eta",),
    "poisson_cg_maxiter": ("poisson_maxiter",),
    "poisson_cg_tol": ("poisson_tol",),
    "k2_min": ("kperp2_min",),
}


def _maybe_get(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    return None


def coerce_system_params(params: DRBSystemParams | eqx.Module, **overrides) -> DRBSystemParams:
    """Return a DRBSystemParams instance populated from compatibility params objects."""

    if isinstance(params, DRBSystemParams) and not overrides:
        return params

    out = DRBSystemParams()
    data: dict[str, object] = {}

    # Collect any attributes on the compatibility params that exist on the new params.
    for name in dir(params):
        if name.startswith("_"):
            continue
        if hasattr(out, name):
            try:
                data[name] = getattr(params, name)
            except Exception:
                continue

    # Apply aliases for renamed fields.
    for target, aliases in _ALIASES.items():
        if target in data:
            continue
        for alias in aliases:
            if hasattr(params, alias):
                data[target] = getattr(params, alias)
                break

    # If compatibility params expose `eta` but not `eta_par`, mirror the value for consistency.
    if "eta" not in data and hasattr(params, "eta_par"):
        data["eta"] = getattr(params, "eta_par")
    if "eta_par" not in data and hasattr(params, "eta"):
        data["eta_par"] = getattr(params, "eta")

    # Neutrals toggle convenience.
    if "neutrals_on" not in data and hasattr(params, "neutrals"):
        neutrals = getattr(params, "neutrals")
        if hasattr(neutrals, "enabled"):
            data["neutrals_on"] = bool(neutrals.enabled)

    # FCI sheath model selector: accept string-based model name if provided.
    if "sheath_bc_model_fci" not in data and hasattr(params, "sheath_bc_model"):
        model = getattr(params, "sheath_bc_model")
        if isinstance(model, str):
            data["sheath_bc_model_fci"] = model

    data.update(overrides)
    return update_params_from_dict(out, data)


def coerce_system_params_if_needed(params: DRBSystemParams | eqx.Module) -> DRBSystemParams:
    return params if isinstance(params, DRBSystemParams) else coerce_system_params(params)
