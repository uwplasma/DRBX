from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp


class RegionBCField(eqx.Module):
    kind: str = eqx.field(static=True)
    value: float = eqx.field(static=True, default=0.0)
    grad: float = eqx.field(static=True, default=0.0)
    nu: float = eqx.field(static=True, default=0.0)

    @classmethod
    def disabled(cls) -> "RegionBCField":
        return cls(kind="none", value=0.0, grad=0.0, nu=0.0)

    def enabled(self) -> bool:
        return self.kind != "none" and self.nu != 0.0


class RegionBC(eqx.Module):
    name: str = eqx.field(static=True)
    mask: jnp.ndarray
    n: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    Te: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    vpar_e: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    vpar_i: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    omega: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    psi: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)
    Ti: RegionBCField = eqx.field(static=True, default_factory=RegionBCField.disabled)


def _parse_field_bc(spec: Any, *, default_nu: float) -> RegionBCField:
    if spec is None:
        return RegionBCField.disabled()
    if isinstance(spec, (int, float)):
        return RegionBCField(kind="dirichlet", value=float(spec), grad=0.0, nu=float(default_nu))
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "dirichlet")).lower()
        if kind in ("none", "off", "disabled"):
            return RegionBCField.disabled()
        value = float(spec.get("value", 0.0))
        grad = float(spec.get("grad", spec.get("gradient", 0.0)))
        nu = float(spec.get("nu", default_nu))
        return RegionBCField(kind=kind, value=value, grad=grad, nu=nu)
    raise TypeError("Region BC spec must be a dict, number, or None.")


def parse_region_bcs(
    policy: dict[str, Any], region_masks: dict[str, jnp.ndarray]
) -> tuple[RegionBC, ...]:
    regions = policy.get("regions", None)
    if not regions:
        return ()
    default_nu = float(policy.get("default_nu", 1.0))
    out: list[RegionBC] = []
    for region in regions:
        name = str(region.get("name", "")).strip()
        if not name or name not in region_masks:
            continue
        mask = jnp.asarray(region_masks[name], dtype=jnp.float64)
        bc_defs = region.get("bc", {})

        def get_spec(field: str):
            if field in bc_defs:
                return bc_defs.get(field)
            return region.get(f"bc_{field}")

        out.append(
            RegionBC(
                name=name,
                mask=mask,
                n=_parse_field_bc(get_spec("n"), default_nu=default_nu),
                Te=_parse_field_bc(get_spec("Te"), default_nu=default_nu),
                vpar_e=_parse_field_bc(get_spec("vpar_e"), default_nu=default_nu),
                vpar_i=_parse_field_bc(get_spec("vpar_i"), default_nu=default_nu),
                omega=_parse_field_bc(get_spec("omega"), default_nu=default_nu),
                psi=_parse_field_bc(get_spec("psi"), default_nu=default_nu),
                Ti=_parse_field_bc(get_spec("Ti"), default_nu=default_nu),
            )
        )
    return tuple(out)
