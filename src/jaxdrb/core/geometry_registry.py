from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.geometry_axisymmetric import build_axisymmetric_field_aligned_adapter
from jaxdrb.core.geometry_axisymmetric_analytic import build_axisymmetric_analytic_adapter
from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.geometry_fci import FCIGeometryAdapter
from jaxdrb.core.geometry_line import LineGeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.axisymmetric_maps import XPointPsi76Config, build_xpoint_psi76_fci_grid
from jaxdrb.geometry import Grid2D, OpenSlabGeometry, SlabGeometry


Builder = Callable[[DRBSystemParams, dict[str, Any]], GeometryAdapter]


@dataclass(frozen=True)
class GeometrySpec:
    kind: str
    builder: Builder
    required: tuple[str, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()
    optional: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    description: str = ""


def _as_float(cfg: dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _as_int(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def _build_plane(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    grid = Grid2D.make(
        nx=_as_int(cfg, "nx", 64),
        ny=_as_int(cfg, "ny", 64),
        Lx=_as_float(cfg, "Lx", 2 * jnp.pi),
        Ly=_as_float(cfg, "Ly", 2 * jnp.pi),
        dealias=bool(cfg.get("dealias", True)),
        bc_x=str(cfg.get("bc_x", "periodic")),
        bc_y=str(cfg.get("bc_y", "periodic")),
        bc_value_x=_as_float(cfg, "bc_value_x", 0.0),
        bc_value_y=_as_float(cfg, "bc_value_y", 0.0),
        bc_grad_x=_as_float(cfg, "bc_grad_x", 0.0),
        bc_grad_y=_as_float(cfg, "bc_grad_y", 0.0),
    )
    return Geometry2DAdapter(grid=grid, params=params)


def _build_line(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    nl = _as_int(cfg, "nl", 64)
    length = _as_float(cfg, "length", 2 * jnp.pi)
    open_field_line = bool(cfg.get("open_field_line", True))
    if open_field_line:
        geom_base = OpenSlabGeometry.make(nl=nl, length=length, shat=0.0, curvature0=0.0)
    else:
        geom_base = SlabGeometry.make(nl=nl, length=length, shat=0.0, curvature0=0.0)
    return LineGeometryAdapter(
        geom=geom_base,
        params=params,
        kx=_as_float(cfg, "kx", 0.0),
        ky=_as_float(cfg, "ky", 0.0),
    )


def _build_fci(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    grid = FCISlabGrid.make(
        nx=_as_int(cfg, "nx", 32),
        ny=_as_int(cfg, "ny", 32),
        nz=_as_int(cfg, "nz", 4),
        Lx=_as_float(cfg, "Lx", 2 * jnp.pi),
        Ly=_as_float(cfg, "Ly", 2 * jnp.pi),
        Lz=_as_float(cfg, "Lz", 2.0),
        Bx=_as_float(cfg, "Bx", 0.0),
        By=_as_float(cfg, "By", 0.0),
        Bz=_as_float(cfg, "Bz", 1.0),
        open_field_line=bool(cfg.get("open_field_line", False)),
    )
    return FCIGeometryAdapter(grid=grid, params=params)


def _build_fci_file(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    path = cfg.get("map_path") or cfg.get("coeff_path") or cfg.get("path")
    if path is None:
        raise ValueError("fci_file geometry requires map_path.")
    grid = FCISlabGrid.from_npz(
        path=str(path),
        open_field_line=cfg.get("open_field_line", None),
        cell_centered=cfg.get("cell_centered", None),
        Bx=cfg.get("Bx", None),
        By=cfg.get("By", None),
        Bz=cfg.get("Bz", None),
    )
    return FCIGeometryAdapter(grid=grid, params=params)


def _build_fci_xpoint_analytic(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    grid = build_xpoint_psi76_fci_grid(
        XPointPsi76Config(
            R_min=float(cfg.get("R_min", 80.0)),
            R_max=float(cfg.get("R_max", 120.0)),
            Z_min=float(cfg.get("Z_min", -60.0)),
            Z_max=float(cfg.get("Z_max", 20.0)),
            nx=_as_int(cfg, "nx", 64),
            ny=_as_int(cfg, "ny", 64),
            nz=_as_int(cfg, "nz", 8),
            Lz=_as_float(cfg, "Lz", 2 * jnp.pi),
            dphi=_as_float(cfg, "dphi", 2 * jnp.pi / max(_as_int(cfg, "nz", 8), 1)),
            B0=_as_float(cfg, "B0", 1.0),
            R0=_as_float(cfg, "R0", 100.0),
            I0=_as_float(cfg, "I0", 40.0),
            sigma0=_as_float(cfg, "sigma0", 6.25),
            R1=_as_float(cfg, "R1", 100.0),
            Z1=_as_float(cfg, "Z1", 0.0),
            Z2=_as_float(cfg, "Z2", -40.0),
            rho_s0=_as_float(cfg, "rho_s0", 1.0),
            open_field_line=bool(cfg.get("open_field_line", True)),
            cell_centered=bool(cfg.get("cell_centered", False)),
            nsteps=_as_int(cfg, "nsteps", 16),
        )
    )
    return FCIGeometryAdapter(grid=grid, params=params)


def _build_field_aligned(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    coeff_path = cfg.get("coeff_path") or cfg.get("coefficients")
    if coeff_path is None:
        raise ValueError("field_aligned geometry requires coeff_path or coefficients.")
    grid = FieldAlignedGrid.make(
        nx=_as_int(cfg, "nx", 32),
        ny=_as_int(cfg, "ny", 32),
        nz=_as_int(cfg, "nz", 32),
        Lx=_as_float(cfg, "Lx", 2 * jnp.pi),
        Ly=_as_float(cfg, "Ly", 2 * jnp.pi),
        Lz=_as_float(cfg, "Lz", 2 * jnp.pi),
        bc_x=str(cfg.get("bc_x", "periodic")),
        bc_y=str(cfg.get("bc_y", "periodic")),
        dealias=bool(cfg.get("dealias", True)),
        open_field_line=bool(cfg.get("open_field_line", False)),
        bc_value_x=_as_float(cfg, "bc_value_x", 0.0),
        bc_value_y=_as_float(cfg, "bc_value_y", 0.0),
        bc_grad_x=_as_float(cfg, "bc_grad_x", 0.0),
        bc_grad_y=_as_float(cfg, "bc_grad_y", 0.0),
    )
    return FieldAlignedGeometryAdapter.from_npz(
        path=str(coeff_path),
        params=params,
        grid=grid,
    )


def _build_salpha(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    return FieldAlignedGeometryAdapter.make_salpha(
        params=params,
        nx=_as_int(cfg, "nx", 32),
        ny=_as_int(cfg, "ny", 32),
        nz=_as_int(cfg, "nz", 32),
        Lx=_as_float(cfg, "Lx", 2 * jnp.pi),
        Ly=_as_float(cfg, "Ly", 2 * jnp.pi),
        Lz=_as_float(cfg, "Lz", 2 * jnp.pi),
        bc_x=str(cfg.get("bc_x", "periodic")),
        bc_y=str(cfg.get("bc_y", "periodic")),
        dealias=bool(cfg.get("dealias", True)),
        open_field_line=bool(cfg.get("open_field_line", False)),
        shat=_as_float(cfg, "shat", 0.796),
        alpha=_as_float(cfg, "alpha", 0.0),
        q=_as_float(cfg, "q", 1.4),
        R0=_as_float(cfg, "R0", 1.0),
        epsilon=_as_float(cfg, "epsilon", 0.18),
        r0=cfg.get("r0", None),
        curvature0=cfg.get("curvature0", None),
        b_min=_as_float(cfg, "b_min", 0.05),
        theta_scale=cfg.get("theta_scale", None),
        curvature_model=str(cfg.get("curvature_model", "vector_xy")),
        B0=cfg.get("B0", None),
        epsilon_x_grad=cfg.get("epsilon_x_grad", None),
        theta_ballooning_on=bool(cfg.get("theta_ballooning_on", False)),
        theta_ballooning_r=cfg.get("theta_ballooning_r", None),
        linear_shear_on=bool(cfg.get("linear_shear_on", False)),
        bc_value_x=_as_float(cfg, "bc_value_x", 0.0),
        bc_value_y=_as_float(cfg, "bc_value_y", 0.0),
        bc_grad_x=_as_float(cfg, "bc_grad_x", 0.0),
        bc_grad_y=_as_float(cfg, "bc_grad_y", 0.0),
    )


def _build_axisymmetric(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    coeff_path = cfg.get("coeff_path") or cfg.get("coefficients")
    if coeff_path is None:
        raise ValueError("axisymmetric geometry requires coeff_path or coefficients.")
    key_overrides = cfg.get("keys", None)
    return build_axisymmetric_field_aligned_adapter(
        params=params,
        coeff_path=str(coeff_path),
        nx=cfg.get("nx", None),
        ny=cfg.get("ny", None),
        Lx=cfg.get("Lx", None),
        Ly=cfg.get("Ly", None),
        bc_x=str(cfg.get("bc_x", "periodic")),
        bc_y=str(cfg.get("bc_y", "periodic")),
        dealias=bool(cfg.get("dealias", True)),
        open_field_line=bool(cfg.get("open_field_line", False)),
        bc_value_x=_as_float(cfg, "bc_value_x", 0.0),
        bc_value_y=_as_float(cfg, "bc_value_y", 0.0),
        bc_grad_x=_as_float(cfg, "bc_grad_x", 0.0),
        bc_grad_y=_as_float(cfg, "bc_grad_y", 0.0),
        keys=key_overrides,
        boundary_policy=cfg.get("boundary_policy", None),
    )


def _build_axisymmetric_analytic(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    return build_axisymmetric_analytic_adapter(params=params, cfg=cfg)


_GEOMETRIES: dict[str, GeometrySpec] = {}
_ALIASES: dict[str, str] = {}


def register_geometry(spec: GeometrySpec) -> None:
    _GEOMETRIES[spec.kind] = spec
    for alias in spec.aliases:
        _ALIASES[alias] = spec.kind


def resolve_geometry_kind(kind: str | None) -> str:
    if kind is None:
        return "plane"
    key = str(kind).lower()
    return _ALIASES.get(key, key)


def available_geometries() -> list[GeometrySpec]:
    return [spec for _, spec in sorted(_GEOMETRIES.items(), key=lambda kv: kv[0])]


def get_geometry_spec(kind: str | None) -> GeometrySpec:
    resolved = resolve_geometry_kind(kind)
    if resolved not in _GEOMETRIES:
        raise ValueError(f"Unknown geometry kind '{kind}'.")
    return _GEOMETRIES[resolved]


def validate_geometry_config(cfg: dict[str, Any]) -> GeometrySpec:
    spec = get_geometry_spec(cfg.get("kind"))
    missing = [key for key in spec.required if key not in cfg]
    for group in spec.required_any:
        if not any(key in cfg for key in group):
            missing.append(" or ".join(group))
    if missing:
        raise ValueError(
            f"Geometry '{spec.kind}' missing required keys: {', '.join(missing)}."
        )
    return spec


def build_geometry(params: DRBSystemParams, cfg: dict[str, Any]) -> GeometryAdapter:
    spec = validate_geometry_config(cfg)
    return spec.builder(params, cfg)


register_geometry(
    GeometrySpec(
        kind="plane",
        builder=_build_plane,
        optional=("nx", "ny", "Lx", "Ly", "dealias", "bc_x", "bc_y"),
        aliases=("2d", "perp", "plane2d"),
        description="2D perpendicular plane.",
    )
)
register_geometry(
    GeometrySpec(
        kind="line",
        builder=_build_line,
        optional=("nl", "length", "open_field_line", "kx", "ky"),
        aliases=("1d", "field_line", "flux_tube"),
        description="1D field-line/flux-tube.",
    )
)
register_geometry(
    GeometrySpec(
        kind="fci",
        builder=_build_fci,
        optional=("nx", "ny", "nz", "Lx", "Ly", "Lz", "Bx", "By", "Bz", "open_field_line"),
        aliases=("fci_slab",),
        description="Analytic slab FCI grid.",
    )
)
register_geometry(
    GeometrySpec(
        kind="fci_file",
        builder=_build_fci_file,
        required_any=(("map_path", "coeff_path", "path"),),
        optional=("open_field_line", "cell_centered", "Bx", "By", "Bz"),
        aliases=("fci_from_file", "fci_map"),
        description="FCI grid loaded from a map file (.npz).",
    )
)
register_geometry(
    GeometrySpec(
        kind="fci_xpoint_analytic",
        builder=_build_fci_xpoint_analytic,
        optional=(
            "R_min",
            "R_max",
            "Z_min",
            "Z_max",
            "nx",
            "ny",
            "nz",
            "Lz",
            "dphi",
            "B0",
            "R0",
            "I0",
            "sigma0",
            "R1",
            "Z1",
            "Z2",
            "rho_s0",
            "open_field_line",
            "cell_centered",
            "nsteps",
        ),
        aliases=("fci_xpoint", "fci_psi76"),
        description="Analytic FCI map from Eq. (76) flux function.",
    )
)
register_geometry(
    GeometrySpec(
        kind="field_aligned",
        builder=_build_field_aligned,
        required_any=(("coeff_path", "coefficients"),),
        optional=("nx", "ny", "nz", "Lx", "Ly", "Lz", "open_field_line"),
        aliases=("aligned", "tabulated"),
        description="Generic field-aligned geometry from coefficients file.",
    )
)
register_geometry(
    GeometrySpec(
        kind="salpha",
        builder=_build_salpha,
        optional=(
            "nx",
            "ny",
            "nz",
            "Lx",
            "Ly",
            "Lz",
            "shat",
            "alpha",
            "q",
            "R0",
            "epsilon",
            "curvature0",
            "curvature_model",
            "b_min",
            "theta_scale",
            "B0",
            "epsilon_x_grad",
            "r0",
            "theta_ballooning_on",
            "theta_ballooning_r",
            "linear_shear_on",
        ),
        aliases=("s-alpha", "s_alpha"),
        description="Analytic s-alpha field-aligned geometry.",
    )
)
register_geometry(
    GeometrySpec(
        kind="axisymmetric",
        builder=_build_axisymmetric,
        required_any=(("coeff_path", "coefficients"),),
        optional=("nx", "ny", "Lx", "Ly", "open_field_line", "keys", "boundary_policy"),
        aliases=("axisym", "tokamak"),
        description="Axisymmetric field-aligned coefficients loaded from file.",
    )
)
register_geometry(
    GeometrySpec(
        kind="axisymmetric_analytic",
        builder=_build_axisymmetric_analytic,
        required=(),
        optional=(
            "nx",
            "ny",
            "nz",
            "Lx",
            "Ly",
            "Lz",
            "open_field_line",
            "shat",
            "alpha",
            "q",
            "R0",
            "epsilon",
            "r_minor",
            "kappa",
            "delta",
            "curvature0",
            "b_min",
            "theta_scale",
            "I0",
            "sigma0",
            "R1",
            "Z1",
            "Z2",
            "rho_s0",
            "B0",
            "epsilon_x_grad",
            "r0",
            "theta_ballooning_on",
            "theta_ballooning_r",
            "linear_shear_on",
            "sheath_windows",
            "sheath_sign",
            "boundary_policy",
        ),
        aliases=("axisymmetric_model", "axisym_analytic"),
        description="Axisymmetric analytic coefficients (salpha, miller, xpoint_psi76).",
    )
)
