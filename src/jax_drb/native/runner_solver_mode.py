from __future__ import annotations

from ..config.boutinp import BoutConfig, NumericResolver


def configured_recycling_transient_solver_mode(config: BoutConfig) -> str | None:
    for section_name in ("runtime", "jax_drb"):
        if not config.has_option(section_name, "recycling_transient_solver_mode"):
            continue
        mode = str(
            config.parsed(section_name, "recycling_transient_solver_mode")
        ).strip()
        allowed = {
            "continuation",
            "bdf",
            "bdf_fixed_full_field_jvp",
            "fixed_bdf2_jax_linearized",
            "fixed_bdf2_jax_linearized_lineax",
            "adaptive_be",
            "adaptive_bdf",
            "adaptive_bdf_sparse_jvp",
            "adaptive_bdf_jax_linearized",
            "adaptive_bdf_jax_linearized_lineax",
            "matrix_free",
            "sparse",
            "sparse_jvp",
            "jax_linearized",
            "jax_linearized_lineax",
        }
        if mode not in allowed:
            raise ValueError(
                f"Unsupported {section_name}.recycling_transient_solver_mode={mode!r}; "
                f"expected one of {sorted(allowed)!r}."
            )
        return mode
    return None


def select_recycling_transient_solver_mode(
    config: BoutConfig,
    *,
    parity_mode: str,
) -> str:
    configured_mode = configured_recycling_transient_solver_mode(config)
    if configured_mode is not None:
        return configured_mode

    if parity_mode != "one_step":
        return "continuation"

    resolver = NumericResolver(config)
    ion_species = 0
    for section_name in config.section_names():
        if not config.has_option(section_name, "charge"):
            continue
        try:
            charge = float(resolver.resolve(section_name, "charge"))
        except Exception:
            continue
        if charge > 0.0:
            ion_species += 1

    return "bdf" if ion_species > 1 else "continuation"


def select_integrated_2d_transient_solver_mode(
    case_name: str,
    *,
    config: BoutConfig,
    parity_mode: str,
) -> str:
    configured_mode = configured_recycling_transient_solver_mode(config)
    if configured_mode is not None:
        return configured_mode

    if case_name in {
        "integrated_2d_production_one_step",
        "tokamak_recycling_one_step",
        "tokamak_recycling_dthe_one_step",
        "tokamak_recycling_dthe_drifts_one_step",
        "tokamak_recycling_dthene_one_step",
    }:
        return "bdf"
    return select_recycling_transient_solver_mode(config, parity_mode=parity_mode)
