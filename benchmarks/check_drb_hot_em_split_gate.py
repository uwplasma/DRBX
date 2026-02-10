"""Operator split mean-invariant gate for hot-ion and EM DRB branches.

This script enforces conservative-subset mean-rate thresholds for:
- hot-ion DRB (mass/charge/current/momentum mean rates),
- electromagnetic DRB (mass/charge/current/momentum mean rates).

It is intended as a CI physics gate for split operator parity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.geometry.slab import SlabGeometry  # noqa: E402
from jaxdrb.models.hot_ion_drb import Equilibrium as HotEquilibrium  # noqa: E402
from jaxdrb.models.hot_ion_drb import State as HotState  # noqa: E402
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs  # noqa: E402
from jaxdrb.models.em_drb import Equilibrium as EMEquilibrium  # noqa: E402
from jaxdrb.models.em_drb import State as EMState  # noqa: E402
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs  # noqa: E402
from jaxdrb.models.invariants import (  # noqa: E402
    em_mean_rates_from_rhs,
    hot_ion_mean_rates_from_rhs,
)
from jaxdrb.models.params import DRBParams  # noqa: E402


def _hot_ion_gate(
    *,
    nl: int,
    amplitude: float,
    ky_values: np.ndarray,
    nseeds: int,
) -> dict[str, float]:
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = HotEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        omega_Ti=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=0.7,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        chi_par_Te=0.0,
        chi_par_Ti=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        nu_sink_n=0.0,
        nu_sink_Te=0.0,
        nu_sink_vpar=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )

    maxima = np.zeros(4, dtype=float)
    keys = jax.random.split(jax.random.key(0), int(nseeds))
    for seed_key in keys:
        y = HotState.random(seed_key, nl, amplitude=amplitude)
        for ky in ky_values:
            dy = hot_rhs(0.0, y, params, geom, kx=0.0, ky=float(ky), eq=eq)
            rates = hot_ion_mean_rates_from_rhs(
                y, dy, params=params, geom=geom, kx=0.0, ky=float(ky), eq=eq
            )
            vals = np.asarray(
                [
                    jnp.abs(rates["dmass_dt"]),
                    jnp.abs(rates["dcharge_dt"]),
                    jnp.abs(rates["dcurrent_dt"]),
                    jnp.abs(rates["dmomentum_dt"]),
                ],
                dtype=float,
            )
            maxima = np.maximum(maxima, vals)

    return {
        "max_abs_dmass_dt": float(maxima[0]),
        "max_abs_dcharge_dt": float(maxima[1]),
        "max_abs_dcurrent_dt": float(maxima[2]),
        "max_abs_dmomentum_dt": float(maxima[3]),
    }


def _em_gate(
    *,
    nl: int,
    amplitude: float,
    ky_values: np.ndarray,
    nseeds: int,
) -> dict[str, float]:
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    eq = EMEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    params = DRBParams(
        omega_n=0.0,
        omega_Te=0.0,
        eta=0.0,
        me_hat=0.2,
        beta=0.4,
        Dpsi=0.0,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=0.0,
        nu_par_i=0.0,
        nu_sink_n=0.0,
        nu_sink_Te=0.0,
        nu_sink_vpar=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )

    maxima = np.zeros(4, dtype=float)
    keys = jax.random.split(jax.random.key(1), int(nseeds))
    for seed_key in keys:
        y = EMState.random(seed_key, nl, amplitude=amplitude)
        for ky in ky_values:
            dy = em_rhs(0.0, y, params, geom, kx=0.0, ky=float(ky), eq=eq)
            rates = em_mean_rates_from_rhs(
                y, dy, params=params, geom=geom, kx=0.0, ky=float(ky), eq=eq
            )
            vals = np.asarray(
                [
                    jnp.abs(rates["dmass_dt"]),
                    jnp.abs(rates["dcharge_dt"]),
                    jnp.abs(rates["dcurrent_dt"]),
                    jnp.abs(rates["dmomentum_dt"]),
                ],
                dtype=float,
            )
            maxima = np.maximum(maxima, vals)

    return {
        "max_abs_dmass_dt": float(maxima[0]),
        "max_abs_dcharge_dt": float(maxima[1]),
        "max_abs_dcurrent_dt": float(maxima[2]),
        "max_abs_dmomentum_dt": float(maxima[3]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=1.0e-2)
    parser.add_argument("--nseeds", type=int, default=4)
    parser.add_argument("--ky-min", type=float, default=0.12)
    parser.add_argument("--ky-max", type=float, default=0.72)
    parser.add_argument("--nky", type=int, default=5)
    parser.add_argument("--max-abs-mean-dt", type=float, default=1e-11)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    ky_values = np.linspace(float(args.ky_min), float(args.ky_max), int(args.nky))
    hot_metrics = _hot_ion_gate(
        nl=int(args.nl),
        amplitude=float(args.amplitude),
        ky_values=ky_values,
        nseeds=int(args.nseeds),
    )
    em_metrics = _em_gate(
        nl=int(args.nl),
        amplitude=float(args.amplitude),
        ky_values=ky_values,
        nseeds=int(args.nseeds),
    )

    metrics = {"hot_ion_operator_gate": hot_metrics, "em_operator_gate": em_metrics}

    print(
        "[drb-hot-em-gate] "
        f"hot_max|dmean/dt|={max(hot_metrics.values()):.3e}, "
        f"em_max|dmean/dt|={max(em_metrics.values()):.3e}",
        flush=True,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures: list[str] = []
    for name, metrics_block in (
        ("hot-ion", hot_metrics),
        ("em", em_metrics),
    ):
        for key, value in metrics_block.items():
            if value > float(args.max_abs_mean_dt):
                failures.append(
                    f"{name} operator gate failed: {key}={value:.3e} > {float(args.max_abs_mean_dt):.3e}"
                )

    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
