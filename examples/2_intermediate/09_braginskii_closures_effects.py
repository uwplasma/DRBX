#!/usr/bin/env python3
"""Braginskii/Spitzer transport scalings in the field-line DRB models.

This example demonstrates the *equilibrium-based* Braginskii closures implemented in `jaxdrb`:

- Spitzer resistivity scaling: η ∝ T_e^{-3/2}
- Spitzer-Härm parallel transport proxy: χ_|| ∝ T^{5/2}
- Parallel viscosity proxy: ν_|| ∝ T^{5/2}

In a linear field-line model, these enter as spatially varying coefficients evaluated on the
equilibrium profiles (here: constant `Te0`). You can vary `Te0` to see the impact on effective
parallel losses and resistive coupling.

Run:
  python examples/2_intermediate/09_braginskii_closures_effects.py --out /tmp/out_braginskii
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from jaxdrb.analysis.plotting import save_json, save_scan_panels, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import DEFAULT_MODEL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--nl", type=int, default=65)
    p.add_argument("--length", type=float, default=6.0)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--ky-min", type=float, default=0.05)
    p.add_argument("--ky-max", type=float, default=1.0)
    p.add_argument("--nky", type=int, default=28)

    p.add_argument("--omega-n", type=float, default=0.8)
    p.add_argument("--eta0", type=float, default=1.0, help="Reference resistivity η at Tref.")
    p.add_argument(
        "--me-hat", type=float, default=0.0, help="0 uses algebraic Ohm (resistive branch)."
    )
    p.add_argument("--chi0", type=float, default=0.2, help="Reference χ_||,e at Tref.")
    p.add_argument("--nu0", type=float, default=0.1, help="Reference ν_||,i at Tref.")
    p.add_argument("--Te0-values", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    geom = OpenSlabGeometry.make(nl=args.nl, length=args.length, shat=0.0, curvature0=0.0)
    ky = np.linspace(args.ky_min, args.ky_max, args.nky)

    base = dict(
        omega_n=float(args.omega_n),
        omega_Te=0.0,
        eta=float(args.eta0),
        me_hat=float(args.me_hat),
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        chi_par_Te=float(args.chi0),
        nu_par_i=float(args.nu0),
        sheath_bc_on=True,
        sheath_bc_model=0,
        sheath_bc_nu_factor=1.0,
        sheath_end_damp_on=True,
    )

    save_json(
        out_dir / "run_config.json",
        {
            "geom": "slab-open",
            "nl": int(args.nl),
            "length": float(args.length),
            "kx": float(args.kx),
            "ky": {"min": float(args.ky_min), "max": float(args.ky_max), "n": int(args.nky)},
            "Te0_values": [float(x) for x in args.Te0_values],
            "base_params": base,
        },
    )

    # Baseline: constant coefficients (no Braginskii scalings).
    params_const = DRBParams(**base, braginskii_on=False)
    res_const = scan_ky(
        params_const,
        geom,
        ky=ky,
        kx=float(args.kx),
        eq=Equilibrium.constant(args.nl, n0=1.0, Te0=1.0),
        model=DEFAULT_MODEL,
        do_initial_value=False,
        verbose=True,
        print_every=5,
    )
    save_scan_panels(
        out_dir,
        ky=res_const.ky,
        gamma=res_const.gamma_eigs,
        omega=res_const.omega_eigs,
        title="Constant transport coefficients",
        filename="scan_constant.png",
    )

    # Braginskii/Spitzer scalings for varying equilibrium Te0.
    gamma = {}
    omega = {}
    for Te0 in args.Te0_values:
        params_b = DRBParams(
            **base,
            braginskii_on=True,
            braginskii_Tref=1.0,
            braginskii_T_floor=1e-4,
            braginskii_T_smooth=1e-4,
        )
        eq = Equilibrium.constant(args.nl, n0=1.0, Te0=float(Te0))
        print(f"\n=== Braginskii Te0={Te0} ===", flush=True)
        res = scan_ky(
            params_b,
            geom,
            ky=ky,
            kx=float(args.kx),
            eq=eq,
            model=DEFAULT_MODEL,
            do_initial_value=False,
            verbose=True,
            print_every=5,
        )
        gamma[str(Te0)] = res.gamma_eigs
        omega[str(Te0)] = res.omega_eigs
        save_scan_panels(
            out_dir,
            ky=res.ky,
            gamma=res.gamma_eigs,
            omega=res.omega_eigs,
            title=rf"Braginskii scalings, $T_{{e0}}={Te0}$",
            filename=f"scan_braginskii_Te0_{Te0}.png",
        )

    np.savez(
        out_dir / "results_braginskii.npz",
        ky=ky,
        gamma_const=res_const.gamma_eigs,
        **{f"gamma_Te0_{k}": v for k, v in gamma.items()},
    )

    # Summary panel
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax = axs[0]
    ax.plot(ky, res_const.gamma_eigs, "o-", label="constant")
    for Te0, g in gamma.items():
        ax.plot(ky, g, "o-", label=rf"Braginskii $T_{{e0}}={Te0}$")
    ax.axhline(0.0, color="k", alpha=0.3)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Growth rate")
    ax.legend(ncols=2)

    ax = axs[1]
    ax.plot(ky, np.maximum(res_const.gamma_eigs, 0.0) / ky, "o-", label="constant")
    for Te0, g in gamma.items():
        ax.plot(ky, np.maximum(g, 0.0) / ky, "o-", label=rf"$T_{{e0}}={Te0}$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$(\gamma/k_y)_+$")
    ax.set_title(r"Transport proxy: $\max(\gamma,0)/k_y$")
    ax.legend(ncols=2)

    fig.savefig(out_dir / "braginskii_panel.png", dpi=220)
    plt.close(fig)

    print(f"Wrote {out_dir / 'braginskii_panel.png'}", flush=True)


if __name__ == "__main__":
    main()
