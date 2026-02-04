from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from jaxdrb.analysis.plotting import (
    save_eigenfunction_panel,
    save_eigenvalue_spectrum,
    save_geometry_overview,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry, SlabGeometry
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.geometry.tokamak import (
    CircularTokamakGeometry,
    OpenCircularTokamakGeometry,
    OpenSAlphaGeometry,
    SAlphaGeometry,
)
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import DEFAULT_MODEL, MODELS, get_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-scan")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL.name)
    parser.add_argument(
        "--geom",
        choices=[
            "slab",
            "slab-open",
            "tabulated",
            "tokamak",
            "tokamak-open",
            "salpha",
            "salpha-open",
        ],
        required=True,
    )
    parser.add_argument("--geom-file", type=str, default=None)
    parser.add_argument("--nl", type=int, default=64)
    parser.add_argument("--length", type=float, default=float(2 * np.pi))
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--R0", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.18)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--curvature0", type=float, default=0.0)

    parser.add_argument("--omega-n", type=float, default=0.8)
    parser.add_argument("--omega-Te", type=float, default=0.0)
    parser.add_argument("--omega-Ti", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--me-hat", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--tau-i", type=float, default=0.0)
    parser.add_argument("--no-curvature", action="store_true")
    parser.add_argument("--no-boussinesq", action="store_true")
    parser.add_argument("--Dn", type=float, default=0.01)
    parser.add_argument("--DOmega", type=float, default=0.01)
    parser.add_argument("--DTe", type=float, default=0.01)
    parser.add_argument("--DTi", type=float, default=0.01)
    parser.add_argument("--Dpsi", type=float, default=0.0)
    parser.add_argument("--kperp2-min", type=float, default=1e-6)

    # Optional parallel closures and volumetric sinks (useful for SOL-like studies).
    parser.add_argument("--chi-par-Te", type=float, default=0.0, help="Parallel Te conduction χ_||")
    parser.add_argument(
        "--nu-par-e", type=float, default=0.0, help="Parallel electron flow diffusion/viscosity"
    )
    parser.add_argument(
        "--nu-par-i", type=float, default=0.0, help="Parallel ion flow diffusion/viscosity"
    )
    parser.add_argument("--nu-sink-n", type=float, default=0.0, help="Volumetric sink on n")
    parser.add_argument("--nu-sink-Te", type=float, default=0.0, help="Volumetric sink on Te")
    parser.add_argument(
        "--nu-sink-vpar",
        type=float,
        default=0.0,
        help="Volumetric sink on vpar_e and vpar_i",
    )

    # Optional SOL/sheath closures (only active for *open* geometries).
    #
    # `--sheath` is kept as a short alias for `--sheath-bc` (Loizu-style MPSE BCs).
    parser.add_argument("--sheath", action="store_true", help="Alias for --sheath-bc (MPSE BCs)")

    parser.add_argument(
        "--sheath-bc", action="store_true", help="Enable Loizu-style MPSE Bohm sheath BCs"
    )
    parser.add_argument(
        "--sheath-bc-model",
        choices=["simple", "loizu2012"],
        default="simple",
        help="MPSE BC model: 'simple' (velocity-only) or 'loizu2012' (full linearized set).",
    )
    parser.add_argument(
        "--sheath-bc-nu-factor", type=float, default=1.0, help="BC enforcement rate factor (~2/L||)"
    )
    parser.add_argument(
        "--sheath-cos2",
        type=float,
        default=1.0,
        help="Proxy for cos^2(incidence angle) in Loizu 2012 vorticity BC (default 1).",
    )
    parser.add_argument(
        "--sheath-lambda", type=float, default=3.28, help="Lambda = 0.5 ln(mi/(2π me))"
    )
    parser.add_argument(
        "--sheath-delta",
        type=float,
        default=0.0,
        help="Ion transmission correction (cold ions -> 0)",
    )

    parser.add_argument(
        "--sheath-loss",
        action="store_true",
        help="Enable volumetric end-loss proxy (nu_sh ~ 2/L||)",
    )
    parser.add_argument(
        "--sheath-loss-nu-factor",
        type=float,
        default=1.0,
        help="Multiplier for the sheath-loss rate (nu_sh ~ 2/L_parallel).",
    )

    parser.add_argument("--ky-min", type=float, required=True)
    parser.add_argument("--ky-max", type=float, required=True)
    parser.add_argument("--nky", type=int, default=32)
    parser.add_argument("--kx", type=float, default=0.0)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--arnoldi-m", type=int, default=40)
    parser.add_argument("--arnoldi-max-m", type=int, default=None)
    parser.add_argument("--arnoldi-tol", type=float, default=1e-3)
    parser.add_argument("--nev", type=int, default=6)
    parser.add_argument("--tmax", type=float, default=30.0)
    parser.add_argument("--dt0", type=float, default=0.01)
    parser.add_argument("--nsave", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-initial-value", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    model = get_model(args.model)

    run_cfg = {
        "model": args.model,
        "geom": args.geom,
        "geom_file": args.geom_file,
        "nl": args.nl,
        "length": args.length,
        "shat": args.shat,
        "q": args.q,
        "R0": args.R0,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "curvature0": args.curvature0,
        "omega_n": args.omega_n,
        "omega_Te": args.omega_Te,
        "omega_Ti": args.omega_Ti,
        "eta": args.eta,
        "me_hat": args.me_hat,
        "beta": args.beta,
        "tau_i": args.tau_i,
        "curvature_on": not args.no_curvature,
        "boussinesq": not args.no_boussinesq,
        "Dn": args.Dn,
        "DOmega": args.DOmega,
        "DTe": args.DTe,
        "DTi": args.DTi,
        "Dpsi": args.Dpsi,
        "kperp2_min": args.kperp2_min,
        "sheath_bc_on": bool(args.sheath or args.sheath_bc),
        "sheath_bc_nu_factor": float(args.sheath_bc_nu_factor),
        "sheath_lambda": float(args.sheath_lambda),
        "sheath_delta": float(args.sheath_delta),
        "sheath_loss_on": bool(args.sheath_loss),
        "sheath_loss_nu_factor": float(args.sheath_loss_nu_factor),
        "ky_min": args.ky_min,
        "ky_max": args.ky_max,
        "nky": args.nky,
        "kx": args.kx,
        "arnoldi_m": args.arnoldi_m,
        "arnoldi_max_m": args.arnoldi_max_m,
        "arnoldi_tol": args.arnoldi_tol,
        "nev": args.nev,
        "tmax": args.tmax,
        "dt0": args.dt0,
        "nsave": args.nsave,
        "seed": args.seed,
        "do_initial_value": not args.no_initial_value,
    }

    if args.geom == "slab":
        geom = SlabGeometry.make(
            nl=args.nl, length=args.length, shat=args.shat, curvature0=args.curvature0
        )
    elif args.geom == "slab-open":
        geom = OpenSlabGeometry.make(
            nl=args.nl, length=args.length, shat=args.shat, curvature0=args.curvature0
        )
    elif args.geom == "tokamak":
        geom = CircularTokamakGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    elif args.geom == "tokamak-open":
        geom = OpenCircularTokamakGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    elif args.geom == "salpha":
        geom = SAlphaGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            alpha=args.alpha,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    elif args.geom == "salpha-open":
        geom = OpenSAlphaGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            alpha=args.alpha,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    else:
        if args.geom_file is None:
            raise SystemExit("--geom-file is required for --geom tabulated")
        geom = TabulatedGeometry.from_npz(args.geom_file)
        if geom.l.size != args.nl:
            raise SystemExit(f"--nl={args.nl} but geometry file has nl={geom.l.size}")

    run_cfg["curvature0_effective"] = float(getattr(geom, "curvature0", args.curvature0))
    (out_dir / "params.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")

    params = DRBParams(
        omega_n=args.omega_n,
        omega_Te=args.omega_Te,
        omega_Ti=args.omega_Ti,
        eta=args.eta,
        me_hat=args.me_hat,
        beta=args.beta,
        tau_i=args.tau_i,
        curvature_on=not args.no_curvature,
        boussinesq=not args.no_boussinesq,
        Dn=args.Dn,
        DOmega=args.DOmega,
        DTe=args.DTe,
        DTi=args.DTi,
        Dpsi=args.Dpsi,
        kperp2_min=args.kperp2_min,
        chi_par_Te=float(args.chi_par_Te),
        nu_par_e=float(args.nu_par_e),
        nu_par_i=float(args.nu_par_i),
        nu_sink_n=float(args.nu_sink_n),
        nu_sink_Te=float(args.nu_sink_Te),
        nu_sink_vpar=float(args.nu_sink_vpar),
        sheath_bc_on=bool(args.sheath or args.sheath_bc),
        sheath_bc_model=1 if args.sheath_bc_model == "loizu2012" else 0,
        sheath_bc_nu_factor=float(args.sheath_bc_nu_factor),
        sheath_cos2=float(args.sheath_cos2),
        sheath_lambda=float(args.sheath_lambda),
        sheath_delta=float(args.sheath_delta),
        sheath_loss_on=bool(args.sheath_loss),
        sheath_loss_nu_factor=float(args.sheath_loss_nu_factor),
    )

    ky_grid = np.linspace(args.ky_min, args.ky_max, args.nky)
    res = scan_ky(
        params,
        geom,
        ky=ky_grid,
        kx=float(args.kx),
        nl=args.nl,
        model=model,
        arnoldi_m=args.arnoldi_m,
        arnoldi_tol=args.arnoldi_tol,
        arnoldi_max_m=args.arnoldi_max_m,
        nev=args.nev,
        seed=args.seed,
        do_initial_value=not args.no_initial_value,
        tmax=args.tmax,
        dt0=args.dt0,
        nsave=args.nsave,
        verbose=True,
        print_every=1,
    )

    np.savez(
        out_dir / "results.npz",
        ky=res.ky,
        gamma_eigs=res.gamma_eigs,
        omega_eigs=res.omega_eigs,
        eigs=res.eigs,
        gamma_iv=res.gamma_iv if res.gamma_iv is not None else np.nan * res.gamma_eigs,
        omega_iv=res.omega_iv if res.omega_iv is not None else np.nan * res.omega_eigs,
        arnoldi_m_used=res.arnoldi_m_used,
        arnoldi_rel_resid=res.arnoldi_rel_resid,
    )

    # Representative eigenfunction/spectrum at ky* maximizing max(gamma,0)/ky
    ratio = np.maximum(res.gamma_eigs, 0.0) / res.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(res.ky[i_star])

    # Summary plots
    save_geometry_overview(out_dir, geom=geom, kx=float(args.kx), ky=float(ky_star))
    save_scan_panels(
        out_dir,
        ky=res.ky,
        gamma=res.gamma_eigs,
        omega=res.omega_eigs,
        gamma_iv=res.gamma_iv,
        title=f"jaxdrb scan ({args.model}, {args.geom})",
        filename="scan_panel.png",
    )
    y_eq = model.equilibrium(args.nl)
    rhs_kwargs = {}
    if model.default_eq is not None:
        rhs_kwargs["eq"] = model.default_eq(args.nl)
    key = jax.random.PRNGKey(args.seed + 1)
    v0 = model.random_state(key, args.nl, amplitude=1e-3)
    matvec_star = linear_matvec_from_rhs(
        model.rhs, y_eq, params, geom, kx=float(args.kx), ky=ky_star, rhs_kwargs=rhs_kwargs
    )
    arn = arnoldi_eigs(matvec_star, v0, m=int(args.arnoldi_m), nev=int(args.nev), seed=args.seed)
    lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
    lead = complex(arn.eigenvalues[lead_idx])
    ritz = arnoldi_leading_ritz_vector(matvec_star, v0, m=int(args.arnoldi_m), seed=args.seed)

    save_eigenvalue_spectrum(out_dir, eigenvalues=arn.eigenvalues, highlight=lead)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=float(args.kx),
        ky=ky_star,
        kperp2_min=float(args.kperp2_min),
        filename="eigenfunctions.png",
    )

    print(f"Wrote {out_dir / 'results.npz'}", flush=True)
    print(f"Wrote {out_dir / 'scan_panel.png'}", flush=True)
    print(f"Wrote {out_dir / 'geometry_overview.png'}", flush=True)
    print(f"Wrote {out_dir / 'spectrum.png'}", flush=True)
    print(f"Wrote {out_dir / 'eigenfunctions.png'}", flush=True)
