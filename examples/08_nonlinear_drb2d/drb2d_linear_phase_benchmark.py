"""Benchmark DRB2D linear-phase growth against the linear flux-tube solver.

We seed a single Fourier mode in DRB2D, measure its early-time growth rate,
then compare against a constant-geometry linear DRB calculation with the same
(kx, ky, kpar) and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.stepper import rk4_scan


@dataclass
class ConstantGeometry:
    kpar: float
    kperp2_value: float
    curvature_coeff: float = 0.0

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:  # type: ignore[override]
        return jnp.asarray([self.kperp2_value])

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return 1j * float(self.kpar) * f

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        if self.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        return -1j * float(self.curvature_coeff) * float(ky) * f


def main() -> None:
    out_dir = Path("out/examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    kx = 1.0
    ky = 1.0
    kpar = 0.0

    drb_params = DRBParams(
        omega_n=0.8,
        omega_Te=0.3,
        eta=0.5,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    # Linear solver (constant geometry, kpar=0).
    geom = ConstantGeometry(kpar=kpar, kperp2_value=kx**2 + ky**2)
    eq = Equilibrium.constant(1, n0=1.0, Te0=1.0)
    y0 = State.zeros(1)
    matvec = linear_matvec(y0, drb_params, geom, kx=kx, ky=ky, eq=eq)
    v0 = State(n=jnp.asarray([1e-6 + 0j]), omega=jnp.asarray([1e-6 + 0j]), vpar_e=jnp.asarray([0.0 + 0j]), vpar_i=jnp.asarray([0.0 + 0j]), Te=jnp.asarray([1e-6 + 0j]))
    lin_res = estimate_growth_rate(matvec, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    gamma_lin = float(lin_res.gamma)

    # DRB2D linear-phase growth.
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    d2_params = DRB2DParams(
        omega_n=drb_params.omega_n,
        omega_Te=drb_params.omega_Te,
        kpar=kpar,
        eta=drb_params.eta,
        me_hat=drb_params.me_hat,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DModel(params=d2_params, grid=grid)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phase = kx * X + ky * Y
    amp = 1e-6
    n = amp * np.cos(phase)
    omega = amp * np.cos(phase)
    vpar_e = np.zeros_like(n)
    vpar_i = np.zeros_like(n)
    Te = amp * np.cos(phase)
    y_state = DRB2DState(
        n=jnp.asarray(n),
        omega=jnp.asarray(omega),
        vpar_e=jnp.asarray(vpar_e),
        vpar_i=jnp.asarray(vpar_i),
        Te=jnp.asarray(Te),
    )

    dt = 0.05
    nsteps = 200
    save_stride = 5
    nframes = nsteps // save_stride
    amps = []
    ts = []
    t = 0.0
    for k in range(nframes):
        _, y_state = rk4_scan(y_state, t0=t, dt=dt, nsteps=save_stride, rhs=model.rhs)
        t = t + dt * save_stride
        kx_idx = 1
        ky_idx = 1
        n_fft = np.fft.fft2(np.asarray(y_state.n))
        w_fft = np.fft.fft2(np.asarray(y_state.omega))
        ve_fft = np.fft.fft2(np.asarray(y_state.vpar_e))
        vi_fft = np.fft.fft2(np.asarray(y_state.vpar_i))
        Te_fft = np.fft.fft2(np.asarray(y_state.Te))
        amp = np.sqrt(
            np.abs(n_fft[kx_idx, ky_idx]) ** 2
            + np.abs(w_fft[kx_idx, ky_idx]) ** 2
            + np.abs(ve_fft[kx_idx, ky_idx]) ** 2
            + np.abs(vi_fft[kx_idx, ky_idx]) ** 2
            + np.abs(Te_fft[kx_idx, ky_idx]) ** 2
        )
        amps.append(amp)
        ts.append(t)

    amps = np.asarray(amps)
    ts = np.asarray(ts)
    mask = ts <= (0.5 * ts[-1])
    slope, _ = np.polyfit(ts[mask], np.log(np.maximum(amps[mask], 1e-30)), 1)
    gamma_nl = float(slope)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(ts, np.log(np.maximum(amps, 1e-30)), lw=2.0, label="DRB2D")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\ln |\hat Y_{k_x,k_y}|$")
    ax.set_title("DRB2D linear-phase growth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "linear_phase_growth.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "metrics.json",
        {
            "gamma_linear": gamma_lin,
            "gamma_drb2d": gamma_nl,
            "kx": kx,
            "ky": ky,
            "kpar": kpar,
            "params": drb_params.__dict__,
        },
    )
    print(f"gamma_lin={gamma_lin:.4e} gamma_drb2d={gamma_nl:.4e}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
