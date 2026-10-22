"""Closed field lines of a VMEC equilibrium traced through VMEX.

Field lines of a converged VMEC equilibrium stay on their flux surface, and the average
slope d theta / d phi over many transits is the rotational transform iota(s) of the wout.
For the Landreman-Paul precise-QA stellarator: one JAX RK4 line per selected surface, traced
in (s, theta, phi) from the Nyquist Fourier tables; the traced iota must match the wout profile
(asserted to 1e-2 relative). The Poincare section at phi = 0 is drawn next to the iota comparison.

Needs a VMEX checkout (``DRBX_VMEX_ROOT``) and a wout (``WOUT_PATH`` or ``DRBX_ESSOS_ROOT``).

    PYTHONPATH=src python examples/geometry-3D/vmex/closed_field_lines.py

Writes ``output/closed_field_lines/closed_field_lines.png``.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drbx.geometry import (
    resolve_essos_landreman_qa_wout,
    load_vmex_wout,
    trace_vmex_field_lines,
    traced_rotational_transform,
    vmex_boundary_rz,
    vmex_half_mesh_s,
    vmex_runtime_available,
    vmex_surface_rz,
    vmex_wout_summary,
)

# PARAMETERS ---------------------------------------------------------------
# Landreman-Paul 2021 precise QA (reactor scale) wout; None -> the copy in the
# ESSOS checkout (DRBX_ESSOS_ROOT). wout files are external inputs, never committed.
WOUT_PATH: Path | None = None
SURFACE_INDICES = (4, 10, 18, 27, 36, 45)  # half-mesh rows to trace (1..ns-1)
THETA_START = 0.0            # poloidal seed angle of every line [rad]
N_TRANSITS = 300             # toroidal transits per line (= Poincare points)
STEPS_PER_TRANSIT = 96       # fixed RK4 steps per toroidal transit
IOTA_RTOL = 1.0e-2           # required traced-vs-wout iota relative agreement
OUTPUT_DIR = Path("output/closed_field_lines")

# setup --------------------------------------------------------------------
if not vmex_runtime_available():
    raise SystemExit(
        "VMEX is not importable. Point DRBX_VMEX_ROOT at a checkout, e.g.\n"
        "    DRBX_VMEX_ROOT=/path/to/VMEX PYTHONPATH=src python "
        "examples/geometry-3D/vmex/closed_field_lines.py"
    )
WOUT_PATH = resolve_essos_landreman_qa_wout(WOUT_PATH)  # raises with a hint if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# stage 1: load the equilibrium --------------------------------------------
print(f"loading VMEC equilibrium via VMEX: {WOUT_PATH.name}")
wout = load_vmex_wout(WOUT_PATH)
summary = vmex_wout_summary(wout)
print(f"  nfp = {summary['nfp']}, ns = {summary['ns']} surfaces, "
      f"aspect ratio = {summary['aspect']:.3f}")
print(f"  R_major = {summary['major_radius']:.3f} m, "
      f"a_minor = {summary['minor_radius']:.3f} m, B0 = {summary['b0']:.3f} T")
print(f"  iota profile: {summary['iota_axis']:.4f} (axis) -> "
      f"{summary['iota_edge']:.4f} (edge)")

s_half = vmex_half_mesh_s(wout)
s_full = np.linspace(0.0, 1.0, summary["ns"])
iotaf = np.asarray(wout.iotaf, dtype=np.float64)

# stage 2 + 3: trace one line per surface and verify iota ------------------
print(f"tracing 1 field line on each of {len(SURFACE_INDICES)} flux surfaces "
      f"({N_TRANSITS} transits, {STEPS_PER_TRANSIT} RK4 steps/transit)...")
poincare_r, poincare_z, traced_iotas, wout_iotas, surface_s = [], [], [], [], []
start = time.perf_counter()
for surface_index in SURFACE_INDICES:
    s_value = float(s_half[surface_index - 1])
    phi_nodes, theta_lines = trace_vmex_field_lines(
        wout,
        s_index=surface_index,
        theta0=np.array([THETA_START]),
        n_transits=N_TRANSITS,
        steps_per_transit=STEPS_PER_TRANSIT,
    )
    iota_traced = float(traced_rotational_transform(phi_nodes, theta_lines)[0])
    iota_wout = float(np.interp(s_value, s_full, iotaf))
    relative_error = abs(iota_traced - iota_wout) / abs(iota_wout)
    print(f"  s = {s_value:.3f}: traced iota = {iota_traced:.5f}, "
          f"wout iotaf = {iota_wout:.5f}, rel. diff = {relative_error:.1e} "
          f"({time.perf_counter() - start:.1f} s elapsed)")
    assert relative_error < IOTA_RTOL, (
        f"traced iota disagrees with the wout profile on s={s_value:.3f}: "
        f"{iota_traced} vs {iota_wout}"
    )
    # Poincare section at phi = 0: the line's poloidal angle each time it
    # completes a full toroidal transit, mapped to the lab frame through the
    # surface's R/Z Fourier tables.
    theta_at_phi0 = theta_lines[0, ::STEPS_PER_TRANSIT]
    r_cross, z_cross = vmex_surface_rz(
        wout, s=s_value, theta=theta_at_phi0, phi=np.zeros_like(theta_at_phi0)
    )
    poincare_r.append(r_cross)
    poincare_z.append(z_cross)
    traced_iotas.append(iota_traced)
    wout_iotas.append(iota_wout)
    surface_s.append(s_value)
print(f"iota verified on all {len(SURFACE_INDICES)} surfaces "
      f"(max rel. diff < {IOTA_RTOL:g}): every field line closes on its flux surface")

# stage 4: plot ------------------------------------------------------------
print("drawing the phi = 0 Poincare section and the iota comparison...")
boundary_r, boundary_z = vmex_boundary_rz(wout, phi=0.0, n_theta=256)
axis_r, axis_z = vmex_surface_rz(wout, s=0.0, theta=np.array(0.0), phi=np.array(0.0))

fig, (axis_poincare, axis_iota) = plt.subplots(1, 2, figsize=(11.5, 5.4))
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(SURFACE_INDICES)))
for r_cross, z_cross, s_value, color in zip(poincare_r, poincare_z, surface_s, colors):
    axis_poincare.scatter(r_cross, z_cross, s=2.5, color=color, linewidths=0,
                          label=f"s = {s_value:.2f}")
axis_poincare.plot(boundary_r, boundary_z, "k--", linewidth=1.2, label="LCFS (wout boundary)")
axis_poincare.scatter([float(axis_r)], [float(axis_z)], marker="+", s=90, color="k",
                      label="magnetic axis")
axis_poincare.set_xlabel("R [m]")
axis_poincare.set_ylabel("Z [m]")
axis_poincare.set_aspect("equal")
axis_poincare.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                     borderaxespad=0.0)
axis_poincare.set_title("Poincare section at phi = 0:\nnested closed flux surfaces")

axis_iota.plot(s_full, iotaf, "-", color="0.4", linewidth=1.5, label="wout iotaf profile")
axis_iota.scatter(surface_s, traced_iotas, s=45, color="#d62728", zorder=3,
                  label="traced field lines")
axis_iota.set_xlabel("normalized toroidal flux s")
axis_iota.set_ylabel("rotational transform iota")
axis_iota.legend(loc="best", fontsize=9)
axis_iota.set_title("traced d(theta)/d(phi) reproduces\nthe equilibrium iota profile")

fig.suptitle("Landreman-Paul precise QA (VMEX): every field line is closed", y=0.99)
fig.tight_layout()
figure_path = OUTPUT_DIR / "closed_field_lines.png"
fig.savefig(figure_path, dpi=170)
plt.close(fig)
print(f"wrote {figure_path}")
