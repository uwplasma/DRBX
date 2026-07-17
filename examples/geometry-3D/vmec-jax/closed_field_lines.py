"""Closed field lines of a VMEC equilibrium traced through vmec_jax.

A converged VMEC equilibrium is, by construction, a set of nested closed flux
surfaces: the magnetic field has no component across a surface (B . grad s =
0), so every field line stays on the surface it starts on forever.  In VMEC
coordinates the line obeys

    d theta / d phi = B^theta(s, theta, phi) / B^phi(s, theta, phi),

and although that pitch oscillates within a toroidal transit (the VMEC
poloidal angle is not a straight-field-line angle), its average slope over
many transits is the rotational transform iota(s) stored in the wout file.
This example makes that concrete for the Landreman-Paul 2021 precise
quasi-axisymmetric stellarator (reactor scale):

1. load the wout NetCDF with vmec_jax's `read_wout` and print the
   equilibrium summary (nfp, aspect ratio, iota range, B0);
2. trace one field line per selected flux surface with a JAX RK4 integrator
   in (s, theta, phi), using the contravariant field from the wout Nyquist
   Fourier tables;
3. verify the physics: the traced average d(theta)/d(phi) of every line must
   reproduce the wout iota profile (asserted to 1e-2 relative);
4. draw the Poincare section at phi = 0 -- each line's crossings trace out
   the closed surface it lives on -- with the LCFS and axis overlaid, next to
   the traced-vs-wout iota profile.

Requires a vmec_jax checkout (env var DKX_VMEC_JAX_ROOT, default
``~/local/vmec_jax``) and a wout file (not committed to this repo):

    PYTHONPATH=src python examples/geometry-3D/vmec-jax/closed_field_lines.py

writes ``output/vmec_jax_closed/closed_field_lines.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dkx.geometry import (
    load_vmec_jax_wout,
    trace_vmec_jax_field_lines,
    traced_rotational_transform,
    vmec_jax_boundary_rz,
    vmec_jax_half_mesh_s,
    vmec_jax_runtime_available,
    vmec_jax_surface_rz,
    vmec_jax_wout_summary,
)

# PARAMETERS ---------------------------------------------------------------
# Landreman-Paul 2021 precise QA (reactor scale) wout from the local
# ESSOS_test checkout; wout files are external inputs, never committed here.
WOUT_PATH = Path.home() / "local" / "ESSOS_test" / "examples" / "input_files" / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
SURFACE_INDICES = (4, 10, 18, 27, 36, 45)  # half-mesh rows to trace (1..ns-1)
THETA_START = 0.0            # poloidal seed angle of every line [rad]
N_TRANSITS = 300             # toroidal transits per line (= Poincare points)
STEPS_PER_TRANSIT = 96       # fixed RK4 steps per toroidal transit
IOTA_RTOL = 1.0e-2           # required traced-vs-wout iota relative agreement
OUTPUT_DIR = Path("output/vmec_jax_closed")

# setup --------------------------------------------------------------------
if not vmec_jax_runtime_available():
    raise SystemExit(
        "vmec_jax is not importable. Point DKX_VMEC_JAX_ROOT at a checkout, e.g.\n"
        "    DKX_VMEC_JAX_ROOT=~/local/vmec_jax PYTHONPATH=src python "
        "examples/geometry-3D/vmec-jax/closed_field_lines.py"
    )
if not WOUT_PATH.exists():
    raise SystemExit(
        f"VMEC wout file not found: {WOUT_PATH}\n"
        "Edit the WOUT_PATH parameter to point at a local stellarator wout NetCDF "
        "(e.g. from an ESSOS or vmec_jax checkout); wout files are not committed here."
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# stage 1: load the equilibrium --------------------------------------------
print(f"loading VMEC equilibrium via vmec_jax: {WOUT_PATH.name}")
wout = load_vmec_jax_wout(WOUT_PATH)
summary = vmec_jax_wout_summary(wout)
print(f"  nfp = {summary['nfp']}, ns = {summary['ns']} surfaces, "
      f"aspect ratio = {summary['aspect']:.3f}")
print(f"  R_major = {summary['major_radius']:.3f} m, "
      f"a_minor = {summary['minor_radius']:.3f} m, B0 = {summary['b0']:.3f} T")
print(f"  iota profile: {summary['iota_axis']:.4f} (axis) -> "
      f"{summary['iota_edge']:.4f} (edge)")

s_half = vmec_jax_half_mesh_s(wout)
s_full = np.linspace(0.0, 1.0, summary["ns"])
iotaf = np.asarray(wout.iotaf, dtype=np.float64)

# stage 2 + 3: trace one line per surface and verify iota ------------------
print(f"tracing 1 field line on each of {len(SURFACE_INDICES)} flux surfaces "
      f"({N_TRANSITS} transits, {STEPS_PER_TRANSIT} RK4 steps/transit)...")
poincare_r, poincare_z, traced_iotas, wout_iotas, surface_s = [], [], [], [], []
for surface_index in SURFACE_INDICES:
    s_value = float(s_half[surface_index - 1])
    phi_nodes, theta_lines = trace_vmec_jax_field_lines(
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
          f"wout iotaf = {iota_wout:.5f}, rel. diff = {relative_error:.1e}")
    assert relative_error < IOTA_RTOL, (
        f"traced iota disagrees with the wout profile on s={s_value:.3f}: "
        f"{iota_traced} vs {iota_wout}"
    )
    # Poincare section at phi = 0: the line's poloidal angle each time it
    # completes a full toroidal transit, mapped to the lab frame through the
    # surface's R/Z Fourier tables.
    theta_at_phi0 = theta_lines[0, ::STEPS_PER_TRANSIT]
    r_cross, z_cross = vmec_jax_surface_rz(
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
boundary_r, boundary_z = vmec_jax_boundary_rz(wout, phi=0.0, n_theta=256)
axis_r, axis_z = vmec_jax_surface_rz(wout, s=0.0, theta=np.array(0.0), phi=np.array(0.0))

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

fig.suptitle("Landreman-Paul precise QA (vmec_jax): every field line is closed", y=0.99)
fig.tight_layout()
figure_path = OUTPUT_DIR / "closed_field_lines.png"
fig.savefig(figure_path, dpi=170)
plt.close(fig)
print(f"wrote {figure_path}")
