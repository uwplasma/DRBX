# Electromagnetic Model

DRBX evolves the parallel vector potential A_par in the four-field FCI model
(deck backend `stellarator_turbulence`). One deck switch adds Ampere's law,
magnetic flutter of every parallel gradient and electron inertia:

```toml
[physics]
electromagnetic = true
beta = 0.1            # mu0 n T / B^2
mi_over_me = 1836.0
```

`drbx run examples/inputs/electromagnetic_turbulence.toml` runs it. With
`electromagnetic = false` (the default) the electrostatic code path is the same
code as before and gives bitwise identical results. The model is differentiable
in forward mode (`jax.jvp`) through the whole RK4 step; reverse mode is not
available through the GMRES potential solve in either model.

## Equations

The evolved `v_electron_parallel` is the canonical electron momentum

```text
w = v_e - (mi/me) A_par
```

Each right-hand-side evaluation solves Ampere's law for A_par,

```text
(-lap_perp + beta (mi/me) n) A_par = beta n (v_i - w),      v_e = w + (mi/me) A_par
```

and adds the flutter term to the parallel gradients of the density, the
potential and both parallel velocities,

```text
grad_par f  ->  grad_par f - {A_par, f} / (rho_star B)
```

so the electron parallel force balance is
`dw/dt = -{phi, v_e}/(rho_star B) + (mi/me) (grad_par phi - Te grad_par n / n)`
and the vorticity equation carries the flutter current divergence
`(B^2/n) grad_par (v_i - v_e)`. The fixed-coefficient Helmholtz operator
`-lap_perp + beta (mi/me) n_ref` is the potential solver with
`regularization_epsilon`; the density variation about `n_ref`
(`reference_density`) is iterated `ampere_iterations` times (one pass is exact
for uniform density).

The linear waves are

```text
omega = k_par v_A / sqrt(1 + k_perp^2 d_e^2),    v_A^2 = B^2 / (beta n),    d_e^2 = 1 / (beta (mi/me) n)
```

for the shear-Alfven wave, and in a density gradient (`omega_* = k_y / L_n`) the
drift-Alfven wave, for which the flutter of the background gradient
`b1.grad n0` (the term `omega_* A_par` in the pressure-gradient force) couples
the Alfven and drift-sound branches.

## Verification

`tests/test_electromagnetic_fci.py` seeds waves of the production RK4 step in a
slab with the eigenvector of the 4x4 local dispersion matrix for
`(phi, n1/n0, v_i, w)` (`drbx.validation.electromagnetic_waves`) and reads the
phase of phi.

| check | measured | analytic | difference |
| --- | --- | --- | --- |
| shear-Alfven, beta 0.5 (k_perp d_e 0.14), 12x3x16 cells | 0.3412 | 0.3501 | -2.5% |
| shear-Alfven, beta 0.005 (k_perp d_e 1.4), 12x3x16 | 2.006 | 2.041 | -1.7% |
| drift-Alfven, L_n 10, beta 0.03, k_par 0.1, 12x16x16 | 0.7934 | 0.7904 | +0.38% |
| electrostatic limit of the same drift-Alfven case | | 1.206 | resolved: 35% lower with A_par |
| beta -> 0: max RHS difference to electrostatic / max RHS | 8.5e-7 (beta 1e-6), 8.5e-9 (1e-8) | linear in beta | |
| energy ExB + magnetic + electron kinetic over one period | drift 0.55% (amplitude 1e-3), 2.7% (0.05) | conserved | ExB and magnetic swing 100% and 74% |
| manufactured solution, N = 8, 16, 24 | A: 3.0e-4, 9.6e-5, 4.2e-5; dw/dt: 0.349, 0.079, 0.034 | order 2 | observed 2.1 |

Each row is anchored as follows. The shear-Alfven dispersion with electron
inertia follows Stegmeir et al., Phys. Plasmas 26, 052517 (2019), and the
Alfven-wave tests of Dudson et al., Comput. Phys. Commun. 180, 1467 (2009). The
drift-Alfven wave, including the flutter of the background gradient, follows
Scott, Phys. Plasmas 10, 963 (2003). The beta -> 0 limit, the energy balance and
the manufactured solution are properties of the equations and the discretization
rather than of a paper: A_par and the flutter are O(beta), the sum of the E x B,
magnetic (`<A j_par>/2`) and electron kinetic energies is the quadratic
invariant of the cold model, and the manufactured source is built from the
analytic fields with `jax.grad`.

![Verification](media/electromagnetic_verification.png)

`examples/electromagnetic/electromagnetic_verification.py` writes the figure:
the relative right-hand-side difference to the electrostatic model against beta
(slope 1, 8.5e-9 at beta = 1e-8), the energies over one period (sum drifts by
0.55%) and the error of A_par and dw/dt against grid size (dw/dt converges at
order 2.1; the A_par error is 3.0e-4, 1.6e-4, 9.6e-5, 4.2e-5 at N = 8, 12, 16, 24).

![Alfven dispersion](media/alfven_dispersion.png)

`examples/electromagnetic/alfven_dispersion.py` writes the figure: the shear-Alfven
frequency against `1/sqrt(1 + k_perp^2 d_e^2)` and the drift-Alfven frequency
against its electrostatic limit.

## Turbulence example

`examples/electromagnetic/electromagnetic_turbulence.py` runs the deck larger
with and without A_par and plots the energies (E x B, magnetic, electron and
ion kinetic), max A_par, max vorticity and the density fluctuation spread, and
a short movie of the density, vorticity and A_par in one toroidal plane.

![Electromagnetic turbulence](media/electromagnetic_turbulence.png)

![Electromagnetic turbulence movie](media/electromagnetic_turbulence.gif)

Electromagnetic runs write `apar_frames` and `energies` beside the usual
arrays, and `final_energy_components` in the summary.

## Cost

`examples/electromagnetic/step_time.py` times one jitted RK4 step.

| device | grid (cells) | electrostatic, ms/step | electromagnetic, ms/step | ratio |
| --- | --- | --- | --- | --- |
| laptop CPU | 8x12x6 | 5.2 (min of 16) | 13.7 | 2.6 |
| laptop CPU | 16x24x12 | 74.3 | 198.2 | 2.7 |
| NVIDIA RTX A4000 (one, shared host) | 8x12x6 | 112.6 | 260.5 | 2.3 |
| NVIDIA RTX A4000 | 16x24x12 | 204.8 | 509.7 | 2.5 |

Minimum of 16 jitted steps after warm-up (medians are 6.3, 101, 154 and 258 ms for
the electrostatic steps and 19.7, 227, 318 and 591 ms for the electromagnetic
ones; the hosts were shared). The closure adds two Helmholtz GMRES solves
(`ampere_iterations = 2`) and four bracket evaluations per right-hand side, so a
step costs about 2.5 times the electrostatic one. These grids are
latency-bound on the GPU, which is slower than the CPU at this size; the closure
does not change that.

Options for the overhead, 16x24x12 CPU grid, minimum of 10 steps, two interleaved
passes (A/B/A/B) on a shared host; temporary buffers of the compiled step are
14 MB (electrostatic) and 16 MB (electromagnetic):

| configuration | ms/step, pass 1 and 2 |
| --- | --- |
| electrostatic | 87, 99 |
| electromagnetic, `ampere_iterations = 2`, `ampere_tol = 1e-8` (default) | 197, 163 |
| `ampere_iterations = 1` | 148, 121 |
| `ampere_iterations = 1`, `ampere_tol = 1e-5` | 118, 117 |
| `ampere_iterations = 2`, `ampere_tol = 1e-5` | 182, 117 |

One iteration is exact for uniform density; with density variation about
`reference_density` it leaves the defect uncorrected. The timings vary by
about 30% between passes, so the ordering of the last two rows is not resolved.
Sharing the potential-solve preconditioner, solving once per stage, fusing the
flutter brackets and warm-starting A from the previous step are not implemented.

## HSX backend

The `fci_braginskii` backend takes the same switch from its own `[physics]` table:

```toml
[physics]
electromagnetic = true
beta = 1.0e-3         # Ampere coefficient mu0 n0 Te0 / (rho_star^2 B0^2)
```

With `electromagnetic = false` (the default) every program and every array of a
history is bitwise the electrostatic one (all 37 arrays of a 4-step 16^3 run, CPU).

The evolved `Ve` slot holds the canonical electron momentum
`w = Ve - (mi/me) A_par`, so the inductive term never appears in the `Ve` row.
Each explicit evaluation and each implicit stage first solves Ampere's law with
electron inertia,

```text
(-lap_perp + beta (mi/me) n) A_par = beta n (Vi - w),     Ve = w + (mi/me) A_par,
```

as one positive definite Helmholtz problem with the exact density in the shift
(the `extra_operator` of the potential solver, with its GMRES configuration and
preconditioner; `A_par` takes the boundary type of the potential with zero
value on the walls), and evaluates the electrostatic right-hand side at the
physical `Ve`; `dw/dt` is that right-hand side's `Ve` row. Histories, snapshots
and restarts hold the physical `Ve` and `apar`.

`beta` is Ampere's law in the metre-normalized variables: `A_par` is in units of `rho_star B0 L` (so that `dA/dt = -grad_par phi`) and the vorticity is `rho_star^2 div(n grad phi / B^2)`, which makes the Alfven speed `B / (rho_star sqrt(beta n))` and the physical beta `rho_star^2 beta`. At the physical `beta` of a real plasma `beta mi/me` is far above `k_perp^2`, `Vi - Ve` is a difference of nearly equal numbers and its solver-tolerance noise exceeds the true current (the `w <-> Ve` round trip loses `beta mi/me / k_perp^2` digits); keep `beta mi/me` of order one until the current is taken from Ampere's law instead.
Magnetic flutter enters both parallel gradient pairs. The field is `B = curl(A b)`, so `b1 = grad A x b / B`; the E x B velocity is `b x grad phi / B`, whose advection of `f` is `+rho_star {phi, f} / B` in the right-hand sides, hence `b1.grad f = -rho_star {A, f} / B`: the same bracket with `phi -> A` and the opposite sign (the four-field model's `-{A, f} / (rho_star B)` is its `1/rho_star` E x B scale). `tests/test_electromagnetic_fci.py::test_flutter_follows_the_perturbed_field_line` fixes the sign from field-line invariance alone. With
`F f = -rho_star {A_par, f} / B` (the centered compatible-flux bracket of the
vorticity equation, constants removed so that `F 1 = 0`) the support pair and the
current/phi pair become `(G + F, D + D_F)` with `D_F = -M^-1 F^T M` from
`jax.linear_transpose`. The pair stays the exact weighted adjoint, so
`sum M D f = 0` and the electron kinetic, E x B and magnetic energies exchange
through `mu G phi` and `D j` only. `F` has no component along `b`, so the flutter
carries no flux through the parallel (sheath) legs, and `A_par = 0` on the radial
wall leaves none through it; the characteristic wall closures act on the
physical `Ve`. The implicit current/phi solve uses the plain pair with `A_par`
of the stage base; the flutter part of the potential force and of the current
divergence stays in the explicit right-hand side, because one flutter bracket
costs about 13 ms on the 16^3 model and a GMRES matvec would pay two.

A backward-Euler step of the cold `(w, omega)` system with `A_par` lagged at the
step base has amplification below one for every step size, but damps the stiff
Alfven branch less than the exact step where `beta (mi/me) > k_perp^2` (0.995
against 0.01 at `h omega = 100`, `beta = 0.1`, `k_perp = 1`). Putting the Ampere
response of the stage inside the implicit solve, as the screened velocity change
`dw - beta (mi/me) H^-1 n dw` with the line preconditioner as `H^-1`, was tried on
the 16^3 model and removed: at `beta = 1e-3` it is identical to the lagged
treatment within noise, at `beta = 0.1` it changes the magnetic energy at
`t = 0.06` by -2.6% (1.2645e-9 against 1.2985e-9; halving the step gives 1.2325e-9
against 1.2656e-9), the same size as the change of halving the step, with the
same GMRES iterations.

Verification (`tests/fci_braginskii/test_hsx_electromagnetic.py`):

| check | measured | analytic |
| --- | --- | --- |
| shear-Alfven frequency, periodic slab with the production support pair and `_conservative_parallel_rates`, beta 1e-3 to 0.1, three `(k_perp, k_par)` | relative difference 6.4e-15 at most; growth rate below 1e-13 of the frequency | `k_par sqrt(mi/me) / (rho_star sqrt(k_perp^2 + beta mi/me))` |
| beta -> 0, beta = 1e-9 | 417.55695 against the electrostatic 417.55733 (9.2e-7) | `beta (mi/me) / (2 k_perp^2) = 9.2e-7` |
| `E_ke + E_mag + E_ExB` exchange through the flutter pair, finite amplitude | `dE/dt` is 2.3e-16 (beta 1e-3) and 1.0e-16 (beta 0.1) of the sum of the terms | 0; without the magnetic term it is O(1) |
| flutter pair on the HSX 16^3 model, dense matrices | `D + M^-1 G^T M` below 1e-12 of `D`, `G 1` below 1e-12 | exact adjoint, constants exact |
| `w -> Ve -> w` through the Helmholtz and Poisson solves (relative to `max Ve`) | 2.0e-3 at the production tolerance, 5e-14 with `tol = 1e-12` | the Helmholtz residual, amplified by the conditioning of the unshifted `-L_perp` solve; the maps are exact (fresh 16^3 geometry on current main) |
| `jax.grad` of one compiled IMEX step (flows of one sign, the ion velocity sourcing Ampere's law) | 0.4752243 against central differences 0.4752243 (4.6e-8) | |

The gradient needs the implicit-function VJP of the Krylov solves.

HSX 16^3 deck (`hsx_fci_blob.toml`, `dt = 7.5e-4`), `beta = 1e-3`, 400 steps to
`t = 0.3`: finite and stable, 34.9 GMRES iterations per solve, no solve accepted
only under the relaxed tolerance. The energies written to `em_energy`
(`<A j>/2`, `-<phi omega>/2`, `<n Ve^2>/2mu`, `<n Vi^2>/2`, `3<n (Te + tau Ti)>/2`):

| t | magnetic | E x B | electron kinetic | ion kinetic | thermal |
| --- | --- | --- | --- | --- | --- |
| 0.06 | 1.33e-11 | 3.53e-6 | 6.57e-8 | 2.10e-8 | 2.1081 |
| 0.18 | 1.15e-10 | 2.02e-4 | 6.47e-7 | 2.17e-7 | 2.0264 |
| 0.30 | 2.70e-10 | 1.21e-3 | 1.55e-6 | 1.33e-6 | 1.9766 |

The magnetic energy is O(beta) and 2e-7 of the E x B energy at `t = 0.3`; at
`beta = 0.1` it is 100 times larger (1.30e-9 against 3.53e-6 at `t = 0.06`) and
the run is equally stable. The wall particle ledger of the existing driver closes
as in the electrostatic run.

Cost of the closure, 12 steps, minimum over A/B/A/B passes: CPU 0.27 and 0.28 s
per electrostatic step against 0.33 and 0.34 s electromagnetic (+21%); one RTX
A4000 on a shared host 0.18-0.19 against 0.25-0.27 s (medians are not
reproducible there, the host load was above 100). The closure adds four
Helmholtz solves per step (22 GMRES iterations from zero, 11 from the previous
`A_par`, 11 ms each on the GPU) and the explicit flutter pairs. Carrying `A_par`
across stages as a warm start is not implemented: it is not part of the state.

## Limits

Reverse-mode gradients through the step require the implicit-function VJP of the Krylov solves (see the gradient PR). The 1D SOL flux tube and the 2D
Hasegawa-Wakatani box do not carry A_par: the flux tube has no
perpendicular structure to invert and the HW box has no parallel direction. The HSX backend carries it only with `[physics] electromagnetic = true`; its implicit current/phi solve keeps `A_par` of the stage base. The
perpendicular Laplacian of the potential solver has a wide stencil with an O(1)
inconsistency at the periodic seam cells in y, so the verification slabs put the
perpendicular structure in x with Dirichlet walls.
