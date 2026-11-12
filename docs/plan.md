# DRBX plan

This is the single plan and logbook for DRBX. It replaces the earlier handoff
note. Every merged batch updates the status table; every PR description is a
self-contained handoff. Revisions add lines, they do not remove history.

Guiding principle (owner, 2026-10-19): fewest lines and files, maximum
performance, efficiency and accuracy. Every PR reports its line and file delta.

## 1. Where DRBX stands (2026-10-21)

On `main`:

- HSX FCI backend (seven-field electrostatic drift-reduced Braginskii,
  IMEX-SSP222, implicit current/phi pair with FGMRES).
- Geometry from the shipped HSX coils and vessel through an in-memory coil
  field table; decks build and cache it (`[geometry]` table).
- Exhaust physics: Bohm sheath wall (default, ambipolar), closed particle,
  momentum and energy ledgers on a flux-form support pair, sheath-limited
  current mode (unstable, #60), diffusive neutrals with AMJUEL rates,
  wall recycling, entrance-temperature limiter (#63).
- `drbx run deck.toml` for every model, ten commented example decks,
  `docs/decks.md` checked against the schema by a test. No argparse outside
  the `drbx` command.
- Preconditioner `auto` (`line-uy-pardiag` where it applies): about 15% per
  step on CPU and faster than `line-u` on GPU (#57).

Open PRs: #64 implicit Spitzer–Härm conduction (ready, Aiken asked to review),
#58 README rewrite (waits on #64 for the HSX movie with the Bohm wall).

Open issues: #41 FCI unification, #42 cross-code V&V ladder, #43, #60
sheath-current instability, #61 GPU cyclic line solves.

Size: `src/` is about 95.6k lines in 144 files. Two FCI families remain
(`drbx/native` + `drbx/geometry` and `drbx/fci_braginskii`), with 67
AST-identical definitions (about 3.5k redundant lines) and near-identical
`native` vs `geometry_build` copies of `fci_boundaries` and `fci_geometry`.

## 2. What the 2026-10-21 review found

Sources: 18 papers in the maintainers' paper folder (Stegmeir 2026 GRILLIX
stellarators, Wiesenberger & Held 2023 FV-FCI, Halpern 2016 GBS, Hill/Shanahan/
Dudson LVF and Varenna 2016, Tecchiolli 2026 GBS HSX, Garcia HSX divertor,
Loizu 2012, Guttenfelder 2009 HSX edge, Akerson 2016 HSX SOL, Bader 2017,
Boeyaert 2025, Wersal & Ricci 2015, Zholobenko 2021, Stangeby), three arXiv
papers (Bassanini mimetic FD 2412.20106, Bassanini GBS IMEX 2609.33677, Garcia
Herreros energy conservation 2609.17425), Hermes-3 arXiv 2303.12131, and the
source of Hermes-3, BOUT++ and FELTOR (GRILLIX, GBS, TOKAM3X and SOLEDGE3X are
not public).

### 2.1 Numerics

- **Parallel operators should be adjoint by construction.** BOUT++ (PR #3330,
  `petsc_operators.hxx`) builds `Div = -dV^-1 Grad^T dV`; FELTOR builds
  `Δ∥ = -∇∥† ∇∥` with transposed interpolation; Wiesenberger & Held Eqs. 21–23
  are the same structure. Interpolated maps break adjointness unless the
  divergence uses the scatter (transpose) of the gather map. Then conduction is
  symmetric negative semi-definite, exactly conservative, and solvable with PCG.
  DRBX's support pair is this structure for the sheath rows only.
- **Non-conservative central FCI converges to the wrong sheath-bounded steady
  state** (Shanahan 2016); a face-flux divergence repairs it.
- **The Ve/Te sawtooth is a known failure of collocated central parallel
  gradients and of raw interpolated maps.** Remedies in the literature:
  staggered or skew-symmetric gradients (Bassanini 2412.20106, GBS),
  map smoothing `[1,4,1]/6` without its inverse (Wiesenberger & Held), and
  energy-consistent parallel diffusion on n and u. Uniform artificial diffusion
  is a large energy sink (31% of source power in a GBS TCV run, Garcia Herreros).
- **Implicit treatment** (Bassanini 2609.33677): BPR(3,4,3) IMEX-RK with the
  shear-Alfvén block (Ve, ω, φ), parallel conduction and parallel viscosity
  implicit, conductivities lagged, Schur-parabolised FGMRES with AMG; 3–5×
  typical and 34× at physical χ∥. Putting linearised sheath rows into that
  implicit block is the most likely route to a stable sheath-current mode.
- **Wall endpoint stencil.** Hill et al. found direct non-uniform endpoint
  stencils more prone to instability than leg-value fill (LVF), which fills the
  missing leg value so the interior stencil is unchanged. DRBX uses the
  endpoint family.
- **Map accuracy.** Hermite or 4-point Lagrange interpolation is O(h³);
  bilinear is O(h²) with over 3× the error. Measure numerical perpendicular
  diffusion with the Hill/Shanahan flux-function test.
- **Energy budget.** Check DRBX includes Ohmic heating J∥²/σ (14% of source
  power in GBS when dropped) and report an energy budget per run.

### 2.2 Sheath, neutrals, HSX

- **Sheath-current boundary conditions.** Loizu 2012 gives the full set
  (v∥i, v∥e with exp(Λ − φ/Te), ∂sφ, ∂sn, ∂sTe = 0, ω). The exp(−φ/Te) electron
  loss is a high-gain feedback that matches the density/Te collapse and φ sign
  alternation we see. Tecchiolli 2026 ran 3D HSX stably with a simplified set:
  φ = ΛTe, V∥e = √Te exp(Λ − φ/Te), ω = 0, ∂sN = ∂sTe = 0.
- **Heat transmission.** DRBX's γe = 2/(1−δ) + max(φ/Te, 0) is consistent with
  Stangeby 25.44 (≈5.0–5.3 for hydrogen; Hermes-3 uses 4.8). Check that φ does
  not also include the ≈0.5 Te presheath drop, and that in current mode the
  electron heat flux multiplies Γe = Γi − j/e. γi = 3.5 with c_s(Ti+Te) is the
  fluid value; do not stack the kinetic reduction on top.
- **Hermes-3 FCI sheath** (`sheath_boundary_fci.cxx:406-407`) may omit the 0.5
  factor used elsewhere in the same file; do not copy numbers from it unchecked.
- **The 1D Hermes-3 gap** (+2.6% n_up, +4.1% T_t): T_t ∝ γ⁻¹ at fixed flow (measured), so +4.1% is a −4%
  shift in effective γ, the size of the difference between a fixed 4.8 and
  2 + φ/Te. Also check ionisation and recycled-neutral energy bookkeeping and
  neutral momentum exchange.
- **Neutrals at the HSX edge.** The CX mean free path (1–10 m) is far larger
  than the ≈2 cm SOL; diffusive neutrals are invalid there (our Knudsen map
  agrees). Zholobenko's and Wersal's models also do not transfer. Akerson finds
  edge ionisation small except a local inboard source; Tecchiolli prescribes it.
- **HSX validation data.** Guttenfelder 2009: QHS, B = 0.5 T, 50 kW ECRH,
  n̄ (1–2)×10¹⁸ m⁻³, Thomson Te at 10 points, LFS and HFS Langmuir probes over
  r/a 0.6–1.0 (Te, n, V_float, δIsat/Isat 10–40%, spectra, kθρs ≈ 0.14,
  L_r ≈ 8ρs). Tecchiolli 2026 targets: Te0 ≈ 40 eV, N0 = 10¹⁸, SOL ≈ 2 cm;
  GBS underestimates fluctuations ≥2× and needed collisionality ×3,
  mi/me = 600 and χ∥ ≈ 10× below Braginskii to stay stable. Akerson 2016
  (B = 1 T, 8/7 island): λn ≈ 0.8 cm, D⊥ ≲ 0.03 m²/s. Garcia's FLARE
  connection-length maps cross-check our wall tracing. No paper gives tables.

### 2.3 Codes

- Hermes-3 FCI is moving fast: merged #657 (Dommaschk stellarator example),
  #662 (`sheath_boundary_fci`), #669 (PETSc conduction operator); open #661,
  #665 (neutrals), #664 (vorticity), #626 (connection-length limiter), #675
  (lagged limiter gradient). Its `tests/integrated/*-fci` MMS cases and
  `stellarator-1D`/`dommaschk-3D` examples are the natural first rungs of #42.
- SOLVAX already has `pcg`, mixed-precision preconditioning, multigrid,
  periodic banded LU and pseudo-transient continuation, none used by DRBX. It
  lacks batched cyclic PCR (#61) and a shard_map-aware inner product (DRBX
  hand-writes about 250 lines in `fci_gmres.py`).
- ESSOS `FieldLineToroidal` could replace DRBX's two field-line tracers.

### 2.4 Hot-path findings (static reading, to be profiled)

- Host `np.linalg.eig` through `pure_callback` on 5×5 matrices at every
  characteristic flux evaluation (`fci_helpers.py`).
- Halo exchange gathers static request tables and `psum`s an
  (n_shards × n_requests) matrix every call: O(P²).
- About 40 host conversions per step in `run.py` (ledgers, timing,
  `block_until_ready`); `run_full_eb` is one 1,830-line function.
- Everything is float64; conduction uses unpreconditioned FGMRES where PCG fits.

## 3. Next steps, in order

Each item is one small PR with its line delta, gate result and tests.

### Phase A: diagnose before rewriting (days)

A1. Assemble the HSX parallel operators D and G as matrices at 16³; check
    G = −Dᵀ in the volume inner product; eigenvalues of the linearised Ve–Te and
    Ve–φ–ω blocks with and without the sheath rows. Wiesenberger's
    step-modulated advection test along b. Numerical perpendicular diffusion:
    Stegmeir 2026 anisotropic heat blob (ε⊥ = 1e-8, 64 planes, L1 norm inside
    the flux-surface volume, support vs naive) and Hariri's zonal-field test;
    report the map-cell corrugation signature.
A2. Profile one HSX 32³ step on CPU and GPU (per phase, host syncs counted).
A3. Energy budget per run: sources, sheath, conduction, artificial diffusion,
    Ohmic heating; check J∥²/σ is present.
A4. γ checks: presheath drop not double counted; Γe in current mode; unit
    test against Stangeby 25.44–25.46 and Table 25.1.

### Phase B: one conservative FCI operator family (#41, weeks)

B1. Slimming S0 + S1: delete verified-dead definitions, single-source the 67
    identical definitions into `drbx/fci/_shared.py` (about −4k lines, bitwise).
B2. Adjoint-by-construction parallel operators: `Div = −dV⁻¹ Gᵀ dV` with the
    scatter of the gather map, for every parallel term (not only sheath rows).
    Conduction becomes SPD: switch #64's solve to SOLVAX PCG with a line
    preconditioner. New gate baseline plus MMS convergence and closed-box
    budgets (mass, momentum, energy) and Shanahan's 1D sheath flow.
    Options (GRILLIX, Stegmeir 2016/2018/2026): toroidally staggered dual
    planes, Q canonical→dual with Q* = −V⁻¹QᵀV; the integration-interpolation
    parameter X (2^2X subcells, built once with the map), never without the
    adjoint pair (unstable). Baseline interpolation: cubic Hermite with direct
    derivatives (error ∝ (k/N)⁴, Hariri & Ottaviani) or 4×4 bipolynomial.
    Wall endpoint options next to LVF: ghost points with a second-order Taylor
    Neumann/Dirichlet fill along the field line (Stegmeir 2016 Eq. 21) and
    first-interior-point smoothing (1−2ε)u_P + ε(u_A + u_Q).
B3. Slimming S2 + S3: collapse the near-identical `geometry_build` copies and
    retire the family-1 `Local*` layer (about −8 to −10k lines). Closes #41.
B4. Move `validation/*` and `dev/` out of `src/` (about −5.5k lines moved).

### Phase C: stability and the sheath (#60)

C1. Based on A1: stagger Ve/Vi (or the fluxes) on toroidally dual planes as
    in GRILLIX, with a skew-symmetric gradient as fallback; perpendicular
    hyper-diffusion (N = 2) as the odd-even damper; add energy-consistent parallel diffusion on n and u in place of
    uniform diffusion; optional map smoothing. Try LVF at the wall endpoint.
C2. Sheath-current mode: adopt the Tecchiolli simplified set first, with an
    implicit sheath conductance (Newton on j(φ)) and a clamped exponent; then
    the full Loizu set. Start-up stabiliser if needed: a parallel resistivity
    ramp through the initial transient (GRILLIX W7-AS used ×18 to 0.042 ms).
C3. Implicit Ve–φ–ω block with linearised sheath rows and a
    Schur-parabolised preconditioner after Chacón, Knoll & Finn 2002 (as in
    GRILLIX 2018 and Bassanini): eliminate δω = Pol δφ − Gφ, substitute δVe,
    solve S = Pol − θ²Δt²c Q*(nD⁻¹)Q plus the linearised sheath conductance
    on wall diagonals by PCG/multigrid with lagged coefficients, back-substitute.
    S is SPD only with the exact adjoint pair, so C3 depends on B2. The
    preconditioner may use first-order upwind fluxes. Lagged conductivities;
    Spitzer–Härm with the q = 2.5, α = 0.8 limiter as a documented option.

C3a. Record: the default 32³ HSX deck fails at t = 0.264 (measured 2026-11-08, main 8420e34a, deck: 800 steps to
t = 0.6, dt = 7.5e-4, `sheath_potential = "ambipolar"`, ε = `parallel_dissipation` = 0.1). Numbers only; the cause is
not established. It belongs with C2/C3 (#60) and the Phase A wall rows above.
1. **Failure.** `invalid imex-ssp222 stage after step 352` (351 clean steps; Te < 0 at step 352). A restart from the
   step-300 checkpoint at the same dt fails at the same step. Failing cell (r, θ, k) = (31, 27, 17), and the 4
   field-period images k = 1, 9, 17, 25. Te there: 0.0785 (t = 0.2265), 0.0501 (0.252), 0.0381 (0.258), 0.0167
   (0.2625), −0.113 (0.264); |Ve|: 5.3, 6.2, 10.2, 37, 1157. n stays ≈ 0.23, |Vi| ≈ 0.43; interior |Ve| (r ≤ 29) ≈ 2.2.
   Growth of |Ve| per 0.0015 time units: ×1.13, 1.15, 1.20, 1.27, 1.43, 2.0.
2. **Geometry.** The cell has both field-line legs on the wall (forward 0.097, backward 0.035; wall endpoints at
   r = 31.5, θ ≈ 26.8 and 27.1). 544 of the 1024 r = 31 cells have both legs on a wall (232 of the r = 30 cells);
   θ = 24…28 on plane 17 are all such cells. It is not a support-core row.
3. **Force breakdown at the cell (step-350 state).** Ve rate from the electron pressure and thermal force
   `−μ(G pe + 0.71 n G Te)/n` = +980 (μ = 1836; Ve self-advection through D is −3e-14). The support-pair gradient
   at that cell has two non-zero entries: `G f_c = 0.1503 (f_c − f_s)`, a one-sided leg of length 6.65 to the core
   cell s = (31, 29, 16) (n = 0.9545, Te = 0.7314; the cell itself has pe = 0.0038). The current/phi pair row of
   the same cell has self weight −18.13 (a 0.055 leg to a φ = 0 wall) and −0.150 on s; φ_c = −1.46 and the
   explicit-form μ G_p(φ) = +4.9e4 (the force is in the implicit pair in production, so the net rate was not
   measured).
4. **Wall-layer energy balance, Te rate at the cell, t = 0.225 (Te = 0.0798, Ve = 1.2).**

   | term | dTe/dt |
   |---|---|
   | sheath wall closure (implicit) | −0.81 |
   | perpendicular diffusion | +0.51 |
   | parallel advection `Ve ∂Te` | −0.37 |
   | E×B bracket | +0.10 |
   | curvature | −0.04 |
   | parallel conduction | +0.005 |

   Mean over the r = 31 cells: wall loss −1.69, perpendicular diffusion +0.20. At t = 0.2625 (Te = 0.0166) the
   parallel advection is −12, then −22, −49, −275 as Ve = 34, 61, 136, 748.
5. **dt.** Restarts from step 350: dt/4 (1.875e-4) has Te ≤ 0 at t ≈ 0.26325; dt/2 (3.75e-4) at ≈ 0.2633 (NaN in a
   LinAlg call); dt = 7.5e-4 between 0.2627 and 0.264. The failure time does not move with dt.
6. **ε = `parallel_dissipation`** (restart from step 300, dt = 7.5e-4):

   | ε | 0.1 | 0.5 | 1 | 2 | 4 |
   |---|---|---|---|---|---|
   | first invalid step | 352 | 355 | 360 | 372 | 410 |
   | t | 0.264 | 0.2663 | 0.270 | 0.279 | 0.3075 |

   Same cell every time. max|Vi| at t = 0.27: 0.43 (ε 0.1), 0.31 (2), 0.24 (4).
7. **PR #74** (conservative perpendicular diffusion, 24dee28a), restart from step 350: fails at step 352, Te = −0.074,
   |Ve| = 758 (main: −0.113, 1157).
8. **Failed variant A: `sheath_potential = "current"`** from the step-300 ambipolar checkpoint: `LinAlgError: Array
   must not contain infs or NaNs` in the phi solve on the first step.
9. **Failed variant B: ambipolar with `_sheath_insulating_pair` applied** (condition `self._sheath_current_mode` →
   `self.sheath is not None` at the pair in `solve_implicit_current_phi_pair`): fails at step 335 (t = 0.2513) with a
   NaN; max|Ve| 8.5 / 12.8 / 18.3 / 20.9 at steps 310 / 320 / 330 / 335, interior |Ve| 6.5 → 12.5, 188 cells above 20% of
   the maximum, mostly r = 29…31, at many θ.
10. **Open candidates (untested).** (a) A stable sheath-current mode (C2/C3): a current-mode Ve row with the
    insulating pair. (b) The long one-sided support-pair stencil that a both-legs-wall cell receives from the
    scatter of a core row (item 3). (c) The implicit current/phi pair solve itself at both-legs-wall cells (the net
    Ve rate it produces from item 3 was not measured).

    **Measurements of 2026-11-08 (branch `fix/sheath-wall-closure`, restarts from the step-300 checkpoint, dt = 7.5e-4,
    ε = 0.1, GPU).** Correction to item 2: 544 of the 1024 r = 31 cells have a *forward* wall leg and 544 a *backward*
    one; **216** have both (r = 30: 56). Every r = 31 cell has both legs shorter than 1 (0.016 to 0.3). The upstream
    request to Aiken on #41 had no reply at this date; his `2D_fci` branch is a different (nodal) code base, and its
    `docs/characteristic_wall_boundary_design.md` does not treat cells with two wall legs, so nothing was ported.
11. **Spectrum of the r = 31 layer** (`scripts/fci_diagnostics.py` with `WALL_LAYER = 31`: the Jacobian of the
    (Te, Ve) block restricted to the 1024 cells of the layer, state of step 350, 2048 columns). Main: spectral
    abscissa +18.5, **216 of 2048 eigenvalues with Re > 0**, all real, eigenvectors on one cell (Ve share 1.00,
    along-b correlation 0, wall-cell share ≥ 0.96), in four degenerate field-period images (+18.5 and +15.9). The
    (Ve, ω) block is stable (−0.35). The diagonal ∂(rate Ve)/∂Ve of every probed r = 31 cell (40 of each wall class)
    is negative (−0.6 … −26), so the growth is the Ve–Te loop through one cell: `G` of a row that is only the scatter
    of a core row has a diagonal of one sign, `δTe' = (G pe)_c/(3/2 n) δVe` and `δVe' = −μ(1.71)(G δTe)_c` multiply
    to a positive number (one-sided differences in both equations of a wave pair give a real exponent, a central
    pair an imaginary one).
12. **Variants** (first invalid step after the restart at step 300; main fails at 52, t = 0.264):

    | variant | change | first invalid step | t |
    |---|---|---|---|
    | A | cells with two wall legs outside the pairs (`G = D = 0`, legs into them dropped from both pairs, zero closure) | 200 | 0.375 |
    | A + B | A, and the insulating pair `G − wG(1)` also in ambipolar mode | 191 | 0.369 |
    | A + ν | A with `electron_collision_frequency = 0.1` (main with it: 52, Ve = 1019 at the same cell) | 284 | 0.438 |
    | C | A, current/φ pair rows = support-pair rows (radial-ghost rows dropped) | 98 | 0.299 |
    | D | A, support pair rows extended to the radial-ghost rows with the Neumann q closure of the current pair | 95 | 0.296 |
    | E | A, and cells whose legs read a radial ghost outside both pairs (r = 31 has no pair row) | 162 | 0.347 |
    | `sheath_potential = "current"` | A, restart from the ambipolar state: NaN in the host 5×5 eigensolve after step 1 (initial wall current −1.08); from t = 0: Ve ≈ 50 at t = 0.04, invalid at step 84 (t = 0.063) | 84 from t = 0 | 0.063 |

    A moves the failure from the two-wall cells to one-wall and interior r = 29…31 cells: at step 450 the
    largest Ve is 216 at (31, 29, k) (backward wall leg 0.275, forward leg 0.295, `G(pe) = 0` because r = 31 rows
    touch the radial ghost and are outside the support pair, `μ G_p(φ) = +1.3e4`, φ = −36 on a two-cell patch).
    E removes that site and fails with Te = −4.7e-3 at an r = 30 cell with |Ve| = 26. A on the main state: 152
    unstable (Te, Ve) modes of the layer, abscissa +10.7 (main: 216, +18.5). The cause is therefore not the
    two-wall cells alone but every cell whose parallel gradient is a one-sided scatter.
13. **Open (next).** The Ve–Te block needs a gradient/divergence pair in which the work term `Ve G(pe)` and the
    thermal force have no one-signed diagonal, or a physical parallel friction in the Ve row (Spitzer, same `κ0` as
    the conduction; the constant `electron_collision_frequency` already delays the failure by 0.063 on A). The
    sheath-current mode needs the Schur-parabolised solve of C3 before it can be a closure.
14. **Spitzer friction and Ohmic heating** (default; `electron_collision_frequency` `0` = Spitzer, `> 0` constant
    override, `< 0` off). `dVe = mu nu n (Vi - Ve)`, `nu = 1.612/(kappa0 Te^1.5)` (0.51 times the collision rate that
    gives `kappa = kappa0 Te^2.5`, so `ln Lambda` and the reference state are those of the conduction), implicit in the
    current/phi solve (`Ve = c (Ve_b + h a Vi + h mu G phi)`, `a = mu nu n`, `c = 1/(1 + h a)`, current weight `n c`,
    the adjoint pair unchanged), and `dTe = nu n (Vi - Ve)^2/1.5` (Ohmic, the slot `ohmic_heating`). Wall-layer
    spectrum on the main state of step 350: abscissa +10.7 (A) to +1.2, but four modes at +4018 on the cells with
    Te = 0.017 (`d nu/d Te ~ Te^-2.5` makes the Ohmic feedback of a large `Ve - Vi` positive; present only on the
    state main has already lost). On the state of step 700 of the E run below: (Te, Ve) +1.1 (Te share 0.82), 32 of 2048
    modes, all real and 1000 times below the stiffness.
15. **Runs with friction** (restart at step 300, `ε = 0.1`): A + friction fails at step 564 (t = 0.423, a hot one-wall cell
    at r = 31 with `G(pe) = 0` and `mu G_p(phi) = 1e4`, Ve = 59); **E + friction** (cells without a complete stencil
    outside both pairs, `G = D = 0`) reaches step 723 (**t = 0.5415**, Te < 0 on an r = 31 cell after phi falls to −43 on
    a two-cell patch while n goes 0.245 → 0.107 in 20 steps; Ve ≈ 0 there). The same code from t = 0 fails at step 442
    (t = 0.3315) with |Ve| = 185 on hot (Te = 1.4) cells without wall legs at r = 29…31, phi = −158 on an r = 31 cell.
    Two sectors remain: the ambipolar layer cells (no parallel current can discharge `omega`: `j_w = 0` per leg) and a
    Ve–phi–omega mode on interior layer cells (+6.8 ± 3.2i in the pair-off block).
    Ledger identity of the wall step (#72) ≤ 8e-12 over the 16³ run (main 5e-10); 16³ at t = 0.6: wall particle flux
    6.73e-2 (main 6.06e-2), N 0.6648 (0.6629), W_e 0.9388 (0.9244), max|Ve| 2.3 (1.9): the friction heats and slows the
    16³ layer, it does not degrade the closure.
16. **Not done:** the skew (central) pair for `Ve G(pe)`, the thermal force and `Ve grad Te`. With friction the real
    (Te, Ve) modes are +1.1 and the failure is the two sectors of item 15, so the skew pair would not move it.

17. **Interior Ve–φ–ω mode of the run from t = 0** (ambipolar, E + friction, state of step 400, `WALL_LAYER = 29`,
    the (Ve, ω) block with the potential from the Poisson solve; the stepper is not a callable, so there is no
    production-path Jacobian). Abscissa +14.9, 154 of 6144 eigenvalues with Re > 0, real, Ve share 1.00, ω share 0.00,
    wall-cell share 0.00, along-b correlation 0 (one cell). The diagonal ∂(rate Ve)/∂Ve is negative at 150 sampled
    r = 29…31 cells without wall legs. For the exact adjoint pair the loop `δVe' = −(mu/rho*²) D^T P D n δVe`
    (`P = (−L)^-1 B²/n`, similar to a positive matrix) is stable, so these modes are not the physical loop; they are
    most likely the derivative of the Poisson/GMRES solve (acceptance 5e-5) read through `jax.jvp` (untested). The
    failure itself is a radial column at (θ, k) = (27, 18), r = 28…31: |Ve| 90 → 185 with alternating sign in 42 steps
    (rate ≈ 140), Te ≈ 1.1, n ≈ 1.1, φ ≈ −4, no wall leg, `G(pe)` and `G_p(phi)` both from the same pair rows.
18. **Cell-local sheath current at the cells outside the pairs.** `current_divergence` returned zero for the rows
    outside the pair, which dropped the sheath current of 872 of the 1024 r = 31 wall cells (their rows read the radial
    ghost): `sheath_potential = "current"` from t = 0 failed at step 22 (density 1 → 0.35 at r = 26 wall cells while φ/Te
    went to −1…−2). Those rows now keep the divergence of their own wall-leg currents (endpoint data only, never the
    bulk current). Result from t = 0: φ/Te at the wall cells 2.3 (Λ ≈ 3), wall current 2e-3 (−2.3 at t = 0), max |Ve| ≤ 1.7,
    no mode at (27, 18), 404 steps (t = 0.303). The ambipolar run fails at t = 0.33.
19. **Failure of the current mode at step 405 (t = 0.304).** Deterministic. Stage 1 is finite; the explicit rate of
    stage 1 at the r = 31 cell (6, 14) (n = 0.097 between 0.71 and 0.48, Te = 0.27, Ti = 0.81) is, per term: density
    curvature 5.1e6, Te curvature 4.8e7, Ti curvature −6.2e8, ω curvature 4.4e7 (all other terms O(1)). The stage-2 base
    then has Ti = −4.7e5 and the stage-2 short-wall solve and the curvature eigensolve (4×4, 80 faces) receive NaN. The
    sheath-current wall removes particles faster than the layer is refilled, leaving density holes; the production
    curvature flux (frozen principal matrix, `max_condition` 1e8, `sigma I` fallback) is singular on them. Next: the
    curvature flux at low n (or a floor-free positivity-preserving wall particle sink), then the full run.

20. **Cause of the step-405 failure of item 19 (measured, 32³ HSX, `sheath_potential = "current"`, restart from the step-404 state).**
    The state is ordinary at stage 1 (n = 0.0972, Te = 0.27, Ti = 0.81, neighbours n = 0.71 and 0.48); the curvature
    terms jump from O(0.06) at step 400 to 5e6/5e7/−6e8 on the same state, so the density hole is the trigger, not
    the cause. Instrumenting `curvature_face_linearized_fluctuations`: the three offenders are the upper-wall faces
    (axis 0, face 32) at (6, 14) and its two field-period images (6, 6), (6, 30), with `spectral_fallback = True`, face
    density **−3.6e-4** (owner 0.097, left/right jump states 0.0972/0.0973, radial profile 0.88 → 0.097 over the last
    cell), flux 1.8e9, Frobenius norm of the principal matrix 1.4e30. Mechanism: the face state of a wall face was the raw
    wall trace (`_curvature_bc_characteristic_wall_states` returned `boundary_trace`), whose density undershoots to
    ≤ 0; the fluctuation matrix floors it at 1e-12, the `4 Te²/(3 n)` entries reach 1e12, the condition test sends it to
    the `sigma I` fallback with `sigma = ||A||_F = 1e30`, and the flux is `sigma` times the jump. The basis of the same
    function was already built at `max(trace, 0.1 owner)` (the Te = 7.6e-5 case of the comment there); the returned face
    state was not. Fix: return that floored state (+1/−1 source lines). Max curvature flux on the same state afterwards:
    1.4, no spectral fallback (was 1.8e9 at 3 faces).
21. **Second failure, after the curvature fix: negative density at an r = 31 cell at t = 0.4855 (step 647 from t = 0).**
    The run from the step-404 state reaches t = 0.4845 and then fails (`invalid imex-ssp222 stage after step 243`); the
    minimum density falls at a constant ≈ 0.4 per unit time (not ∝ n): 0.0898 (t = 0.318) → 0.0143 (0.45) → 2.3e-4
    (0.4845). Density terms at the minimum cell (31, 24, 17) of the t = 0.45 state (n = 0.0143, φ = 0.94 against 0.48, −0.02,
    0.36, 0.34 around it): **E×B bracket −0.712**, curvature −0.019, perpendicular diffusion +0.444, sheath wall sink
    −0.126; the sum −0.41 is the observed rate. The sink is not the wall (it is ∝ n, 9 n); at the step-404 state the wall
    sink was −1.04 against +0.54 of perpendicular diffusion. The bracket is the advective form
    `D(U q_face) − q_centre D(U)` with the third-order upwind face value `qL = q0 + (δ⁻ + 2 δ⁺)/6`
    (δ⁻ = q0 − q−1, δ⁺ = q1 − q0): at a minimum of a positive scalar the outflow face value exceeds the cell value, so
    the advection lowers the minimum by an amount independent of n. Fix in `_third_order_scalar_face_states_from_halo`
    (positive scalars n, Te, Ti only, i.e. `positivity_floor` set): first-order face value at the minimum along the
    axis (`q0 < q−1 & q0 < q1` for the left state, `q1 < q2 & q1 < q0` for the right). The same density term of the same
    state: −0.712 → **+0.055**.
22. **Result, 32³ HSX, `sheath_potential = "current"`, from t = 0:** 800 steps to **t = 0.6**, no invalid stage, all fields
    finite at the end (n 0.0210…1.39, Te 0.0156…4.42, Ti 0.175…6.13, Ve −4.1…2.0, Vi −0.22…0.31, φ −8.2…18.6). The minimum
    density falls from 0.097 (t = 0.30) to 0.021 and stays near 0.02 from t = 0.5 on. CPU step cost, local, idle: 2.89 s (mean
    of 800 steps; operators 0.50 s, GMRES 2.38 s, 59.8 iterations per step); the curvature-only fix measured 3.03 s
    (median, with a test suite running alongside), the unfixed code 2.9–3.0 s before it stopped. Ledger, 16³ default deck to t = 0.6 (ambipolar), final code: operator-level wall balance
    (`test_wall_balance_is_independent...`, the ≈ 8e-12 figure) passes at < 1e-11; the run-level `[wall-step]` applied-versus-face
    mismatch (a linearisation error, not round-off) has the same maxima as with the curvature fix alone (n 1.6e-4, W_e 6.3e-4,
    W_i 1.9e-4, P 9e-6), `[wall-run]` closure 9.3e-5 of the wall loss; t = 0.6 wall particle flux 6.85e-2, N 0.6644, W_e 0.9383,
    max|Ve| 2.4 (item 15: 6.73e-2, 0.6648, 0.9388, 2.3). 16³ step 0.298 s (final) against 0.314/0.367 (parent, two runs) and
    0.335/0.339 (curvature fix) in A/B/A/B of 30 steps. GPU (office A4000, idle card, `nice`, no preallocation): 16³, 40 steps A/B/A/B, parent 0.650/0.533 s, final 0.539/0.559 s per step (warm-up included, parity); 32³ restart from step 404, final code, 1.25 s per step (operators 0.38 s, GMRES 0.86 s; the pre-fix run on a shared card measured 1.96 s). Not done: a positivity-preserving *wall
    sink* (item 21 shows the wall sink is not what empties the cell).

### Phase D: performance (each with A/B/A/B timings and a gate)

D1. Replace the host 5×5 `eig` callbacks with an in-JAX eigensolve (Rusanov
    fallback kept); removes a host sync per flux and unlocks GPU.
D2. Static-routing halo exchange (precomputed send lists, `ppermute`).
D3. #61 in SOLVAX: batched cyclic PCR or periodic banded LU as a batched
    matmul; move DRBX's SPMD inner products and generic line preconditioners to
    SOLVAX (about −600 lines).
D4. Remove per-step host conversions; split `run_full_eb`; evaluate
    mixed-precision preconditioning.

### Phase E: neutrals

E1. Prescribed ionisation source option (decouples sheath work from neutrals).
E2. Ballistic neutrals with Knudsen-cosine wall emission, reflection fraction
    and attenuation exp(−∫(ν_iz + ν_cx) dl / v); keep diffusive neutrals for 1D,
    with a local Knudsen-number warning. Adopt Hermes-3's harmonic limiter
    blend and lagged limiter gradient.
E3. Split the energy ledger into plasma sheath cooling and surface power
    (+ I_iz Γ); secondary emission δ as an explicit input.

### Phase E (revision 2026-11-08): one differentiable 3D fluid neutral model, stages N0-N6

Goal: one full fluid neutral model on the FCI grid (n_n, parallel/3D momentum, T_n; not Monte Carlo, not kinetic)
replacing the diffusive model (`fci_neutrals.py`: diffusive N, fixed T_n, mean-free-path cap) as the default once
validated; then the old model is deleted. Closure: advanced fluid neutral (Horsten 2017, Van Uytven 2022):
tau_n^-1 = n_i K_cx + n_e K_i, pressure diffusion Gamma = -(tau_n/m) grad p_n, eta = p_n tau_n, kappa = 5 p_n tau_n/(2m),
flux limiter Gamma_fl = Gamma_c [1 + (|Gamma_c|/(alpha Gamma_lim))^gamma]^(-1/gamma), alpha = 1, gamma = 2, on particle,
heat and viscous fluxes; nine-point (cross-metric) terms kept. Valid for Kn <~ 0.1-0.3, so every run reports a Knudsen map.
Rates: AMJUEL 2020 in 3D; the SD1D fits stay only for the SD1D 1% parity path (deck option for the rates set,
default AMJUEL in 3D).

- N0 (done, #73): one rates module, `native/neutrals/atomic_rates.py`, used by the SD1D and FCI paths.
- N1 (#73): `native/fci_neutral_fluid.py`: implicit conservative (n_n, T_n) transport with limited D_p and kappa, exact
  exchange with the plasma through `reaction_source`; nuclei and neutral energy change only by fluxes and sources.
- N2: parallel neutral momentum (advection, pressure gradient, limited viscosity, CX/ionisation friction with the ions)
  and exact momentum/energy exchange tests (radiation is the only energy sink).
  Done in the N2 PR: `(n_n, M = n_n u_n)` implicit acoustic coupling through p_n with perpendicular pressure diffusion
  (Horsten model 2), implicit upwind advection and limited viscosity of u_n, pressure work and viscous heating in the T_n
  equation, friction as the explicit reaction momentum exchange (stability number `max_dt_nu`); curvature of b neglected.
- N3: wiring into the FCI run: deck option `neutral_model = "fluid"`; history, ledger and restart carry n_n, u_n, T_n;
  reflection wall BC (energy reflection coefficient, Franck-Condon energy for the dissociated remainder); recycling
  from the face ion flux (uses `wall_step_check` from #72); a conservation check independent of the step that fails
  on a 1% injected leak.
  Done in the N3 PR (stacked on #72, which it merges): deck keys `neutral_model`, `neutral_wall_absorption`,
  `neutral_wall_energy_reflection`; the run carries n_n, u_n, T_n through history, ledger and restart; recycling energy is
  deposited as thermal neutral energy; the Maxwellian wall flux is absorbed with probability a and energy-reflected with
  R_E (implicit sinks, ledgered); the recycling ledger of #72 is the independent check (1% leak test).
- N4: validation: 16^3 HSX run with neutrals, Knudsen map, comparison with the diffusive model, 1D slab penetration
  length against the mean free path.
  Done in the N4 PR (32^3 HSX, the only artifact available; a 16^3 one needs the NERSC mgrid): slab penetration length
  matches lambda_mfp/sqrt(nu_iz/nu) to 0.2%; HSX 6-step fluid vs diffusive ionisation rate within 1.5%; Knudsen map: 99.8% of
  the cells have Kn_grad >= 1 (mfp 6 cm at n0 = 5e18 m^-3), so at these plasma parameters neither neutral model is in its
  validity range and the fluid particle limiter reaches 0.01-0.03 in its most limited cells.
- N5: fluid becomes the default; the diffusive model is deleted (fewest lines).
  Done in the N5 PR: `neutrals = true` runs the fluid model; `neutral_step`, `implicit_diffusion`, `neutral_diffusivity`, the
  mean-free-path cap (`neutral_mfp_max`) and the `neutral_model` key are deleted (the SD1D path keeps its own fits).
- N6: molecules / hybrid (kinetic or ballistic) for Kn > 0.3 regions, only if N4 shows the fluid model is outside its range.
  N6 (revision 2026-11-08, additive): N4 showed the fluid model outside its range (99.8% of the HSX 32^3 cells have Kn_grad >= 1,
  mean free path 6 cm). Options weighed from the AFN assessment (Van Uytven 2022, Horsten 2017) and Borodin 2022:
  (a) calibrated Kn-dependent flux limiter and diffusion factor, (b) free-streaming/ballistic correction, (c) few-direction
  discrete-ordinates atom field on the FCI grid (a new field set, an angular operator and its own boundary conditions),
  (d) fluid-kinetic hybrid with an MC solver (Borodin's CFD-EIRENE/APMC: stochastic, not differentiable, excluded by the
  brief). Chosen: (a), because it adds no state and no solver, keeps the exact flux-form conservation (the limiter and the
  factor only scale transport coefficients) and `jax.grad`, and costs 0 new source lines (three constants). (b) is contained
  in it (the limiter caps the flux at alpha Gamma_lim). (c) is the next step only if (a) is not enough: its cost is a new field set and solver,
  worth it only if the residual error is not grid-scale gradients.
  Oracle: a deterministic 1D BGK/charge-exchange atom transport solution (exact cell-wise streaming on 60-100 Gauss
  speeds, dense scattering solve, ~25 lines in `tests/fci_braginskii/test_fci_neutral_fluid.py`). Calibration (alpha, gamma, c)
  on the mean ionisation depth of a uniform slab, Kn_grad = lambda_mfp/lambda_pen = sqrt(nu_iz/nu) = 0.09...0.8, then
  checked on a thin layer (thickness lambda_mfp/Kn, Kn = 0.3...10, absorbing wall + mirror) for the ionised fraction.
  Result: limiter alpha = 3, gamma = 4 (was 1, 2: it cut the penetration depth to 0.65 of the oracle at Kn_grad 0.6) and
  tau_n = (1 + 0.8 nu_iz/nu)/nu give depth/oracle = 0.995, 1.014, 1.018, 1.004, 0.987 at Kn_grad = 0.09, 0.42, 0.60, 0.74, 0.80
  (old: 0.993, 0.79, 0.65, 0.56, 0.53). Thin-layer ionised fraction / oracle: 0.79, 0.87, 0.95, 1.00 at Kn = 0.3, 1, 3, 10
  (old 0.66, 0.85, 0.95, 1.00); the Kn = 3 and 10 results were already within 5% (ballistic transit is carried by the limiter) and are
  kept as regression tests. Open: the 13-21% undershoot of the ionised fraction at Kn = 0.3-1 in the thin layer is a wall
  Knudsen-layer error of the Maxwellian wall sink (the injected beam is counted in the cell density that the sink
  acts on); a kinetic wall albedo is the follow-up. alpha = 3 exceeds the pure-beam flux bound alpha = 2 (n vbar/2), so
  ballistic fluxes in a vacuum-like gap can be overestimated by up to 1.5x; that regime is not covered by the oracle yet.
  N6 follow-up (2026-11-08): (1) Robustness. The NaN seen at 64-128 cells over 0.1 lambda with a strong wall source was not the
  pressure work or any T_n sink (switching the pressure-work and viscous-heating source off, or writing the energy equation in
  a positivity-preserving advective form, left it unchanged); it was the implicit solves losing convergence: the acoustic
  coupling is stiff for cells much smaller than the mean free path and GMRES(60) stagnated at relative residuals 1e-2 to 1,
  after which T_n < 0. `gmres_restart` 60 -> 200 restores 1e-6 residuals (96 and 128 cells, 40 steps stay positive; 140 is the
  smallest restart tried that passes 128). The residual is in the ledger (`max_relres`); a line preconditioner is the next step
  if finer cells are needed. (2) Wall albedo `a/(1 + 2 Kn_w^2)` (Knudsen-layer correction, one constant fitted to the oracle):
  thin-layer ionised fraction / oracle 0.996, 1.009, 0.988, 1.002 at Kn = 0.3, 1, 3, 10 (was 0.79, 0.87, 0.95, 1.00); the slab
  table is unchanged (the albedo only acts at an absorbing wall). (3) Vacuum gap (3 mean free paths at 1e-3 plasma density, then
  plasma, absorbing wall): ionised fraction / oracle 0.983 for alpha = 3, gamma = 4 (1.05 for the old 1, 2). The neutral
  density inside the gap is 21% low for alpha = 3 (0.79) and 20% low for 2-2.5 at gamma = 4, 9% low for (2, 2), 41% high for
  (1, 2): it is not controlled by alpha above 2, and the constants that fix the gap inventory ((2, 2), (1, 2)) break the
  slab depth (0.95-0.79 of the oracle at Kn_grad 0.42-0.8). Trade-off kept: slab depth and ionisation first, gap inventory -21%.
  HSX 32^3 re-run with the follow-up (same deck, CPU): ionisation summed 1.696e-4 (+2.8% vs 1.651e-4), last step 8.441e-6 (+5.4%),
  neutral inventory 5.029e-3 (+1.3%), wall-absorbed nuclei 2.5e-5 (was 9.4e-5: the Knudsen-layer albedo, capped at a third of
  the sink, applies at Kn_w >= 1 where the HSX wall cells are, outside the fitted range 0.04-0.9), peak n_n 0.106, ionisation
  depth 2.16 cells, wall slab 27% of the ionisation, min particle limiter 0.032, Kn_grad >= 1 in 97.2% of the cells, implicit
  residual 5e-15. The wall-loss change is the least validated number of this PR; the first (pre-albedo) comparison follows.
  HSX 32^3, 30 steps of 7.5e-4 (recycling R = 0.9, absorption 0.1, energy reflection 0.5; CPU), old vs N6 limiter/factor:
  ionisation summed over the run 1.651e-4 vs 1.687e-4 (+2.2%; +4.5% at the last step), neutral inventory 4.965e-3 vs 4.984e-3
  (+0.4%), wall-absorbed nuclei 9.4e-5 vs 7.1e-5, peak n_n 0.149 vs 0.105, ionisation-weighted depth from the wall 1.49 vs 2.17
  cells (wall slab holds 40% vs 26% of the ionisation), minimum particle limiter factor 0.007 vs 0.032, Kn_grad >= 1 in 98.1% vs
  97.1% of the cells (mean free path 7.2 cm). Ledgers close as before (transport residual 5e-20, implicit residual 2e-12).

### Phase F: verification and validation (#42)

F1. 1D Hermes-3 gap: swap test with γe = 4.8, γi = 3.5, c_s(Ti+Te), identical
    AMJUEL rates, recycled-neutral energy and conduction law.
    Done in #50 (`benchmarks/hermes3_1d/README.md`, last section): the 1D model uses SD1D's fixed γ = 6, and the
    face-temperature, face-velocity and guard-extrapolation swaps are null. Hermes-3 and DRBX recycled-neutral energies differ by 1.5.
    The gap (+3.4 % n_up, +5 % T_t at equal recycled energy) is open: it needs per-process energy diagnostics.
F2. Port Hermes-3 `tests/integrated/*-fci` MMS cases; run Hermes-3's
    `stellarator-1D` and Dommaschk-3D examples as code-to-code rungs.
F3a. TJ-K (Coelho 2023), the cheaper first stellarator rung: 6 periods,
    R0 = 0.6 m, a ≈ 0.1 m, B ≈ 70 mT, ι ≈ 0.28, n ≈ 5×10¹⁷ m⁻³, Te ≈ 10 eV,
    Ti < 1 eV, no neutrals; compare the n profile, ExB flux vs poloidal angle,
    kyρs spectrum (peak ≈ 0.4, slope); GBS underestimates fluctuations ≥2× and
    misses the Te and φ profiles.
F3b. HSX: reproduce the Tecchiolli/Guttenfelder case (profiles, SOL width,
    fluctuation levels, kθρs, V_float); cross-check wall connection lengths
    against Garcia's FLARE maps; then Akerson's island SOL and Boeyaert's EMC3
    case.
F4. Ask the HSX group for tabulated probe, Thomson, H-alpha, neutral-pressure
    and target flux data.

### Phase G: usability

G1. Deck presets and `--set key=value` overrides; a `drbx.run(config)` API.
G2. A small shipped HSX geometry so real-geometry tests never skip.
G3. README (#58) and tutorial kept in step with each phase.

### Physics extensions after A–F

VMEX + extender finite-beta field provider; electromagnetic (A∥) terms; GPU
production runs and scaling.

### B2 results (2026-10-25, PR #68; 16^3 HSX, default Bohm deck)

- **Pairs.** One exact weighted-adjoint pair `D = -V^-1 G^T V` (scatter of the mapped gather) carries every
  parallel term: the five material rows including Ve, the zero-flow wall, parallel diffusion `D(c G f)`,
  conduction and `grad omega`; the current/phi pair is the second exact pair; a compact forward-leg pair
  `(G+, D+)` carries the face-flux dissipation. Real HSX 16^3: `|D + V^-1 G^T V|/|D|` 1.7e-16 (support), 8e-17
  (leg), `sum V D(f)` 2e-17, `G(1)` 1.4e-16, conduction symmetry 1.4e-16 and `max eig` of its symmetric part 4e-20.
- **Wall rows.** The core-only pair left `D` inconsistent on the two cells next to a wall (1D Bohm flux tube:
  growth rate `c_s/(2dx)`, blow-up). Wall rows now close with the owner value (finite-volume divergence with zero
  wall flux, Neumann `G`). The Ve wall law was the other source of a wall-localised mode: Ve-Te block of the
  16^3 snapshot t=0.075 with Bohm rows: main (from #66) 18 modes with Re>0, max +0.65; with the characteristic Ve
  wall leg on the new pair 16 modes, max +3.7 (95% in wall cells, 100% Ve); with a flux-form Ve leg 224 modes, max
  +19; with no Ve wall leg 4 modes, max +0.07. The final code has no Ve wall leg.
- **Advection.** `-D` is not skew: on sheared maps it has eigenvalues of positive real part (toy +0.1/dz;
  HSX 16^3 `+0.77/dx`; 8-leg step advection min/max -12.8/+15.6, L2 growth 9.6; old pair -48/+51). The skew form
  `-(D+G)/2` is exactly neutral but not conservative. A staggered one-map flux form `-D+ A f` conserves but still grows
  (toy +0.05 to +0.18/dz). `parallel_dissipation` (eps, default 0.1) adds the Rusanov face flux `eps a (f+ - f)` with
  `a = |V| + c_s` (`sqrt(mu Te)` for Ve): unit-speed step advection L2 growth 4.3 (0.1), 2.3 (0.25), 1.5 (0.5),
  1.15 (1), 0.98 (2): neutral at `eps a/|u| ~ 2`, i.e. flows below ~5% of the signal speed at eps = 0.1. It is a
  numerical term without physical scale; its kinetic-energy sink is 0.29% of the wall power at t = 0.6 (0.6% with the
  previous Ve wall law), mass, ion momentum and thermal energy are untouched.
- **Heat blob (Stegmeir).** Perpendicular leakage is a map-interpolation property and is unchanged: L1 error of the
  aligned steady state 0.94 to 1.6 (central `D G`) and 1.15 to 1.5 (compact `D+ G+`) at 16^3.
- **Runs.** Default deck to t = 0.6 at 16^3: finite (Te >= 0.035, |Ve| <= 1.9; main-style legacy Ve gave 7.5), mass /
  Te / Ti energy / momentum ledger imbalance <= 4e-15 / 2e-15 / 4e-15 / 2e-12 of the wall loss (**by construction,
  see "Ledger correction 2026-11-01": an identity of the update, not a conservation result**). The wall electron
  energy loss is about 500 times the volumetric input power (there is no heat source in the default blob: a decaying
  blob, not a steady SOL).

### Ledger correction 2026-11-01 (independent conservation checks)

Aiken Xie's review of #68 found that `wall_step_ledger` could not fail: wall = -sum V dt (implicit increment), other =
sum V dt (explicit increment), and the inventory change is exactly sum V dt (implicit + explicit), so the "imbalance"
was round-off for any scheme. **Every figure quoted from it (1e-12, 4e-15, 2e-15, 2e-12, 1e-11, 2.0e-12 ... of the wall
loss, in this plan, `docs/fci_braginskii_hsx_backend.md` and the PR #68 and #53-#56 descriptions) was by
construction and is retracted.** It is replaced by `LocalFciDrbEBRhs.wall_step_check`: per implicit stage, the
inventory rate the stage produced (stage minus base state) against the rate the wall faces of the stage state give
(`wall_face_fluxes` through `wall_flux_ledger`), plus the explicit density sum, which has to vanish on its own. A
mutation test (`test_wall_balance_is_independent_and_fails_on_a_leak_or_a_wrong_wall_flux`: 1% volumetric leak,
2% wrong wall flux, production operators) fails the check; the unperturbed operator closes to 4e-16.

| Check | Status before | Now |
|---|---|---|
| `wall_step_ledger` n / W_e / W_i / P imbalance, HSX 1e-12 table row | tautological | removed; `wall_step_check` |
| `wall_flux_ledger` face fluxes and inventories | real (face states, owner sums) | kept |
| applied vs contract flux of the step states (5e-3 test) | real | kept in effect: stage-face check, trapezoid of end states agrees to 3e-5 |
| recycling `nuclei_change` vs `step_particle_other - (1-R) wall` | tautological (IMEX part is an identity) | wall loss from the face states, explicit sum from the explicit rate |
| `recycled_particles == R wall` at 1e-12 | tautological (deposit built from the same rate) | against the face loss, 5e-3 |
| neutral split nuclei / energy | real but weak: plasma and neutral updates share the reaction rate; relative to the inventory | kept, stated as such |
| `wall_influx` equals `recycled_particles` | bookkeeping (one number recorded twice) | kept, not evidence |
| neutral diffusion unit tests (null of D, closed box, influx) | real | kept |
| conduction (#64): sum V 3/2 n dT = 0 per stage, closed-wall test | real (property of the SPD solve) | kept; the cross term with the stage n appears in applied-face |
| #53 leg rows equal the face flux; 1D flux tube steady state | real | kept |
| zero-flow HSX "closed box" (`diff(inventory) == other`) | tautological | face and applied wall terms both 1e-12, inventory moves by the explicit sum |
| momentum `P` ledger | tautological | applied vs face, scaled by the wall momentum term |
| Phase A: "ledger closes to -2.8e-17" | tautological | retracted above |
| SD1D `detachment_ledger` (README 1e-9) | real (steady residual, target flux separate) | kept |
| conservative parallel rates sum-free with reported sources | real | kept |

Real numbers, default deck, 16^3 HSX, dt = 7.5e-4, 800 steps to t = 0.6 (office; the wall loss integrates to 6.13e-2
particles, N 0.7243 to 0.6629):

- Applied against face, per step: particles max 1.6e-4 (mean 8.9e-5), W_e max 6.3e-4 (mean 3.8e-4), W_i max 1.9e-4
  (mean 1.4e-4), P at most 1.8e-5 of the wall momentum term. This is the first-order error of the frozen-Jacobian
  backward-Euler stage, not round-off. It shrinks as the loss rate falls (n 1.6e-4 to 6.7e-5, W_e 6.3e-4 to 2.8e-4).
- Whole run, particles: dN = -6.1323e-2 against the integrated face flux -6.1329e-2 and the integrated explicit sum
  +2.0e-7: closure 9.4e-5 of the wall loss (running maximum 1.6e-4). No particle leak. The explicit density sum is
  3.3e-6 of the integrated wall loss, up to 1.75e-3 of the instantaneous loss at the end (perpendicular transport, not
  separated from the radial-edge flux; its absolute sum is up to 0.58 of the loss, so the sum is a cancellation).
- Energy does **not** close to the wall flux. W_e fell by 0.16197 against an integrated face flux of 0.15912 (2.85e-3
  more, 1.8% of the wall loss); W_i by 0.12296 against 0.12123 (1.7e-3, 1.4%); total W_e + W_i + K_i fell by 0.28433
  against the integrated wall energy flux 0.28035 (3.97e-3, 1.4% of the wall loss). The wall flux is not the whole
  budget. Term by term at t = 0.6 (rates as a share of the electron wall loss): perpendicular diffusion -6.7%,
  Poisson bracket -1.4%, curvature -0.7%, parallel work source S_e -0.1%; for ions -4.8%, -1.2%, -0.3%, +0.7%. The
  perpendicular diffusion is `D lap(T)` and `D lap(n)` of the primitive fields with one D (1e-5) for all six diffused
  fields, so `sum V 3/2 (n D lap T + T D lap n)` is -3 D sum V grad n . grad T plus edge terms, not zero (analysis;
  the interior and the edge part were not separated numerically): thermal energy is not conserved by it in the
  interior. It is a model choice (an n-weighted flux `div(n D grad T)` would
  conserve it), not a wall or indexing error, and it is not a bug in the new check. Left alone: changing the
  diffusion form changes the physics, so it is the owner's call. The Poisson bracket and curvature terms are the
  usual non-conservative drift forms (energy exchanged with the E x B motion, not tracked).

What #58 needs (do not edit its branch from here): rebase onto main and take main's
`docs/fci_braginskii_hsx_backend.md` (its copy still carries `wall_step_ledger`, the 1e-12 per-step bullet, the
"Applied-step imbalance" table row, the 1e-11 neutrals bullet and the recycling ledger text; the README of #58 quotes
no HSX ledger figure, its "1e-9" is the SD1D steady-state ledger and stays valid); re-run its HSX movie only if the
movie caption quotes a conservation figure.

### Revision 2026-10-25 (after review of Phase A)

Aiken Xie's `2D_fci` branch already has the full parallel operator this plan was going to build on main:
a mass-weighted adjoint pair, characteristic jump upwinding, a characteristic SAT wall closure and
split-form advection, now in MMS and performance work. Main's FCI backend has further convergence and
conservation problems that this branch would not fix. Decision:

- Do not rebuild that operator on main. C1 (staggered or skew-symmetric gradients, damping) and the
  cross-field leakage work move to `2D_fci`; B2 on main is reduced to an intermediate setup.
- Intermediate setup (#68): the exact adjoint pair for the parallel terms with `parallel_dissipation`
  on by default (ε > 0). Accept it if the default 16^3 deck reaches t = 0.6 with the wall ledger closed;
  32^3 is reported, not required. Until #68 merges, `parallel_diffusion = 1.0` is the interim setting
  for Bohm-wall runs.
- When `2D_fci` finishes, it replaces the parallel operator on main; B3 (retiring the family-1 layer)
  and C2/C3 (sheath-current mode, implicit Ve–φ–ω) are planned on top of it.
- Phases D (performance outside the parallel operator), E (neutrals), F (V&V) and G (usability)
  continue on main.

### Phase A results (2026-10-25; `scripts/fci_diagnostics.py`)

Setup: 16^3 HSX, default deck (200 steps, dt 7.5e-4), snapshot at t = 0.075 (step 100), CPU, x64. Done: A1
(adjointness, advection/heat-blob first pass, eigenvalues), A2 (phase timing, host syncs), A3 (term rates), A4.
Not done: GPU, 32^3 (resolution trend of the heat-blob leakage not measured).

- A1 adjointness, ||D + V^-1 G^T V|| / ||D|| (homogeneous wall data, 3216 active cells):
  support pair 1.7e-16 (exact; sheath rows and #64 conduction use it; conduction V(DG) is symmetric to
  1.4e-16 and dissipative); current/phi pair 1.3e-16; **legacy mapped pair 1.9 and support core + legacy wall
  rows 1.9** (not adjoint at all: these carry the generic Ve/Vi/advection terms);
  generic `parallel_diffusion` operator: V L asymmetric by 10%, sum V L(f) not zero (0.96 relative), so it is
  neither conservative nor symmetric. Implies B2: adjoint-by-construction must cover all parallel terms.
- A1 advection of a step-modulated field-aligned blob (central, no upwind): the support-pair D grows
  modes (min/max f -0.65/2.6 after 2 legs, -48/51 after 8, label variance x3.3 after 2 legs); the
  legacy D blows up (1e30). A field-aligned function gets |G P| dx/|P| = 0.37 (support) and 1.0
  (legacy) instead of 0, i.e. numerical perpendicular/parallel error of order one cell at 16^3. The heat
  blob (eps_perp = 0) never approaches the aligned steady state (L1 error 0.67 at short times, 0.86 at chi t/dx^2 = 1,
  >1 later): heat leaks across field lines at O(1). 16^3 is coarse; trend with resolution not measured.
  Implies C1: central operators need stagger/skew-symmetric gradients or damping.
- A1 eigenvalues (dense 6432 x 6432 Jacobians by JVP over the 3216 active cells, `scipy.linalg.eigvals`;
  ARPACK `eigs` with `which="LR"` did not converge on the Ve-Te block in 60 restarts because the spectrum is
  stiff and oscillatory, so the full spectrum is used; eigenvectors by inverse iteration):
  | block | sheath rows (Bohm) | zero-flow (no sheath rows) |
  |---|---|---|
  | Ve-Te | **18 of 6432 eigenvalues with Re > 0**, max +0.65 (complex pairs, +-0.3i) | stable, max -4.3e-3 |
  | Ve-phi-omega | stable, max -0.42 | stable, max -0.42 |

  **The unstable Ve-Te eigenvectors are not period-2 along b.** The along-b neighbour correlation
  <f, f_plus>/<f, f> (+1 smooth, -1 period 2) is -0.12 to -0.14 for Te and +0.03 to +0.07 for Ve, i.e.
  uncorrelated/rough, not a checkerboard (-1). They are localized at the wall: 80% of the Te part and 60% of the
  Ve part sit in cells with a physical-boundary field-line endpoint, and the
  Ve part dominates the energy norm. The growth is slow (e-folding about 1.5 time units, |lambda| dt ~ 5e-4), so
  it is a physical-rate instability of the sheath-closed system, not a time-step-limited numerical mode. The
  Ve-phi-omega modes are Ve-dominated (94-99%), also not period 2 (correlation -0.26..+0.05). Without sheath
  rows the leading Ve-Te modes are smooth (correlation +0.35..+0.75). Implies B2/C1: the sheath rows (Bohm outflow
  coupling of Ve and Te to the wall legs), not the interior central stencil, create the positive spectrum;
  the interior advection tests below remain a separate accuracy issue.

  Reviewer note (Aiken Xie, 2026-10-25): the same wall-localised, Ve-dominated modes with Re > 0 appear on
  `2D_fci` even with an exactly adjoint pair; there they come from the wall closure combined with the
  flux-form advection, which is not skew-symmetric (the discrete ∇·b term). So "physical-rate" above means
  slow, not physical: the growth is a property of the discretisation. They were resolved (tentatively) with
  a characteristic SAT wall closure (penalising only incoming characteristics, from SBP-SAT work) and
  split-form advection. Also, `zero-flow` in this A/B switches the bulk n/Te/Ti/Vi rows back to the upwinded
  characteristic rows, not only the wall: with the Bohm wall and only the upwinded bulk rows restored, the
  default 16^3 deck ran to t = 0.6 with max |Ve| ≈ 1 (20 Oct). The central bulk flux form is part of the
  cause, not only the sheath rows.
- A2: default step 0.58-0.89 s at 16^3 on CPU (operator 0.21-0.28 s, GMRES/solves 0.34-0.62 s). One stage of
  each phase (compiled, median of 5): explicit RHS 0.093 s, implicit wall legs 0.027 s, Spitzer conduction
  solve 0.037 s, current/phi pair solve 0.088 s, phi reconstruction 0.009 s. **Host callbacks in the jaxprs:
  18 in the explicit RHS, 6 in the wall-leg step, 0 in the solves** (on CPU the 5x5 eig is `jnp.linalg.eig`; the
  `pure_callback` path is only taken off CPU). Host synchronisations of `run_full_eb`, difference of a 6- and
  a 2-step run: 6.25 `block_until_ready` per step, no `__bool__`/`__float__`/`__array__` (so 6 per step are
  the stage/solve boundaries; nothing hidden in the loop). GPU not measured.
- A3 (default `Ve_nu = 0`, `parallel_diffusion = 0`, perpendicular diffusion 1e-5; at t = 0.075, units
  n0 Te0 c_s0 L^2 per normalised time): the only positive Te term is the parallel work, **+9.8e-4 = the whole
  input power**; Poisson bracket -4.0e-4 (-41%), curvature -1.5e-4 (-15%), perpendicular diffusion -6.9e-4
  (-70%). The implicit wall legs remove **0.51, i.e. 519 times the input power** (= 49% of the electron
  inventory W_e = 1.04 per unit time); the Spitzer conduction integral is zero to 1e-18 (conservative). The
  J^2/sigma that Spitzer would give at this state, 1.4e-3 (143% of the input power), is absent from the
  equations. Over the 200-step run W_e fell 1.086 -> 1.006; the ledger closes to -2.8e-17 (**by construction**:
  inventory change against the sum of the increments that made it; integrated wall legs
  8.05e-2, explicit terms -6.3e-5, 3/2 dn dT +6.1e-5). So the default blob run is a wall-loss decay; the
  volumetric terms are 3 orders smaller than the sheath sink.
- A4: `SheathClosure.gamma_e` equals Stangeby 25.44-25.46 exactly: H Ti/Te=0: 4.84, Ti/Te=1: 4.49 (D 4.84
  at Ti=0 not 5.0-5.3). In current mode `current_gamma_e(phi/Te)` equals it at phi = Lambda Te, and the
  electron energy flux is gamma_e Te (Gamma_i - j) (Gamma_e as asked: yes). phi is the owner-cell
  potential used as the entrance potential: the ~0.5-0.7 Te presheath drop is not applied (n_w = n/2 is,
  phi is not), so the floating phi is Lambda Te rather than Lambda Te + ln 2 Te. Test:
  `tests/test_sheath_heat_transmission.py`.

## 4. References in hand

Also read on 2026-10-21: Hariri & Ottaviani, CPC 184 (2013) 2419 (original
FCI); Chacón, Knoll & Finn, JCP 178 (2002) 15 (physics-based preconditioning);
Coelho et al., PPCF 65 (2023) 085018 (GBS TJ-K); Stegmeir et al., CPC 318
(2026) 109874 (GRILLIX for tokamaks and stellarators); Stegmeir et al. 2016
(advances in FCI) and 2018 (GRILLIX). None of them changes the order of the
phases; they sharpen B2, C1, C2, C3 and add the TJ-K rung F3a.

Still wanted: tabulated HSX edge data (probes, Thomson, H-alpha, neutral
pressure, target fluxes) from the HSX group.

## 4b. Performance lane (2026-11-08)

Profiling of the HSX 32^3 default deck (IMEX-SSP222, Bohm wall), 10-20 steps after compile, on an Apple-silicon Mac (CPU),
an office 36-core node (CPU) and an RTX A4000 (both shared with other jobs: spreads are wide, GPU rows are best-of).

### Ranked bottlenecks (before the lane)

| # | Item | Share of a step / cost | Evidence |
|---|---|---|---|
| 1 | FGMRES solves (4 per step, ~63 iterations each at 1e-8) | CPU 80% of the step (3.7 s of 4.7 s); GPU 75% (1.2 s of 1.6 s) | phase timers |
| 1a | Arnoldi orthogonalisation against the zero-padded 101-row basis (pytree path, custom inner product) | about half of an iteration on CPU | HLO ops `solvax.gmres.arnoldi_reductions` |
| 1b | Sequential Givens-rotation loop (static 100 trips per Arnoldi step) in SOLVAX | 64% of all GPU kernel launches (~410 of ~640 events per iteration); device busy only 35% of the step: launch bound | profiler trace |
| 1c | Radial-tree line preconditioner: two `fori_loop`s with dynamic scatters, pivots re-eliminated every application | 1.07 ms/apply on the A4000 (~430 launches), 0.37 ms on CPU | micro-benchmark |
| 2 | Host `pure_callback` 4x4 eig in the curvature fluxes | CPU 0.75 s of 4.7 s (16%); GPU 0.4 s of 1.6 s (25%) | profiler, phase timers |
| 3 | Compile: lowering 42 s + compile 77 s (CPU, cold); GPU cold 343-687 s; initial phi 87-116 s cold | one-time | log |
| 4 | Peak host RSS 5.3 GB: 1.4 GB after geometry and initial state, +2.8 GB lowering and compiling the implicit+phi kernel, +1.1 GB stepping | not the Krylov basis (2 x 26 MB) | RSS sampled per stage |
| 5 | Per-step host work: `_assert_owner_sparse` (7 fields to host), 6 `block_until_ready`, ordered phase-timing callbacks | small (<2%) | not changed |
| 6 | SD1D 1-D detachment case (100 cells, 2 scan points): 6.3 s in-process, of which 4.3 s tracing, 1.7 s lowering/compile, < 1 s run | trace-bound | cProfile; not changed |

No recompiles per step and no hidden host syncs were found (6 `block_until_ready` per step at the stage boundaries).

### Changes (each its own PR, tests identical to the unchanged path)

| PR | Change | CPU Mac step (3 runs, median) | GPU A4000 |
|---|---|---|---|
| #84 | flat-array SOLVAX FGMRES on a single shard | 2.79 -> 2.30 s (-18%) | fewer basis passes |
| #85 | radial tree: pivots once, fused sweeps | 2.59 -> 2.44 s (-6%); cold compile 99 -> 73 s | 1.07 -> 0.08 ms per application (13x) |
| #86 | closed-form 4x4 eig, no host callback | 2.59 -> 2.14 s (-17%) | operator phase 0.40 -> 0.04-0.08 s |
| #84+#85+#86 | combined | 2.59 -> 1.78 s (-31%); cold compile 99 -> 69 s | step 1.05 -> 0.66 s best-of (1.6x); cold compile 687 -> 340 s; initial phi 87 -> 41 s |
| #87 | neutral implicit solves: banded x-line preconditioner, restart 200 -> 30 (stacked on #82) | stiff slab test 20-22 -> 10.6 s | not measured |

Office 36-core CPU (load average 34, one A/B pair only, so indicative): step 26.4 -> 20.0 s (-24%), cold compile 238 -> 195 s, peak RSS 6.9 -> 5.0 GB.
Peak RSS on the Mac is unchanged within the run-to-run spread (5.2-5.4 GB warm). GPU peak memory 0.47 GB (1.5 GB during the initial cold phi).
Equivalence: wall-ledger lines (10 digits) of a 4-step run agree with main; the eig matches LAPACK to 2.5e-13 relative in the
absolute-value action; the tree solve matches a dense solve to 1e-12; flat GMRES matches the custom-product path in iteration
count and to 1e-8 in the solution (the solver tolerance).

Re-measurement 2026-11-09 on main dfc5eeb1 (macOS CPU, HSX 16^3 artifact, default deck, 30 steps, first excluded, one process under a lock, A/B/A/B x 2, mean step): main 0.310 / 0.310 s; #84 0.280 / 0.279 s (-10%); #85 0.324 / 0.312 s (+2.6%, within spread: its gain is the GPU micro-benchmark, not re-measured); #86 0.265 / 0.261 s (-15%). With fluid neutrals and recycling: main 0.310 / 0.309 s, #82 0.311 / 0.310 s, #87 0.310 / 0.306 s (parity). GMRES iterations per step (avg4) 37.5 in every variant. The rotation-matrix patch is now SOLVAX draft PR #185: on top of #84, 0.316 / 0.328 s (SOLVAX main) against 0.286 / 0.299 s (patch), mean -9%.

### Tried and not adopted
- Pallas/Triton single-kernel radial-tree solve (one program per eta plane, in-register sweeps, 64-bit): exact (1e-16) but 0.36 ms
  per application on the A4000 against 0.08 ms for the fused XLA form above; layout conversions between the 32-wide vectors
  dominate. Dropped (needed >= 1.3x end to end).
- Replacing the 100-trip rotation loop by an accumulated rotation matrix in SOLVAX (matvec instead of a sequential loop): same
  iterates (5e-17), 4x per iteration on a matrix-free 32^3 FGMRES(100) on a shared A4000 (0.47 vs 1.8 ms), but end to end inside the
  contended HSX step it was within noise (0.89 s vs 0.66 s best-of). Kept as a SOLVAX patch (branch only, no PR opened).
- `restart` below 100 would also shorten the rotation loop but costs extra cycles on the solves that need ~90 iterations.

### Merge wave (2026-11-10, additive)
#84, #86, #85, #82 and #87 merged in that order, each rebuilt on the then-current main and green in CI before merging.
HSX 16^3 default deck, 30 IMEX steps, local macOS CPU, one process at a time, A/B/A/B x 2, median step: main before the wave 0.31 s,
after #84 0.28 s, after #86 0.23 s (-26% in total; #85 and #82/#87 are parity on this deck). Fields of the last step equal to
main in float32 history storage except phi (1.8e-9 relative, the solver tolerance) and the wall ledger (1e-19); GMRES iterations per
step (avg4) 37.5 in every variant. Office A4000 (idle at start, cold then warm cache, pre-#84 main 0.44 / 0.43 s): #84 alone 0.50 / 0.47 s
(+10%: the flat path still runs the 100-trip rotation loop and the chunked orthogonalisation on the GPU), #85 0.40 / 0.40 s (-8%),
#86 0.29 / 0.28 s (-35%).
Review findings fixed before merging: the closed-form 4x4 eigendecomposition (#86) returned NaN vectors for the zero matrix (a
face with `q = 0`) and for a scalar block, overflowed beyond |A| ~ 1e100, and could return a plausible but wrong basis for a defective
spectrum; it now scales the matrix, takes the axes for a rank <= 1 block, and checks its own residual (a failing case is reported
non-real so the callers fall back). Edge-case tests were added for every PR (flat FGMRES: restart 1/2, zero rhs, first-iteration
breakdown, NaN; tree solver: ratios 1/2/3, single ring, ny = 1, nothing active, invalid profiles; N6 closure at zero gradient and
near-zero rates; line solver on lines shorter than the colour pattern). Stiff neutral tests: slab penetration 9.0 -> 3.1 s each with #87.
Found, not fixed: a plasma with zero density everywhere gives a NaN `jax.grad` of the neutral fluid step (on main before N6 too).
SOLVAX #185 (accumulated rotation matrix) is reviewed against the pytree rotation loop (restart 1/2/m, breakdown, complex, float32,
fixed work); its DRBX GPU measurement is recorded in that PR.

### Still open
Relax the 1e-8 inner target (acceptance is 5e-5; ~25% fewer iterations, changes results), reuse the previous step's Krylov
information, lower the 2.8 GB lowering peak (split the implicit+phi kernel), idle-GPU timings, SOLVAX pytree-path rotation fix for
sharded runs, jit the SD1D residual/Jacobian trace.

## 4c. Solver and capability review from source (2026-11-11, additive)

No code changed. Sources read as data (shallow clones, nothing built or run): PETSc (`src/ksp`), SuperLU_DIST (`SRC`), FFTW,
Hermes-3, BOUT++, FELTOR (`inc/dg`), DEGAS2 (`src`), OpenFOAM-dev (`src/OpenFOAM/matrices/lduMatrix`, `src/finiteVolume/cfdTools`),
lineax, optimistix, diffrax, jax-fem, jax-cfd, jax-md, JAX (`pallas`), and SOLVAX main (0.36.1) next to DRBX main (8dc7bcfc).
Not readable as source: MUMPS (the public mirror is a build wrapper; judged from SOLVAX's wrapper and PETSc's interface), GRILLIX,
GBS and SOLEDGE3X (not public; judged from the literature already in this plan). Gains below are estimates unless a number is
quoted; every item names the measurement that confirms or kills it. Line cost = net DRBX + SOLVAX source lines.

What is already done (so it is not re-proposed): flat-array FGMRES with fixed preconditioner (#84), accumulated rotation matrix
(SOLVAX #185), fused radial-tree sweeps (#85), closed-form 4x4 eig (#86), banded neutral line solver (#87), persistent
compilation cache (`runtime/performance.py`), PCG for Spitzer-Harm conduction on a single shard (`solve_implicit_parallel_conduction`).

### Two facts found while reading

1. **Reverse-mode `jax.grad` cannot pass through the potential solves.** `solvax_gmres_solve` (`fci_gmres.py`) calls
   `solvax.krylov.gmres`, a `lax.while_loop` with no VJP rule (`krylov.py` has no `custom_vjp`/`custom_linear_solve`), and DRBX does
   not use `fixed_work` or any `solvax.implicit` wrapper. Check on a 64-unknown model: `jax.grad` through `gmres` raises
   "Reverse-mode differentiation does not work for lax.while_loop"; the same solve inside `solvax.implicit.linear_solve` returns
   the gradient (-0.187392253 against the analytic value -0.187392253, relative difference 1e-10). The existing gradient gate
   (`tests/test_fci_differentiable.py`) differentiates the explicit 2-field RHS, not an implicit stage.
2. **The four solves per step are `implicit(current)`, `solve_phi(stage_2_base)`, `implicit(stage_2_base)`, `solve_phi(next_state)`**
   (`run.py`, `ssp222_stages`). Each `LocalPerpLaplacianInverseSolver._solve_common` call rebuilds its preconditioner: tree pivots
   and, for `line-uy-pardiag`, four colored probes of the parallel operator (four extra operator applications, about 10% of a
   37-iteration solve), then starts from `state.phi` with the 1e-8 target (acceptance 5e-5).

### Ranked recommendations

| # | Technique | Bottleneck / file | Gain (estimate) | Difficulty | Lines | Differentiability |
|---|---|---|---|---|---|---|
| 1 | Implicit-function VJP around every Krylov solve (`solvax.implicit.linear_solve` or `recycled_linear_solve`) | no reverse mode through `fci_gmres.py::solvax_gmres_solve` | capability: `jax.grad` of any multi-step objective; memory independent of iteration count (no 101-row basis stored per step) | low | +20 | is the point: one transposed solve; tree preconditioner is symmetric so it serves the adjoint |
| 2 | Short-recurrence PCG on the potential and current/phi solves (symmetric Galerkin owner operator) | FGMRES is 80% of the CPU step; Arnoldi was half an iteration (1a) and the rotation loop 64% of GPU launches (1b) | model 32^3 operator, 37 vs 38 iterations, 1e-8: FGMRES 1.34 ms/iteration, PCG 0.14 ms/iteration (9x; DRBX's costlier matvec dilutes this, expect 1.5-2x on the solve, 25-40% of the step) | medium | +45 | `pcg_linear_solve` is already a `custom_linear_solve` (SOLVAX `pcg.py`) |
| 3 | Better starting point: time-extrapolated or projected initial guess, plus GCROT recycling across the four solves | `_solve_common` starts from `state.phi` | 20-35% fewer iterations (plan 4b "reuse previous step's Krylov information") | low-medium | +30 | guess is `stop_gradient`; adjoint independent of it |
| 4 | Inexact solves: tolerance tied to the time-truncation error | 1e-8 inner target against 5e-5 acceptance (`run.py`, `gmres_target_tolerance`) | about -25% iterations (plan 4b); changes results at the new tolerance | trivial | +6 | tighten the adjoint solve separately so gradients keep 1e-8 |
| 5 | Coarse-grid correction for the smooth global modes the tree and `pardiag` do not see | `fci_operators.py::build_solvax_perp_laplacian_preconditioner` | iterations 37 -> ~10 would multiply with #2; unproven | high | +150..250 | preconditioner is outside the adjoint |
| 6 | Lag the preconditioner build across stages and steps | `_build_angular_agglomeration_line_pardiag_preconditioner` (4 probes + pivots per solve) | 5-10% of solver time | low | +15 | none (flexible GMRES, preconditioner does not change the answer) |
| 7 | Remove or replace `solve_phi(stage_2_base)` when `_sheath_current_mode` is off | `run.py::ssp222_stages` | up to one solve of four; verify first | medium (needs an equivalence test) | about -5 | simplifies the graph |
| 8 | Fewer synchronisation points per Krylov iteration: test convergence every k iterations, fused CGS2 | GPU device busy 35% of the step (launch bound) | 1.2-1.5x on the GPU solve, unmeasured | medium | +30 in SOLVAX | none |
| 9 | Deterministic uncollided-flight (optical-depth, track-length) neutral correction | N6 gap: Kn_grad >= 1 in 99.8% of HSX 32^3 cells | accuracy, not speed | high | +200..300 | smooth in the plasma rates, hence differentiable |
| 10 | Neutral solves on SOLVAX flat FGMRES with the banded line preconditioner, pressure eliminated first | `fci_neutral_fluid.py::_gmres` (`jax.scipy` GMRES, left-preconditioned) | shares #84/#185 gains; neutral stiff test is still 10.6 s | low | -10 | already implicit-differentiable in `jax.scipy`; keep with #1 |

### Evidence per item

**1. Implicit adjoint.** Where: SOLVAX `implicit.py::linear_solve` (`jax.lax.custom_linear_solve`, one extra transposed solve),
`recycled_linear_solve` (hands the forward Krylov recycle pair to the adjoint, `_recycled_solve_bwd`); jax-fem
`solver.py::ad_wrapper`/`implicit_vjp` implements the same rule for FEM solves; lineax `linear_solve` differentiates the same way.
Why here: wrap `apply_A`, `rhs` of `solvax_gmres_solve` and the `extra_operator` closure (all parameters are closed over, so
gradients reach density, Te, kappa and `h`). The wrapper must keep the residual-correction `lax.cond` outside the black box.
Cost: a second solve per backward pass. Test: gradient of a 3-step objective against central differences, like
`test_fci_differentiable.py`. Literature: Blondel et al., NeurIPS 2022; Christianson, Optim. Methods Softw. 3 (1994) 311.
Alternative `fixed_work=True` (scan) is reverse-compatible but stores every iterate.

**2. PCG.** Where: SOLVAX `pcg.py` (`custom_linear_solve`, line 412 onwards); PETSc `KSPCG`; FELTOR `inc/dg/pcg.h`. Why: the operator
is `-L_perp + h^2 mu/rho*^2 P kappa D n G` and the code states `E` is symmetric positive (`fci_operators.py`, `pardiag` docstring);
`D = -V^-1 G^T V` makes the fine-grid operator V-self-adjoint, and the tree is symmetric (the preconditioner `T V` is
V-self-adjoint). The caveat is documented in `tests/fci_braginskii/test_implicit_current_phi_consistency.py`: `E P - I` is not
zero on merged owners, so the owner-space operator is not exactly symmetric, and the sheath-current linearisation `J phi` may add
asymmetry. First step (10 lines): measure `|<Ax,y> - <x,Ay>|_V / (|Ax||y|)` on the HSX 16^3 stage operator. If it is above 1e-10,
build the owner operator as the Galerkin product `P^T A_fine P` (restriction = transpose of prolongation, SOLVAX
`galerkin_deflation` states the same symmetry argument) and keep FGMRES only for the sheath-current mode. On a sharded domain
SOLVAX PCG has no global inner product (comment in `solve_implicit_parallel_conduction`); adding the `inner_product` argument that
FGMRES already has is a 10-line SOLVAX change that also lifts that FGMRES fallback.

**3. Initial guess and recycling.** Where: PETSc `src/ksp/ksp/guess/impls/fischer/fischer.c` (`KSPGuessFormGuess_Fischer_1/3`,
projection onto stored solutions, `-ksp_guess_type fischer`); FELTOR `inc/dg/extrapolation.h` (`least_squares`, polynomial
extrapolation in time of the solution of a Poisson/Helmholtz solve); SOLVAX `krylov.py::gcrot` with a `(C, U)` recycle pair
(Parks et al. 2006) and `harmonic_recycle`; PETSc `lgmres.c` (augmentation by error approximations) and `dgmres/`. Why: the four
solves of a step have right-hand sides that differ by O(dt) and the same Krylov limiter modes (smooth, slowly damped); the
solve starts from `state.phi` (zero-order guess). A 2-3 step linear/quadratic extrapolation of phi in time costs two `axpy`s of
stored arrays; projection onto the last two to four solutions costs a few dots. Fischer assumes a fixed `A`; here `kappa = B^2/n`
changes slowly, so measure iterations per solve for: state.phi (now), linear extrapolation, Fischer-3 with 4 vectors, gcrot k=10
recycle; keep the winner. Literature: Fischer, CMAME 163 (1998) 193; Parks et al., SISC 28 (2006) 1651; Morgan, SISC 24 (2002) 20;
Baker, Jessup, Manteuffel, SIMAX 26 (2005) 962.

**4. Inexact solves.** Where: BOUT++ `src/invert/laplace/impls/petsc3damg/petsc3damg.cxx` (3D potential solve, GMRES + GAMG,
default `rtol` 1e-5); OpenFOAM `lduMatrix` solvers stop on separate `tolerance` and `relTol` entries set per case, so inner linear solves run
to a fraction of the outer residual; diffrax `_root_finder/_with_tols.py::with_stepsize_controller_tols` ties the nonlinear solver tolerance to the step
controller; BOUT++ `solver/impls/petsc/petsc.cxx` lags Jacobian and preconditioner. DRBX asks 1e-8 inside a second-order
scheme whose acceptance floor is 5e-5. Measure phi, ledger lines and iteration counts at 1e-6, 1e-7; the plan 4b open item
estimates 25%. Keep the strict target as the adjoint tolerance. Literature: Eisenstat and Walker, SISC 17 (1996) 16; Knoll and
Keyes, JCP 193 (2004) 357.

**5. Coarse-grid correction.** Where: PETSc `src/ksp/pc/impls/gamg/agg.c` (smoothed aggregation; the default of BOUT++'s 3D
Laplacian) and `pc/impls/deflation/`; SOLVAX `precond.py::multigrid`, `semicoarsening_hierarchy` (coarsen only the weakly coupled
axes; Trottenberg ch. 5), `galerkin_deflation` (symmetric, PCG-ready), `low_rank_corrected`; PETSc `fieldsplit.c` Schur
`SELFP` (assembled `A11 - A10 diag(A00)^-1 A01`) is what `pardiag` already does with a lumped diagonal. The plan records that
eta-line corrections "add nothing beyond the diagonal": what remains after tree + diagonal is the smooth global part, the part a
coarse grid removes. The angular agglomeration already gives nested aggregates (`angular_group_sizes`), so the Galerkin coarse
operator can be formed by aggregation of the existing tree graph with the tree as the smoother, without a new mesh. Chacon 2008
(cited in SOLVAX `alfven_block`) and Bassanini (arXiv 2609.33677, plan section 2) both reach for Schur plus AMG. Do after #2, since PCG makes the
coarse correction cheap to test (`galerkin_deflation` is symmetric).

**6. Lag the preconditioner.** Where: BOUT++ `petsc.cxx` (`SNESSetLagPreconditioner`, `SNESSetLagJacobianPersists`), diffrax
`_verychord.py` (linear state kept across steps), optimistix `newton_chord.py` (Chord). `_solve_common` rebuilds pivots and four
probes at every call although `density_dense_g` and `kappa_dense` change by O(dt) between stages. Cache the built closure in the
stage pytree and rebuild every step (or when `max|dn/n|` exceeds a threshold). A flexible Krylov solver tolerates this.

**7. Redundant stage solve.** `ssp222_stages` solves phi for `stage_2_base` and then `implicit(stage_2_base)` solves the coupled
current/phi system again. `solve_implicit_current_phi_pair` uses `state.phi` only as the guess, the sheath linearisation point
(`_sheath_current_mode`) and the face data; with the sheath-current mode off, the first may be replaced by the extrapolated guess
of #3. Needs a 4-step equivalence test on the HSX 16^3 deck (wall ledger and phi to the solver tolerance) before it is trusted.

**8. Synchronisation.** Where: PETSc `ksp/impls/gmres/pipefgmres/pipefgmres.c` (one fused reduction per iteration, overlapped
with the preconditioner and matvec; Sanan, Schnepp, May, SISC 38 (2016) C441; Ghysels et al., SISC 35 (2013) C48); SOLVAX
already has CGS2. On one GPU the cost is launches and host-visible loop predicates, not network latency: the profile shows the
device busy 35% of the step. Test (a) checking the residual every 4 iterations through an inner `fori_loop`, (b) one fused
reduction kernel for the two CGS passes. After #185 the profile must be repeated first; if the busy fraction is above 60%,
drop this item. Pallas (JAX `pallas/ops/gpu`) offers no reduction cheaper than XLA's here: the existing tree-solve trial was 4x
slower than the fused XLA form.

**9. Neutral kinetic correction without sampling.** Where: DEGAS2 `src/tally.hweb` (track-length estimators), `src/collide.web`
(weight fractions, Russian roulette), `src/flight.web`; micro-macro closure in Horsten, Samaey and Baelmans (Contrib. Plasma Phys.
58 (2018) 572). DEGAS2 is Monte Carlo and so excluded by the differentiability brief, but its track-length tally is the line
integral of the rate along each flight, whose expectation is a deterministic optical-depth integral `exp(-tau)`: the uncollided
flux from a wall source cell to every cell. Ray marching it on the FCI grid (a fixed set of directions per wall cell, cumulative
`tau` from the existing AMJUEL rates) gives a positive, smooth, `jax.grad`-compatible first-flight source, and the fluid model
handles the collided population (the micro-macro split). It addresses the N6 gap (Kn_grad >= 1 in 99.8% of cells) that
calibration of the limiter only masks. Cost is the new angular set; rank it after the solver items, before neutrals are
declared validated.

**10. Neutral solves.** `_gmres` (`fci_neutral_fluid.py`) uses `jax.scipy.sparse.linalg.gmres` with a left `M`; SOLVAX's flat FGMRES
takes a right preconditioner and carries the #84/#185 improvements. Hermes-3 `neutral_mixed.cxx::NeutralMixed::precon` is a
block-LU: lower Schur step for the pressure with a Laplacian solve, then back-substitution for N and NV. DRBX's segregated solves
with lagged coefficients already follow that ordering, so only the solver swap is proposed.

### Evaluated, not recommended

- **MUMPS, SuperLU_DIST** (multifrontal/supernodal, block low-rank `ICNTL(35)`, `ReplaceTinyPivot` static pivoting in
  `superlu_defs.h`, `p?gsrfs.c` refinement). The operator is matrix-free, SPD-like and changes every stage (n, Te in kappa), so a
  factorisation is rebuilt per stage; 3D fill grows as N^(4/3) and the work as N^2 (32^3: tens of MB, seconds on the host, off the
  GPU). SOLVAX `sparse_direct.sparse_solve` already wraps both with implicit adjoints and a factor cache; use it only for the
  transposed/tangent solves that reuse one factorisation, or for a coarse level. Static pivoting is not needed for SPD banded solves.
- **FFTW real-to-real, planning.** Coefficients vary with the metric, so there is no exact separable solve; cyclic theta rings are
  already O(n) Thomas (`cyclic_tridiagonal_solve`), cheaper than an FFT; jax-cfd `pressure.py::solve_fast_diag` and SOLVAX
  `separable.FastDiagonal` stay available for a uniform-box test problem. Planning is mirrored by the persistent compile cache.
- **Matrix-free finite-difference Jacobians** (PETSc `mffd/ds.c`, `wp.c`): SOLVAX `newton_krylov` uses exact JVPs.
- **OpenFOAM `bound`** (`finiteVolume/cfdTools/general/bound/bound.C`) clips and averages neighbours, which breaks the exact
  conservation the ledger checks; DRBX's flux-form limiter already keeps the positivity floor. BOUT++ monotonic Hermite clipping
  (`hermite_spline_xz.cxx`) is a nonlinear map and would break the adjoint pair (B2).
- **Hermes-3 `relax_potential`** removes the elliptic solve but makes potential relaxation a stiff ODE with a dt limit.
- **Mixed precision** (`solvax.precond.mixed_precision`): the tree application is 0.08 ms on the A4000; nothing to save.
- **ESDIRK/ARK variants** (diffrax `kencarp*`, `kvaerno*`): SSP222 has two implicit stages and no explicit first stage to save.
- **jax-md, optimistix Newton/LM:** no technique that applies to a linearised IMEX stage; optimistix `ImplicitAdjoint` is the same
  rule as item 1.

### Order of work

1 (and its gradient test), 4 and 3 first (all small, independent, measured on the 16^3 default deck with A/B/A/B), then 2 behind the
symmetry probe, then 6 and 7, then 10; 5 and 8 after the 2-3 results are in; 9 when the neutral model is next revisited.
Every item is its own PR with the line delta, an equivalence test against main, and the profile that justifies it.

### Citations (not previously in section 4)

Blondel et al., Efficient and modular implicit differentiation, NeurIPS 35 (2022); Christianson, Reverse accumulation and
attractive fixed points, Optim. Methods Softw. 3 (1994) 311; Fischer, Projection techniques for iterative solution of Ax=b with
successive right-hand sides, CMAME 163 (1998) 193; Parks et al., Recycling Krylov subspaces for sequences of linear systems, SISC 28
(2006) 1651; Morgan, GMRES with deflated restarting, SISC 24 (2002) 20; Baker, Jessup, Manteuffel, A technique for accelerating the
convergence of restarted GMRES, SIMAX 26 (2005) 962; Eisenstat and Walker, Choosing the forcing terms in an inexact Newton method,
SISC 17 (1996) 16; Knoll and Keyes, Jacobian-free Newton-Krylov methods, JCP 193 (2004) 357; Sanan, Schnepp, May, Pipelined, flexible
Krylov subspace methods, SISC 38 (2016) C441; Ghysels et al., Hiding global communication latency in the GMRES algorithm, SISC 35
(2013) C48; Vanek, Mandel, Brezina, Algebraic multigrid by smoothed aggregation, Computing 56 (1996) 179; Stueben, A review of
algebraic multigrid, JCAM 128 (2001) 281; Chacon, Phys. Plasmas 15 (2008) 056103; Horsten, Samaey, Baelmans, Nucl. Fusion 57 (2017)
116043 and Contrib. Plasma Phys. 58 (2018) 572; Stotler and Karney, Neutral gas transport modeling with DEGAS 2, Contrib. Plasma
Phys. 34 (1994) 392; Amestoy et al., MUMPS, SIMAX 23 (2001) 15; Li and Demmel, SuperLU_DIST, ACM TOMS 29 (2003) 110; Frigo and
Johnson, The design and implementation of FFTW3, Proc. IEEE 93 (2005) 216; Carson and Higham, SISC 40 (2018) A817; Bassanini,
arXiv 2609.33677 (IMEX with physics-based preconditioning for GBS, already in section 2).
### Solver lane items 7 and 8 (2026-11-11, additive; plan section 4c of PR #92)
Item 7 (stage-2 base potential solve): `ssp222_stages` no longer calls `solve_phi(stage_2_base)` unless the sheath-current wall is
on (`_solves_stage_2_phi`); implicit(stage_2_base) re-solves phi from the same right-hand side and nothing else reads the base
potential. Equivalence on HSX, default deck, against main: 16^3 (6 steps) and 32^3 (4 steps) `density`, `Te`, `Ti` bitwise
identical; `phi`/`Ve`/`vorticity`/`Vi` differ by at most 3.4e-8 / 7.2e-8 / 1.7e-8 / 5.6e-11 (16^3) and 8.5e-8 / 8.7e-8 / 4.0e-8 /
1.0e-8 (32^3) of their maxima, i.e. the 1e-8 solver target; wall ledger within 2e-14. Four solves per step become three (the
`avg4` log label now means the mean over the step's solves). Timings and profile are in the PR body.
Item 8 (fewer synchronisation points per Krylov iteration) is not adopted. Convergence tested every 4 Arnoldi steps (inner
masked block, results bitwise identical to the per-step test) was implemented in SOLVAX and measured on the office A4000: an
Arnoldi-plus-stencil micro-benchmark gives 105 / 105 us per iteration (4096 unknowns) and 281 / 300 us (32768) for check_every 1 / 4,
and the DRBX 16^3 step does not speed up (the traced step has the same ~34k device kernels, device busy 28.5% against 22.5%). The
launches are in the matvec and the radial-tree preconditioner (about 220 kernels per iteration), not in the Arnoldi
bookkeeping or the loop predicate, so the lever is fusing those, not the Krylov wrapper. No SOLVAX PR was opened.
### Solver lane item 4: inner tolerance tied to the time step (2026-11-11, additive)
`gmres_target_tolerance = 0` (new default) resolves to `max(1e-8, min(0.2 * acceptance, 10 dt^2))`: 5.6e-6 at dt = 7.5e-4; a positive value is used as given (`1e-8` restores the old behaviour). Rationale: a residual `r` in a potential solve enters a step as `dt r`, the SSP222 local error is `dt^3`.
Measured on the HSX 16^3 default deck, 30 steps (dt 7.5e-4, t = 0.0225), macOS CPU, GMRES iterations per solve (pair1, plain2, pair2, plain-next), mean of the last 29 steps: target 1e-8 `[42.3, 33.0, 42.5, 32.6]` mean 37.6; 1e-7 32.8; 1e-6 27.2; 1e-5 22.9; auto (5.6e-6) `[26.0, 21.5, 26.1, 20.3]` mean 23.5 (-37.5%). Neutrals + recycling deck: 37.6 -> 23.4. 32^3 (10 steps): 64.9 -> 40.4.
Field change against the time-truncation error (last frame, relative L2; truncation = same deck at dt/2): phi 8.1e-6 vs 1.1e-1, Ve 2.7e-6 vs 1.3e-1, vorticity 2.5e-7 vs 1.7e-2, Vi 5.8e-9 vs 1.4e-4, density 2.6e-9 vs 1.2e-6; Te, Ti unchanged in float32 storage (truncation 6.3e-6, 1.1e-6). The solver-induced change is 1e-4 to 1e-5 of the truncation change in every field. 32^3, 10 steps: phi 2.5e-4 vs 2.0e-1, Ve 1.4e-5 vs 1.8e-1, vorticity 1.6e-6 vs 3.9e-2, Vi 1.8e-8 vs 5.4e-4 (ratio 1e-3 at worst, in phi).
Not changed: the conduction PCG (1e-12) and the strict `1e-8` remains the value to give the adjoint solve of the implicit-VJP lane.

## 5. Status log

| Date | Merged | Notes |
|---|---|---|
| 2026-10-19 | #44, #45, #49, #38, #46, #47, #39, #37 | FCI unification PRs 0–1, threaded host eig, ESSOS adapter, HSX from coils, examples pattern, tutorial, vessel shipped |
| 2026-10-20 | #59, #52–#56 (#62), #51, #57, #63 | decks; exhaust stack; integration; preconditioner `auto`; Bohm limiter |
| 2026-10-21 | — | literature and code review (24 papers, Hermes-3/BOUT++/FELTOR source); this plan |
| 2026-11-02 | #68 | B2 merged (Aiken approved). Ledger correction: the wall-step ledger was by construction; independent `wall_step_check` replaces it, figures above retracted (see "Ledger correction 2026-11-01") |
| 2026-11-02 | physics/conservative-perp-diffusion (stacked on #72) | Perpendicular T diffusion in flux form of n T (energy rate D lap(nT), sum-free); one `reference_density_m3` (neutrals now 1e19, was 5e18). Unit test added. Measured on the default 16^3 HSX deck (dt 7.5e-4, 800 steps to t = 0.6), before -> after: dW_e - int face -2.852e-3 -> -7.648e-4 (3.7x smaller), dW_i - int face -1.735e-3 -> -3.301e-4 (5.3x), total W_e+W_i+K_i minus integrated faces -3.98e-3 -> -4.87e-4 (8.2x); particles closed to 9.5e-5 of the wall loss (was 9.4e-5). Remaining bulk residual: -0.5% of the electron wall energy loss, not yet attributed term by term. Neutral rates change for decks that relied on the old 5e18 reference |
| 2026-11-08 | perf lane | #84-#87 (open); ranked bottleneck table and results in section 4b |
| 2026-11-10 | #84, #86, #85, #82, #87 | perf lane merge wave: CPU step 0.31 -> 0.23 s on the 16^3 default deck; review fixes and edge-case tests; details in section 4b |
| 2026-11-08 | docs/examples-current (stacked on #78, merges #74) | Docs and examples brought current: fluid neutrals as the default 3D model (validity Kn <~ 0.3; HSX 99.8% Kn >= 1, so N6 kinetic correction is the open gap), independent wall check output, conservative perpendicular energy diffusion, single reference density. New: `examples/neutrals/fluid_neutral_slab.py`, `examples/inputs/hsx_fci_neutrals.toml`. Retracted-figure and removed-key wording removed from docs/fci_braginskii_hsx_backend.md |
| 2026-11-10 | solvax pin (open PR) | `solvax>=0.36.1` (FGMRES flat path with the accumulated Givens rotation matrix, SOLVAX #185). 16^3 default and neutrals decks, 4 steps, against solvax 0.36.0: every field array bitwise identical; only `phi` (at most 4.5e-9 of its maximum, the GMRES solve) and the wall ledger differ, at rounding level. |
| 2026-11-10 | zero-density NaN (open PR) | `jax.grad` of the neutral fluid step with n = 0 everywhere was NaN (the step value was NaN too). Two causes: `apply_volumetric_source` divided 0/0 for the momentum and temperatures of an empty cell (it now keeps the old Vi, Te, Ti there), and `transport_coefficients` floored nu at 1e-300, so tau_n = 1/nu and its derivative 1/nu^2 overflowed to inf * 0 (floor now 1e-100). Nonzero-density results bitwise unchanged; 16^3 default and neutrals decks bitwise identical to main over 4 steps. |
| 2026-11-11 | solver source review (docs PR) | Section 4c: ten ranked solver/capability techniques from PETSc, FELTOR, Hermes-3, BOUT++, SOLVAX, diffrax, DEGAS2 and others; reverse-mode gap in the potential solves found; no code changes |
| 2026-11-11 | gmres implicit VJP (open PR) | Section 4c item 1: `solvax_gmres_solve` now runs inside `solvax.implicit.linear_solve` (one transposed Krylov solve per backward solve, same preconditioner), so `jax.grad` passes through the potential, current/phi and sharded conduction solves; neutral solves (`jax.scipy` gmres) and single-shard conduction (`pcg_linear_solve`) were already differentiable. Forward fields bitwise identical to main on the 16^3 default and neutrals decks (4 steps). Gradient of a scalar of one 16^3 IMEX step against central differences: relative difference 2.7e-5 (eps 1e-3). Cost and memory in the PR body. The driver's diagnostic `pmin`/`pmax` have no JVP rule and need `stop_gradient` for end-to-end step gradients (done in the test, not in the driver). |
| 2026-11-11 | slim lane: dead source (open PR) | Removed unreferenced code, each symbol with a repo-wide reference count of 1 (its definition): `validation/geometry_lineouts.py`, `validation/geometry_slices.py`, `native/array_backend.py` (+ its test), and 14 methods/functions in `geometry_build/*`, `native/fci_drb_EB_rhs.py` and `geometry/fci_geometry.py` (-594 lines, 16 files). `detachment_sol_model.steady_jvp` stays: it is registered through `@steady.defjvp`, which a static unreferenced-symbol scan misses. Default 16^3 blob and neutrals decks, 4 steps: every history array bitwise identical to main; 552 focused tests pass. Open candidates left to the owner: `native/electromagnetic.py` (test-only, documented feature) and the five `validation/stellarator_*_campaign.py` modules (one test + one example). |
| 2026-11-11 | slim lane: stray scripts (open PR) | Removed `w/` (4 scratch scripts) and `dev/fci_analysis/` (15 post-processing scripts for old shifted-torus runs); no references in src, tests, docs, CI or pyproject, not packaged, no `drbx` code touched (-6814 lines, 19 files). Supersedes the dev/ half of B4. |
| 2026-11-11 | solver lane item 7 (open PR) | Stage-2 base phi solve skipped unless the sheath-current wall is on: 4 -> 3 phi solves per step, fields equal main to the solver tolerance; item 8 measured and not adopted (section 4c lane note) |
| 2026-11-11 | solver lane item 4 (open PR) | Inner potential-solve tolerance tied to dt: iterations -37.5% on the 16^3 default and neutrals decks, -38% on 32^3; field changes 1e-4 to 1e-5 of the time-truncation change |
| 2026-11-11 | slim lane: pillow and stellarator campaigns (open PR) | `pillow` is no longer a base dependency (it stays in the `viz` extra; every `PIL` import is covered because `matplotlib`, which stays in the base set, requires pillow itself). Removed the five `validation/stellarator_{fci_suite,metric_mms,neutral_physics,sheath_recycling,vorticity}_campaign.py` modules, their tests in `tests/test_validation_stellarator_fci_campaigns.py`, the `stellarator-fci/validation_campaign.py` example and the docs and `drbx.validation` exports that referenced them. Owner-approved removals. |
| 2026-11-11 | neutral solves on SOLVAX FGMRES (open PR) | Plan 4c item 10: `_gmres` in `fci_neutral_fluid.py` is `solvax.krylov.gmres` (flat path, right-applied banded line preconditioner, flexible because the preconditioner is zero on inactive cells) inside `solvax.implicit.linear_solve`, so reverse mode costs one transposed solve; the old `jax.scipy` GMRES, its `M` and the recomputed arguments are gone (net -1 line). Neutrals deck (16^3, 4 steps) against main: neutral density within 8e-11 (relative), T_n within 1.3e-6, ionisation rate 3e-11, ledger 5e-11. `neutral_fluid_step` on a 96x16x16 slab with a wall source, A4000 (idle, nice): 0.153 -> 0.099 s (-35%, three A/B pairs). Slab-penetration, thin-layer and vacuum-gap oracle tests pass. Item 9 (uncollided-flight correction) prototyped on the 1D oracle cases and not built: adding a ray-integrated first flight (flux-weighted Maxwellian emission, cell-averaged `exp(-tau/(v mu))`, charge-exchange births as the fluid source) raises the vacuum-gap density from 0.76 to 1.05 of the oracle and the gap ionised fraction from 0.94 to 1.05, but moves the thin-layer ionised fraction (Kn = 0.3, 1, 3, 10) from 0.995, 1.009, 0.988, 1.002 to 1.094, 1.134, 1.105, 1.030, because the limiter and the Knudsen-layer albedo were fitted to the full-flux model and already carry part of the first flight; a micro-macro split needs those refitted first. |
