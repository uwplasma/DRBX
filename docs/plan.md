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
- **The 1D Hermes-3 gap** (+2.6% n_up, +4.1% T_t): T_t ∝ γ⁻², so +4.1% is a −2%
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

### Phase F: verification and validation (#42)

F1. 1D Hermes-3 gap: swap test with γe = 4.8, γi = 3.5, c_s(Ti+Te), identical
    AMJUEL rates, recycled-neutral energy and conduction law.
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
  Te / Ti energy / momentum ledger imbalance <= 4e-15 / 2e-15 / 4e-15 / 2e-12 of the wall loss. The wall electron
  energy loss is about 500 times the volumetric input power (there is no heat source in the default blob: a decaying
  blob, not a steady SOL).

## 4. References in hand

Also read on 2026-10-21: Hariri & Ottaviani, CPC 184 (2013) 2419 (original
FCI); Chacón, Knoll & Finn, JCP 178 (2002) 15 (physics-based preconditioning);
Coelho et al., PPCF 65 (2023) 085018 (GBS TJ-K); Stegmeir et al., CPC 318
(2026) 109874 (GRILLIX for tokamaks and stellarators); Stegmeir et al. 2016
(advances in FCI) and 2018 (GRILLIX). None of them changes the order of the
phases; they sharpen B2, C1, C2, C3 and add the TJ-K rung F3a.

Still wanted: tabulated HSX edge data (probes, Thomson, H-alpha, neutral
pressure, target fluxes) from the HSX group.

## 5. Status log

| Date | Merged | Notes |
|---|---|---|
| 2026-10-19 | #44, #45, #49, #38, #46, #47, #39, #37 | FCI unification PRs 0–1, threaded host eig, ESSOS adapter, HSX from coils, examples pattern, tutorial, vessel shipped |
| 2026-10-20 | #59, #52–#56 (#62), #51, #57, #63 | decks; exhaust stack; integration; preconditioner `auto`; Bohm limiter |
| 2026-10-21 | — | literature and code review (24 papers, Hermes-3/BOUT++/FELTOR source); this plan |
