# Hermes-3 vs DRBX: SD1D-matched 1D divertor leg (ladder rungs L3/L4, issue #42)

A grid-converged cross-code comparison between Hermes-3 and DRBX's `detachment_sol_model`
(which follows SD1D's 13.6 eV scan term by term). Each code is run on five grids
(100 to 1600 cells) at three operating points. Observed order, Richardson extrapolation and GCI are
computed per code, and the extrapolated values are compared. The tolerance is the sum of both GCIs
and a measured model-difference bound.

## Pinned versions

| Item | Version |
|---|---|
| Hermes-3 | `955c8722970e0ad88bc4914fa4be3659d3bd4bf5` plus `hermes3_matching.patch` (sha256 `ec9c6a32...7e9`). Built locally as commit `01807655` |
| BOUT++ | `10ec8defb901657ad1c66f95f2bf532ae6ae6099` (Hermes-3 submodule). SUNDIALS 7.2.1 and netcdf-cxx4 downloaded by BOUT++ |
| Toolchain | conda-forge: gcc 15.3, Open MPI 5.0.11, PETSc 3.26.0. Build recipe in issue #42 |
| DRBX | main `bd296ebe` |

The patch adds four opt-in options to Hermes-3. Without them Hermes-3 cannot express SD1D's choices.
- `neutral_parallel_diffusion`: `max_mean_free_path` (0.1 m), `neutral_temperature_floor` (0.5 eV) and `neutral_neutral_cross_section`. Together these give SD1D's closure `D = dneut vth^2 / (nu_cx + nu_iz + vth/min(lambda_nn, 0.1 m))`.
- `simple_conduction`: `coulomb_log`. SD1D freezes the Coulomb log at its normalisation values.
- `ionisation`: publishes the ionisation collision frequency `h_h+_iz` to the neutral closure.

## How Hermes-3 is matched to SD1D (`deck.BOUT.inp`)

| SD1D choice | Hermes-3 deck |
|---|---|
| Single temperature | T_e and T_i are evolved, with `braginskii_collisions:ei_multiplier = 1e5`. The residual \|T_e − T_i\|/(T_e + T_i) at the target is 1e-6 to 4e-3, and enters the bound |
| Electron conduction 3.2 n T tau_e / m_e, Coulomb log frozen | `simple_conduction` on electrons only, `kappa_coefficient = 3.19595`, `coulomb_log = 13.5078` |
| Willett ionisation fit, 13.6 eV per ionisation | `ionisation` component, `Eionize = 13.6` |
| SD1D CX fit at E = 10 eV | New rate file `json_database/AMJUEL_SD1D_CX10.json` (same polynomial), evaluated at T_e |
| SD1D recombination fit | **Not expressible**: AMJUEL H.4 2.1.8 is kept. Bounded with a DRBX toggle (below) |
| Neutral D with 0.1 m mfp cap, 0.5 eV floor, nn collisions | Patched `neutral_parallel_diffusion`, `equation_fix = false` |
| Sheath, total gamma = 6 | `sheath_boundary_simple`, `gamma_e = gamma_i = 3` |
| Recycling 0.99 at 3.5 eV | `target_recycle_multiplier = 0.99`, `target_recycle_energy = 3.5`. This is 3.5 eV per neutral (T_n = 2.33 eV), whereas SD1D's 3.5 eV is a temperature (5.25 eV per neutral). See the swap tests |
| Area 1 to 2 below the X-point, SD1D dy(y) | Same expressions in `[mesh]` (`J`, `dy`) |
| 5e7 W m^-2 over 10 m | 2.5e7 into each of P_e and P_i. Both codes rescale the source by 10 m / (discrete source length) so that the integrated power is grid independent |

Different control variable: Hermes-3 runs with a fixed particle source and no feedback. At steady
state with 99 % recycling this fixes the target flow, Gamma_t A_t = S x 10 m / 0.01, and
n_up becomes the output. DRBX holds n_up fixed (SD1D's controller). The comparison is therefore
made at equal Gamma_t A_t. DRBX is scanned over n_up = 1.6 to 9e19 on every grid, and its
extrapolated curves are interpolated (in log n_up) at the Hermes-3 flow. The target-flux
observable is reported as "n_up needed for this target flow".

## Commands

```bash
# Hermes-3: build per issue #42, then
git -C hermes-3 checkout 955c8722 && git -C hermes-3 apply DRBX/benchmarks/hermes3_1d/hermes3_matching.patch
cmake --build hermes-3/build
benchmarks/hermes3_1d/run_all_hermes.sh <hermes-3 dir> <out>/hermes      # 15 runs, 1 rank each, under an hour
# DRBX (JAX_PLATFORMS=cpu OMP_NUM_THREADS=4)
python benchmarks/hermes3_1d/run_drbx.py <out>/drbx base 1.6e19 1.8e19 2.0e19 2.2e19 2.5e19 3.0e19 \
    3.5e19 4.0e19 4.5e19 5.0e19 6.0e19 7.0e19 8.0e19 9.0e19                 # about 40 min
HERMES_JSON=<hermes-3 dir>/json_database python benchmarks/hermes3_1d/run_drbx.py <out>/drbx amjuel_rec \
    2.0e19 3.0e19 4.0e19 5.0e19 6.0e19 7.0e19 8.0e19 9.0e19
python benchmarks/hermes3_1d/compare.py <out>/hermes <out>/drbx
```

The Hermes-3 solver is backward Euler with SNES and a direct LU preconditioner. With hypre/euclid,
SNES stalls on grids of 400 cells and above. The run outputs are not in git. On the reference
host, the 126 JSON records (`h_*.json`, `drbx_*.json`) are listed in a sha256 manifest whose own
sha256 is `583e47a42f344e7f757df7e4d23b96e94bf5bee765420a7e23ae6b3d88fd54e2`. The `compare.py`
output has sha256 `b6966ddd9b0e9525b1d13d7d815160f2baaab6d740c4b8979d84fa4829b38c31`.

## Results (extrapolated value ± GCI, grids 400/800/1600)

| Gamma_t A_t [s^-1 m^-2] | Observable | Hermes-3 (p) | DRBX (p) | Diff | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| 2.231e24 | n_up [1e19 m^-3] | 2.207 ± 0.025 (1.05) | 2.081 ± 0.060 (0.97) | +6.0 % | 0.085 | **differ** |
| | T_t [eV] | 21.19 ± 0.50 (0.72) | 21.05 ± 0.84 (0.88) | +0.7 % | 1.4 | agree |
| | T_up [eV] | 58.77 ± 0.26 | 58.82 ± 0.05 (1.16) | −0.1 % | 0.31 | agree |
| | L_iz [cm] | 6.05 ± 0.24 (0.83) | 6.18 ± 0.08 (0.94) | −2.1 % | 0.35 | agree |
| 6.885e24 | n_up | 5.44 ± 0.62 (0.78) | 4.42 ± 0.33 (0.71) | +23 % | 0.95 | **differ** |
| | T_t | 5.63 ± 0.64 (0.53) | 5.91 ± 0.35 (0.56) | −4.7 % | 0.99 | agree |
| | T_up | 57.80 ± 0.04 | 57.82 ± 0.04 (1.21) | −0.03 % | 0.07 | agree |
| | L_iz | 2.44 ± 0.18 (1.08) | 2.68 ± 0.06 (0.91) | −9.0 % | 0.28 | agree |
| 1.040e25 | n_up | 9.7 ± 2.6 (−0.25, not asymptotic) | 7.68 ± 0.38 (1.40) | +27 % | 3.0 | agree (inconclusive) |
| | T_t | 3.675 ± 0.07 (oscillatory) | 3.66 ± 0.56 (−0.11) | +0.5 % | 0.63 | agree |
| | T_up | 57.36 ± 0.03 | 57.42 ± 0.02 (1.69) | −0.11 % | 0.056 | **differ** (by 0.06 eV) |
| | L_iz | 2.14 ± 0.49 (0.62) | 2.37 ± 0.06 (1.27) | −9.5 % | 0.62 | agree |

Definitions:
- T_t is the last-cell temperature (Hermes-3 uses (T_e + T_i)/2).
- L_iz is the ionisation-weighted distance of the front from the target, with the same post-processing for both codes.
- The GCI uses Fs = 1.25 when 0.5 <= p <= 2.5. Otherwise Fs = 3, and for oscillatory sequences 3 x the maximum change.

Ledgers on the finest grids: Hermes-3 particle balance is within 2e-8 and power balance within 7e-8
(sources against target, radiation and V grad P work). DRBX closes both to 1e-15.
Five DRBX 1600-cell runs at n_up <= 2.5e19 stop at residual 2e-9 to 5e-8 (target 1e-9). This is
2 to 4 orders below the grid differences.

## Model differences and attribution

Measured bounds (in `compare.py`):
- Recombination rate, AMJUEL vs SD1D: DRBX with the AMJUEL rate moves every observable by 3e-2 % or less.
- Two temperatures: at most 0.4 % on T_t (low density, 1600 cells).

Discretisation differences vanish in the extrapolated values and sit inside the GCIs. These are the MC vs MinMod limiter and cell-centred vs Simpson rate integration.

What agrees: the target temperature, the upstream temperature (to 0.1 %), the ionisation-front
location, and the closed ledgers, all at equal target flow over the full range from
attached to high recycling.

What differs: Hermes-3 needs 6 % (attached) to about 23 % (high recycling) more upstream density
for the same target flow. Both codes give the same T_t and the same Gamma_t, so the target
pressure is the same. The difference is therefore in the upstream-to-target total pressure loss.
Neither the recombination rate nor T_e != T_i can account for it, since together they bound less
than 0.5 %. The attribution below assigns most of it to two named model differences.

## Attribution of the n_up difference

Method. `momentum_budget.py` sums each code's discrete steady ion momentum equation over the leg,
`P_up + sum P dA - F_fric = A_t P_f + A_t (m n v^2)_f`, at matched Gamma_t A_t on the same grid.
Each candidate was then switched off or aligned in one code, and the n_up gap at equal flow was
re-measured per grid (DRBX n_up interpolated in log Gamma_t A_t on the same grid).

Budget, attached flow (2.231e24), 800 cells, as published (Pa per upstream m^2):

| Term | Hermes-3 | DRBX | H − D |
|---|---|---|---|
| P_up | 408.8 | 377.2 | +31.6 |
| expansion force sum P dA | 406.3 | 374.9 | +31.4 |
| friction to neutrals | 165.4 | 121.7 | **+43.7** |
| face pressure A_t P_f | 341.6 | 337.6 | +4.0 |
| face momentum flux A_t (m n v^2)_f | 308.0 | 292.8 | **+15.3** |
| CX reactions [s^-1 m^-2] | 4.37e24 | 4.43e24 | −1 % |
| CX-weighted slip <v_i − v_n> [m/s] | 11 922 | 9 858 | +21 % |

The extra upstream pressure is 70 % extra friction and 30 % a larger sheath momentum flux. The CX
reaction count is the same, so the extra friction comes from slower neutrals. The CX slip is 21 %
larger, and the ionisation momentum gain is smaller.

| Candidate | Test | n_up gap, attached (400 / 800 / 1600 cells) | n_up gap, 6.885e24 (400 / 800 / 1600) |
|---|---|---|---|
| As published | Hermes base vs DRBX base | +9.5 / +8.3 / +7.3 % (extrapolated +6.0 %) | +5.6 / +14.9 / +18.9 % |
| CX rate form and temperature | Hermes `Fhh+_cx` against n n_n <sigma v>(T_e) m (v_i − v_n) | identical (ratio 1.0000); no effect | — |
| Recombination rate; T_e != T_i | earlier bounds | < 0.5 % | < 0.5 % |
| Sheath density and pressure extrapolation (exponential vs linear) | Hermes `density_boundary_mode = pressure_boundary_mode = 2` | +10.0 / +8.5 % | +7.5 / +16.4 % |
| **Hermes-3 neutral viscosity** (`neutral_parallel_diffusion:perpendicular_viscosity`, default on; SD1D has none) | off | +8.6 / +7.0 % | +15.9 % (400) |
| **Hermes-3 neutral momentum flux through the no-flow wall** (Lax term of `Div_par_fvv` at the boundary) | plus `h:fix_momentum_boundary_flux = true` | +7.5 / +6.7 % | — |
| **SD1D/DRBX sheath momentum flux**: MinMod face value of NV_i times V_out (SD1D default `bndry_flux_fix = false`, NV_i guard = N_out V_out), 14 % below Gamma_t c_s at 800 cells | DRBX `sheath_momentum_flux=face` sets it to N_out V_out^2 (vs Hermes base) | +0.9 / +3.2 % | — |
| Hermes-3 ion sheath momentum flux set to n_mid v_mid^2 | `h+:fix_momentum_boundary_flux = true` (vs DRBX base) | +5.5 / +11.4 / +6.3 % | +5.5 / +10.7 / +14.4 % |
| **All aligned** | Hermes: viscosity off, ion and neutral `fix_momentum_boundary_flux = true`; DRBX: `sheath_momentum_flux=face` | +1.8 / +2.0 / +2.5 % (100, 200 cells: +2.2, +1.9 %) | +3.2 / +0.9 / +5.1 % |

Aligned budget (800 cells): friction now agrees (117.9 vs 119.6 Pa, CX slip within 8 %). What
remains (+2 %) sits entirely in the face terms. Hermes-3's T_t is 5 % higher at the same flow, and
the face momentum scales as Gamma_t sqrt(T_t).

Aligned, extrapolated (`compare.py`, DRBX with `sheath_momentum_flux=face`):
- **Attached.** n_up 2.149e19 ± 0.025 vs 2.094e19 ± 0.003, a +2.6 % difference. T_t 21.93 vs 21.06 eV, +4.1 %. Both are formally outside the (now small) GCIs.
- **6.885e24.** n_up +4.4 % (agree within 7e18). The aligned Hermes-3 sequence is not asymptotic: 4.53, 4.49, 4.71e19. The 1600-cell state was reached from both the 800-cell and the base 1600-cell states.
- **1.04e25.** Inconclusive. The aligned and ion-fixed Hermes-3 runs on 400 and 800 cells do not reach steady state (drift 1e-4 and 1.5e-5).

Conclusions:
1. **Neutral momentum: a Hermes-3 vs SD1D model difference.** By default Hermes-3 adds a parallel
   neutral viscosity, and it loses neutral momentum through the no-flow wall via the Lax flux.
   SD1D/DRBX have neither. Both slow the neutrals near the target and raise the CX friction.
   Hermes-3 can be configured to remove both (above), which brings the friction into agreement.
2. **Sheath momentum flux: discretisation, not physics.** SD1D's (and DRBX's) boundary momentum
   flux is the MinMod-reconstructed NV_i times V_out. Across the last half cell the flow jumps from
   Mach about 0.77 to 1, so this flux is about 14 % below Gamma_t c_s at 800 cells. The error
   vanishes only slowly with the grid:
   - base DRBX n_up at the attached flow is 1.71, 1.84, 1.94, 2.00 and 2.04e19 on 100 to 1600 cells;
   - with `sheath_momentum_flux=face` it is 2.101, 2.109, 2.104, 2.099 and 2.096e19.
   The Richardson value of the base runs (2.081e19) is close to the `face` limit, so this term
   mostly inflates the per-grid gap and the GCIs. It is not a DRBX bug: DRBX reproduces SD1D's
   default. The option `sheath_momentum_flux=face` is an opt-in benchmark variant (default `minmod`), not a model change.
3. **Not attributed (+2.6 % on n_up, attached).** It goes together with a 4 % higher Hermes-3 T_t
   at the same flow, so it lies in the target energy closure. One candidate is the kinetic-energy
   correction that Hermes-3 applies in the pressure equation when `fix_momentum_boundary_flux` is
   on; SD1D has no kinetic energy term. The 0.1 % T_up difference at the highest flow is also not
   attributed.

Reproduce:
- Hermes-3 aligned runs: run `run_hermes.sh` with
  `HERMES_OPTS="h+:fix_momentum_boundary_flux=true h:fix_momentum_boundary_flux=true neutral_parallel_diffusion:perpendicular_viscosity=false"`.
  Start each run from the base steady state on the same grid, via `FROM=<base run dir>`.
- DRBX aligned runs: `run_drbx.py <out> sheath_momentum_flux=face 1.8e19 2.0e19 2.2e19 2.4e19 4.0e19 4.5e19 5.0e19 5.5e19 6.0e19 7.0e19 8.0e19 9.0e19 1.0e20`
  (the variant argument is any comma-separated `DetachmentSolParameters` fields).
- Budget: `python momentum_budget.py <hermes run dir> <drbx json>`.

## Swap tests on the remaining gap (plan F1)

Point: attached, Gamma_t A_t = 2.231e24, Hermes-3 aligned as above, DRBX with
`sheath_momentum_flux=face`. Gap = Hermes-3 / DRBX - 1 at equal flow; DRBX n_up and T_t are
interpolated in log Gamma_t A_t on the same grid (`swap_gap.py`). The DRBX swaps were made with
temporary options (commit 836d4e51, removed afterwards because every one of them was null).

Upstream status and premises. Hermes-3 PR #688 (FCI bugfixes) and #687 (neutral bugfixes) are both
open. #688 only touches `sheath_boundary_fci`, `sound_speed`, viscosity and FCI tests; #687 touches
the `neutral_mixed` preconditioner and a guard-cell communicate in `braginskii_conduction`. Neither is on the
`sheath_boundary_simple` / `neutral_parallel_diffusion` path of this deck, so Hermes-3 was not rebuilt.
The comparison does not involve a gamma_e of 4.8 or 2/(1-delta) + phi/T_e: SD1D's total gamma = 6 is used by both
codes. Hermes-3 `gamma_e = gamma_i = 3` minus the explicit (2.5 T + m v^2/2) n v advected part adds
no net energy sink at v = c_s, and DRBX has `sheath_gamma = 6` with advection only. Hermes-3's
`sheath_boundary_simple` does not extrapolate the face velocity. It uses
`V_sheath = max(|V_cell|, c_s)` with `c_s` from the face temperature, the same clamp as DRBX.

DRBX-side swaps (gap in n_up / T_t, 100 / 200 / 400 / 800 / 1600 cells, then extrapolated 400-800-1600):

| DRBX variant | n_up gap per grid | T_t gap per grid | Extrapolated n_up / T_t |
|---|---|---|---|
| reference (aligned) | +2.2 / +1.8 / +1.8 / +2.0 / +2.5 % | +12.1 / +9.0 / +6.3 / +5.0 / +4.4 % | +2.63 / +4.11 % |
| c_s from the face temperature | same to 0.01 % | same to 0.1 % | +2.63 / +4.11 % |
| V floor from the extrapolated face velocity | identical (floor never active) | identical | +2.63 / +4.11 % |
| N, P, T guards exponential (Hermes-3 `limitFree`) | -0.3 / +0.3 / +0.9 / +1.5 / +2.3 % | +6.0 / +5.0 / +4.0 / +3.7 / +3.9 % | +2.8 / +4.6 % (p = 0.8, not asymptotic) |
| recycled-neutral energy equal to Hermes-3 (`recycled_temperature = 2.333`) | +3.0 / +2.6 / +2.5 / +2.8 / +3.3 % | +13.4 / +10.4 / +7.7 / +6.3 / +5.8 % | +3.44 / +5.45 % |
| `sheath_gamma = 5.5` (sensitivity only) | -1.3 / -1.6 / -1.6 / -1.4 / -0.9 % | +3.3 / +0.5 / -2.0 / -3.3 / -3.8 % | -0.81 / -4.10 % |

Hermes-3-side swaps, same point, restarted from the aligned steady state (gap against the DRBX reference):

| Hermes-3 variant | n_up gap per grid (100 / 200 / 400 / 800 / 1600) | T_t gap per grid | 1600-cell n_up, T_t |
|---|---|---|---|
| aligned | +2.2 / +1.8 / +1.8 / +2.0 / +2.5 % | +12.1 / +9.0 / +6.3 / +5.0 / +4.4 % | 2.149e19, 22.01 eV |
| `p_div_v = true` (P div V work form, as DRBX) | -1.7 / -0.8 / +0.3 / +1.3 / +2.5 % | +2.7 / +2.6 / +2.6 / +3.2 / +4.4 % | 2.150e19, 22.01 eV |
| `numerical_viscous_heating = true` | -2.3 / -1.1 / +0.0 / +1.1 / +2.8 % | +0.6 / +1.6 / +1.9 / +2.7 / +5.1 % | 2.155e19, 22.14 eV |
| `target_recycle_energy = 5.25` (energy per neutral equal to DRBX) | +2.9 / +2.6 / +2.6 / +2.8 / +3.2 % | +13.4 / +10.3 / +7.6 / +6.3 / +5.7 % | 2.165e19, 22.26 eV |

Grid convergence at this point. Hermes-3 n_up (1e19) 2.147, 2.148, 2.142, 2.141, 2.149 and T_t
(eV) 23.99, 23.14, 22.49, 22.14, 22.01 (extrapolated 21.93 +- 0.10). DRBX n_up 2.101, 2.109, 2.104,
2.099, 2.097 and T_t 21.41, 21.23, 21.15, 21.09, 21.07 (extrapolated 21.061 +- 0.014, p = 1.5).

Conclusions:
1. **No sheath closure explains the gap.** The face temperature in `c_s`, the face-velocity floor and the
   exponential vs linear guard extrapolation move the extrapolated T_t gap by at most 0.5 points. On the
   Hermes-3 side the pressure-work form and the numerical viscous heating change the coarse-grid values
   (T_t gap 12 % down to 1 to 3 %), but the 1600-cell state is the same to 0.6 %. So the large per-grid T_t
   gap of the base runs is an O(h) sheath-cell energy error in Hermes-3, and the continuum gap is
   about +4.4 to +5 % in T_t and +2.5 to +2.8 % in n_up in every variant.
2. **T_t is a direct measure of the energy delivered to the target.** At fixed flow, `sheath_gamma` 6 to 5.5
   raises T_t by 8.5 % (T_t ~ 1/gamma, not gamma^-2). A 4 % T_t gap is therefore a 0.25 T n v difference in
   the energy reaching the target, about 1.9 MW of the 50 MW input. The two codes agree on the
   ionisation radiation (4.81 MW vs 4.83 MW at the matched flow) and on T_up (0.1 %). The Hermes-3 / DRBX T ratio at 1600 cells is +0.4 % at 5.5 m from the target, +1.2 % at 1.7 m,
   +2.2 % at 0.57 m, +3.0 % at 0.22 m and +3.6 % in the last 0.07 m, so the difference accumulates over the
   whole recycling zone, not in the sheath cell.
3. **A deck mismatch was found.** Hermes-3 `target_recycle_energy = 3.5` is the energy per recycled
   neutral (T_n = 2.33 eV). SD1D, and so DRBX, put a pressure source N x 3.5 eV into the neutral
   pressure equation, which is 5.25 eV per neutral. The README table above called this matched. Aligning it in
   either direction raises the T_t gap from +4.1 % to +5.2 to +5.5 % and the n_up gap from +2.6 % to +3.4 %,
   so it does not explain the gap; it hides part of it.
4. **Not attributed:** the remaining +3.4 % n_up and +5.2 to +5.5 % T_t (same recycled energy). It lies in the plasma-neutral
   energy exchange (at the matched flow the CX energy transfer from the plasma is 5.71 MW in Hermes-3 and 6.08 MW in DRBX). A
   channel-by-channel ledger was not completed, because the Hermes-3 dump has no ionisation energy diagnostic and a
   reconstruction from the dump leaves about 1.5 MW unaccounted. The next step is to add per-process energy
   diagnostics for ionisation and CX to the Hermes-3 patch and compare the two ledgers.
