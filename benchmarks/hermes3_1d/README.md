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
| Recycling 0.99 at 3.5 eV | `target_recycle_multiplier = 0.99`, `target_recycle_energy = 3.5` |
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
pressure is the same. The difference is therefore in the upstream-to-target total pressure loss
(momentum removed by CX, ionisation and recombination friction, and neutral momentum transport).
Neither the recombination rate nor T_e != T_i can account for it, since together they bound less
than 0.5 %. It is not attributed to a named model difference here. The remaining candidates are
SD1D choices that this deck does not reproduce: SD1D's neutral-pressure relaxation where
Nn < 1e-5, the neutral momentum diffusion, and Hermes-3's split ion/electron momentum with
`electron_force_balance`. The next step is a DRBX toggle that removes the SD1D neutral momentum
diffusion. The 0.1 % T_up difference at the highest flow is small but resolved, and is also not attributed.
