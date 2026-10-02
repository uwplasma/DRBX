# P08 B-evaluator comparison: `bfield_toroidal="spline"` vs `"compact_c3"`

## Purpose

Compare the two toroidal interpolations of the magnetic-field evaluator (`p_shared.bfield`) for the P-path perpendicular
operators **on identical sampled owners** at N32 / N48 / N64: per region and per "magnetic-knot class" the N-R errors of the
bracket, curvature and total terms and the N-O, O-R, N-R errors of the diffusion term, for a Dirichlet-phi variant
(`main_phi_dirichlet`) and a physical-normal-Neumann variant (`main_phi_neumann`), and how much the geometry coefficients of the
closure themselves change between the evaluators.  Each evaluator is internally consistent (the operator rows, the closure geometry
and every reference of one run use the same evaluator), so what is compared is error magnitude and order, not the two operators
against each other.

## Commands (from `DRBX/scripts`, env `drb`)

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m p08_bfield_eval.run 32 --mode spline     --out DIR [--per-cell 6] [--seed 0]
python -m p08_bfield_eval.run 32 --mode compact_c3 --out DIR [--per-cell 6] [--seed 0]
# ... likewise 48 and 64, then
python -m p08_bfield_eval.analyze DIR              # report.md + summary.json from every grid with both modes present
pytest tests/test_p08_bfield_eval.py -m "not slow" # fast synthetic tests;  -m slow adds the N32 smoke (1 owner per cell)
```

Options fixed in `run.py`: `curvature="autodiff"`, `face_quadrature="q2"`, `inner_support="fixed_radius"`, campaign `p06n`
(owner values of the P06N state from the oracle path, B-independent cell averages).  Parameters are the step-5.3 ones
(`G33_PARAMS`: rho_star 0.05, tau 1, diffusion 1e-2 for n, Te, Ti, vorticity).  One process, threads limited to 1.  N32 with
`--per-cell 6` (72 owners): about 105 s and 3.5 GiB (spline) / 4.4 GiB (C3) peak RSS on the development laptop; the rows
(~80 s) dominate.  Only small files are written (a few MB per run).

## Owner sample (identical for both evaluators)

A deterministic function of the grid topology (`env.t`) and the owner-centroid geometry (`sampling.py`); written to `results.json`.

* Region (disjoint, priority order): `axis_core` (owners with a raw cell of radius index 0), `physical_wall`, `last_two_layers`
  (`transverse_last_two_layers & ~physical_wall`), `transition`, `aggregate`, `ordinary` (`p06n_field_derived_global.core.regional_masks`).
* Knot class: cylindrical angle phi of the owner centroid (volume-weighted circular mean of the angles of the owner's raw-cell midpoints;
  Cartesian positions from the metric evaluator) against the MAKEGRID planes `phi[0] + k dphi` of the B evaluator (HSX: 90 planes per
  field period, `dphi` = 1 degree).  `plane`: distance to the nearest plane `< 0.2 dphi`; `mid`: within `0.2 dphi` of a midpoint
  between planes; otherwise unclassified and never sampled.
* `--per-cell K` owners per (region, knot) cell with `np.random.default_rng(seed)` without replacement (fewer when unavailable; the
  available / drawn / kept counts are recorded).  Owners whose rows cannot be built are dropped and recorded
  (`build_rows_safe`, the pattern of `p08_inner_support_eval`); `analyze` then works on the common owners.

## What is compared

Per owner, variant, field (n, Te, Ti, omega) and term (`poisson_bracket`, `curvature`, `perpendicular_diffusion`, `total`):

* N: the combined RHS `perpendicular_rhs(closure.plan, ...)` (as `step3_gates.run_g33`);
* R (all terms): `perpendicular_reference_rhs.reference_rhs`; its `perpendicular_diffusion` is the exact-face-flux reference **O** (O_q3);
* R (diffusion midpoint): `references.diffusion_midpoint_reference`, the generalization of
  `p08_step5_combined.references.psi_midpoint_reference` to every field, in the same `+D_f div(P grad f)` convention as O: raw midpoints,
  autodiff divergence of `J P`, exact gradient and Hessian, projected by raw volume over owner volume.

Comparisons: `N-R` for bracket / curvature / total; for the diffusion term `N-O`, `O-R` and `N-R` (against the midpoint R).  `total` is
`reference_rhs`'s total (which uses O for the diffusion part).  Metrics per comparison and per subset (each region, each knot class,
each region x knot class, pooled): owner-volume-weighted L2, max abs, relative L2 / max normalized as `step3_gates.term_metrics` does
(fallback scale = the largest reference L2 / max over the terms of the field, used for degenerate references), least-squares fit scale.
`results.json["diffusion_convention_check"]` compares O and R pooled: relative difference and `<O,R>/<R,R>` (about 1; a sign error would
give -1); they are two discretizations of the same continuum term and differ at the discretization-error level (about 10% at N32).

Coefficient table (`analyze`): on identical points (asserted equal across modes), the relative change C3 vs spline of the closure
arrays that carry B: `p05_{raw,face}_h`, `p06_{raw,face}_B`, `p06_{raw,face}_K`, `p07_{raw,face}_tensor`, `p07_raw_divergence`, and the
metric-only controls `p05_*_jacobian`, `p06_*_J` (expected 0); per array and subset the RMS and max over points of `|dA|/|A|` and
`rms(dA)/rms(A)`.  A raw cell belongs to its owner; a face to its lower owner when that is a sampled owner, else the upper one.

Flags (`report.md`, `summary.json`): C3 worse than spline by more than 1.5x in relative L2 on some grid, or an observed order lower
than spline's by more than 0.5 between consecutive grids.

## Outputs

`DIR/N{n}_{mode}/results.json` (schema, options, sample and counts, dropped owners, metrics, convention check, sanity, receipt with
`env.ref.provenance["bfield_toroidal"]`, git HEAD, timings, peak RSS) and `arrays.npz` (per-owner N / R / Rmid, owners, volumes, region and knot
codes, the closure coefficient arrays with their points and owner maps); `DIR/report.md` and `DIR/summary.json` from `analyze`.

## Limits

* A bounded sample (about 6 owners per cell): orders from three grids are indicative, small cells are noisy, and the grids sample different
  owners (same classes, not the same places).
* The knot class is a property of the owner centroid.  At N32 a raw cell spans about 11 MAKEGRID planes in phi, so its quadrature
  points are not at the classified phi; the class marks where the owner sits, not where each point is.
* `axis_core` owners are aggregates of dozens of raw cells; they dominate the raw-point counts of the coefficient table.
* The midpoint reference R and the exact-face-flux reference O differ at the discretization level, so `N-R` of the diffusion term is
  dominated by the reference mismatch; use `N-O` for the operator error and `O-R` as the reference consistency measure.
* The B evaluator is the only difference between the two modes; the geometry of the metric (`J`, points) is identical.
