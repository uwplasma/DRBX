# ESSOS Imported FCI Validation

!!! note "Plan authority"
    This page is an imported-geometry validation report. The active execution
    plan is [Research-Grade Execution Plan](research_grade_execution_plan.md).
    If this page conflicts with that plan, follow the execution plan and update
    this page afterward.

This page documents the first downstream use of externally traced
Landreman-Paul QA field lines inside `jax_drb` FCI operators. ESSOS supplies
the coil-field evaluation and adaptive trajectories. `jax_drb` converts those
trajectories into fixed-shape plane-to-plane maps, builds a lightweight
VMEC-shaped metric for the imported logical grid, and then evaluates JAX-native
sheath/recycling and neutral reaction-diffusion closures on those maps.

The current authoritative promotion sequence is in
[Research-Grade Execution Plan](research_grade_execution_plan.md#current-authoritative-open-lane-implementation-plan).
In that sequence, `coil`, `vmec`, and `hybrid` have different meanings:
direct-coil open-field maps must pass their own endpoint and connection-length
gates before a movie can be promoted; VMEC maps are closed-field controls; and
hybrid maps are the current bridge that combines smooth VMEC map coordinates
with coil-derived endpoint masks. The convenience workflow
`examples/geometry-3D/essos-field-lines/direct_coil_open_sol_demo.py` records
that gate order in one script. The live FCI stage now also feeds
`direct_coil_source_profile_gate`, a machine-readable JSON check on the exact
target-label, heat-load, neutral-source, radial-profile, and source-balance
artifacts used for open-SOL promotion. When live FCI arrays exist, that same
gate also writes a standalone source/profile PNG showing the consumed
directional target labels, target heat-load response, neutral ionisation
source, target particle-loss flux when regenerated with the current artifact
schema, and normalized radial profiles. Its optional `RUN_LIVE_MEDIA_GATE`
stage writes the direct-coil GIF/PNG/NPZ diagnostic media only after the user
explicitly enables it; the output remains diagnostic unless the same workflow
summary also shows green geometry, endpoint/source/profile, refinement, and
visual-QA evidence. The workflow summary now writes
`promotion_rejection_reasons`, `promotion_blocking_stages`, and `next_actions`
even for the default dry run. A clean-clone contract therefore reports
`no_live_promotion_gates_ran` rather than silently producing
`promotion_ready = false` with no explanation.

A local live direct-coil FCI/source-profile check on the Landreman-Paul QA
assets in `/Users/rogerio/local/ESSOS/examples/input_files` passes the FCI and
source/profile stages: target fraction `0.90375`, magnetic-field modulation
`1.4167`, particle-recycling relative error `2.2e-15`, neutral-particle
relative error `3.7e-18`, current and neutral-momentum balance errors below
reported precision, and target-label reconstruction exactly matching the
consumed endpoint masks. The workflow summary still reports
`promotion_ready = false` because endpoint-label refinement, adjacent-step
refinement, stationarity, and media QA were not run in that lightweight pass.
Running the live pure-coil endpoint-label refinement gate on the same assets
keeps direct-coil media unpromoted: coarse-to-middle all-label and endpoint
agreement are both `0.444`, middle-to-fine all-label agreement is `0.616`, and
middle-to-fine endpoint agreement is `0.573`, below the `0.90` all-label and
`0.80` endpoint thresholds. This is the current reason to treat pure
direct-coil movies as diagnostic and to use the hybrid VMEC/coil lane as the
promotion path for open-SOL media.
The current endpoint-label gate also records the endpoint-union population in
each nested comparison and, for the direct-coil open-SOL workflow, requires a
nonzero endpoint population. This prevents a misleading pass in which all
levels agree only because nearly every cell is classified as non-target.
Generic closed-field diagnostics can still set that threshold to zero because
closed maps are validated by periodic/return-map metrics instead of target
contact.
A live rerun with this stricter gate confirms that the blocker is true
directional endpoint instability rather than missing target contact:
minimum endpoint-union population is `0.898`, while the minimum all-label and
endpoint-agreement fractions remain `0.444`. Rebuilding the diagnostic from
the same live label levels classifies both nested-pair failures as
`directional_endpoint_mismatch`, so the next direct-coil geometry fix should
inspect forward/backward target classification and bidirectional cells at
collocated seeds rather than only increasing endpoint population.
The component-level report shows this is not a single sign error: forward and
backward endpoint bits both have false positives and false negatives. The
dominant direction-component error is
`balanced_forward_backward_components`, which points to endpoint projection and
seed-collocation stability rather than a one-sided target-label bug.
The endpoint-label gate now also classifies where the mismatch lives. It builds
endpoint-presence and directional transition shells on the coarse and
restricted labels, then reports whether failed cells are concentrated near
target-boundary transitions or spread through the bulk map. The direct-coil
workflow summary surfaces `dominant_endpoint_boundary_localization`,
`target_boundary_projection_suspected`, and a
`projection_recommended_next_action`. This keeps the next live diagnostic
actionable: a boundary-localized failure points to wall-hit projection and
forward/backward target classification, while a bulk mismatch points to field
line tracing, map source, or coordinate restriction.
A separate odd-ratio live rerun using `(3, 5, 9) -> (7, 15, 27)` grids keeps
coarse periodic seed angles collocated on the refined grid. That diagnostic
also fails promotion: all-label agreement is `0.474`, endpoint agreement is
`0.458`, endpoint-union population is `0.970`, and the dominant component
error is `forward_component`. This rules out even-ratio non-collocation as the
only cause of the pure-coil blocker.
A rerun of the same odd-ratio diagnostic after adding transition-shell
localization classifies the blocker as
`direction_boundary_localized`, with `target_boundary_projection_suspected =
true`, all label mismatches lying on directional transition shells, and zero
mismatch fraction outside those shells. The next direct-coil code work should
therefore focus on target-boundary projection and forward/backward wall-hit
classification before changing the bulk field-line map. A follow-up
boundary-excluded report gives all-label and endpoint-label agreement `1.0`
outside the transition shell, but the boundary-excluded valid fraction is only
`0.022` on the coarse odd-ratio probe. This is useful localization evidence,
not a promotion pass: the target boundary still occupies too much of the
coarse direct-coil diagnostic grid to advertise a pure-coil open-SOL movie.
The direct-coil workflow therefore applies a stricter boundary-excluded
coverage requirement of `0.20`; the same live run fails that requirement while
passing boundary-excluded agreement. The next promoted pure-coil gate must use
a target projection or grid choice with enough non-boundary interior support,
not merely perfect agreement on a tiny interior subset.
A larger live endpoint-label comparison using `(7, 15, 27) -> (11, 25, 45)`
grids is now exposed as the optional
`RUN_LIVE_BOUNDARY_RESOLVED_ENDPOINT_LABEL_REFINEMENT_GATE` stage in
`direct_coil_open_sol_demo.py`. It gives enough interior support for this
criterion, with boundary-excluded valid fraction `0.248` and
boundary-excluded all-label and endpoint-label agreement `1.0`. The full
endpoint-label gate still fails, with endpoint agreement `0.763`, because the
mismatch remains concentrated on the target transition shell. This is positive
evidence for the bulk direct-coil FCI map and negative evidence for the current
target-boundary projection. Pure-coil open-SOL media therefore remains
diagnostic until the target projection, adjacent-step refinement, and consumed
source/profile gates pass on the same map.
The endpoint-label report now also includes a projection-neighborhood
diagnostic for coordinate-restricted comparisons. For each mismatched coarse
cell, it checks whether the coarse endpoint label appears in a one-cell local
fine-grid neighborhood around the nearest projected sample. Large
`projection_neighborhood_mismatch_support_fraction` values mean the mismatch is
consistent with a discontinuous target-boundary projection rather than a bulk
field-line-map failure. Small values mean the coarse label is absent even in
the local fine neighborhood, so the next investigation should move to field
line tracing, wall-hit retention, or the target classifier.

The published FCI validation figures and arrays are restored by
`python scripts/fetch_example_artifacts.py --skip-baselines`. The regeneration
script follows the same top-level-parameter style as the SIMSOPT examples:
edit `MAP_SOURCES_TO_RUN`, `DRY_RUN`, `WRITE_DRY_RUN_ARTIFACTS`, grid size, and
optional external input paths at the top of
`examples/geometry-3D/essos-field-lines/imported_fci_campaign.py`, then run the
file. Regenerating the import from the external coil geometry is a developer
workflow and requires the geometry source checkout:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_fci_campaign.py
```

By default the script performs a safe dry run for `coil`. Set
`MAP_SOURCES_TO_RUN = ("coil", "vmec", "hybrid")` to regenerate the published
`coil`, `vmec`, and `hybrid` artifact directories in one run. Set
`MAP_SOURCES_TO_RUN = ("hybrid",)`, `OUTPUT_ROOT = Path("tmp/hybrid")`, and
`CASE_LABEL = "custom"` for a custom single-map artifact root.

The multi-grid connection-length refinement machinery also has a clean-clone
example that does not require the external geometry checkout:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_connection_length_refinement_demo.py
```

That script runs a manufactured non-axisymmetric nested-grid gate and writes a
JSON/NPZ/PNG package under
`docs/data/essos_imported_connection_length_refinement_artifacts/`. Live
`coil`, `vmec`, or `hybrid` connection-length arrays can be generated by
setting `LIVE_IMPORT = True` in the same script or by calling
`create_live_essos_imported_connection_length_refinement_package(...)`. The
example accepts `MAP_SOURCES_TO_RUN = ("coil", "vmec", "hybrid")` so a local
promotion pass can regenerate all imported map-source reports in one command.
With `CONNECTION_QUANTITY = "auto"`, the script uses
`adjacent_step_length` for pure coil FCI-map refinement and
`parallel_step_per_toroidal_radian` for VMEC and hybrid adjacent-map
refinement, avoiding the common mistake of comparing raw adjacent-plane length
across different toroidal grid spacings. Live runs compare non-collocated grids
by interpolating the fine level at the coarse radial, toroidal, and poloidal
coordinates. For endpoint or wall-hit studies, set `CONNECTION_QUANTITY`
explicitly to `target_exit_length` and treat the result as a target-distance
diagnostic rather than an adjacent-map FCI convergence proof.
The live template now uses three nested grids and sets
`REQUIRE_OBSERVED_ORDER = True`, so the command fails if the generated report
does not contain an actual observed-order convergence measurement. Two-level
live checks remain useful for fast debugging, but they are advisory and should
not be used as publication-grade refinement evidence. Live controls use
`LIVE_CONVERGENCE_THRESHOLD` and `LIVE_LINF_THRESHOLD`, separately from the
stricter manufactured thresholds, because the live coordinate-interpolation
gate measures imported-map consistency rather than a manufactured analytic
solution.
Each run also writes a compact sweep summary JSON next to the report files.
For live runs with `MAP_SOURCES_TO_RUN = ("coil", "vmec", "hybrid")`, that
summary records the source, refinement quantity, finest errors, observed order,
finite-overlap threshold, `promotion_ready`, and `evidence_role` for all three
map sources in one file. This is the preferred artifact for deciding which
geometry source is ready for a turbulence/movie claim without manually opening
each report.
For definitions of one-sided, target-to-target, and effective parallel
connection length, and for the exact code paths used by each geometry source,
see [Connection Length](connection_length.md).

The hybrid open-SOL promotion path has a single workflow ledger:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/hybrid_open_sol_demo.py
```

The default run is self-contained and writes a dry-run contract under
`artifacts/essos_hybrid_open_sol/`. It is intentionally not a promoted physics
result. Set the live flags at the top of the script only after ESSOS coil and
VMEC inputs are available. The live stages run in the required order:
FCI/source-profile, target-label/source accounting, hybrid
parallel-step-per-radian refinement, reduced-transient stationarity, grid/time
movie refinement, and optional media generation. The summary JSON must report
`promotion_ready=true` before any hybrid open-SOL figure or movie can be used
as README or publication evidence. The live source/profile stage writes the
same standalone target/source/profile PNG as the direct-coil workflow, so the
hybrid bridge can be compared against the pure-coil lane with identical
diagnostics. Current regenerated FCI arrays also include
`particle_loss_toroidal`, so target particle flux can be inspected as a map
instead of only through the radial particle-loss profile.

A local live hybrid pass on the same Landreman-Paul QA assets now clears the
first three live gates. The FCI/source stage passes with target fraction
`0.90375`, `|B|` modulation `1.4167`, particle-recycling relative error
`2.2e-15`, neutral-particle relative error `2.5e-18`, zero current-balance
and neutral-momentum errors, exact consumed endpoint-mask reconstruction, and
a positive target particle-loss map. The regenerated FCI and source/profile
figures were visually checked. The hybrid parallel-step-per-radian refinement
gate also passes on three nested grids with finite overlap `1.0`, observed
order `1.20`, finest normalized RMS error `5.90e-2`, finest normalized
\(L_\infty\) error `1.18e-1`, and monotone RMS/\(L_\infty\) reduction. The
workflow still refuses promotion until the promotion stationarity, grid/time
movie-refinement, media, and visual-QA stages are run on the same map.

Direct-coil open-field promotion now has a separate categorical endpoint-label
refinement gate. This is necessary because `target_exit_length` is a
wall/endpoint distance and is discontinuous when a neighboring cell changes
from non-target to target. The endpoint-label gate compares the directional
labels consumed by the sheath/recycling kernels, using the convention
`0` for no target, `1` for a forward endpoint, `2` for a backward endpoint, and
`3` for a bidirectional endpoint. Live nested grids are compared by
nearest-neighbor restriction at the coarse logical coordinates; self-contained
manufactured tests use block-majority restriction. The JSON report records
all-label agreement, endpoint-union agreement, endpoint-union population,
forward/backward/bidirectional confusion matrices, valid overlap, endpoint
false positives, endpoint false negatives, and directional mismatches. The
report also records endpoint-presence and directional transition-shell
localization for each failed pair, so target-boundary projection errors are
separated from bulk map/restriction errors before any movie is promoted. The
scalar `adjacent_step_length`
refinement remains the smooth FCI-map quality gate, while scalar
`target_exit_length` refinement is retained as a target-distance diagnostic
and not as the sole promotion blocker for a direct-coil movie.

The first live direct-coil endpoint-label rerun on June 21, 2026 kept the
direct-coil open-field lane diagnostic. The FCI endpoint/source gate passed:
the interior connection-resolution roughness had `p95 = 2.98e-2`, endpoint
roughness was correctly localized to endpoint-touching faces, target-exit
lengths were finite only on endpoint cells, and source accounting closed. The
new endpoint-label nested-refinement gate did not pass. The
`(3, 4, 6) -> (6, 8, 12)` pair had all-label and endpoint-only agreement
`0.444`; the `(6, 8, 12) -> (12, 16, 24)` pair improved to all-label
agreement `0.616` and endpoint-only agreement `0.573`, but remained below the
promotion thresholds. The smooth `adjacent_step_length` gate still had small
finest error but weak observed order (`0.104`). These results keep pure
direct-coil open-field media out of README/paper promotion and move the next
promoted visual path to VMEC closed-field controls and the hybrid
VMEC-coordinate/coil-endpoint open-SOL lane.

The self-contained gate now records more than the finest-grid error. It stores
successive RMS and \(L_\infty\) error-reduction factors and explicitly requires
monotonic error reduction when three or more nested levels are available. The
checked-in example also requires observed-order availability, preventing a
two-level run from being promoted accidentally. The report now carries the same
classification that downstream docs and movie scripts should use:
`promotion_ready=true` is the only state that supports publication or README
movie promotion; `advisory_only=true` records useful debugging evidence without
supporting a physics claim; and `evidence_role` distinguishes observed-order,
monotonicity, threshold, and finite-data rejection modes. The current
manufactured artifact is `promotion_ready` and passes with finest normalized RMS
`6.71e-3`, finest normalized \(L_\infty\) `1.14e-2`, observed order `1.78`,
minimum RMS reduction factor `3.45`, and minimum \(L_\infty\) reduction factor
`3.31`.

The first live June 15, 2026 raw-length checks are intentionally retained as
negative promotion evidence. Raw `coil` and `hybrid` runs on
`(3, 4, 6) -> (6, 8, 12) -> (12, 16, 24)` returned normalized RMS `0.356`,
\(L_\infty\) `2.52`, observed order `0.137`, and non-monotonic
\(L_\infty\) error. Those values show that the mixed raw length is not a
grid-invariant refinement quantity. After the importer was split into
`raw_connection_length`, `adjacent_step_length`, and `target_exit_length`, live
VMEC and hybrid controls using `parallel_step_per_toroidal_radian` pass the
live three-level observed-order control with normalized RMS `5.90e-2`,
\(L_\infty\) `1.18e-1`, observed order `1.20`, and monotonic RMS and
\(L_\infty\) reduction. They do not satisfy the stricter manufactured
thresholds and therefore remain connection-length controls, not full
turbulence/movie promotion evidence by themselves. A June 18, 2026 pure-coil
`adjacent_step_length` rerun improved the finest errors to normalized RMS
`1.05e-2` and \(L_\infty\) `1.98e-2`, but the observed order was only `0.101`;
pure-coil adjacent-step tracing is therefore classified as
`negative_observed_order_control` rather than promotion evidence. The local
June 18 live rerun used `minimum_finite_pair_fraction=0.25` for this open-field
adjacent-step quantity; both pairwise comparisons had finite overlap fraction
`0.5`, so the failure is now specifically the weak observed order, not missing
endpoint-cell adjacent lengths. This keeps the hybrid VMEC-map/coil-mask lane
as the current open-field bridge while pure-coil map refinement remains active
work.

![Manufactured nested-grid connection-length refinement](data/essos_imported_connection_length_refinement_artifacts/images/essos_imported_connection_length_refinement.png)

Set `WRITE_DRY_RUN_ARTIFACTS = True` to write a self-contained JSON contract
under the resolved artifact root; that contract records the live artifact
paths, grid/refinement settings, required report fields, required NPZ array
keys, and the connection-length/refinement/consumed-map diagnostic schema
without reading the coil JSON or VMEC wout file. Set `COIL_JSON_PATH`,
`VMEC_WOUT_PATH`, or `ESSOS_ROOT` when the external checkout is not located at
the default path used by the importer. `MAP_SOURCES_TO_RUN` accepts three
imported-map semantics:

- `coil` traces the external Biot-Savart coil field to adjacent toroidal
  planes and keeps the resulting open-field endpoint masks.
- `vmec` evaluates a VMEC-coordinate field-line map from
  \(d\theta/d\phi=B^\theta/B^\phi\), preserving closed flux surfaces and
  disabling target endpoint masks.
- `hybrid` uses the VMEC-coordinate map locations but keeps the coil-derived
  endpoint masks, connection-length proxy, and \(|B|\) modulation. This is the
  intended bridge for open-field SOL closure tests while the VMEC map supplies
  smooth non-axisymmetric interpolation coordinates.

The script chooses source-specific defaults, so `MAP_SOURCES_TO_RUN = ("vmec",)`
writes `docs/data/essos_imported_fci_vmec_artifacts/` and
`MAP_SOURCES_TO_RUN = ("hybrid",)` writes
`docs/data/essos_imported_fci_hybrid_artifacts/` unless `OUTPUT_ROOT` or
`CASE_LABEL` is set for a custom single-source run.

The committed report JSON can also be audited without rerunning the external
geometry import:

```bash
PYTHONPATH=src .venv/bin/python \
  examples/geometry-3D/essos-field-lines/imported_artifact_schema_audit.py
```

The audit compares the checked-in JSON reports against the fields produced by
the current validation code. It is useful before promoting README figures or
paper plots because it flags stale reports whose PNG/movie assets may still
exist but whose JSON no longer contains the current connection-length,
endpoint, target-label, map-quality, refinement, or consumed-map diagnostics.
As of the June 21, 2026 regeneration, the committed `coil`, `vmec`, and
`hybrid` imported-FCI JSON reports match the current schema.
Quick regeneration keeps `REQUIRE_CONNECTION_RESOLUTION = False`, so the
single-grid roughness diagnostic is recorded but remains advisory. Promotion
runs for publication figures, README movies, or release evidence should set
`REQUIRE_CONNECTION_RESOLUTION = True`; this makes the
`connection_length_resolution_diagnostics["passed"]` flag a hard acceptance
gate in the generated report and stores
`connection_length_resolution_required=true` in the artifact metadata. A
strict failure means the map needs more field-line resolution, a better
interpolation grid, or a successful multi-grid refinement campaign before the
physics result should be advertised.

## Geometry Import

The imported grid is a scaled VMEC Landreman-Paul QA flux-surface shell
centered on the magnetic axis reported by the external Biot-Savart field
object. The VMEC Fourier boundary is read from
`wout_LandremanPaul2021_QA_reactorScale_lowres.nc`, then rescaled and
translated onto the ESSOS coil-field coordinate system so that the rendered
surface has the QA non-axisymmetric cross-section while the traced field lines
remain in the coordinate system used by the coil JSON. The stellarator-symmetric
surface evaluation uses

\[
R(s,\theta,\phi)=\sum_{mn} R_{mn}(s)\cos(m\theta-n\phi),\qquad
Z(s,\theta,\phi)=\sum_{mn} Z_{mn}(s)\sin(m\theta-n\phi),
\]

Forward and backward coil trajectories are traced from every seed. For each
seed, the adapter interpolates the external trajectory to the adjacent toroidal
planes \(\phi\pm\Delta\phi\), projects the endpoint onto the nearest structured
VMEC-shaped target plane, and marks a boundary if the endpoint leaves the
resolved shell or lands on a radial edge. For VMEC-coordinate maps the adapter
instead integrates

\[
\frac{d\theta}{d\phi} = \frac{B^\theta(s,\theta,\phi)}{B^\phi(s,\theta,\phi)}
\]

with a fixed-step RK4 rule over one toroidal-plane spacing and stores the
resulting poloidal interpolation coordinate at fixed \(s\). Boundary map
indices are stored as finite placeholders and the boundary mask carries the
physics meaning; this keeps the JAX interpolation kernels shape-stable and
safe under `jit`, `vmap`, `jvp`, and future implicit residual promotion.

The metric is computed from the Cartesian embedding
\(\mathbf{x}(\rho,\phi,\theta)\). The covariant basis vectors are finite
differences of the scaled VMEC surface coordinates, \(g_{ij} =
\partial_i\mathbf{x}\cdot\partial_j\mathbf{x}\), \(J=\sqrt{\det g_{ij}}\), and
the contravariant metric is the matrix inverse of \(g_{ij}\). This keeps the
closure accounting on the same non-axisymmetric surface used for visualization.

## Physics Gates

The sheath/recycling gate applies a normalized Bohm target flux to every
forward or backward field-line endpoint,

\[
\Gamma_i = N_i\sqrt{(T_e+T_i)/m_i},
\]

reconstructs the electron particle flux from zero-current balance, and checks
that recycled particle and neutral-energy sources exactly close their global
accounting identities. The neutral gate then evaluates FCI parallel diffusion,
perpendicular metric diffusion, ionisation, recombination, and charge exchange
on the same imported maps. The report records endpoint fractions, magnetic
field modulation, connection-length statistics, target heat-load contrast,
particle balance residuals, current residuals, and neutral momentum balance.
The imported-map diagnostics now separately report connection-length finite and
nonnegative fractions, radial connection-length means, grid/refinement metadata,
single-grid connection-length resolution diagnostics, map-coordinate
displacement proxies, a map-quality summary, a consumed-map check requiring the
sheath endpoint count to match the forward-plus-backward FCI boundary masks, a
direction-aware target label diagnostic, and an endpoint-length diagnostic. The
target labels use
`0` for closed/non-target cells, `1` for forward exits, `2` for backward exits,
and `3` for bidirectional exits; the report verifies that these labels exactly
reconstruct the endpoint counts consumed by the sheath and recycling closures.
For open-field `coil` and `hybrid` maps the endpoint-length gate
requires finite, nonnegative `target_exit_length` values on a nonzero subset of
endpoint cells and finite, nonnegative `adjacent_step_length` values where the
adjacent map exists. It also requires finite, nonnegative forward and backward
target-exit lengths on the corresponding imported boundary masks. This prevents
an aggregate wall-hit array from hiding a missing direction in a bidirectional
open-field map. The compact NPZ and PNG artifacts now include `target_exit_toroidal` and
`adjacent_step_toroidal`, plus `target_label_toroidal`; the summary plot shows
directional target labels and the target-exit map for open-field artifacts, and
falls back to endpoint counts and the connection-length proxy for closed VMEC
maps. The resolution diagnostics record normalized neighbor jumps, per-axis
95th-percentile jumps, per-axis underresolved-face fractions, dominant rough
and underresolved directions, endpoint-touch versus interior roughness, an
underresolved-face fraction, and an advisory pass flag. For open-field maps the
diagnostic is endpoint-aware: if large jumps are localized on physical
endpoint-mask faces while the interior map is resolved, the report sets
`endpoint_aware_passed=true` and `interior_resolution_passed=true` while still
recording the large `endpoint_touch_normalized_jump_p95`. This avoids treating
a physical target-exit discontinuity as an interior FCI interpolation failure.
The `map_quality_diagnostics` block then converts these low-level numbers into
a short recommendation. In the current committed artifacts, the `coil` and
`hybrid` reports are toroidally dominated and endpoint-touch dominated, but
their interior FCI map resolution is green and their endpoint/source accounting
closes. They still require endpoint-mask refinement, target/source plots, and
nested refinement evidence before movie promotion; the closed `vmec` report is
an interior-only closed-map control and should not be used for open-target
sheath/recycling claims. These diagnostics catch grid-scale connection-length
roughness before a live imported run is promoted, but they are not a
replacement for a multi-grid refinement campaign. Set
`require_connection_resolution=True` in the campaign API, or
`REQUIRE_CONNECTION_RESOLUTION = True` in the example script, when the
single-grid diagnostic should reject the imported map instead of only
annotating it.
For that promotion step,
`build_essos_imported_connection_length_refinement_diagnostics` compares
nested connection-length grids either after conservative block restriction or,
when live coordinate payloads are supplied, by interpolating the fine grid at
the coarse logical coordinates. The report records normalized RMS,
95th-percentile, and \(L_\infty\) errors for every coarse/fine pair plus the
observed order when three or more levels are supplied. It also records
successive RMS and \(L_\infty\) error-reduction factors and requires monotonic
error reduction for three-or-more-level refinement claims. Promotion runs can
set `require_observed_order=True`, which makes two-level reports fail even when
the finest-grid error is small. The report-level `promotion_ready` flag is true
only when finite pair data, finest-grid error thresholds, monotonic reduction,
and an explicitly required observed-order check all pass. Reports with small
errors but no required observed-order check are retained as `advisory_only`;
reports with poor observed order are retained as negative controls through
`evidence_role`. Live reports also record `minimum_finite_pair_fraction`.
Manufactured and closed-map controls default to full finite pair coverage,
whereas pure-coil `adjacent_step_length` and `target_exit_length` live probes
use a finite-overlap threshold because open-field endpoint cells are validated
by the endpoint and target-label diagnostics rather than by the adjacent-plane
map comparison. For live imported geometry, the available
quantities are `raw_connection_length`, `adjacent_step_length`,
`target_exit_length`, and `parallel_step_per_toroidal_radian`; only the
adjacent-step quantities are appropriate for FCI-map convergence. The
self-contained
`imported_connection_length_refinement_demo.py` campaign exercises that exact
report and plotting path with manufactured nested grids, so CI can protect the
refinement logic even without the external field-line runtime. Imported-field
turbulence movies should not be used as publication evidence until the same
multi-grid connection-length gate passes on the live `coil`, `vmec`, or
`hybrid` map source used by the movie.
For `vmec` maps the consumed-map count must be zero; for `coil` and `hybrid`
maps it must be nonzero and exactly consumed by the sheath/recycling masks.

## Current Artifacts

![ESSOS imported FCI coil validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_artifacts__images__essos_imported_fci_campaign.png)

![ESSOS imported FCI VMEC-coordinate validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_vmec_artifacts__images__essos_imported_fci_vmec_campaign.png)

![ESSOS imported FCI hybrid validation](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__essos_imported_fci_hybrid_artifacts__images__essos_imported_fci_hybrid_campaign.png)

The first figure shows the default `coil` artifact: imported VMEC-shaped QA
cross-section, endpoint map structure, connection-length proxy, sheath
heat-load response, neutral ionisation response, and radial diagnostics. The
`vmec` artifact is the closed-field surface-preservation control; it has zero
target endpoint fraction and zero target heat load while still exercising
metric diffusion and neutral source accounting on the VMEC-coordinate map. The
`hybrid` artifact uses the VMEC-coordinate map positions but keeps the
coil-derived endpoint masks, connection-length proxy, and \(|B|\), making it
the preferred open-field SOL bridge. All three routes pass and feed the same
JAX-native closure kernels used by the synthetic non-axisymmetric validation
suite.

The next imported-map gate is documented in
[ESSOS imported PyTree/JVP validation](essos_imported_pytree_validation.md).
It drives the fixed-layout drift-reduced Braginskii PyTree RHS, `jax.jvp`, and
`jax.vmap` checks from the same external field-line map construction.

## Artifact Files

- `docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.json`
- `docs/data/essos_imported_fci_artifacts/data/essos_imported_fci_campaign.npz`
- `docs/data/essos_imported_fci_artifacts/images/essos_imported_fci_campaign.png`
- `docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.json`
- `docs/data/essos_imported_fci_vmec_artifacts/data/essos_imported_fci_vmec_campaign.npz`
- `docs/data/essos_imported_fci_vmec_artifacts/images/essos_imported_fci_vmec_campaign.png`
- `docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json`
- `docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.npz`
- `docs/data/essos_imported_fci_hybrid_artifacts/images/essos_imported_fci_hybrid_campaign.png`
- `docs/data/essos_imported_connection_length_refinement_artifacts/data/essos_imported_connection_length_refinement.json`
- `docs/data/essos_imported_connection_length_refinement_artifacts/data/essos_imported_connection_length_refinement_summary.json`
- `docs/data/essos_imported_connection_length_refinement_artifacts/data/essos_imported_connection_length_refinement.npz`
- `docs/data/essos_imported_connection_length_refinement_artifacts/images/essos_imported_connection_length_refinement.png`
