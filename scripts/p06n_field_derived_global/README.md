# P06N: batched global static campaign for the accepted P06 curvature operator
# under physical-normal Neumann rows

## What this qualifies

`p06_structured_global` accepted a curvature/E×B operator (the q1 midpoint
material/remainder volume term plus the q3 characteristic face correction)
under an all-Dirichlet boundary contract. P06N qualifies the *same* algebra
(`_continuum_terms`, `_face_geometry`, `_principal_matrix`, `_absolute_action`,
`_curvature_bc_characteristic_wall_states`, all imported unchanged from
`p06_structured_global.numerics`) with the transported states (density, Te,
Ti) supplied through **physical-normal Neumann rows** instead: the wall datum
is `g_N = a.grad_x f` (`a` the physical outward normal), reconstructed through
the same lifted-row / Neumann-eliminated machinery
`p05n_field_derived_global`/`p07n_field_derived_global` use, with the
**recovered-trace wall contract** (`boundary_trace = interior`) for the
characteristic wall solve.

phi enters only through its gradient in q1's remainder term
(`K.grad(phi)`); omega never affects the action at all (no omega column in
`curvature_principal_matrix`, and `_principal_matrix`/q3 never reads
`state[...,3]` either) -- see `operator.py`'s module docstring and
`check_omega_independence`.

## Frozen catalogue

`p06n_catalogue.json` is frozen (never edit it) and holds seven cases, each a
5-slot `(n, Te, Ti, omega, phi)` assignment of physical fields, with an
unsuffixed field name meaning a Neumann state and a `:dirichlet` suffix
meaning a Dirichlet state:

| case | n | Te | Ti | omega | phi |
|---|---|---|---|---|---|
| `main_phi_neumann` | rich_f | rich_a | rich_c | field_b1_minus1:D | rich_a_minus1 |
| `main_phi_dirichlet` | rich_f | rich_a | rich_c | field_b1_minus1:D | zero_trace_generator:D |
| `heldout_phi_neumann` | **heldout_rich_h** | rich_c | rich_f | field_b1_minus1:D | rich_f_minus1 |
| `heldout_phi_dirichlet` | **heldout_rich_h** | rich_c | rich_f | field_b1_minus1:D | zero_trace_generator:D |
| `dirichlet_rich` | rich_f:D | rich_a:D | rich_c:D | field_b1_minus1:D | zero_trace_generator:D |
| `control_constant_neumann` | constant | constant | constant | constant:D | constant |
| `control_constant_dirichlet` | constant:D | constant:D | constant:D | constant:D | constant:D |

**`heldout_rich_h` is frozen held-out**: no preparation step (tests, preflight,
equivalence, timing) evaluates or prints an N−R comparison for either
`heldout_*` case. They run through the same code path, so they are structurally
checked. Their accuracy is first computed by the campaign run, and they are
gated there like every other nonconstant case, following the P05N/P07N
held-out convention. `dirichlet_rich` re-qualifies the accepted P06 upwinded (U)
operator with the *corrected* face census (see "Face census" below);
accepted run `58880303` counted the periodic seam twice, invisible there
because those corrections were ≲1e-12.

`fields.py` reuses the P05N field catalogue verbatim (by file-path import,
never copy-modified) and adds two P06N-specific rich fields (`rich_c`,
`heldout_rich_h`) plus shifted variants (`*_minus1`, shift-invariant in
gradient/normal-datum). `rows.py` adds the one row primitive P05N never
needed, `batched_face_common_value` (q3's face-common *value*, mirroring
`core.batched_face_common_gradient` line for line). `operator.py` is the
frozen per-owner oracle (`owner_q1`, `owner_q3`, `owner_all_faces`, `Case`,
`make_role`, `check_omega_independence`) that `verify-equivalence` checks the
batched path in `core.py` against.

## Face census: accepted-with-duplicate vs. deduplicated

The accepted P06 face-index space allocates `n+1` theta/eta face slots (0..n)
per ring; slot `n` is the *same physical face* as slot 0, but the original
accepted campaign counted its q3 correction on both slots (`_face_incidence`
maps both to the identical `(lo, hi)` owner pair). `operator.owner_all_faces`
reproduces that double-count verbatim (`dedupe=False`, the default) so it can
serve as the gate-(a) replay oracle against the saved accepted artifact;
`dedupe=True` drops the duplicate slot (`p06_structured_global.numerics`'s own,
already-fixed `_periodic_duplicate_face` predicate), which is what this
package's own campaign reduction (`core.face_chunk(..., dedupe=True)`,
`campaign.py`'s `faces` stage) and `dirichlet_rich`'s re-qualification use.
`preflight.py`'s gate (c) checks the algebraic relationship between the two
censuses directly at a theta-seam and an eta-seam owner.

Both censuses use the **"add both sides independently"** scatter convention
accepted P06 uses (`_merge`): an internal aggregation seam face (where
`lo == hi == owner`, e.g. an agglomerated axis-core owner) contributes *both*
its lower and upper corrections to that one owner -- this is not the
antisymmetric `+jump`/`-jump` convention P05N/P07N use for their own (genuinely
antisymmetric) bracket jump.

## Reduction outputs

Per grid, per variant (7 cases × {as-declared N, matched-Dirichlet-diagnostic
D} = 14), per equation (density, Te, Ti, and omega "for completeness" --
`configuration.json`'s `equations`; only density/Te/Ti are ever gated), the
campaign reports, at owner resolution:

- `N_centered` -- q1's material+remainder alone (no face correction).
- `N_U` -- `N_centered`'s material term plus the q3 face correction divided
  by the owner's accepted-P06 *evolution volume* (`sum(cell_quadrature_weight
  * J / max(B, eps))` over that owner's raw members -- **not** the owner's
  stored physical volume `t.vol`; this is the same divisor
  `p06_structured_global.numerics`'s own upwind formula uses), plus the
  unchanged remainder term.
- `D_centered`, `D_U` -- identical formulas for the matched-Dirichlet
  diagnostic (every originally-Neumann slot switched to that same field's
  Dirichlet role; already-Dirichlet slots unchanged).
- `R` -- the exact analytic reference, from an independent second call to the
  *same* `_continuum_terms` on the case's exact (value, gradient) (the
  pointwise operator is `O == R` by code identity at the q1 stage: `N−R`
  therefore measures reconstruction and closure, not a residual MMS source).

Two norms are reported: the **owner-weighted global L2** matching
`p06_structured_global.numerics._statistics`/`_compact_statistics` (owner
physical volume `t.vol` as weight -- the accepted campaign's own headline
norm) and the **stored-owner-volume L2** (P05N convention; identical weight
choice here, reported separately for cross-package comparability).

Regions reported: physical wall ring, last two (radial) layers, transition
(profile-count boundary), aggregate (agglomerated, non-transition,
non-wall), ordinary, interior (`core.regional_masks`, mirroring
`p05n_field_derived_global.campaign.regional_masks`).

## Gate

`N−R` global-L2 convergence order ≥ 1.8 on both intervals (32→48, 48→64), for
every case in `gated_cases` (`main_phi_neumann`, `main_phi_dirichlet`,
`heldout_phi_neumann`, `heldout_phi_dirichlet`, `dirichlet_rich`) × equation in
`gated_equations` (density, Te, Ti) × candidate (centered, U), on the total
term. Material/remainder and `N−D` are reported, never gated (the acceptance
decision from `N−D` and the O≡R structural identity is the user's, made after
results, exactly as P05N's README documents). Constant controls (both
`control_constant_*` cases and their D diagnostics): `|action| ≤ 1e-8` on both
candidates. The gate is returned unchanged, uninterpreted, by
`reduce_global`/`summary.json`.

## Preflight (`preflight.py`)

Three campaign preflight gates, all through the batched kernels:

- **(a) replay:** the four accepted all-Dirichlet P06 states
  (`p06_structured_global.numerics.FIELD_NAMES`) run through
  `core.raw_chunk`/`face_chunk` with accepted-state `FieldTables`, in
  accepted-census mode (the periodic seam duplicate included). They are checked
  against the accepted campaign's saved `candidate_centered`/`candidate_U`/
  `evolution_volume` (vendored by `preflight_fixtures/extract.py` into
  `preflight_fixtures/N{n}.replay.npz`). This also checks that the run
  environment reproduces the accepted numbers.
  - Tolerance atol=1e-9, rtol=1e-12. It absorbs the measured cross-platform gap
    in the accepted finite-difference curvature: 1.2e-10, 1.6e-10 and 2.7e-10
    absolute at N32/N48/N64 on values up to about 3.
  - The local N32 headroom is 0.12.
- **(b) structural:** on the P06N bounded selection (vendored byte-identical):
  - row linearity;
  - constant controls;
  - the physical-wall characteristic correction, which is zero by the
    recovered-trace contract;
  - a zero jump on `radial_n_minus_1`/`transverse_last_two_layers` for every
    variant;
  - ω-independence;
  - the exact-input (`O==R`) defect;
  - the Neumann row condition number (≤ 1e8).
- **(c) seam:** at one θ-seam and one η-seam owner, the accepted-census
  correction minus the deduplicated-census correction equals exactly the
  duplicate face's own contribution, for every variant.

`campaign.py verify-equivalence` is the one-off code-equivalence check. It
compares the batched kernels with the per-owner oracle
(`operator.owner_q1`/`owner_q3`) for every catalogue variant (candidate, D and R,
q1 and q3) on a covering owner subset, atol=rtol=1e-12, and writes
`equivalence.json`. It runs once per code change, not on every campaign.

## Timing / chunking

Three stages: `observations` (owner-averaged live fields), `raw` (q1, chunked
over raw-cell ids, `n**3` per grid), `faces` (q3, chunked over the accepted
P06 face-index space *directly* -- `core.all_face_ids(n)`, **not**
`p07_combined_global.topology`'s deduplicated owner-boundary census, since
P06N must be able to enumerate internal aggregation-seam faces too; see "Face
census" above). Chunk sizes (`configuration.json`): `raw_chunk=512`,
`face_chunk=256`, `observation_chunk=2048` -- tuned for the ~14-variant ×
5-field row/algebra cost per target; adjust if profiling shows a different
sweet spot before running the full campaign (not done as part of this
preparation task).

## Do not run the full campaign here

Per the task that built this package, the global campaign (`run`) is not
executed as part of this preparation; only `verify-inputs`, `preflight`
(N32/N48/N64), a one-off `verify-equivalence` (N32), and a small
`--max-units` timing sample at N64 are. `heldout_rich_h`'s N−R is never
evaluated or printed, in any of these.

## Files to commit

`fields.py`, `rows.py`, `operator.py`, `p06n_catalogue.json` (pre-existing,
frozen where noted) plus everything this task added: `core.py`, `campaign.py`,
`configuration.json`, `input_manifest.json` (byte-identical to
`p07n_field_derived_global/input_manifest.json`), this `README.md`,
`preflight.py`, `preflight_fixtures/` (`extract.py`, `fixtures_manifest.json`,
`N{32,48,64}.selection.json`, `N{32,48,64}.replay.npz`), and
`tests/test_p06n_field_derived_campaign.py`.
