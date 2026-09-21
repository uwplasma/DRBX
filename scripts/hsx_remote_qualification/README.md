# Clean remote HSX global qualification

This is a new research campaign for the matched-q3 centered perpendicular
bracket, not a continuation of the historical v1/v2 donor graph. Production
operators, evolved-state layouts, MMS fields, boundary model, geometry,
quadrature and the global >=1.8 two-interval acceptance target are unchanged.
The revised donor policy applies to all N32/N48/N64 reconstructions.

The remote handoff is [repository commands only](REMOTE_COMMANDS.md). Commands
run from the DRBX repository. The existing input workspace is addressed by
`--input-root`; its default layout and immutable content are specified in
`configuration.json` and `input_manifest.json`. Output is always a new campaign
directory. There are no environment, scheduler, transfer, or cluster-launch
instructions and no remote scaling study. The planned allocation uses 64 CPU
workers; one numerical-library thread per worker is enforced before imports.

Use the committed campaign source in a clean checkout. The source manifest
covers its transitive local Python imports, including the HSX drivers, rather
than unrelated research scripts. Local edits to a covered dependency require
a new manifest and a new campaign directory.

[Local implementation validation](VALIDATION.md) records HSX selection,
fresh-preparation, parallel-equivalence, and checkpoint-recovery evidence.

## What is rebuilt

`selection.py` uses a single convention for every octant boundary and for
near-equal distances. Radius-squared groups are anchored at their smallest
distance, with tolerance 128*float64 epsilon times the squared coordinate
scale. Canonical owner IDs order each group. The indexed query includes the
whole near-cutoff shell before truncating, and is checked against exhaustive
selection. Sectors snap to every integer octant within 128*epsilon in octant
coordinates and use counterclockwise half-open intervals. This is a declared
new policy, not a patch designed to reproduce historical IDs. Finite tolerances
do not promise bitwise portability for every conceivable geometry; retain
actual donor hashes and cross-platform diagnostic outputs.

Preparation rebuilds the physical radial-boundary derivative rows actually
needed by the final operator. It records their donors. Face and cell workers
build all remaining reconstructions under the same policy and record donor
hashes per entity. No old global derivative, bounded fit, cross, factor, or
local global-action cache is consumed. Rebuilding unrelated historical
experiments or unused interior derivative caches is unnecessary.

Only immutable geometry, original manufactured owner data, magnetic/metric
inputs and independent qualified reference values/uncertainties are reused.
Source and input manifests are checked in full, including the magnetic input.
A new source/configuration/input identity needs a new output directory.
Never relabel old chunks to satisfy v3 checks. Historical results remain
available for comparison, outside this campaign's dependency chain.

## Execution and scientific reporting

`verify-inputs` validates content and freezes the campaign identity.
`preflight` rebuilds preparation and exercises actual HSX faces/cells, including
axis, physical boundary, agglomerated regions and the reported N32 cutoff
location. It checks indexed/exhaustive donor agreement. Historical donor
equality is not a preflight gate. It is a correctness check, not a scaling test.

`run` computes deterministic non-overlapping face/cell chunks with persistent
spawn workers, atomic output and exclusive writer locks. Repeating the same
command resumes only validated v3 chunks. Workers are periodically recycled
without changing numerical output. It completes each resolution, assembles
the physical-volume-weighted errors and regional diagnostics, then merges all
three resolutions. `validate` rechecks full chunk coverage and assembly
metadata and regenerates the summary. The same source/input manifest must
remain in place throughout the campaign.

`summary.json` distinguishes computation completion, invariant diagnostics,
reference qualification and convergence. A scientifically failed convergence
study is still a completed computation. Every field must meet >=1.8 on both
intervals, with the existing reference uncertainty below 10% of its new
candidate error. Reused absolute reference uncertainties are redivided by the
new candidate errors for all three fields. Actual-action/constant identities
remain scientific diagnostics, not extra convergence vetoes. Independent
solution checks and production promotion remain later roadmap work.

P is paused locally. Only the remote task owns the remote processes and
monitoring. Q's separate optimization assignment is unaffected, and this
runner does not implement or authorize Q's enriched global campaign.
