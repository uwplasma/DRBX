# Local readiness evidence — 24 September 2026

This validates campaign implementation and a bounded boundary-adjacent policy;
it does not establish global convergence. N48/N64 complete preflight and all
full-domain computations remain remote prerequisites/work.

## Numerical provenance and boundary check

The interior kernel is extracted from the completed actual-HSX face-projection
and fourth-order experiments. Those numerical comparisons improved all eighteen
field/track/resolution regional RMS comparisons, using true owner observations
and shared physical face fluxes. It preserves the old selector's three ordinary
cubic candidates and geometry-only fallbacks; no selector tuning is introduced.

A new domain audit exposed 115 invalid legs in 13,800 tested N32 interior-face
legs. The user explicitly authorized a bounded test of boundary treatment.
The frozen geometry-only interval-halving policy is described in README.md and
validation/boundary_design.md. Actual six boundary/inward pairs at N32/N48/N64
were checked with q5 and full-pair q7 construction, q9/q11 exact flux references,
independent host/128-step tracing and actual numerical endpoint transfer.

All tested exits were resolved with at most six halvings. Candidate constants
were below 3.85e-12 and sampled cubic endpoint residuals below 2.70e-10 (scaled).
The shortest-leg 64/128 endpoint differences were below 8.89e-16. The old q11
owner reference replay differs by at most 2.68e-17. Each face has one canonical
flux and both complete owners are scored; exterior increments have their own
ledger. Maps were frozen across all six cases before errors. The exact executed
prototype source hash and numerical summaries are in boundary_summary.json.
Prototype numerical work used 388 CPU seconds (508 accounted with reserve),
peak RSS 1.438 GiB.

Accuracy remains mixed: 12/18 pair RMS comparisons improve against the archived
single-seed return. Largest regression is 4.196x on the N64 hotspot mixed field.
The numerical endpoint transfer now dominates several near-wall cases; the
candidate's q5-to-q7 construction change can be material. Independent q9/q11
reference changes are much smaller. These limitations are explicitly retained;
operationally defined boundary-adjacent evaluation is not an accuracy certificate.
The remote run must report the full-domain result, including a scientific fail,
without tuning the method or changing quadrature.

## Tests and executable runner

`python -m pytest -q tests/test_q_fci_projected_campaign.py
 tests/test_q_fci_return_campaign.py` passes 14 tests. Coverage includes:

- Fast field values/gradients against the independent frozen MMS implementation.
- Independent eta cubic transfer, on-plane behavior and periodic wrapping.
- Richardson polynomial cancellation under different accepted spans, and the
  wrong-denominator negative control.
- Domain-halving mechanics, unchanged ordinary legs, reuse of half legs and
  rejection of bad magnetic-field data rather than hiding it by shortening.
- CSR serialization and shared-face incidence.
- Missing/corrupt/incompatible checkpoint handling and exclusive writer lock.

A real N32 one-owner smoke run with two spawn workers completed all q5/q7 map,
trace, field, q9/q11 face reference, q5/q7 volume and strong-volume stages,
including sparse action replay and complete reference output. It took 25.3 s
wall time on the local machine. This is an execution check, not a scaling study.

The complete N32 preflight then passed with two workers: seven complete owners,
139 incident faces, axis/aggregate/transition/wall/inward/seam coverage,
q7 candidate on every preflight face, independent strong-volume controls,
full trace/map/field/reference receipts and deterministic assembly. Runtime was
143.2 s locally. The constant maximum is 7.20e-14; largest signed incidence
balance defect is 1.28e-21. All 115 initially invalid primary legs were resolved;
maximum accepted span reduction was H/64. The exact preflight output is saved
in validation/preflight_n32.json. Its bounded scientific-pass flag is false by
definition; computation_complete is true. Do not treat bounded error slopes as
global orders.

The local archive also contains a simulated interrupted smoke run: one face
payload/receipt was omitted from a copied campaign while keeping prepared maps
and trace checkpoints. Repeating the supported command reconstructs only the
missing result, preserving the map/trace payload hashes; validation/resume
receipt accompanies this document. The original run is unchanged. Corruption
of a receipted payload is separately tested as a hard failure, not a reason to
silently recompute or delete provenance.

## Optimization and limits

Compared with the old catalogue-return global runner, this runner removes the
global footprint catalogue/return fit, shares projection geometry with primary
exact integration, avoids unused scalar Hessians and quadratic diagnostic fits,
uses compiled batched RK4, reuses half legs during interval reduction, and
checkpoints traces/maps before field work. Every computational stage including
preflight is scheduled through the node-local process pool. Scalar fields reuse
one numerical map. Full/half maps determine fourth without duplicate storage.

Source/configuration hashes and all eight immutable inputs are verified before
work. The pinned geometry archive isolates this research runner from unrelated
uncommitted package edits. Only new campaign files and the focused test are part
of this change. The shared physical-wall MMS contract remains explicitly limited
and the existing static second-order/reference gates remain unchanged.
