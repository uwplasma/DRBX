# JCP Readiness Audit

This page is the explicit ship/publication decision record for `jax_drb`.
It answers a narrower question than the broad development plan:

- is the code ready to ship as a research-grade public codebase?
- is it ready for a Journal of Computational Physics paper with strong numerical and physics claims?

The short answer is:

- `jax_drb` is close to a strong public research release;
- `jax_drb` is **not yet** ready for the strongest possible JCP claim boundary if that claim is "general, parity-complete replacement across the full intended DRB reference-workflow matrix".

## External bar

The criteria below are aligned with the verification/validation expectations visible in the relevant reference-code literature:

- rigorous code-verification evidence, especially Method of Manufactured Solutions and observed-order checks;
- benchmark validation against published reference cases with clear diagnostics and compare surfaces;
- explicit distinction between verification, validation, and reduced/scaffolded evidence;
- reproducible artifacts, scripts, and input decks for the published figures;
- runtime/scaling evidence on promoted numerical paths, not only on synthetic kernels.

Useful external anchors:

- [MMS verification paper](https://arxiv.org/abs/1602.06747)
- [Multi-component plasma CPC paper](https://arxiv.org/abs/2303.12131)
- [TCV-X21 edge turbulence validation benchmark paper](https://arxiv.org/abs/2109.01618)
- [TCV-X21 SOLPS validation paper](https://arxiv.org/abs/2310.17390)
- [Recent TCV-X21 turbulence validation paper](https://arxiv.org/abs/2506.12180)

## What is already strong

- Manufactured-solution and convergence evidence is in-tree on promoted native paths.
- The code now has broad local Hermes-backed parity tooling for `one_rhs`, `one_step`, bounded transient windows, controller diagnostics, reactions/collisions, impurity/radiation, and selected 3D reduced rungs.
- Public artifacts are reproducible and sanitized: JSON summaries, NPZ arrays, publication plots, movies, and GIFs are committed for the promoted demo/validation surfaces.
- The release/runtime surface is significantly stronger than before: verbose progress, run logs, artifact manifests, restart bundles, and public release-surface regression checks are all in place.
- The 3D infrastructure is no longer tokamak-only in architecture: tokamak, traced-field-line, and VMEC/stellarator adapters now share the same manifest/observable/parity/runtime artifact model.
- Reduced native JAX profiling is now explicit on the promoted non-tokamak 3D kernels: compile, first-execute, warm-execute, and Perfetto trace artifacts are committed for the traced-field-line and VMEC reduced native surfaces.

## What is still missing for a strong JCP claim

These are the real remaining blockers, not wishlist items:

1. Full neutral transient closure is not finished.
   The public native `neutral_mixed_short_window` lane now clears a bounded full short-window centerline gate on the matrix-free path, but the broader global short-window mass/pressure surface is still not promoted.

2. The direct tokamak recycling transient family is not fully widened.
   `tokamak_recycling_dthe_one_step` is exact on its committed first-output surface, the compact D/T `nout=2` window is bounded operationally, and the richer drift-enabled `nout=2` window is now bounded too, but longer-window and neon-enabled direct tokamak recycling surfaces are still not promoted.

3. Controller-oriented temperature/detachment physics is only reduced-promoted, not full-production.
   `controller_feedback_campaign` and the reduced `temperature_feedback_campaign` are useful validation packages, and `detachment_controller_campaign` now promotes a bounded reduced Hermes-backed detachment-controller lane, but the bounded local Hermes Tt-control run still does not finish inside the current ten-minute policy and there is not yet a broader production temperature/detachment-control workflow.

4. The 3D native claim boundary is still reduced, not full-production.
   The repo has native reduced tokamak and non-tokamak 3D rungs plus benchmark/scaffold packages, but it does not yet have a broad end-to-end native 3D production workflow comparable to the strongest 2D promoted lanes.

5. Coverage is now split into a real release gate and a broader hardening target.
   `scripts/run_closeout_coverage.py` now enforces `95%` on the bounded controller/runtime/profile/audit closeout slice, and that gate is passing. Repo-wide monolithic coverage is still broader and slower than the local release gate, so it remains a hardening target rather than a ship blocker.

## Shipping decision

If the question is "can this be shipped publicly as a serious research codebase with explicit claim boundaries?", the answer is **yes**, provided the release notes say clearly:

- which lanes are `native_exact`;
- which lanes are `native_operational`;
- which 3D surfaces are reduced/scaffolded/native-reduced rather than full production;
- which controller and neutral families remain open.

If the question is "is the code ready for the strongest JCP-style paper claiming broad standalone parity/replacement across the whole target DRB/Hermes workflow matrix?", the answer is **not yet**.

## Required pre-paper closeout

Before starting the main JCP manuscript, the following should still be closed:

1. Promote a broader production temperature/detachment-control lane beyond the reduced detachment-controller gate.
2. Promote the broader global `neutral_mixed_short_window` surface beyond the current centerline-only gate.
3. Promote at least one longer-window or neon-enabled direct tokamak recycling surface beyond the current bounded `nout=2` gates.
4. Decide the paper claim boundary explicitly:
   either "selected promoted native lanes plus general 3D infrastructure"
   or
   "broad standalone parity-complete DRB solver".

Until those four items are done, the code is best described as:

- a strong research-grade public codebase with unusually good parity/validation tooling;
- not yet the final broad-claim JCP submission target.
