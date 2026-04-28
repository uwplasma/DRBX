# Implicit Solver Profile Audit

This package is the numerical-methods companion to the Hermes offender
register. It isolates the shared sparse implicit backend before the full
recycling physics stack is involved, so runtime changes can be checked against
a controlled finite-difference Jacobian problem and a small nonlinear Newton
solve.

The audit writes:

- [summary JSON](data/implicit_solver_profile_audit_artifacts/data/implicit_solver_profile_audit.json)
- [summary figure](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__implicit_solver_profile_audit_artifacts__images__implicit_solver_profile_audit.png)

The left panel compares colored sparse finite-difference Jacobian construction,
the precomputed CSC/color extraction plan, thread-parallel finite differences,
and the JAX-linearized sparse-JVP path in both serial and batched color-push
modes. The right panel shows the finite-difference sparse Newton phase timing;
the JSON also records the matching sparse-JVP Newton solve under
`newton_sparse_jvp`, including residual norm, solution error, phase timings,
fallback use, and `jacobian_mode`.

The JAX timing bars are warmed once before sampling, so they measure steady
derivative execution rather than first-trace overhead. On this small controlled
residual the main result is not a universal speed claim; it is that batched
JVP construction matches the serial JVP construction exactly, agrees with the
finite-difference reference to roundoff-level finite-difference error, and
removes the finite-difference perturbation from the derivative path.

This is not a replacement for the full `recycling_dthe_one_step` Hermes
comparison. Its role is narrower and deliberate: it proves that the optimized
Jacobian assembly path is algebraically identical to the original path, that
the JAX sparse-JVP batched path matches the serial JVP path, and that both can
be checked against a finite-difference reference before the full physics
runtime plots are generated.

The current JSON therefore contains two solver-backend gates:

- `newton`: sparse Newton with finite-difference Jacobian assembly
- `newton_sparse_jvp`: sparse Newton with grouped JVP Jacobian assembly

Run the package locally with:

```bash
PYTHONPATH=src python examples/engineering/implicit_solver_profile_audit_demo.py
```
