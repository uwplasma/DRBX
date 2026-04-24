# Implicit Solver Profile Audit

This package is the numerical-methods companion to the Hermes offender
register. It isolates the shared sparse implicit backend before the full
recycling physics stack is involved, so runtime changes can be checked against
a controlled finite-difference Jacobian problem and a small nonlinear Newton
solve.

The audit writes:

- [summary JSON](data/implicit_solver_profile_audit_artifacts/data/implicit_solver_profile_audit.json)
- [summary figure](data/implicit_solver_profile_audit_artifacts/images/implicit_solver_profile_audit.png)

The left panel compares colored sparse finite-difference Jacobian construction
with and without the precomputed CSC/color extraction plan. The right panel
uses the sparse Newton step diagnostics now attached to solver step info:
residual time, Jacobian assembly time, linear-solve time, line-search time,
and fallback use.

This is not a replacement for the full `recycling_dthe_one_step` Hermes
comparison. Its role is narrower and deliberate: it proves that the optimized
Jacobian assembly path is algebraically identical to the original path and
provides a small, reproducible figure for the numerical-methods section before
the full physics runtime plots are generated.

Run the package locally with:

```bash
PYTHONPATH=src python examples/engineering/implicit_solver_profile_audit_demo.py
```
