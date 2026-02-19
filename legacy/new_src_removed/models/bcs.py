from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC1D


class LineBCs(eqx.Module):
    """Optional user-defined 1D boundary conditions for the field-line coordinate `l`.

    These BCs are enforced weakly (SAT/relaxation) as extra RHS terms localized to the
    boundary nodes. This is intended for:

    - benchmarking alternative end conditions (Dirichlet/Neumann) against MPSE/sheath models,
    - early nonlinear preparation work where boundary behavior matters.

    By default, `jaxdrb` uses:
      - periodic derivatives for closed field lines,
      - open derivatives for open field lines,
      - MPSE (Loizu-style) sheath entrance BCs via `DRBParams.sheath_bc_*` knobs.

    If both MPSE and user BCs are enabled, the two mechanisms will both contribute and
    can conflict. In most studies you should use one or the other.
    """

    enabled: bool = False

    n: BC1D = eqx.field(static=True, default=BC1D.periodic())
    omega: BC1D = eqx.field(static=True, default=BC1D.periodic())
    vpar_e: BC1D = eqx.field(static=True, default=BC1D.periodic())
    vpar_i: BC1D = eqx.field(static=True, default=BC1D.periodic())
    Te: BC1D = eqx.field(static=True, default=BC1D.periodic())
    Ti: BC1D = eqx.field(static=True, default=BC1D.periodic())
    psi: BC1D = eqx.field(static=True, default=BC1D.periodic())

    @classmethod
    def disabled(cls) -> "LineBCs":
        return cls(enabled=False)

    @classmethod
    def all_dirichlet(cls, *, value: float = 0.0, nu: float = 1.0) -> "LineBCs":
        bc = BC1D.dirichlet(left=value, right=value, nu=nu)
        return cls(
            enabled=True,
            n=bc,
            omega=bc,
            vpar_e=bc,
            vpar_i=bc,
            Te=bc,
            Ti=bc,
            psi=bc,
        )

    @classmethod
    def all_neumann(cls, *, grad: float = 0.0, nu: float = 1.0) -> "LineBCs":
        bc = BC1D.neumann(left=grad, right=grad, nu=nu)
        return cls(
            enabled=True,
            n=bc,
            omega=bc,
            vpar_e=bc,
            vpar_i=bc,
            Te=bc,
            Ti=bc,
            psi=bc,
        )


def bc_relaxation_1d(f: jnp.ndarray, *, bc: BC1D, dl: float) -> jnp.ndarray:
    """Return an RHS term that relaxes boundary values toward the BC targets."""

    if bc.kind == 0 or bc.nu == 0.0:
        return jnp.zeros_like(f)

    f = jnp.asarray(f)
    n = int(f.size)
    mask = bc.mask(n, f.dtype)

    target = f
    if bc.kind == 1:
        target = target.at[0].set(bc.left_value)
        target = target.at[-1].set(bc.right_value)
    elif bc.kind == 2:
        # Use implied boundary values from a 1st-order one-sided derivative relation.
        target = target.at[0].set(f[1] - dl * bc.left_grad)
        target = target.at[-1].set(f[-2] + dl * bc.right_grad)
    else:
        raise ValueError(f"Unknown BC kind: {bc.kind}")

    return -float(bc.nu) * mask * (f - target)
