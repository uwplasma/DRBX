from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .grid import FCISlabGrid
from .parallel import parallel_derivative_centered_3d


class FCISlabParams(eqx.Module):
    """Minimal 3D slab model used to validate FCI operators and sheath budgets."""

    nu_par: float = 0.0
    sheath_nu: float = 0.0
    open_field_line: bool | None = None


class FCISlabState(eqx.Module):
    f: jnp.ndarray


class FCISlabModel(eqx.Module):
    params: FCISlabParams
    grid: FCISlabGrid

    def rhs(self, t: float, y: FCISlabState) -> FCISlabState:
        _ = t
        open_field_line = (
            self.grid.open_field_line
            if self.params.open_field_line is None
            else self.params.open_field_line
        )
        dpar = parallel_derivative_centered_3d(
            y.f,
            map_fwd=self.grid.map_fwd,
            map_bwd=self.grid.map_bwd,
            open_field_line=open_field_line,
        )
        df = -dpar

        if self.params.nu_par != 0.0:
            dpar2 = parallel_derivative_centered_3d(
                dpar,
                map_fwd=self.grid.map_fwd,
                map_bwd=self.grid.map_bwd,
                open_field_line=open_field_line,
            )
            df = df + self.params.nu_par * dpar2

        if self.params.sheath_nu != 0.0:
            df = df - self.params.sheath_nu * self.grid.sheath_mask * y.f

        return FCISlabState(f=df)

    def mass(self, y: FCISlabState) -> jnp.ndarray:
        return jnp.mean(y.f)

    def energy(self, y: FCISlabState) -> jnp.ndarray:
        return 0.5 * jnp.mean(y.f**2)

    def mass_rate(self, y: FCISlabState, dy: FCISlabState) -> jnp.ndarray:
        _ = y
        return jnp.mean(dy.f)

    def energy_rate(self, y: FCISlabState, dy: FCISlabState) -> jnp.ndarray:
        return jnp.mean(y.f * dy.f)

    def sheath_mass_loss(self, y: FCISlabState) -> jnp.ndarray:
        if self.params.sheath_nu == 0.0:
            return jnp.asarray(0.0)
        return self.params.sheath_nu * jnp.mean(self.grid.sheath_mask * y.f)

    def sheath_energy_loss(self, y: FCISlabState) -> jnp.ndarray:
        if self.params.sheath_nu == 0.0:
            return jnp.asarray(0.0)
        return self.params.sheath_nu * jnp.mean(self.grid.sheath_mask * y.f**2)
