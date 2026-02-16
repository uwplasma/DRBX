import jax.numpy as jnp

from jaxdrb.bc import bc2d_from_strings
from jaxdrb.nonlinear.fd import inv_laplacian_fd_fft, laplacian


def _grid_from_bc(
    nx: int, ny: int, Lx: float, Ly: float, bc
) -> tuple[jnp.ndarray, jnp.ndarray, float, float]:
    dx = Lx / nx if bc.kind_x == 0 else Lx / (nx - 1)
    dy = Ly / ny if bc.kind_y == 0 else Ly / (ny - 1)
    x = jnp.linspace(0.0, Lx, nx, endpoint=(bc.kind_x != 0))
    y = jnp.linspace(0.0, Ly, ny, endpoint=(bc.kind_y != 0))
    return x, y, float(dx), float(dy)


def test_inv_laplacian_fd_fft_periodic_recovers_solution() -> None:
    nx, ny = 32, 36
    Lx, Ly = 2.0 * jnp.pi, 2.0 * jnp.pi
    bc = bc2d_from_strings(bc_x="periodic", bc_y="periodic")
    x, y, dx, dy = _grid_from_bc(nx, ny, Lx, Ly, bc)
    xx = x[:, None]
    yy = y[None, :]
    u = jnp.sin(2.0 * jnp.pi * xx / Lx) * jnp.sin(2.0 * jnp.pi * yy / Ly)
    rhs = laplacian(u, dx, dy, bc)
    u_rec = inv_laplacian_fd_fft(rhs, dx=dx, dy=dy, bc=bc)
    err = jnp.max(jnp.abs(u_rec - u))
    assert float(err) < 2e-4


def test_inv_laplacian_fd_fft_dirichlet_recovers_solution() -> None:
    nx, ny = 33, 35
    Lx, Ly = 1.0, 1.2
    bc = bc2d_from_strings(bc_x="dirichlet", bc_y="dirichlet", value_x=0.0, value_y=0.0)
    x, y, dx, dy = _grid_from_bc(nx, ny, Lx, Ly, bc)
    xx = x[:, None]
    yy = y[None, :]
    u = jnp.sin(jnp.pi * xx / Lx) * jnp.sin(jnp.pi * yy / Ly)
    rhs = laplacian(u, dx, dy, bc)
    u_rec = inv_laplacian_fd_fft(rhs, dx=dx, dy=dy, bc=bc)
    err = jnp.max(jnp.abs(u_rec - u))
    assert float(err) < 2e-4


def test_inv_laplacian_fd_fft_neumann_recovers_solution_up_to_constant() -> None:
    nx, ny = 34, 32
    Lx, Ly = 1.1, 0.9
    bc = bc2d_from_strings(bc_x="neumann", bc_y="neumann", grad_x=0.0, grad_y=0.0)
    x, y, dx, dy = _grid_from_bc(nx, ny, Lx, Ly, bc)
    xx = x[:, None]
    yy = y[None, :]
    u = jnp.cos(jnp.pi * xx / Lx) * jnp.cos(jnp.pi * yy / Ly)
    rhs = laplacian(u, dx, dy, bc)
    u_rec = inv_laplacian_fd_fft(rhs, dx=dx, dy=dy, bc=bc)
    u_rec = u_rec - jnp.mean(u_rec)
    u = u - jnp.mean(u)
    err = jnp.max(jnp.abs(u_rec - u))
    assert float(err) < 2e-4
