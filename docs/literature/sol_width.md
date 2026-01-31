# SOL width via $\max(\gamma/k_y)$ (gradient removal)

This page documents the **SOL profile length** estimate used in the scrape-off-layer literature
based on the "gradient removal" saturation paradigm, and how it is implemented in `jaxdrb`.

## The idea in the literature

In Halpern et al. (Phys. Plasmas 20, 052306 (2013)), the turbulent particle/pressure transport is
estimated from linear growth rates using a gradient-removal rule. The key proxy is:

$$
\Gamma \propto \left(\frac{\gamma}{k_y}\right)_{\max},
$$

where the maximization is taken over $k_y$ at fixed parameters and geometry.

The profile scale length $L_p$ is then determined from a balance of perpendicular transport and
parallel losses. In the reduced model discussion around Eq. (20) of Halpern (2013), this leads to
the fixed-point condition:

$$
\left(\frac{\gamma}{k_y}\right)_{\max}(L_p) = \frac{L_p}{q},
\qquad\text{equivalently}\qquad
L_p = q\,\left(\frac{\gamma}{k_y}\right)_{\max}(L_p).
$$

The important practical point is that **$\gamma$ depends on $L_p$**, since the equilibrium gradients
depend on $L_p$. So $L_p$ must be found self-consistently.

## How `jaxdrb` implements it

`jaxdrb` provides a convenience routine:

- `jaxdrb.analysis.lp.solve_lp_fixed_point`

which performs the fixed-point iteration:

1. choose an initial guess $L_p^{(0)}$,
2. map $L_p$ to the model's gradient drive (see below),
3. scan $\gamma(k_y)$ using matrix-free Arnoldi and compute $(\gamma/k_y)_{\max}$,
4. update $L_p^{(n+1)} \leftarrow q\,(\gamma/k_y)_{\max}$ (with optional relaxation),
5. repeat until convergence.

### Mapping $L_p$ to `jaxdrb` parameters

In a reduced fluid model, $L_p$ enters through equilibrium gradients.

In v1 of `jaxdrb`, the simplest mapping is to interpret the density-gradient drive as:

$$
\omega_n \sim \frac{R}{L_p}.
$$

Since `jaxdrb` uses dimensionless parameters, we implement the mapping as:

$$
\omega_n \equiv \frac{\omega_{n,\mathrm{scale}}}{L_p}.
$$

This is what `solve_lp_fixed_point(..., omega_n_scale=...)` controls.

> If you want to treat *pressure* gradients more faithfully, you can include a temperature-gradient
> drive `omega_Te` and relate $L_p$ to both $L_n$ and $L_{T_e}$, but the v1 examples keep
> `omega_Te=0` for clarity.

## A worked example

Run:

```bash
python examples/3_advanced/03_halpern2013_gradient_removal_lp.py
```

This script:

- computes $\gamma(k_y)$ and $\gamma/k_y$ for two gradient strengths,
- demonstrates the fixed-point solve for $L_p$ on a tokamak-like geometry.

To access the core routine directly:

```python
from jaxdrb.analysis.lp import solve_lp_fixed_point
from jaxdrb.models.params import DRBParams

res = solve_lp_fixed_point(
    params=DRBParams(...),
    geom=geom,
    q=3.0,
    ky=ky_grid,
    Lp0=20.0,
    omega_n_scale=1.0,
)
print(res.Lp, res.ky_star, res.gamma_over_ky_star)
```

## Practical guidance

- Use a **positive** $k_y$ grid (the maximization uses $\gamma/k_y$).
- Ensure your scan includes the region where $\gamma>0$; otherwise the fixed-point update will fail.
- The result can be sensitive to:
  - the ky range,
  - numerical diffusion (stabilization),
  - geometry coefficients,
  - boundary conditions (not modeled in v1).

For SOL studies that are boundary-condition dominated (line-tied, sheath, etc.), consider v1 results
as a *methodology demo* rather than a predictive model.
