# Initial-value growth rate estimation

Initial-value methods are robust and easy to automate, but can suffer from:

- transient behavior (multiple modes present),
- large dynamic range (overflow/underflow),
- sensitivity to initial conditions.

`jaxdrb` uses a **renormalized** formulation that behaves like a continuous-time power method.

## Renormalized system

We want to integrate:

$$
\frac{dv}{dt} = A v
$$

where `A = J` is the linearized operator and `v` is the perturbation vector.

Write:

$$
v(t) = \exp(a(t))\,u(t),
$$

where `u` is a normalized direction vector and `a` is the accumulated log-norm.

Choose `a'(t)` to remove the growth from `u`:

$$
a'(t) = \gamma(t),\qquad
u'(t) = A u - \gamma(t)\,u.
$$

Here $\gamma(t)$ is taken as the (real part of) a Rayleigh quotient:

$$
\gamma(t) = \Re\left[\frac{u^\dagger A u}{u^\dagger u}\right].
$$

We additionally track an accumulated “phase” slope via the imaginary part:

$$
\omega(t) = \Im\left[\frac{u^\dagger A u}{u^\dagger u}\right].
$$

In the implementation, because we store complex states as real vectors for Diffrax, we compute
these parts explicitly from the real/imag blocks.

## Implementation details

- Time integration uses Diffrax (`diffrax.diffeqsolve`).
- Adaptive step size control uses a PID controller (`rtol`, `atol`).
- Growth rate is estimated by least squares fitting of `a(t)` over a late-time window.

See [`src/jaxdrb/linear/growthrate.py`](https://github.com/uwplasma/jax_drb/blob/main/src/jaxdrb/linear/growthrate.py).
