# Characteristic-wall boundary design

This note records the agreed boundary-condition architecture for the parallel
five-field system. The generic two-incoming-mode solver and validation
no-flow selector are implemented; the later physical closures below remain a
development plan.

## Characteristic coordinates and `R_in`

At a physical endpoint, orient the material flux Jacobian with the outward
normal, $A_n = \partial F_n/\partial q$. Let its right eigenvectors be the
columns of $R$, with $A_n R = R\Lambda$. `R_in` means the submatrix formed
by the columns whose oriented characteristic speeds are incoming (negative
under the code's outward-normal convention):

Here $R_{\mathrm{in}}=[r_k:\lambda_k\text{ is incoming}]$ maps incoming
characteristic amplitudes `a` to primitive-state changes,
$q_{\mathrm{wall}}=q_{\mathrm{interior}}+R_{\mathrm{in}}a$.

Outgoing and stationary characteristic content is retained from the interior.
The active incoming count is `N_in`; it can change near a sonic/rank-changing
state and must be diagnosed rather than silently assumed. The implemented
solver currently has an intentional exact-two-mode gate: it accepts only
`N_in = 2` and requires numerical rank two for both `R_in` and the composed
boundary Jacobian. A sonic limit or a contact/zero-speed mode can change the
incoming/stationary count, so those cases must fail the gate explicitly until
their characteristic treatment is designed.

## General nonlinear boundary-law interface

The reusable abstraction should be a boundary residual with exactly `N_in`
independent scalar equations, $F(q_{\mathrm{wall}},q_{\mathrm{retained}},
\phi_{\mathrm{wall}},\mathrm{wall\ data})=0$, where
$F\in\mathbb{R}^{N_{\mathrm{in}}}$.

The incoming solve finds `a` in
$q_{\mathrm{wall}}=q_{\mathrm{interior}}+R_{\mathrm{in}}a$. Local
well-posedness requires the residual Jacobian to constrain every incoming
direction: $\operatorname{rank}((\partial F/\partial q_{\mathrm{wall}})
R_{\mathrm{in}})=N_{\mathrm{in}}$.

This is an exact constraint count, not a least-squares preference for more
primitive rows. A law may be nonlinear and state dependent, but it must not
provide fewer constraints (underdetermined) or more independent constraints
(incompatible unless an additional physical variable is solved globally).
The interface should expose the residual, its Jacobian (analytic or AD), wall
context, expected rank, and admissibility checks. Solver failure, nonfinite
states, singular Jacobians, and rank changes should be explicit diagnostics,
never a silent fallback.

## Current implementation and its limits

`DRBX/src/drbx/native/characteristic_wall_residual.py` currently provides the
generic strict nonlinear solver plus two compatibility mechanisms:

* `solve_nonlinear_incoming_characteristic_boundary` parameterizes the wall
  trace with exactly two incoming right-eigenvector columns and solves exactly
  two residuals with Newton iterations. It uses autodiff for the residual
  Jacobian when one is not supplied, and rejects rank-deficient bases,
  rank-deficient or ill-conditioned residual Jacobians, nonconvergence, and
  inadmissible states. `no_flow_boundary_residual` supplies the implemented
  two-velocity validation law. The `fci_parallel_production_flux` selector
  is `parallel_characteristic_wall_law="velocity-no-flow"`.

For the landed `velocity-no-flow` path, wall-state classification is performed
at $q_{\mathrm{class}}=(n,T_e,T_i,0,0)$. At this state the contact mode is
stationary and excluded from the incoming solve; exactly the two acoustic
incoming modes are supplied to the nonlinear solver. The live/interior
eigensystem remains the one used for flux splitting. These are deliberately
separate roles: the constrained wall classification determines the law's
incoming subspace, while the live state determines the numerical material
flux. Future Bohm--Chodura and magnetic-presheath laws must revisit this
classification, including sonic and glancing/contact modes, rather than
assuming the no-flow two-acoustic count.

* `solve_incoming_characteristic_state` projects a primitive `target` onto the
  incoming subspace by minimizing a weighted residual. This is the current
  `primitive-least-residual` policy, useful for compatibility and debugging.
  It remains a compatibility path rather than the general nonlinear law.
* `apply_maximally_dissipative_characteristic_wall` copies outgoing/stationary
  amplitudes and replaces incoming amplitudes with a source, zero by default.
  The production wiring in
  `DRBX/src/drbx/native/fci_parallel_production_flux.py` uses an explicit
  equilibrium reference for this `energy-absorbing` policy.

The first policy can over-constrain the two incoming degrees of freedom when
asked to fit a full primitive target. The second is a useful stable numerical
baseline, but its zero incoming amplitude is defined relative to a reference
state; it is therefore not a reference-free material-wall model. Neither is
the final physical sheath or magnetic-presheath closure.

## Agreed implementation progression

1. **No-flow validation closure (validation only).** Impose the two velocity
   residuals $F=(V_i,V_e)$ for the two incoming modes. Check exact residual
   satisfaction and preservation of outgoing/stationary content. This closure
   is intentionally reflecting and is not a production wall model.
2. **Zero-current Bohm--Chodura closure.** Use state-dependent ion sonic/Bohm
   flow and zero parallel current, for example
   $F=(V_i-\sigma\sqrt{T_e+\tau T_i},\;n(V_i-V_e))$, with the precise
   normalization and orientation supplied by the model. This is the first
   nonlinear physical test and does not require an electrical-wall solve.
3. **Electron sheath and electrical wall.** Add the electron response to the
   plasma-to-wall drop $\phi_b-\phi_w$, then support a grounded wall
   (prescribed `phi_w`), a
   local floating policy, or a globally floating conductor with one shared
   potential/current constraint. The potential policy is wall data/global
   coupling, not an extra local incoming characteristic equation.
4. **Full magnetic presheath.** Extend the physical wall-law bundle with the
   coupled normal derivative and polarization conditions required at the
   magnetic-presheath entrance, while retaining the same incoming solve for
   the hyperbolic subset. See Loizu, Ricci, Halpern & Jolliet, *Boundary
   conditions for plasma fluid models at the magnetic presheath entrance*
   ([EPFL PDF](https://infoscience.epfl.ch/server/api/core/bitstreams/c693bbf8-aa98-4bcc-8c56-2ded2e106038/content)),
   and Giacomin et al., *Journal of Computational Physics* 463 (2022) 111294
   ([EPFL PDF](https://infoscience.epfl.ch/server/api/core/bitstreams/cf0f95bd-a12b-48aa-ab29-5cb77a412129/content)).

## Hyperbolic versus normal-derivative closures

The incoming-characteristic solve handles only information entering through
the hyperbolic parallel system. It must not be made responsible for every
primitive boundary quantity. Density and temperature normal derivatives,
potential derivatives, and polarization/vorticity conditions are separate
operator-level closures. The eventual magnetic-presheath wall-law object may
provide all of them, but each consumer must receive the appropriate trace,
flux, or derivative; adding derivative equations to `F` would incorrectly
change the hyperbolic incoming count.

## Acceptance tests and diagnostics

Every closure should be tested in eager and JIT/compiled paths, for both wall
orientations, with batched traces and invalid spectra. Required checks are:

* `N_in` and the rank/condition number of
  $(\partial F/\partial q)R_{\mathrm{in}}$;
* nonlinear residual norm and convergence/failure status;
* retained-mode error (outgoing and stationary content unchanged);
* finite and thermodynamically admissible wall state;
* characteristic boundary power/incoming modal energy;
* consistency of the wall state consumed by all local-BE, SAT, and flux paths.

The no-flow gate should have an analytic answer and near-roundoff velocity
residuals. The first end-to-end validation is a short 48^3 replay, recording
the same diagnostics and checking that no fallback or hidden extra constraint
is used. Passing this gate validates machinery only; it does not promote the
no-flow closure to production.
