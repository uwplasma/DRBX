# References

The goal of this page is to provide a starting point for the literature most relevant to
`jaxdrb`’s current scope (drift-reduced fluid SOL/edge linear analysis and near-axis stellarator
geometry).

## Drift-reduced SOL / edge workflows

- A. Mosetto, F. D. Halpern, S. Jolliet, and P. Ricci,
  *Low-frequency linear-mode regimes in the tokamak scrape-off layer*,
  **Physics of Plasmas 19**, 112103 (2012).
  [DOI: 10.1063/1.4758809](https://doi.org/10.1063/1.4758809)

- F. D. Halpern, S. Jolliet, J. Loizu, A. Mosetto, and P. Ricci,
  *Ideal ballooning modes in the tokamak scrape-off layer*,
  **Physics of Plasmas 20**, 052306 (2013).
  [DOI: 10.1063/1.4807333](https://doi.org/10.1063/1.4807333)

- R. Jorge, P. Ricci, F. D. Halpern, N. F. Loureiro, and C. Silva,
  *Plasma turbulence in the scrape-off layer of the ISTTOK tokamak*,
  **Physics of Plasmas 23**, 102511 (2016).
  [DOI: 10.1063/1.4964783](https://doi.org/10.1063/1.4964783)

- P. Ricci et al.,
  *Simulation of plasma turbulence in scrape-off layer conditions*,
  **Plasma Physics and Controlled Fusion 54**, 124047 (2012).
  (See `Ricci_2012_Plasma_Phys._Control._Fusion_54_124047.pdf` in `drb_literature/`.)

- P. Ricci,
  *Turbulence in the scrape-off layer of tokamaks* (EPFL thesis, 2011),
  (See `EPFL_TH6197.pdf` in `drb_literature/`.)

## Near-axis stellarator geometry

- R. Jorge and M. Landreman,
  *The use of near-axis magnetic fields for stellarator turbulence simulations*,
  **Plasma Physics and Controlled Fusion 63**, 014001 (2021).
  [DOI: 10.1088/1361-6587/abc862](https://doi.org/10.1088/1361-6587/abc862)

## Sheath / MPSE boundary conditions

- J. Loizu, P. Ricci, F. D. Halpern, and S. Jolliet,
  *Boundary conditions for plasma fluid models at the magnetic presheath entrance*,
  **Physics of Plasmas 19**, 122307 (2012).
  [DOI: 10.1063/1.4771573](https://doi.org/10.1063/1.4771573)

- R. Chodura,
  *The ion velocity (Bohm–Chodura) boundary condition at the entrance to the magnetic presheath in the presence of diamagnetic and $E\times B$ drifts in the scrape-off layer*,
  **Physics of Plasmas 2**, 707–715 (1995).
  [DOI: 10.1063/1.871421](https://doi.org/10.1063/1.871421)

## Nonlinear drift-wave testbed + numerics

- A. Hasegawa and M. Wakatani,
  *Plasma edge turbulence*,
  **Physical Review Letters 50**, 682 (1983).
  [DOI: 10.1103/PhysRevLett.50.682](https://doi.org/10.1103/PhysRevLett.50.682)

- M. Wakatani and A. Hasegawa,
  *Collisional drift wave description of plasma edge turbulence*,
  **Physics of Fluids 27**, 611 (1984).
  [DOI: 10.1063/1.864660](https://doi.org/10.1063/1.864660)

- O. Panico et al.,
  *On the importance of flux-driven turbulence regime to address tokamak plasma edge dynamics*,
  **Journal of Plasma Physics 91**, E26 (2025).
  [DOI: 10.1017/S0022377824001624](https://doi.org/10.1017/S0022377824001624)

- A. Arakawa,
  *Computational design for long-term numerical integration of the equations of fluid motion: Two-dimensional incompressible flow. Part I*,
  **Journal of Computational Physics 1**, 119–143 (1966).
  [DOI: 10.1016/0021-9991(66)90015-5](https://doi.org/10.1016/0021-9991(66)90015-5)

- S. A. Orszag,
  *On the elimination of aliasing in finite-difference schemes by filtering high-wavenumber components*,
  **Journal of the Atmospheric Sciences 28**, 1074–1074 (1971).
  [DOI: 10.1175/1520-0469(1971)028<1074:OTEOAI>2.0.CO;2](https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2)

## Kelvin–Helmholtz shear-layer benchmarks

- A. Michalke,
  *On the inviscid instability of the hyperbolic-tangent velocity profile*,
  **Journal of Fluid Mechanics 19**, 543–556 (1964).

- M. Frank, M. Jones, and W. Nowak,
  *Stratified Kelvin–Helmholtz turbulence of compressible shear flows*,
  **Nonlinear Processes in Geophysics 25**, 457–468 (2018).

## 2D drift-wave / HW turbulence

- J. A. Camargo, H. Biskamp, and B. Scott,
  *Turbulent transport in the Hasegawa–Wakatani model*,
  **Physics of Plasmas 2**, 48–62 (1995).
  (See `camargo_biskamp_scott95.pdf` in `drb_literature/`.)

## Hermes-2 / SOL blob benchmarks

- B. D. Dudson and J. Leddy,
  *Hermes: global plasma edge fluid turbulence simulations*,
  **Plasma Physics and Controlled Fusion 59**, 054010 (2017).
  (Accepted version: `hermes-2/hermes_paper.pdf` in the local Hermes-2 checkout.)

## Plasma–neutral interactions (SOL context)

- F. D. Halpern et al.,
  *[SOL turbulence modeling and code validation; includes model extensions such as neutrals]* (2016).
  (See the PDF in `drb_literature/` included with this repository clone.)

- G. Bufferand et al.,
  *[Review: SOL turbulence and edge/SOL modeling]*,
  **Nuclear Fusion 61**, 116052 (2021).
  (See the PDF in `drb_literature/` included with this repository clone.)

## Drift-reduced Braginskii background

For a general overview of Braginskii closures and drift-reduced fluid modeling, see the
standard plasma fluid literature and the references cited in the papers above.

- S. I. Braginskii,
  *Transport processes in a plasma*,
  in **Reviews of Plasma Physics, Vol. 1** (Consultants Bureau, 1965).

- L. Spitzer Jr. and R. Härm,
  *Transport phenomena in a completely ionized gas*,
  **Physical Review 89**, 977 (1953).

## Conservative nonlinear DRB

- B. De Lucca et al.,
  *Conservative formulation of the drift-reduced fluid plasma model* (2026),
  arXiv: [2601.05704](https://arxiv.org/abs/2601.05704)
