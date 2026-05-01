from __future__ import annotations

from pathlib import Path

from jax_drb.reference.paths import default_reference_root


_REQUIRED_REFERENCE_FILES = (
    Path("tests/integrated/1D-recycling/data/BOUT.inp"),
    Path("tests/integrated/1D-recycling-dthe/data/BOUT.inp"),
    Path("examples/tokamak-2D/recycling-dthene/BOUT.inp"),
)


def reference_root_or_ci_fixture(tmp_path: Path) -> Path:
    """Return the external reference root, or a compact CI fixture with real decks.

    Hosted CI does not have the larger external reference checkout available.  The
    fixture keeps closeout coverage from skipping the reference-root campaign
    tests while still driving the production validation code through representative
    single-species, multispecies, and D/T/He/Ne inputs.
    """

    reference_root = default_reference_root()
    if reference_root is not None and all((reference_root / relative_path).exists() for relative_path in _REQUIRED_REFERENCE_FILES):
        return reference_root

    fixture_root = tmp_path / "reference-fixture"
    _write_fixture_decks(fixture_root)
    return fixture_root


def _write_fixture_decks(root: Path) -> None:
    decks = {
        Path("tests/integrated/1D-recycling/data/BOUT.inp"): _ONE_D_RECYCLING_INPUT,
        Path("tests/integrated/1D-recycling-dthe/data/BOUT.inp"): _ONE_D_RECYCLING_DTHE_INPUT,
        Path("examples/tokamak-2D/recycling-dthene/BOUT.inp"): _TOKAMAK_RECYCLING_DTHENE_INPUT,
    }
    for relative_path, text in decks.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


_ONE_D_RECYCLING_INPUT = """\
nout = 1
timestep = 1
MXG = 0

[mesh]
nx = 1
ny = 8
nz = 1
length = 30
length_xpt = 10
dymin = 0.1
dy = (length / ny) * (1 + (1-dymin)*(1-y/pi))
J = 1
source = length_xpt / length
y_xpt = pi * ( 2 - dymin - sqrt( (2-dymin)^2 - 4*(1-dymin)*source ) ) / (1 - dymin)
ixseps1 = -1
ixseps2 = -1

[hermes]
components = (d+, d, e, sheath_boundary, braginskii_collisions, recycling, reactions, electron_force_balance, neutral_parallel_diffusion)
Nnorm = 1e19
Bnorm = 1
Tnorm = 100

[solver]
type = pvode
mxstep = 100
atol = 1e-7
rtol = 1e-5

[sheath_boundary]
lower_y = false
upper_y = true

[neutral_parallel_diffusion]
dneut = 10

[d+]
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
noflow_lower_y = true
noflow_upper_y = false
charge = 1
AA = 2
thermal_conduction = true
diagnose = true
recycle_as = d
target_recycle = true
target_recycle_multiplier = 1.0
target_recycle_energy = 0.0

[Nd+]
function = 1
source_shape = H(mesh:y_xpt - y) * 1e20

[Pd+]
function = 1
powerflux = 2.5e7
source = (powerflux*2/3 / (mesh:length_xpt))*H(mesh:y_xpt - y)

[NVd+]
function = 0

[d]
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
charge = 0
AA = 2
thermal_conduction = true
diagnose = true

[Nd]
function = 0.001

[Pd]
function = 0.0001

[e]
type = quasineutral, evolve_pressure, zero_current, noflow_boundary
noflow_upper_y = false
charge = -1
AA = 1/1836
thermal_conduction = true
diagnose = true

[Pe]
function = `Pd+:function`
source = `Pd+:source`

[recycling]
species = d+

[reactions]
diagnose = true
type = (d + e -> d+ + 2e, d+ + e -> d, d + d+ -> d+ + d)
"""


_ONE_D_RECYCLING_DTHE_INPUT = """\
nout = 1
timestep = 1
MXG = 0

[mesh]
nx = 1
ny = 8
nz = 1
length = 30
length_xpt = 10
dymin = 0.1
dy = (length / ny) * (1 + (1-dymin)*(1-y/pi))
J = 1
source = length_xpt / length
y_xpt = pi * ( 2 - dymin - sqrt( (2-dymin)^2 - 4*(1-dymin)*source ) ) / (1 - dymin)
ixseps1 = -1
ixseps2 = -1

[hermes]
components = (d+, d, t+, t, he+, he, e, sheath_boundary, braginskii_collisions, recycling, reactions, electron_force_balance, neutral_parallel_diffusion, braginskii_ion_viscosity)
Nnorm = 1e19
Bnorm = 1
Tnorm = 100

[solver]
mxstep = 100
atol = 1e-12
rtol = 1e-7

[sheath_boundary]
lower_y = false
upper_y = true

[neutral_parallel_diffusion]
dneut = 10

[d+]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
noflow_lower_y = true
noflow_upper_y = false
charge = 1
AA = 2
thermal_conduction = true
recycle_as = d
target_recycle = true
target_recycle_multiplier = 1.0
target_recycle_energy = 0.0

[Nd+]
function = 1
source_shape = H(mesh:y_xpt - y) * 1e20

[Pd+]
function = 1
powerflux = 2.5e7
source = (powerflux*2/3 / (mesh:length_xpt))*H(mesh:y_xpt - y)

[NVd+]
function = 0

[d]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
charge = 0
AA = 2
thermal_conduction = true

[Nd]
function = 0.001

[Pd]
function = 0.0001

[t+]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
noflow_lower_y = true
noflow_upper_y = false
AA = 3
charge = 1
thermal_conduction = true
recycle_as = t
target_recycle = true
target_recycle_multiplier = 1.0
target_recycle_energy = 0.0

[Nt+]
function = 1
source_shape = H(mesh:y_xpt - y) * 1e20

[Pt+]
function = 1
powerflux = 2.5e7
source = (powerflux*2/3 / (mesh:length_xpt))*H(mesh:y_xpt - y)

[NVt+]
function = 0

[t]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
charge = 0
AA = 3
thermal_conduction = true

[Nt]
function = 0.001

[Pt]
function = 0.0001

[he+]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
noflow_lower_y = true
noflow_upper_y = false
AA = 4
charge = 1
thermal_conduction = true
recycle_as = he
target_recycle = true
target_recycle_multiplier = 1.0
target_recycle_energy = 0.0

[Nhe+]
function = 0.01

[Phe+]
function = 0.01

[NVhe+]
function = 0

[he]
diagnose = true
type = (evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)
charge = 0
AA = 4
thermal_conduction = true

[Nhe]
function = 0.00001

[Phe]
function = 0.000001

[e]
diagnose = true
type = quasineutral, evolve_pressure, zero_current, noflow_boundary
noflow_upper_y = false
charge = -1
AA = 1/1836
thermal_conduction = true

[Pe]
function = `Pd+:function`
source = `Pd+:source`

[braginskii_collisions]
diagnose = true
electron_electron = true
electron_ion = true
electron_neutral = true
ion_ion = true
ion_neutral = true
neutral_neutral = true

[reactions]
diagnose = true
type = (d + e -> d+ + 2e, d+ + e -> d, d + d+ -> d+ + d, t + e -> t+ + 2e, t+ + e -> t, t + t+ -> t+ + t, d + t+ -> d+ + t, t + d+ -> t+ + d, he + e -> he+ + 2e, he+ + e -> he)

[recycling]
species = d+, t+, he+

[braginskii_thermal_force]
override_ion_mass_restrictions = true
"""


_TOKAMAK_RECYCLING_DTHENE_INPUT = """\
nout = 50
timestep = 50
MZ = 1

[mesh]
file = "tokamak.nc"

[mesh:paralleltransform]
type = shifted

[solver]
mxstep = 10000

[hermes]
components = (d+, d, t+, t, he+, he, ne+, ne, e, braginskii_collisions, braginskii_friction, braginskii_heat_exchange, sheath_boundary, reactions, braginskii_conduction, recycling)
Nnorm = 2e18
Bnorm = 1
Tnorm = 5

[d+]
type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion
AA = 2
charge = 1
anomalous_D = 2
anomalous_chi = 1
thermal_conduction = true
recycle_as = d
target_recycle = true
target_recycle_multiplier = 0.99

[Nd+]
function = 1
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[Pd+]
function = 1
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.01)

[Td+]
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[d]
type = neutral_mixed
AA = 2

[t+]
type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion
AA = 3
charge = 1
anomalous_D = 2
anomalous_chi = 1
thermal_conduction = true
recycle_as = t
target_recycle = true
target_recycle_multiplier = 0.99

[Nt+]
function = 1
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[Pt+]
function = 1
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.01)

[Tt+]
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[t]
type = neutral_mixed
AA = 3

[he+]
type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion
AA = 4
charge = 1
anomalous_D = 2
anomalous_chi = 1
thermal_conduction = true
recycle_as = he
target_recycle = true
target_recycle_multiplier = 0.99

[Nhe+]
function = 0.1
bndry_core = dirichlet(0.1)
bndry_all = dirichlet(0.1)

[Phe+]
function = 0.01
bndry_core = dirichlet(0.1)
bndry_all = dirichlet(0.01)

[The+]
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[he]
type = neutral_mixed
AA = 4

[ne+]
type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion
AA = 20
charge = 1
anomalous_D = 2
anomalous_chi = 1
thermal_conduction = true
recycle_as = ne
target_recycle = true
target_recycle_multiplier = 0.99

[Nne+]
function = 0.01
bndry_core = dirichlet(0.01)
bndry_all = dirichlet(0.01)

[Pne+]
function = 0.001
bndry_core = dirichlet(0.01)
bndry_all = dirichlet(0.001)

[Tne+]
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[ne]
type = neutral_mixed
AA = 20

[e]
type = quasineutral, evolve_pressure, zero_current, anomalous_diffusion
AA = 1/1836
charge = -1
anomalous_D = `d+`:anomalous_D
anomalous_chi = 0.1
thermal_conduction = true

[Pe]
function = 1
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.01)

[Te]
bndry_core = dirichlet(1.0)
bndry_all = dirichlet(0.1)

[recycling]
species = d+, t+, he+, ne+

[reactions]
type = (d + e -> d+ + 2e, t + e -> t+ + 2e, he + e -> he+ + 2e, he+ + e -> he, ne + e -> ne+ + 2e, ne+ + e -> ne)
"""
