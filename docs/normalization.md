# Normalization

The unified DRB solver accepts **physical inputs** via a normalization block that
converts those values into the solver's normalized units. This keeps input files
consistent with Hermes/GBS conventions while letting the core remain unitless.

## Configuration

```toml
[normalization]
enabled = true
mode = "physics"          # physics | manual
Te0_eV = 50.0
Ti0_eV = 50.0
n0 = 1e19
B0 = 2.0
m_i_amu = 2.0
Z_i = 1
length_unit = "rho_s"     # rho_s | lref
# Lref_m = 1.0             # used when length_unit = "lref"

# Physical sections that will be converted into normalized values
[geometry_physical]
Lx = 0.1
Ly = 0.1
Lz = 6.283185
R0 = 2.0
r0 = 0.2
B0 = 2.0

[physics_physical]
omega_n = 20.0             # 1/m

[transport_physical]
Dn = 0.5                   # m^2/s

[closures_physical.sol]
sol_width = 0.02
sol_relax_open = 1e5       # 1/s
```

The converted values are merged into the corresponding normalized sections:
`geometry`, `physics`, `transport`, `closures`, and `initial`.

## CLI Usage

```
jaxdrb /path/to/input.toml
```

## Physical Sections

The following optional sections are recognized and converted when
`[normalization].enabled = true`:

- `geometry_physical`
- `physics_physical`
- `transport_physical`
- `closures_physical`
- `initial_physical`

## Notes

- `tau_i` is set automatically from `Ti0_eV/Te0_eV` if not explicitly provided.
- If `normalization.enabled = false` or the block is omitted, inputs are assumed
  to already be normalized.
- Use `mode = "manual"` if you want to provide explicit unit scales instead of
  plasma parameters.
