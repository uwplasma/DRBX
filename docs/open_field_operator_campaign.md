# Open-Field Operator Campaign

This campaign is an operator-level verification surface for the open-field
parallel-gradient, electron-force-balance, and target-recycling kernels used by
the promoted sheath-connected and recycling cases.

The campaign is deliberately smaller than a full Hermes comparison. It verifies
the local mathematics that must be correct before a longer reference-code or
experimental validation run is meaningful:

- the centered parallel derivative
  \(D_y f_j = (f_{j+1} - f_{j-1})/(2\Delta y)\);
- the electron force balance
  \(E_\parallel = (-D_y p_e + S_{\mathrm{mom},e})/\max(n_e,n_{\mathrm{floor}})\);
- the target recycling particle source
  \(S_N = R\max[0, s(n_i+n_g)(v_i+v_g)/4]A_\parallel/V\);
- the returned neutral energy source
  \(S_E = S_N(1-R_f)E_{\mathrm{recycle}}\);
- the differentiability of the force-balance objective by comparing `jax.grad`
  with a centered finite difference.

The verification style follows the method-of-manufactured-solutions refinement
logic used in the plasma-fluid verification literature, including the Dudson et
al. MMS verification paper. The sheath and recycling source formulas are matched
to the Hermes-3 boundary-condition documentation, where the sheath particle flux
and target recycling model are written explicitly. The campaign therefore
connects a mathematically controlled refinement test to the same boundary
operator family used in the multi-component edge/SOL model.

![Open-field operator campaign](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__images__open_field_operator_campaign.png)

The generated artifact bundle contains:

- `docs/data/open_field_operator_campaign_artifacts/data/open_field_operator_campaign.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__data__open_field_operator_campaign.npz`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__open_field_operator_campaign_artifacts__images__open_field_operator_campaign.png`

Run it with:

```bash
python examples/engineering/open_field_operator_campaign_demo.py
```

The figure has four panels. The first two show error refinement and observed
order for the parallel-gradient and electron-force-balance operators. The third
panel compares the computed target recycling source against the analytic
finite-volume identity. The fourth panel compares the autodiff sensitivity of a
force-balance objective against a centered finite-difference estimate.

References:

- Dudson et al., method-of-manufactured-solutions verification:
  <https://arxiv.org/abs/1602.06747>
- Hermes-3 boundary conditions and recycling equations:
  <https://hermes3.readthedocs.io/en/latest/boundary_conditions.html>
- Hermes-3 multi-component edge/SOL model paper:
  <https://arxiv.org/abs/2303.12131>
