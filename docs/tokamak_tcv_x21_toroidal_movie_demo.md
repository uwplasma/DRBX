# TCV-X21 Toroidal Movie Demo

This demo rebuilds a toroidal 3D visualization from the committed TCV-X21 scaffold arrays instead of showing only a 2D slice GIF.

Run:

```bash
PYTHONPATH=src python examples/tokamak-3D/tcv-x21/toroidal_movie_demo.py
```

Outputs:

- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__data__tokamak_tcv_x21_toroidal_arrays.npz`
- `docs/data/tokamak_tcv_x21_toroidal_movie_artifacts/data/tokamak_tcv_x21_toroidal_summary.json`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__images__tokamak_tcv_x21_toroidal_poster.png`
- `https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif`

What it shows:

- a toroidal outer-shell fluctuation surface built from the staged benchmark field history;
- two orthogonal poloidal cuts carrying the instantaneous 2D cross-section dynamics;
- a clearer device-scale view for the README and docs surface than the original flat scaffold slice movie.

![TCV-X21 toroidal movie](https://github.com/uwplasma/jax_drb/releases/download/validation-artifacts-2026-04-28/docs__data__tokamak_tcv_x21_toroidal_movie_artifacts__movies__tokamak_tcv_x21_toroidal.gif)
