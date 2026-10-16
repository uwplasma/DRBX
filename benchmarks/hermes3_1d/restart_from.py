"""Overwrite the evolved fields of a fresh Hermes-3 restart file with a coarser run's steady state.

Usage: python restart_from.py COARSE_RUN_DIR NEW_RUN_DIR
NEW_RUN_DIR/BOUT.restart.0.nc must exist (written by a nout = 0 run of the new deck, so the metric
and guard layout are correct). Fields are interpolated linearly in the cell-centre index coordinate
(j + 1/2)/ny, which is the coordinate the SD1D dy(y) formula is written in; guard cells copy the
nearest interior cell. Only a starting guess: the new run is then relaxed to its own steady state.
"""

import sys

import netCDF4
import numpy as np

src, dst = sys.argv[1:3]
a = netCDF4.Dataset(f"{src}/BOUT.restart.0.nc")
b = netCDF4.Dataset(f"{dst}/BOUT.restart.0.nc", "r+")
g = int(a["MYG"][...])
for k, v in b.variables.items():
    if v.dimensions != ("x", "y", "z"):
        continue
    old = np.asarray(a[k][0, g:-g, 0])
    n_old, n_new = old.size, v.shape[1] - 2 * g
    x_old, x_new = (np.arange(n_old) + 0.5) / n_old, (np.arange(n_new) + 0.5) / n_new
    new = np.interp(x_new, x_old, old)
    v[0, g:-g, 0] = new
    v[0, :g, 0], v[0, -g:, 0] = new[0], new[-1]
b.close()
