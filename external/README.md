# External Code Cache (Local)

This directory is reserved for **local, out‑of‑tree copies** of external codes
used for benchmarking (`gbs`, `hermes-2`, `hermes-3`, and their dependencies).
These repositories and any compiled artifacts are **not tracked in git** to keep
the main repo lightweight and avoid licensing/size issues.

Expected local layout:

```
external/
  gbs/
  hermes-2/
  hermes-3/
  deps/        # optional build deps (e.g., fftw/hdf5/netcdf/petsc)
```

If you need these in version control, use a private Git LFS store or submodules.
