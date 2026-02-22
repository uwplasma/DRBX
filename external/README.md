# External Code Cache (Local)

This directory is reserved for **local, out‑of‑tree copies** of external codes
used for cross‑code comparisons and tooling (plus their dependencies).
These repositories and any compiled artifacts are **not tracked in git** to keep
the main repo lightweight and avoid licensing/size issues.

Expected local layout:

```
external/
  code_a/
  code_b/
  code_c/
  deps/        # optional build deps (e.g., fftw/hdf5/netcdf/petsc)
```

If you need these in version control, use a private Git LFS store or submodules.
