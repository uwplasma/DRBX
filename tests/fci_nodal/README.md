# FCI nodal test layers

Run `pytest -m fast tests/fci_nodal` for data-free checks. Both resolution 16 profiles carry `slow`.

Set `DRBX_FCI_NODAL_REFERENCE_CASE` to a complete current-format, one-owner
resolution 16 case with current source receipts. Use a reference built on the
same machine and with the same JAX version as the test. The reference must stay
where it was built because its stage receipts store absolute product paths.

| Layer | Profile |
|---|---|
| Pinned | `-m pinned_case`. Adopt the saved metric, field table, polygon and indexed geometry. Build the owner operators and run the saved manufactured profile. Compare one-owner perpendicular operators and run outputs with the reference byte for byte. The two-rank test compares distributed output with the local two-owner run. |
| Rebuild | `-m rebuild_case`. Read the build and run options from the reference receipts, build one owner from device inputs, and compare the stage arrays byte for byte. Set `DRBX_FCI_NODAL_DEVICE_DIR` to a different directory containing the same-named MAKEGRID and vessel files. |

The tests write per-array difference JSON before asserting identity. The
two-owners-against-one comparison uses the recorded bounds in
`support/owner_size_bounds.py`.
