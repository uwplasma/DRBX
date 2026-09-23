# Local release validation

Validated on 2026-09-22 from a fresh directory beneath `/private/tmp`, with
the repository campaign script outside the extracted source tree and the
immutable inputs supplied through `--input-root`.

- Source archive: 1,186 files, 15,854,509 bytes,
  SHA-256 `d90f28eeb4e33cdfa1579bee42bac03ce5d7dcaae21954f70ec7f75d1e5362cb`.
- Source manifest SHA-256:
  `8678c54a45dfd9b99da18c347b8bfd20f90720b153e3c47a1331e8883cf396a9`.
- External-input manifest SHA-256:
  `66a3a435bb6bf2daf35ed98ca01658211e587aa552f997e02b7833a81c036bf5`.
- Campaign identity:
  `d0ce435f63b7858fd0e5cdfa52177718364f019eb3009e2a190023c1f2d54682`.
- The relocated one-plane numerical preflight completed in 30.90 seconds at
  1.109 GiB recorded worker peak RSS.
- Recipients and tile indices matched exactly. Volume, moment, and source
  arrays were bitwise identical to the archived local preflight (maximum
  absolute difference `0.0` for each array family).
- Re-running the preflight in the same campaign folder reused the valid plane
  checkpoint and completed successfully without recomputation.
- Both Python entry points pass byte-code compilation and the repository diff
  passes whitespace validation.

The earlier optimized local production launch also exercised two persistent
workers concurrently before it was intentionally cancelled for this remote
handoff. Its completed units were not copied into this campaign, so the remote
run starts from independently verified inputs and produces its own receipts.
