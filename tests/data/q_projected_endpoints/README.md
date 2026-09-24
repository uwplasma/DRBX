These fixtures retain the complete eta-index-zero owner plane from the canonical
actual-HSX N32/N48/N64 geometries. The fifteen raw-volume owner moments through
transverse degree four, centroids and logical grids come directly from the
frozen OwnerMoments constructor; no observations or geometry are synthesized.
Owner IDs are compactly renumbered within the plane and the original IDs are
retained. labels contains that one plane only; tests query its actual eta center.

The 21 geometry-selected targets per resolution cover radial indices
0, 1, N//8, N//4, N//2, N-2, N-1 and theta indices 0, N//4, N-1, with an angular
offset of 0.37 cell widths. They exercise axis/aggregate, interior and wall
support and periodic theta handling. They test implementation equivalence, not
convergence. Parent input identities and fixture hashes are in manifest.json.
Broader saved-trace and complete face-map evidence is documented in the
campaign's validation/endpoint_optimization.json and ENDPOINT_OPTIMIZATION.md.
