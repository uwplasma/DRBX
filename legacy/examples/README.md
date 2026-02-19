# Examples (Unified DRB System)

These examples are **usage-focused** (not validation). Physics verification lives in `tests/` and `benchmarks/`.

## 1D / line (flux-tube)
- `01_line/line_cold_es.py`: cold-ion, electrostatic line model.
- `01_line/line_hot_es.py`: hot-ion, electrostatic line model.
- `01_line/line_em.py`: electromagnetic line model (psi on).
- `01_line/line_sheath_mpse.py`: MPSE/Loizu-style sheath BCs on an open field line.

## 2D (perpendicular plane)
- `02_2d/drb2d_es.py`: nonlinear DRB2D electrostatic subset.
- `02_2d/drb2d_hot_ion.py`: hot-ion DRB2D subset.
- `02_2d/drb2d_em.py`: electromagnetic DRB2D subset.
- `02_2d/hw2d.py`: Hasegawa–Wakatani (HW) subset.

## 3D / FCI
- `03_fci/fci_drb3d_es.py`: minimal electrostatic FCI 3D example.
- `03_fci/fci_drb3d_em_hot.py`: electromagnetic + hot-ion FCI example (full core).
