from __future__ import annotations

import numpy as np

from jaxdrb.verification.gdb2018 import saw_linear_matrix, saw_phase_speed_sq


def test_gdb2018_saw_dispersion_relation_matches_matrix_eigs() -> None:
    """Check that the linear matrix reproduces the analytic SAW phase speed.

    This mirrors the shear-Alfvén verification test described in Zhu et al. (2018, CPC).
    """

    Te = 1.7
    ky = 2.0
    kpar = 0.3
    n0 = 1.0
    alpha_m = 2e-4
    alpha_d = 2e-3
    eps_R = 0.6
    de2 = 6.4e-7

    M = np.asarray(
        saw_linear_matrix(
            kpar=kpar,
            Te=Te,
            ky=ky,
            n0=n0,
            alpha_m=alpha_m,
            alpha_d=alpha_d,
            eps_R=eps_R,
            de2=de2,
        )
    )
    evals = np.linalg.eigvals(M)

    # Extract the oscillatory frequency from the imaginary eigenvalues.
    w = np.max(np.abs(np.imag(evals)))
    v2_num = (w / abs(kpar)) ** 2
    v2_ex = saw_phase_speed_sq(
        Te=Te, ky=ky, n0=n0, alpha_m=alpha_m, alpha_d=alpha_d, eps_R=eps_R, de2=de2
    )

    assert np.isfinite(v2_num)
    assert abs(v2_num - v2_ex) / max(abs(v2_ex), 1e-30) < 1e-12
