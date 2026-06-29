"""Chapter 15 -- the FIRM (Richtmyer / BDF4) variant of the sparse CV solver.

Companion to :mod:`serm.ch15_sparse_finite_differences`.  The standalone
notebook ``Extra Notebooks/chapter15/sparseTri.nb`` offers two time-stepping
schemes for the single-species quasireversible CV on the expanding grid:

* **FIFD** -- fully implicit (backward Euler) finite differences.  This is the
  ``solveCV1`` / ``solveCV3`` scheme already ported as
  :func:`serm.ch15_sparse_finite_differences.simulate_cv_single`.  It is first
  order accurate in time.
* **FIRM** -- "fully implicit with Richtmyer modification" (``solveCV2`` /
  ``solveCV4``).  The single backward-Euler time derivative is replaced by the
  four-point **BDF4** combination

  .. math::
      \\tfrac{25}{12} c_j^{(k)} - 4 c_j^{(k-1)} + 3 c_j^{(k-2)}
        - \\tfrac{4}{3} c_j^{(k-3)} + \\tfrac14 c_j^{(k-4)}
        = (\\text{implicit spatial operator}),

  which is fourth-order accurate in time.  The only structural change to the
  matrix is the main-diagonal constant: ``1 + (1+a) D_M a^(3-2j)`` becomes
  ``25/12 + (1+a) D_M a^(3-2j)`` (cf. ``makeDiagonals2`` and the ``25./12.`` in
  ``sparseMat2``).  The right-hand side becomes the weighted sum of the previous
  four time levels.  Because BDF4 is not self-starting, the first three steps are
  taken with FIFD (``c1=FoldList[solveCV3, initial, Range[2,4]]``) before the
  Richtmyer recursion begins at step 5.

Both schemes share the surface Butler--Volmer elimination of
:mod:`serm.kinetics` and the expanding-grid stencil of the FIFD port, so the FIRM
result must converge to the same voltammogram as FIFD as the time grid is
refined -- the reduction-to-validated-limit check used here.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import spsolve

from .ch15_sparse_finite_differences import (
    CVParams, potential_xi, _interior_diagonals,
)
from .kinetics import bv_surface_factor, bv_surface_conc


# BDF4 weights on the four most recent time levels [c_{k-4}, c_{k-3}, c_{k-2},
# c_{k-1}] for the right-hand side (the source's MapThread combination).
_BDF4_RHS = np.array([-0.25, 4.0 / 3.0, -3.0, 4.0])
_BDF4_DIAG = 25.0 / 12.0


def simulate_cv_firm(p: CVParams, backend: str = "sparse"):
    """Single-species quasireversible CV with the FIRM (BDF4) time scheme.

    Port of ``solveCV4`` / ``solveCV2`` from ``sparseTri.nb``: the first three
    advanced steps use backward-Euler (FIFD) to bootstrap BDF4, then every
    subsequent step uses the four-point Richtmyer combination.

    Parameters
    ----------
    p : CVParams
        Same dimensionless parameters as the FIFD solver.
    backend : {"sparse", "dense"}
        Linear-solver backend, as in
        :func:`serm.ch15_sparse_finite_differences.simulate_cv_single`.

    Returns
    -------
    profiles : ndarray, shape (n, m)
        Concentration of O at every node and time increment (row 0 = initial
        condition); surface value is column 0.
    """
    m = p.m_space
    M = m - 2
    sub, main, sup = _interior_diagonals(p)
    tail = p.D_M * p.a ** (5 - 2 * m)

    # Two main-diagonal variants: FIFD uses `main`, FIRM uses main shifted by
    # (25/12 - 1) on every interior node.
    main_firm = main + (_BDF4_DIAG - 1.0)
    y1_e, z1 = main[0], sup[0]
    y1_f = main_firm[0]

    def assemble(diag):
        if backend == "dense":
            A = np.zeros((M, M))
            idx = np.arange(M)
            A[idx, idx] = diag
            A[idx[1:], idx[:-1]] = sub[1:]
            A[idx[:-1], idx[1:]] = sup[:-1]
            return A
        if backend == "sparse":
            A = sp.lil_matrix((M, M))
            A.setdiag(diag)
            A.setdiag(sub[1:], -1)
            A.setdiag(sup[:-1], 1)
            return A
        raise ValueError("backend must be 'sparse' or 'dense'")

    A_fifd = assemble(main)
    A_firm = assemble(main_firm)

    def solve(A, b):
        if backend == "dense":
            return dense_solve(A, b)
        return spsolve(A.tocsc(), b)

    def step(A, y1, rhs_inner, k):
        """One implicit step: rhs_inner is the (length-M) interior RHS vector."""
        xi = potential_xi(k, p)
        tmp = bv_surface_factor(xi, p.ks_star, p.alpha)
        b = rhs_inner.copy()
        b[0] += tmp * p.D_M * p.ks_star * xi ** (1.0 - p.alpha)
        b[-1] += tail
        A[0, 0] = y1 - 4.0 * p.D_M * tmp
        A[0, 1] = z1 + p.D_M * tmp
        inner = solve(A, b)
        c0 = bv_surface_conc(inner[0], inner[1], xi, p.ks_star, p.alpha, tmp)
        return np.concatenate(([c0], inner, [1.0]))

    cur = np.ones(m)
    profiles = [cur]

    # --- bootstrap: first three advanced steps with FIFD ---------------------
    for k in range(2, min(5, p.n_time + 1)):
        cur = step(A_fifd, y1_e, profiles[-1][1:M + 1].copy(), k)
        profiles.append(cur)

    # --- BDF4 / Richtmyer recursion from step 5 ------------------------------
    for k in range(5, p.n_time + 1):
        # weighted combination of the last four full profiles (interior part)
        stack = np.array([profiles[-4][1:M + 1], profiles[-3][1:M + 1],
                          profiles[-2][1:M + 1], profiles[-1][1:M + 1]])
        rhs_inner = _BDF4_RHS @ stack
        cur = step(A_firm, y1_f, rhs_inner, k)
        profiles.append(cur)

    return np.array(profiles)
