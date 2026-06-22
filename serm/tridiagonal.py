"""Tridiagonal linear-system solvers (Thomas algorithm).

Python port of ``Electrochem/Tridiagonal.m`` by Mike Honeychurch (2002), from
*Simulating Electrochemical Reactions in Mathematica* (SERM).

The original ``TridiagSolver[x, y, z, b]`` solves ``A . u == b`` where ``A`` is
tridiagonal with sub-diagonal ``x``, main diagonal ``y`` and super-diagonal
``z`` (see the package's own docstring)::

         y[0] z[0]
         x[0] y[1] z[1]
              x[1]  .    .
                         .    z[n-2]
                    x[n-2] y[n-1]

It uses "regular Gaussian elimination ... but no pivoting" -- i.e. the classic
Thomas algorithm.  We reproduce that here, and also expose a banded wrapper
around :func:`scipy.linalg.solve_banded` as the recommended production path.

Index convention (matches the original, 0-based here):
    x : sub-diagonal,   length n-1  (x[i] is A[i+1, i])
    y : main diagonal,  length n
    z : super-diagonal, length n-1  (z[i] is A[i, i+1])
    b : right-hand side, length n
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def tridiag_solve(x, y, z, b):
    """Solve a tridiagonal system ``A . u == b`` with the Thomas algorithm.

    Direct port of the original ``TridiagSolver`` (no pivoting).  This is a
    forward sweep (eliminate the sub-diagonal) followed by a back substitution.

    Parameters
    ----------
    x : array_like, shape (n-1,)
        Sub-diagonal of ``A`` (entries below the main diagonal).
    y : array_like, shape (n,)
        Main diagonal of ``A``.
    z : array_like, shape (n-1,)
        Super-diagonal of ``A`` (entries above the main diagonal).
    b : array_like, shape (n,)
        Right-hand side.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Solution vector ``u``.

    Notes
    -----
    Like the original, this performs *no pivoting*: a zero pivot produces
    ``inf``/``nan`` (the original warns "Infinity introduced if pivot becomes
    zero").  For diagonally dominant systems -- the usual case for implicit
    finite-difference diffusion problems -- it is stable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]
    if y.shape[0] != n:
        raise ValueError("y must have the same length as b")
    if x.shape[0] != n - 1 or z.shape[0] != n - 1:
        raise ValueError("x and z must have length len(b) - 1")

    # alpha: modified main diagonal after elimination; f: modified RHS.
    alpha = np.empty(n)
    f = np.empty(n)
    alpha[0] = y[0]
    f[0] = b[0] / alpha[0]
    for j in range(1, n):
        # x[j-1] is the sub-diagonal entry A[j, j-1]; z[j-1] is A[j-1, j].
        alpha[j] = y[j] - x[j - 1] * z[j - 1] / alpha[j - 1]
        f[j] = (b[j] - x[j - 1] * f[j - 1]) / alpha[j]

    # Back substitution.
    u = np.empty(n)
    u[-1] = f[-1]
    for j in range(n - 2, -1, -1):
        u[j] = f[j] - z[j] * u[j + 1] / alpha[j]
    return u


def tridiag_solve_banded(x, y, z, b):
    """Solve the same system via :func:`scipy.linalg.solve_banded`.

    This is the recommended production path: SciPy calls LAPACK's banded solver
    (``gtsv``-class routines via the banded interface), which *does* pivot and
    is more robust than the bare Thomas algorithm.  Same argument convention as
    :func:`tridiag_solve`.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    # solve_banded wants ab[0] = super-diagonal (shifted), ab[1] = diagonal,
    # ab[2] = sub-diagonal (shifted), with l=u=1.
    ab = np.zeros((3, n))
    ab[0, 1:] = z          # super-diagonal placed in row 0, columns 1..n-1
    ab[1, :] = y           # main diagonal
    ab[2, :-1] = x         # sub-diagonal placed in row 2, columns 0..n-2
    return solve_banded((1, 1), ab, b)
