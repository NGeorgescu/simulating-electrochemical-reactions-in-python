"""serm -- Simulating Electrochemical Reactions in Python.

A Python-native adaptation of Michael Honeychurch's *Simulating Electrochemical
Reactions in Mathematica* (SERM).  The original Mathematica notebooks are the
reference for the science and algorithms; this package is an idiomatic numpy /
scipy / matplotlib re-implementation.

This module exposes the explicit finite-difference solver for the Chapter 2
pilot (the diffusion-limited current transient of ``O + e- <-> R`` after a
potential step), ported from ``ExplicitFD.nb``.
"""
from __future__ import annotations

import numpy as np

from .grids import make_grid, space_points, dx_dimensionless
from . import (
    tridiagonal, filters, grids, plotting, waveforms, echem, kinetics,
    boundary, semiintegration, convdiff2d, currentdist, rrde,
)

__all__ = [
    "explicit_solve",
    "electrode_current",
    "cottrell_dimensionless",
    "make_grid",
    "space_points",
    "dx_dimensionless",
    "tridiagonal",
    "filters",
    "grids",
    "plotting",
    "waveforms",
    "echem",
    "kinetics",
    "boundary",
    "semiintegration",
    "convdiff2d",
    "currentdist",
    "rrde",
]


def explicit_solve(c, D_M):
    """Advance the concentration grid in time with the explicit FD scheme.

    Port of ``explicitSolve`` / ``explicitSolve2`` from ``ExplicitFD.nb``.

    The forward-difference (fully explicit) discretisation of Fick's second law

        dc/dt = d^2 c / dx^2     (dimensionless)

    is

        c[j, k] = D_M * c[j-1, k-1] + (1 - 2*D_M) * c[j, k-1]
                  + D_M * c[j+1, k-1]

    where ``D_M = dt / dx^2`` is the dimensionless model diffusion coefficient.
    This is the exact update in the original procedural ``explicitSolve``; here
    we vectorise the spatial sweep (the interior of each column) as in the
    original ``explicitSolve2`` (``ListCorrelate[{D, 1-2D, D}, ...]``).

    The scheme is stable only for ``D_M <= 0.5``.

    Parameters
    ----------
    c : ndarray, shape (m, n)
        Grid from :func:`serm.grids.make_grid`, with IC/BCs already applied.
        Modified in place and also returned.
    D_M : float
        Dimensionless model diffusion coefficient.

    Returns
    -------
    ndarray, shape (m, n)
        The filled grid.
    """
    m, n = c.shape
    stencil = np.array([D_M, 1.0 - 2.0 * D_M, D_M])
    for k in range(1, n):
        prev = c[:, k - 1]
        # interior points 1 .. m-2 updated from the three-point stencil;
        # np.convolve in 'valid' mode gives exactly len(prev)-2 values.
        c[1:-1, k] = np.convolve(prev, stencil[::-1], mode="valid")
        # boundaries (rows 0 and m-1) keep the values set by make_grid.
    return c


def electrode_current(c, D_M):
    """Dimensionless current transient at the electrode.

    Port of the ``i1`` expression in ``ExplicitFD.nb``::

        i1 = (-4 c[2] + c[3]) * 0.5 * Sqrt[D_M (n-1)]   (1-based indices)

    Per time slice (column ``k``), the flux at the electrode is approximated by
    a one-sided finite difference of the concentration gradient.  With the
    surface concentration pinned to zero (``c[0] = 0``), the 3-point one-sided
    derivative ``(-3 c[0] + 4 c[1] - c[2]) / (2 dx)`` reduces to
    ``(4 c[1] - c[2]) / (2 dx)``.  Multiplying the gradient by the dimensionless
    factor ``Sqrt(D_M (n-1)) = 1/dx`` converts grid units to the dimensionless
    current (the chapter's ``(tn/D)^{1/2}`` scaling).

    Returns the magnitude (positive) of the current for each time column.

    Parameters
    ----------
    c : ndarray, shape (m, n)
    D_M : float

    Returns
    -------
    ndarray, shape (n,)
        Dimensionless current at each time index; ``i[0]`` (t=0) is set to nan.
    """
    m, n = c.shape
    factor = 0.5 * np.sqrt(D_M * (n - 1))
    # (4 c[1] - c[2]) per column -> rows index 1 and 2.
    i = (4.0 * c[1, :] - c[2, :]) * factor
    i = np.asarray(i, dtype=float).copy()
    i[0] = np.nan  # gradient undefined at the initial instant
    return i


def cottrell_dimensionless(n):
    """Analytical Cottrell dimensionless current ``1/sqrt(pi*tau)``.

    This is the *dimensionless* Cottrell transient on the chapter's ``tau``
    grid (``tau = (k-1)/(n-1)``), taking the number of time steps ``n`` as its
    only argument.  It is distinct from :func:`serm.echem.cottrell_current`,
    which evaluates the dimensional Cottrell current ``i(t)`` in amperes.

    Port of ``z = Table[-1/Sqrt[Pi (k-1)/(n-1)], {k, 2, n-1}]`` from
    ``ExplicitFD.nb``.  For a potential step to the diffusion-limited region,
    the flux at a planar electrode follows the Cottrell form; in the chapter's
    dimensionless variables (``tau = (k-1)/(n-1)``) this is ``1/sqrt(pi*tau)``.

    Returns the magnitude for each time index ``k = 0 .. n-1``; ``tau = 0``
    gives ``inf`` and is returned as nan.

    Parameters
    ----------
    n : int

    Returns
    -------
    ndarray, shape (n,)
    """
    k = np.arange(n)
    tau = k / (n - 1)
    with np.errstate(divide="ignore"):
        i = 1.0 / np.sqrt(np.pi * tau)
    i[0] = np.nan
    return i
