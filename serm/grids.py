"""Finite-difference grid helpers for the SERM diffusion simulations.

These reproduce the grid construction from ``ExplicitFD.nb`` (Honeychurch,
SERM, Chapter 2).  The original ``makeGrid[m, n]`` builds an ``m x n`` array
``c`` of dimensionless concentration, where ``m`` indexes space (the electrode
at the first row, bulk at the last) and ``n`` indexes time.

Initial / boundary conditions for a potential step to the diffusion-limited
region of ``O + e- <-> R`` (the reactant ``O`` is consumed at the electrode):

* Initial condition (t = 0, first time column): ``c = 1`` everywhere
  (bulk concentration, normalised to 1).
* Electrode boundary (first space row, all later times): ``c = 0``
  (Cottrell step -- surface concentration driven to zero).
* Bulk boundary (last space row, all times): ``c = 1``.
"""
from __future__ import annotations

import math

import numpy as np


def space_points(D_M, n):
    """Number of spatial grid points ``m`` for ``n`` time points.

    Port of ``m = 1 + Ceiling[6 Sqrt[D_M (n-1)]]`` from ``ExplicitFD.nb``.

    The dimensionless diffusion-layer extent is fixed at ``x_dv = 6`` (six
    diffusion lengths -- effectively semi-infinite over the experiment), and the
    number of space steps needed to reach it is ``6 Sqrt(D_M (n-1))`` because
    the dimensionless space step is ``dx = 1 / Sqrt(D_M (n-1))`` (see
    :func:`dx_dimensionless`).

    Parameters
    ----------
    D_M : float
        Model (dimensionless) diffusion coefficient ``D_M = D*dt/dx^2``.
    n : int
        Number of time grid points.

    Returns
    -------
    int
    """
    return 1 + math.ceil(6.0 * math.sqrt(D_M * (n - 1)))


def dx_dimensionless(D_M, n):
    """Dimensionless space step ``dx = 1 / sqrt(D_M (n-1))``.

    Derived in the chapter: with dimensionless time step ``dt = 1/(n-1)`` and
    ``D_M = dt / dx^2``, we get ``dx = sqrt(dt / D_M) = 1/sqrt(D_M (n-1))``.
    """
    return 1.0 / math.sqrt(D_M * (n - 1))


def make_grid(m, n):
    """Build the ``m x n`` concentration grid with IC/BCs applied.

    Port of ``makeGrid[m, n]`` from ``ExplicitFD.nb``.  Returns a float array
    ``c`` with ``c[:, 0] = 1`` (initial condition), ``c[0, 1:] = 0`` (electrode
    boundary for t > 0) and ``c[-1, 1:] = 1`` (bulk boundary).

    Parameters
    ----------
    m : int
        Number of space points (row 0 = electrode, row m-1 = bulk).
    n : int
        Number of time points (column 0 = initial condition).

    Returns
    -------
    numpy.ndarray, shape (m, n)
    """
    c = np.ones((m, n), dtype=float)   # initial condition: c = 1 everywhere
    c[0, 1:] = 0.0                     # electrode boundary, t > 0
    c[-1, 1:] = 1.0                    # bulk boundary (already 1, explicit for clarity)
    return c
