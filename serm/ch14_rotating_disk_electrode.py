"""Rotating disk electrode (RDE) finite-difference solvers.

Chapter-specific helpers for Chapter 14 of *Simulating Electrochemical
Reactions in Python*, adapted from Michael Honeychurch's *Simulating
Electrochemical Reactions in Mathematica* (SERM), chapter "Rotating disk
electrode voltammetry" and the companion notebooks ``ImplicitRDESS.nb``
(steady state) and ``ImplicitRDE.nb`` (non-steady state).

The physics
-----------
At a rotating disk the only surviving transport coordinate is ``z`` (normal to
the disk).  The one-dimensional convection--diffusion equation is

.. math::
    \\frac{\\partial c}{\\partial t}
        = D\\,\\frac{\\partial^2 c}{\\partial z^2}
          - v_z(z)\\,\\frac{\\partial c}{\\partial z},

with the first term of the von Karman / Cochran velocity series

.. math::
    v_z(z) \\approx -0.510\\,\\omega^{3/2}\\nu^{-1/2}\\,z^2 .

Distance is scaled by the Levich diffusion-layer thickness
:math:`\\delta = 1.61\\,D^{1/3}\\nu^{1/6}\\omega^{-1/2}` (``Z = z/\\delta``).
With this scaling the dimensionless convection coefficient becomes the constant

.. math::
    0.510 \\times 1.61^3 \\approx 2.13496,

so that ``v_z\\,\\delta/D = -2.13496\\,Z^2`` and the limiting (mass-transport)
plateau of the dimensionless current is exactly ``-1`` -- which is precisely the
Levich current once the dimensional prefactor ``nFADc^*/\\delta`` is restored.
This is the constant Honeychurch hard-codes as ``2.13496`` in ``makeDiagonals``.

The surface boundary condition is Butler--Volmer with a dimensionless
heterogeneous rate ``ksStar`` and ``xi = exp[f(E-E^0)]``.  The current is read
off from a three-point one-sided derivative at the electrode,
``chi = (3 c_0 - 4 c_1 + c_2)(m-1)/(2 z_max)``.
"""
from __future__ import annotations

import numpy as np

from .tridiagonal import tridiag_solve_banded
from .kinetics import bv_surface_factor, bv_surface_conc

# Dimensionless convection coefficient  0.51023 * 1.61**3  ~ 2.13496
# (von Karman leading velocity coefficient times the cube of the Levich
# diffusion-layer prefactor).  Honeychurch hard-codes 2.13496.
CONV = 2.13496


def _interior_beta(m: int, z_max: float) -> np.ndarray:
    """Convection coefficient ``beta_j`` at interior nodes ``j = 2 .. m-1``.

    ``beta_j = CONV * z_max^3 * (j-1)^2 / (2 (m-1)^3)`` -- the discretised
    ``v_z delta / D`` weighting (the ``z^2`` dependence appears as ``(j-1)^2``).
    """
    j = np.arange(2, m)  # 1-based interior indices, length m-2
    return CONV * z_max ** 3 * (j - 1) ** 2 / (2.0 * (m - 1) ** 3)


def steady_state_voltammogram(
    e_grid: np.ndarray,
    *,
    m: int = 200,
    z_max: float = 2.0,
    ks_star: float = 2000.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Steady-state RDE voltammogram (dimensionless current ``chi``).

    Port of ``solveRDESS`` from ``ImplicitRDESS.nb``.  At steady state the time
    derivative vanishes and the discrete operator is ``c'' - 2.13496 Z^2 c' = 0``
    with diagonals ``x = 1 - beta``, ``y = -2``, ``z = 1 + beta``.  Each potential
    is solved independently (no time stepping is required).

    Parameters
    ----------
    e_grid : ndarray
        Dimensionless potentials ``f(E - E^0)`` at which to evaluate the
        voltammogram (typically swept from positive to negative for a reduction).
    m : int
        Number of spatial nodes.
    z_max : float
        Outer edge of the (dimensionless) domain, in units of ``delta``.
    ks_star : float
        Dimensionless heterogeneous rate constant.  Large values give a
        reversible (Nernstian) wave.
    alpha : float
        Transfer coefficient.

    Returns
    -------
    ndarray
        Dimensionless current ``chi`` at each potential.  The mass-transport
        plateau approaches ``-1`` for a reduction.
    """
    beta = _interior_beta(m, z_max)
    x = 1.0 - beta            # sub-diagonal template, length m-2
    y = np.full(m - 2, -2.0)  # main diagonal
    z_sup = 1.0 + beta        # super-diagonal template, length m-2
    x1, y1, z1 = x[0], y[0], z_sup[0]

    jM = m - 1
    z_outer = 1.0 + CONV * z_max ** 3 * (jM - 1) ** 2 / (2.0 * (m - 1) ** 3)

    chi = np.empty(e_grid.shape)
    for k, e in enumerate(e_grid):
        xi = np.exp(e)
        tmp = bv_surface_factor(xi, ks_star, alpha)

        b = np.zeros(m - 2)
        b[0] -= tmp * x1 * ks_star * xi ** (1.0 - alpha)
        b[-1] -= z_outer

        yy = y.copy()
        zz = z_sup.copy()
        yy[0] = y1 + 4.0 * x1 * tmp
        zz[0] = z1 - x1 * tmp

        sol = tridiag_solve_banded(x[1:], yy, zz[:-1], b)
        c0 = bv_surface_conc(sol[0], sol[1], xi, ks_star, alpha, tmp)
        chi[k] = (3 * c0 - 4 * sol[0] + sol[1]) * (m - 1) / (2.0 * z_max)
    return chi


def steady_state_chi(e: float, **kwargs) -> float:
    """Dimensionless current ``chi`` at a single dimensionless potential ``e``.

    Convenience wrapper around :func:`steady_state_voltammogram` for one point;
    used by the Koutecky--Levich analysis, which probes a fixed potential while
    the rotation rate (and hence ``ks_star``) is varied.
    """
    return float(steady_state_voltammogram(np.array([e]), **kwargs)[0])


def levich_delta(D: float, nu: float, omega: float) -> float:
    """Levich diffusion-layer thickness ``delta = 1.61 D^{1/3} nu^{1/6} omega^{-1/2}``.

    Parameters are in CGS-consistent units (``D``, ``nu`` in cm^2/s, ``omega``
    in rad/s); ``delta`` is returned in cm.
    """
    return 1.61 * D ** (1.0 / 3.0) * nu ** (1.0 / 6.0) * omega ** (-0.5)


def non_steady_state_voltammogram(
    *,
    n_steps: int = 400,
    m: int | None = None,
    z_max: float = 2.0,
    DM: float = 10.0,
    Delta: float = 0.3,
    ks_dim: float = 20.0,
    alpha: float = 0.5,
    upper_limit: float = 10.0,
    lower_limit: float = -10.0,
):
    """Non-steady-state (cyclic) RDE voltammogram via a fully implicit scheme.

    Port of ``solveRDE`` from ``ImplicitRDE.nb``.  Time is retained, so the
    diagonals carry the mesh ratio ``DM``: ``x = -DM + beta'``, ``y = 1 + 2 DM``,
    ``z = -DM - beta'`` with ``beta' = DM * beta``.  The potential is swept down
    and back up (a cyclic voltammogram); the sweep rate enters through
    ``Delta = sigma delta^2 / D``, the ratio of sweep rate to rotation rate.

    Returns
    -------
    e_axis : ndarray
        Dimensionless potential ``f(E - E^0)`` at each step.
    chi : ndarray
        Dimensionless current at each step.
    """
    big_T = 2.0 * (upper_limit + abs(lower_limit))
    tau = big_T / (n_steps - 1)
    if m is None:
        m = 1 + int(np.ceil(z_max * np.sqrt((Delta * DM) / tau)))
    ks_star = 2.0 * ks_dim * z_max / (m - 1)

    j = np.arange(2, m)
    beta = CONV * z_max ** 3 * DM * (j - 1) ** 2 / (2.0 * (m - 1) ** 3)
    x = -DM + beta
    y = np.full(m - 2, 1.0 + 2.0 * DM)
    z_sup = -DM - beta
    x1, y1, z1 = x[0], y[0], z_sup[0]

    jM = m - 1
    z_outer = -DM - CONV * z_max ** 3 * DM * (jM - 1) ** 2 / (2.0 * (m - 1) ** 3)

    c = np.ones(m)  # bulk initial condition
    chi = np.empty(n_steps)
    e_axis = np.empty(n_steps)
    half = (n_steps + 1) / 2.0
    for k in range(1, n_steps + 1):
        # forward (reductive) sweep then reverse, matching the Wolfram If[...]
        if k > half:
            e = upper_limit - big_T + tau * (k - 1)
        else:
            e = upper_limit - tau * (k - 1)
        xi = np.exp(e)
        tmp = bv_surface_factor(xi, ks_star, alpha)

        b = c[1:m - 1].copy()
        b[0] -= tmp * x1 * ks_star * xi ** (1.0 - alpha)
        b[-1] -= z_outer

        yy = y.copy()
        zz = z_sup.copy()
        yy[0] = y1 + 4.0 * x1 * tmp
        zz[0] = z1 - x1 * tmp

        sol = tridiag_solve_banded(x[1:], yy, zz[:-1], b)
        c0 = bv_surface_conc(sol[0], sol[1], xi, ks_star, alpha, tmp)
        c = np.concatenate(([c0], sol, [1.0]))
        chi[k - 1] = (
            (3 * c[0] - 4 * c[1] + c[2]) / np.sqrt(Delta) * (m - 1) / (2.0 * z_max)
        )
        e_axis[k - 1] = e
    return e_axis, chi
