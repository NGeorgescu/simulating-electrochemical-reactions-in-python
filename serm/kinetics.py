"""Shared electrode-kinetics and dimensionless-sweep primitives.

The finite-difference voltammetry chapters (5, 6, 10, 13, 14, 15) all build a
cyclic-voltammetry simulation out of the same handful of dimensionless
ingredients that Honeychurch develops in *Simulating Electrochemical Reactions
in Mathematica* (SERM):

* the inverse thermal voltage ``f = F /(R T)`` (the ``ScriptF`` of the source
  notebooks);
* the *number of spatial nodes* rule ``m = 1 + ceil(6 sqrt(D_M (n-1)))``;
* the *triangular sweep* potential axis (ramp down from ``upper`` to the vertex,
  then back up);
* the dimensionless standard rate constant ``ks_star = 2 ks_dim
  sqrt(T /(D_M (n-1)))``;
* the *Butler--Volmer surface elimination* factor ``tmp`` and the eliminated
  surface concentration ``c0`` (a one-sided three-point flux balance).

Each chapter previously re-inlined these identical expressions; collecting them
here removes that duplication while leaving every chapter's distinctive solver
assembly (its diagonals, boundary patches and current normalisation) in place.
The functions reproduce the source expressions exactly, so the numerical results
are unchanged.

The Butler--Volmer surface condition (SERM Ch. 6) eliminates the unknown surface
concentration ``c_{O,1}`` from the flux balance.  With ``xi = exp[(nF/RT)(E-E0)]``
the elimination factor is

.. math::
    \\mathrm{tmp} = \\frac{\\xi^{\\alpha}}{3\\xi^{\\alpha} + k_s^*(1+\\xi)},
    \\qquad
    c_{O,1} = \\bigl(k_s^*\\xi^{1-\\alpha} + 4 c_2 - c_3\\bigr)\\,\\mathrm{tmp},

which collapses to the Nernstian Dirichlet value ``xi/(1+xi)`` as
``ks_star -> inf``.
"""
from __future__ import annotations

import math

import numpy as np

# Physical constants (SI), matching serm.echem / serm.waveforms.
F = 96485.33212        # Faraday constant, C/mol
R = 8.314462618        # gas constant, J/(mol K)


def f_thermal(temperature: float = 298.15) -> float:
    """Return ``f = F /(R T)`` (1/V), the inverse thermal voltage per electron."""
    return F / (R * temperature)


def space_points_6sigma(D_M: float, n: int, x_extent: float = 6.0) -> int:
    """Number of spatial nodes ``m = 1 + ceil(x_extent sqrt(D_M (n-1)))``.

    The semi-infinite diffusion problems of SERM resolve the diffusion layer out
    to ``x_extent`` diffusion lengths; with the dimensionless space step
    ``dx = 1/sqrt(D_M (n-1))`` that needs ``x_extent sqrt(D_M (n-1))`` steps.

    Parameters
    ----------
    D_M : float
        Model (dimensionless) diffusion coefficient.
    n : int
        Number of time / potential grid points.
    x_extent : float
        Domain length in diffusion lengths (6 in the source notebooks).

    Returns
    -------
    int
    """
    return 1 + math.ceil(x_extent * math.sqrt(D_M * (n - 1)))


def ks_star_sweep(ks_dim: float, T: float, D_M: float, n: int) -> float:
    """Dimensionless standard rate constant ``2 ks_dim sqrt(T /(D_M (n-1)))``.

    The grid-scaled rate constant ``ksStar`` of SERM Chapters 6 and 15, where
    ``ks_dim`` is the dimensional standard rate constant ``k^o`` (cm/s), ``T`` the
    total dimensionless sweep length and ``n`` the number of time steps.
    """
    return 2.0 * ks_dim * np.sqrt(T / (D_M * (n - 1)))


def triangular_sweep_potential(n: int, tau: float, T: float,
                               upper_limit: float) -> np.ndarray:
    """Dimensionless potential ``nF(E - E0)/RT`` along a triangular CV sweep.

    Ramps down from ``upper_limit`` until the vertex at ``k = (n+1)/2`` and back
    up, matching the ``cv2`` mapping of SERM (forward branch
    ``upper - (k-1) tau``; reverse branch ``upper - T + (k-1) tau``, with ``k``
    1-based over ``1 .. n``).

    Parameters
    ----------
    n : int
        Number of potential steps.
    tau : float
        Dimensionless step size in units of ``RT/nF``.
    T : float
        Total dimensionless sweep length ``2(upper + |lower|)``.
    upper_limit : float
        Dimensionless potential at the sweep ends.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    k = np.arange(1, n + 1)
    forward = k <= (n + 1) / 2
    return np.where(forward,
                    upper_limit - (k - 1) * tau,
                    upper_limit - T + (k - 1) * tau)


def bv_surface_factor(xi, ks_star: float, alpha: float):
    """Butler--Volmer surface-elimination factor ``tmp``.

    ``tmp = xi**alpha / (3 xi**alpha + ks_star (1 + xi))`` (SERM Ch. 6).  As
    ``ks_star -> inf`` this gives the Nernstian Dirichlet limit.

    Parameters
    ----------
    xi : float or array_like
        Surface ratio ``exp[(nF/RT)(E - E0)]``.
    ks_star : float
        Dimensionless standard rate constant.
    alpha : float
        Transfer coefficient.
    """
    xa = xi ** alpha
    return xa / (3.0 * xa + ks_star * (1.0 + xi))


def bv_surface_conc(c1, c2, xi, ks_star: float, alpha: float, tmp=None):
    """Eliminated Butler--Volmer surface concentration ``c0``.

    ``c0 = (ks_star xi**(1-alpha) + 4 c1 - c2) tmp`` where ``c1, c2`` are the
    first two interior nodes and ``tmp`` is :func:`bv_surface_factor` (recomputed
    if not supplied).  This is the surface value eliminated from the one-sided
    three-point flux balance (SERM Ch. 6).

    Parameters
    ----------
    c1, c2 : float or array_like
        Concentration at the first and second interior nodes.
    xi : float or array_like
        Surface ratio ``exp[(nF/RT)(E - E0)]``.
    ks_star : float
        Dimensionless standard rate constant.
    alpha : float
        Transfer coefficient.
    tmp : float or array_like, optional
        Precomputed :func:`bv_surface_factor`; computed from ``xi`` if omitted.
    """
    if tmp is None:
        tmp = bv_surface_factor(xi, ks_star, alpha)
    return (ks_star * xi ** (1.0 - alpha) + 4.0 * c1 - c2) * tmp
