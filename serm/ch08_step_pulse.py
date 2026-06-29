"""Chapter 8 helpers: quasi-reversible step / pulse and staircase simulators.

Chapter 8 of *Simulating Electrochemical Reactions in Python* (after
Honeychurch's SERM) builds the whole family of potential-step and pulse
techniques -- chronoamperometry, normal-pulse, differential-pulse and
staircase voltammetry -- out of *one* implicit (backward-Euler) finite-difference
diffusion solver, by swapping the excitation waveform.  The reversible
(Nernstian) versions live inline in the chapter notebook; this module adds the
*quasi-reversible* Butler--Volmer variants and the standalone staircase machinery
so the notebooks stay short and the algorithm has a single, tested home.

Everything here reuses the validated shared infrastructure:

* the implicit surface elimination of :mod:`serm.boundary` /
  :mod:`serm.kinetics` (``bv_row_patch`` / ``bv_surface_state``), which collapses
  to the Nernstian Dirichlet value ``xi/(1+xi)`` as ``ks_star -> inf``;
* the banded tridiagonal solve of :mod:`serm.tridiagonal`.

The dimensionless conventions match the rest of the package: ``eta = nF(E-E0)/RT``
is the dimensionless overpotential, ``xi = exp(eta)`` the surface ratio,
``D_M = dtau/dX**2`` the model diffusion number, and the dimensionless current is
the one-sided three-point surface gradient ``3 c0 - 4 c1 + c2`` times a
technique-specific scale.
"""
from __future__ import annotations

import numpy as np

from .tridiagonal import tridiag_solve_banded
from .boundary import bv_row_patch, bv_surface_state, bv_dirichlet_surface

__all__ = [
    "chrono_step_bv",
    "staircase_eta",
    "staircase_simulate",
    "sample_at_beta",
]


def _space_points(D_M: float, n: int) -> int:
    """SERM rule for the number of spatial nodes ``ceil(6 sqrt(D_M (n-1)))``."""
    return int(np.ceil(6.0 * np.sqrt(D_M * (n - 1))))


def chrono_step_bv(D_M, n, xi_t, ks_star, alpha=0.5, *, m=None,
                   scale=None):
    """Implicit FD step/pulse current with a quasi-reversible BV surface.

    Generalises the Nernstian ``chrono_step`` of the chapter notebook: instead of
    imposing a Dirichlet surface concentration, the surface node is *eliminated*
    through the Butler--Volmer flux balance (:func:`serm.boundary.bv_row_patch`,
    :func:`serm.boundary.bv_surface_state`), so the same engine handles finite
    electrode kinetics.  As ``ks_star -> inf`` it reproduces the reversible
    solver to round-off.

    Parameters
    ----------
    D_M : float
        Dimensionless model diffusion number ``dtau/dX**2`` (the implicit scheme
        is unconditionally stable in it).
    n : int
        Number of time levels (``k = 0 .. n-1``).
    xi_t : array_like, shape (n,)
        Surface ratio ``xi = exp(eta)`` at each time level (the excitation
        waveform).  ``xi_t[0]`` is unused (the step is singular at ``k = 0``).
    ks_star : float
        Dimensionless (grid-scaled) standard rate constant.
    alpha : float
        Transfer coefficient.
    m : int, optional
        Number of spatial nodes; defaults to ``ceil(6 sqrt(D_M (n-1)))``.
    scale : float, optional
        Multiplier applied to the dimensionless surface gradient to form the
        current.  Defaults to the chronoamperometric ``0.5 sqrt(D_M (n-1))``.

    Returns
    -------
    chi : numpy.ndarray, shape (n,)
        Dimensionless current ``(3 c0 - 4 c1 + c2) * scale``; ``chi[0] = nan``.
    c_final : numpy.ndarray, shape (m,)
        Concentration profile of O at the last time level.
    """
    xi_t = np.asarray(xi_t, dtype=float)
    if m is None:
        m = _space_points(D_M, n)
    if scale is None:
        scale = 0.5 * np.sqrt(D_M * (n - 1))

    c = np.ones(m)                              # bulk everywhere at k=0
    chi = np.empty(n)
    chi[0] = np.nan

    M = m - 2
    sub = np.full(M - 1, -D_M)
    base_diag = 1.0 + 2.0 * D_M
    base_sup = -D_M
    diag = np.full(M, base_diag)
    sup = np.full(M - 1, base_sup)

    for k in range(1, n):
        xi = float(xi_t[k])
        # Inject the BV surface elimination into the first interior row.
        dd, sd, rd = bv_row_patch(xi, ks_star, alpha, D_M)
        diag[0] = base_diag + dd
        sup[0] = base_sup + sd
        b = c[1:-1].copy()
        b[0] += rd
        b[-1] += D_M * c[-1]                     # known bulk node (=1)
        c[1:-1] = tridiag_solve_banded(sub, diag, sup, b)
        c0, _ = bv_surface_state(c[1], c[2], xi, ks_star, alpha)
        c[0] = c0
        chi[k] = (3.0 * c[0] - 4.0 * c[1] + c[2]) * scale

    return chi, c


def staircase_eta(n, upper, dEs_dimless, tN):
    """Dimensionless staircase overpotential ``eta = f(E - E0)``.

    The staircase holds a flat potential for ``tN`` time levels, then steps down
    by ``dEs_dimless``.  This is the SERM ``upper - dEs ((k - Mod[k, tN])/tN)``
    waveform written with numpy's ``%``.

    Parameters
    ----------
    n : int
        Number of time levels.
    upper : float
        Dimensionless start overpotential ``f(E_start - E0)``.
    dEs_dimless : float
        Dimensionless staircase step height ``f * dEs`` (``dEs`` the step in V).
    tN : int
        Number of time levels per staircase step.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    k = np.arange(n)
    return upper - dEs_dimless * ((k - (k % tN)) / tN)


def staircase_simulate(D_M, upper, dEs_dimless, tN, cycles, *,
                       ks_star=1e9, alpha=0.5, m=None):
    """Simulate a staircase voltammogram; return ``(eta, chi)`` over all levels.

    A backward sawtooth (the staircase) drives the quasi-reversible step solver.
    The dimensionless current is normalised the SERM way for a staircase of total
    range ``T = cycles * dEs_dimless``::

        chi = (3 c0 - 4 c1 + c2) * sqrt(D_M (n-1) / (2 T)),

    which makes ``chi`` a sweep-rate-like current that can be sampled at any
    fraction of the step (see :func:`sample_at_beta`).

    Parameters
    ----------
    D_M : float
        Model diffusion number.
    upper : float
        Dimensionless start overpotential.
    dEs_dimless : float
        Dimensionless staircase step height.
    tN : int
        Time levels per step.
    cycles : int
        Number of staircase steps.
    ks_star : float
        Dimensionless rate constant (``1e9`` -> effectively reversible).
    alpha : float
        Transfer coefficient.
    m : int, optional
        Spatial nodes.

    Returns
    -------
    eta : numpy.ndarray, shape (n,)
        Dimensionless overpotential at every time level (``n = cycles * tN``).
    chi : numpy.ndarray, shape (n,)
        Dimensionless current at every time level; ``chi[0] = nan``.
    """
    n = cycles * tN
    eta = staircase_eta(n, upper, dEs_dimless, tN)
    xi_t = np.exp(eta)
    T = cycles * dEs_dimless
    scale = np.sqrt(D_M * (n - 1) / (2.0 * T))
    chi, _ = chrono_step_bv(D_M, n, xi_t, ks_star, alpha, m=m, scale=scale)
    return eta, chi


def sample_at_beta(eta, chi, tN, beta):
    """Sample the staircase current at a fraction ``beta`` of each step.

    Staircase voltammetry samples one current per step.  Sampling at the *end* of
    the step (``beta = 1``) gives an analogue-like wave only as ``dEs -> 0``;
    sampling earlier in the step (the chapter finds ``beta ~ 0.33--0.35`` for
    reversible couples) recovers a peak position matching analogue linear-sweep
    voltammetry even at finite step size.

    Parameters
    ----------
    eta, chi : numpy.ndarray
        Per-level overpotential and current from :func:`staircase_simulate`.
    tN : int
        Time levels per step.
    beta : float
        Sampling fraction in ``(0, 1]``.

    Returns
    -------
    eta_s : numpy.ndarray
        Overpotential at the sampled points.
    chi_s : numpy.ndarray
        Sampled current.
    """
    idx0 = max(1, int(round(beta * tN)) - 1)
    idx = np.arange(idx0, len(chi), tN)
    idx = idx[idx < len(chi)]
    return eta[idx], chi[idx]
