"""Chapter 5 helpers: reversible cyclic voltammetry by finite differences.

This module re-implements, in vectorised numpy, the explicit and implicit
finite-difference simulators for a reversible (Nernstian) couple
``O + n e- <=> R`` that Honeychurch develops in *Simulating Electrochemical
Reactions in Mathematica* (SERM), Chapter 5 and the accompanying notebooks
``ExplicitCVRev.nb`` and ``ImplicitCVRev.nb``.  It also provides the
semi-analytic Volterra-integral reference (``CVNumerical.nb``) used to validate
the simulators independently.

The whole problem is cast in dimensionless variables.  Distance is scaled by a
diffusion length, time/potential by ``sigma = n F v /(R T)``, and concentration
by the bulk value of O.  The applied potential enters only through the
Nernstian surface ratio

.. math::
    \\theta = \\exp\\!\\big[\\tfrac{nF}{RT}(E - E^{0})\\big],
    \\qquad c_O(0,\\tau) = \\frac{1}{1+\\theta^{-1}}
    = \\frac{\\xi}{1+\\xi}, \\quad \\xi = \\theta .

Honeychurch parameterises the sweep by a dimensionless potential window
``[lower_limit, upper_limit]`` measured in units of ``RT/nF`` about the formal
potential.  With ``xi = exp(upper_limit - (k-1) tau)`` on the forward
(cathodic) sweep and ``xi = exp(upper_limit - T + (k-1) tau)`` on the reverse
sweep (``T = 2(upper_limit + |lower_limit|)``), the surface boundary value is
``xi/(1+xi)`` -- exactly the code fragment quoted in chapter5.nb.

Nothing here is a line-by-line transliteration of Wolfram: the time loop is a
vectorised stencil application and the implicit step calls
:func:`serm.tridiagonal.tridiag_solve_banded`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tridiagonal import tridiag_solve_banded
from .kinetics import space_points_6sigma, triangular_sweep_potential


@dataclass
class CVGrid:
    """Container for a dimensionless CV simulation grid and its parameters.

    Attributes
    ----------
    c : numpy.ndarray, shape (m, n)
        Dimensionless concentration of O.  Row 0 is the electrode surface,
        row ``m-1`` the bulk; column ``k`` is the ``k``-th potential step.
    n : int
        Number of potential steps (odd, so the vertex falls on a grid point).
    m : int
        Number of spatial points.
    D_M : float
        Model diffusion coefficient ``D_M = D dt / dx**2`` (``<= 0.5`` for
        explicit stability).
    tau : float
        Dimensionless step size ``T/(n-1)`` in units of ``RT/nF``.
    T : float
        Total dimensionless sweep length ``2(upper_limit + |lower_limit|)``.
    lower_limit, upper_limit : float
        Dimensionless potential limits (units ``RT/nF`` about ``E0``).
    """

    c: np.ndarray
    n: int
    m: int
    D_M: float
    tau: float
    T: float
    lower_limit: float
    upper_limit: float


def space_points(D_M: float, n: int, x_extent: float = 6.0) -> int:
    """``m = 1 + ceil(x_extent * sqrt(D_M (n-1)))`` (SERM ``ExplicitCVRev.nb``).

    The diffusion layer is resolved out to ``x_extent`` diffusion lengths,
    which is effectively semi-infinite over a CV experiment.  Thin wrapper over
    :func:`serm.kinetics.space_points_6sigma`.
    """
    return space_points_6sigma(D_M, n, x_extent)


def surface_ratio(k: np.ndarray, tau: float, T: float, upper_limit: float) -> np.ndarray:
    """Nernstian surface fraction ``xi/(1+xi)`` of O at potential step ``k``.

    ``xi = exp(upper_limit - (k-1) tau)`` on the forward sweep (``k`` up to the
    vertex) and ``xi = exp(upper_limit - T + (k-1) tau)`` on the reverse sweep,
    matching the ``xi[y]`` definitions in ``ExplicitCVRev.nb``.  Vectorised over
    an array of step indices ``k`` (1-based, as in the Wolfram code).
    """
    k = np.asarray(k, dtype=float)
    n_steps = int(round(T / tau)) + 1
    vertex = (n_steps + 1) / 2
    forward = k <= vertex
    exponent = np.where(
        forward,
        upper_limit - (k - 1) * tau,
        upper_limit - T + (k - 1) * tau,
    )
    xi = np.exp(exponent)
    return xi / (1.0 + xi)


def make_cv_grid(D_M: float, n: int, lower_limit: float = 8.0,
                 upper_limit: float = 8.0) -> CVGrid:
    """Allocate and initialise the dimensionless CV grid.

    The initial condition is ``c_O = 1`` everywhere (only O present in bulk).
    Column 0 is the initial state; the surface value at column 0 is overwritten
    with its Nernstian value so the first stencil step sees a consistent
    boundary.

    Parameters
    ----------
    D_M : float
        Model diffusion coefficient.
    n : int
        Number of potential steps; forced odd so the switching potential lands
        on a grid point (``(n+1)/2``).
    lower_limit, upper_limit : float
        Dimensionless potential limits about ``E0`` (units ``RT/nF``).
    """
    if n % 2 == 0:
        n += 1
    T = 2.0 * (upper_limit + abs(lower_limit))
    tau = T / (n - 1)
    m = space_points(D_M, n)
    c = np.ones((m, n), dtype=float)
    c[0, 0] = surface_ratio(np.array([1]), tau, T, upper_limit)[0]
    return CVGrid(c=c, n=n, m=m, D_M=D_M, tau=tau, T=T,
                  lower_limit=lower_limit, upper_limit=upper_limit)


def explicit_cv(grid: CVGrid) -> CVGrid:
    """Solve the reversible CV by the explicit (forward-difference) scheme.

    Vectorised port of ``explicitCV`` from ``ExplicitCVRev.nb``: at each step
    the interior points use the three-point stencil
    ``c_j = D_M c_{j-1} + (1 - 2 D_M) c_j + D_M c_{j+1}`` (applied over the whole
    interior at once), the surface is pinned to its Nernstian value and the
    outer node is held at the bulk value 1.  Requires ``D_M <= 0.5`` for
    stability.
    """
    c = grid.c
    D_M = grid.D_M
    k_all = np.arange(1, grid.n + 1)
    surf = surface_ratio(k_all, grid.tau, grid.T, grid.upper_limit)
    for k in range(1, grid.n):
        prev = c[:, k - 1]
        c[1:-1, k] = (D_M * prev[:-2]
                      + (1.0 - 2.0 * D_M) * prev[1:-1]
                      + D_M * prev[2:])
        c[0, k] = surf[k]          # 0-based column k == (k+1)-th step
        c[-1, k] = 1.0
    return grid


def implicit_cv(grid: CVGrid) -> CVGrid:
    """Solve the reversible CV by the fully implicit (backward-Euler) scheme.

    Port of ``solveCV`` from ``ImplicitCVRev.nb``.  Each interior node satisfies
    ``-D_M c_{j-1} + (1 + 2 D_M) c_j - D_M c_{j+1} = c_j^{old}``; the known
    surface and bulk values are folded into the right-hand side and the
    tridiagonal system is solved with the pivoting banded solver from
    :mod:`serm.tridiagonal`.  Unconditionally stable, so ``D_M`` may exceed
    ``0.5``.
    """
    c = grid.c
    D_M = grid.D_M
    m = grid.m
    n_int = m - 2                          # interior unknowns
    sub = np.full(n_int - 1, -D_M)
    sup = np.full(n_int - 1, -D_M)
    diag = np.full(n_int, 1.0 + 2.0 * D_M)
    k_all = np.arange(1, grid.n + 1)
    surf = surface_ratio(k_all, grid.tau, grid.T, grid.upper_limit)
    for k in range(1, grid.n):
        b = c[1:-1, k - 1].copy()
        b[0] += D_M * surf[k]              # surface contribution
        b[-1] += D_M * 1.0                 # bulk contribution
        c[1:-1, k] = tridiag_solve_banded(sub, diag, sup, b)
        c[0, k] = surf[k]
        c[-1, k] = 1.0
    return grid


def dimensionless_current(grid: CVGrid) -> np.ndarray:
    """Dimensionless current ``sqrt(pi) * chi`` at each potential step.

    Uses Honeychurch's three-point one-sided surface gradient

    .. math::
        \\sqrt{\\pi}\\,\\chi = (3 c_0 - 4 c_1 + c_2)\\,
        \\frac{\\sqrt{D_M (n-1)}}{\\sqrt{4 T}},

    i.e. ``cv1 = Map[(3 c0 - 4 c1 + c2) Sqrt[DM (n-1)]/Sqrt[4 T] ...]`` from
    ``ExplicitCVRev.nb``.  A positive value is a cathodic (reduction) current.
    """
    c = grid.c
    grad = 3.0 * c[0, :] - 4.0 * c[1, :] + c[2, :]
    return grad * math.sqrt(grid.D_M * (grid.n - 1)) / math.sqrt(4.0 * grid.T)


def potential_axis(grid: CVGrid) -> np.ndarray:
    """Dimensionless potential ``nF(E - E0)/RT`` at each step.

    Forward sweep: ``upper_limit - (k-1) tau``; reverse sweep:
    ``upper_limit - T + (k-1) tau`` (the ``cv2`` mapping in ``ExplicitCVRev.nb``).
    """
    return triangular_sweep_potential(grid.n, grid.tau, grid.T, grid.upper_limit)


def volterra_lsv(sigma_t: np.ndarray, init: float = 10.0) -> np.ndarray:
    """Semi-analytic dimensionless LSV current from the Volterra solution.

    Honeychurch's ``CVNumerical.nb`` evaluates the reversible linear-sweep
    voltammogram as

    .. math::
        \\sqrt{\\pi}\\,\\chi(\\sigma t) =
        -\\frac{1}{2\\sqrt{\\pi}} \\int_0^{\\sigma t}
        \\sqrt{\\sigma t - z}\\;
        \\frac{\\tanh[(\\mathrm{init}-z)/2]}{\\cosh^2[(\\mathrm{init}-z)/2]}\\,dz ,

    where ``init = ln(theta_initial)`` is the (large) starting dimensionless
    potential.  The integrand's only mild feature is the square-root weight at
    the upper limit; the integral is evaluated here by adaptive quadrature.

    Returns the cathodic-sweep current ``sqrt(pi) chi`` (sign flipped so that a
    reduction peak is positive, matching :func:`dimensionless_current`).

    Parameters
    ----------
    sigma_t : array_like
        Dimensionless time/potential coordinate ``sigma t >= 0``.
    init : float
        Initial dimensionless potential ``ln(theta)``; 10 corresponds to
        ``theta ~ 2.2e4`` (essentially all O at the start).
    """
    from scipy.integrate import quad

    def integrand(z, st):
        a = (init - z) / 2.0
        return math.sqrt(st - z) * math.tanh(a) / math.cosh(a) ** 2

    out = np.empty_like(np.asarray(sigma_t, dtype=float))
    for idx, st in np.ndenumerate(np.asarray(sigma_t, dtype=float)):
        if st <= 0.0:
            out[idx] = 0.0
            continue
        val, _ = quad(integrand, 0.0, st, args=(st,), limit=200)
        out[idx] = val / (2.0 * math.sqrt(math.pi))
    return out
