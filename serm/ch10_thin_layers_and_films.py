"""Fully implicit finite-difference solvers for thin-layer and thin-film
cyclic voltammetry (SERM Chapter 10).

The diffusion space is *finite*: the electroactive species is confined to a
gap of thickness ``L`` and the spatial coordinate is scaled by ``L`` so that
``x in [0, 1]``.  Two geometries are treated:

* **Thin film** -- one electrode at ``x = 0`` and an *impermeable barrier*
  (zero-flux) at ``x = 1``.
* **Thin layer** -- *two* electrodes, one at ``x = 0`` and one at ``x = 1``,
  driven by the same potential, with identical reaction at each face.

Both use the fully implicit (backward-Euler) scheme of SERM Chapter 6, solved
with a tridiagonal system each time step.  This is a re-implementation of the
algorithm in ``ImplicitTFV.nb`` / ``ImplicitTLV.nb`` in idiomatic numpy; the
tridiagonal solve is delegated to :func:`serm.tridiagonal.tridiag_solve_banded`.

Symbols (matching the source notebooks):

``L_param``
    Dimensionless diffusion parameter ``L^2 sigma / D`` (the Wolfram
    ``DoubleStruckCapitalL``), where ``sigma = n F v / (R T)`` is the
    dimensionless sweep rate.  Small ``L_param`` => slow sweep / true
    thin-layer limit; large ``L_param`` => fast sweep / semi-infinite-like.
``DM``
    Model diffusion coefficient ``T / (L_param * dx^2 * (n - 1))``.  Because
    the implicit scheme is unconditionally stable, ``DM`` may be large.
``ks_dim``
    Dimensionless heterogeneous rate constant.  Large values approach the
    reversible (Nernstian) limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tridiagonal import tridiag_solve_banded
from .kinetics import bv_surface_factor, bv_surface_conc, triangular_sweep_potential

# Chapter 10 uses the source notebook's rounded constants (these set the number
# of sweep steps via mV_step * f); kept as-is to reproduce ImplicitTLV/TFV.nb.
F = 96485.0      # Faraday constant, C/mol
R = 8.3144       # gas constant, J/(mol K)


@dataclass
class CVResult:
    """Result of a thin-layer / thin-film CV simulation.

    Attributes
    ----------
    potential : numpy.ndarray
        Dimensionless potential axis ``nF (E - E0) / RT`` for each step.
    current : numpy.ndarray
        Dimensionless current ``chi = i / (n F A L c* sigma)``.
    x : numpy.ndarray
        Dimensionless distance grid, ``x in [0, 1]``.
    conc : numpy.ndarray
        Concentration field, shape ``(n_steps, m)`` (dimensionless ``c / c*``).
    DM : float
        Model diffusion coefficient actually used.
    m : int
        Number of spatial nodes.
    """

    potential: np.ndarray
    current: np.ndarray
    x: np.ndarray
    conc: np.ndarray
    DM: float
    m: int


def _sweep_setup(temperature, upper, lower, mV_step):
    """Common dimensionless sweep bookkeeping shared by both geometries."""
    f = F / (R * temperature)
    total = 2.0 * (upper + abs(lower))          # forward + reverse duration
    n = int(round(total / (mV_step * 1e-3 * f)))
    tau = total / (n - 1)
    return f, total, n, tau


def _potential_axis(n, tau, total, upper):
    """Dimensionless potential nF(E-E0)/RT at each step index k = 1..n.

    Delegates to the shared :func:`serm.kinetics.triangular_sweep_potential`
    used by the other voltammetry chapters, giving a symmetric triangular sweep
    (ramp down from ``upper`` to the vertex at ``k = (n+1)/2`` and back up).
    """
    return triangular_sweep_potential(n, tau, total, upper)


def simulate_thin_layer(
    L_param: float = 10.0,
    ks_dim: float = 1.0e4,
    dx: float = 0.01,
    alpha: float = 0.5,
    upper: float = 10.0,
    lower: float = -10.0,
    mV_step: float = 1.0,
    temperature: float = 298.0,
) -> CVResult:
    """Simulate a thin-layer cyclic voltammogram (reaction at both faces).

    Re-implementation of ``ImplicitTLV.nb``.  The boundary condition at each of
    the two electrodes (``x = 0`` and ``x = 1``) is a one-sided three-point
    Butler--Volmer flux condition; in the large-``ks_dim`` limit it becomes
    Nernstian (reversible).

    Returns
    -------
    CVResult
    """
    f, total, n, tau = _sweep_setup(temperature, upper, lower, mV_step)
    DM = total / (L_param * dx * dx * (n - 1))
    m = 1 + int(np.ceil(1.0 / dx))
    ks_star = 2.0 * ks_dim * L_param * dx

    N = m - 2                       # interior unknowns (nodes 1 .. m-2)
    y1 = 1.0 + 2.0 * DM
    conc = np.ones(m)
    surf3 = np.empty((n, 3))        # store first three nodes for current calc
    full = np.empty((n, m))

    pot = _potential_axis(n, tau, total, upper)
    xi_all = np.exp(pot)

    for k in range(n):
        xi = xi_all[k]
        tmp = bv_surface_factor(xi, ks_star, alpha)
        drive = tmp * DM * ks_star * xi ** (1.0 - alpha)

        sub = np.full(N - 1, -DM)
        sup = np.full(N - 1, -DM)
        diag = np.full(N, 1.0 + 2.0 * DM)
        b = conc[1:m - 1].copy()
        b[0] += drive
        b[-1] += drive
        diag[0] = y1 - 4.0 * DM * tmp
        diag[-1] = y1 - 4.0 * DM * tmp
        sub[-1] = -DM + DM * tmp
        sup[0] = -DM + DM * tmp

        interior = tridiag_solve_banded(sub, diag, sup, b)
        c0 = bv_surface_conc(interior[0], interior[1], xi, ks_star, alpha, tmp)
        cL = bv_surface_conc(interior[-1], interior[-2], xi, ks_star, alpha, tmp)
        conc = np.concatenate(([c0], interior, [cL]))
        full[k] = conc
        surf3[k] = conc[:3]

    # Dimensionless current chi = (-c2 + 4 c1 - 3 c0) / (L_param * dx).
    current = (-surf3[:, 2] + 4.0 * surf3[:, 1] - 3.0 * surf3[:, 0]) / (L_param * dx)
    x = np.linspace(0.0, 1.0, m)
    return CVResult(pot, current, x, full, DM, m)


def simulate_thin_film(
    L_param: float = 10.0,
    ks_dim: float = 1.0e4,
    dx: float = 0.01,
    alpha: float = 0.5,
    upper: float = 10.0,
    lower: float = -10.0,
    mV_step: float = 1.0,
    temperature: float = 298.0,
) -> CVResult:
    """Simulate a thin-film cyclic voltammogram (one electrode, zero-flux wall).

    Re-implementation of ``ImplicitTFV.nb``.  The electrode is at ``x = 0``; the
    barrier at ``x = 1`` is impermeable, so the three-point one-sided derivative
    is set to zero there, which modifies the last entries of the main and
    sub-diagonals (``1 + (2/3) DM`` and ``-(2/3) DM``) and gives the wall node
    ``c_m = (4 c_{m-1} - c_{m-2}) / 3``.

    Returns
    -------
    CVResult
    """
    f, total, n, tau = _sweep_setup(temperature, upper, lower, mV_step)
    DM = total / (L_param * dx * dx * (n - 1))
    m = 1 + int(np.ceil(1.0 / dx))
    ks_star = 2.0 * ks_dim * L_param * dx

    N = m - 2
    y1 = 1.0 + 2.0 * DM
    conc = np.ones(m)
    surf3 = np.empty((n, 3))
    full = np.empty((n, m))

    pot = _potential_axis(n, tau, total, upper)
    xi_all = np.exp(pot)

    for k in range(n):
        xi = xi_all[k]
        tmp = bv_surface_factor(xi, ks_star, alpha)
        drive = tmp * DM * ks_star * xi ** (1.0 - alpha)

        sub = np.full(N - 1, -DM)
        sup = np.full(N - 1, -DM)
        diag = np.full(N, 1.0 + 2.0 * DM)
        # Impermeable wall at x = 1: zero-flux three-point closure.
        diag[-1] = 1.0 + (2.0 / 3.0) * DM
        sub[-1] = -(2.0 / 3.0) * DM

        b = conc[1:m - 1].copy()
        b[0] += drive
        # electrode boundary only at x = 0:
        diag[0] = y1 - 4.0 * DM * tmp
        sup[0] = -DM + DM * tmp

        interior = tridiag_solve_banded(sub, diag, sup, b)
        c0 = bv_surface_conc(interior[0], interior[1], xi, ks_star, alpha, tmp)
        cwall = (4.0 * interior[-1] - interior[-2]) / 3.0
        conc = np.concatenate(([c0], interior, [cwall]))
        full[k] = conc
        surf3[k] = conc[:3]

    # Note the factor 2 in the film current normalisation (single electrode).
    current = (-surf3[:, 2] + 4.0 * surf3[:, 1] - 3.0 * surf3[:, 0]) / (2.0 * L_param * dx)
    x = np.linspace(0.0, 1.0, m)
    return CVResult(pot, current, x, full, DM, m)


def forward_sweep_charge(result: CVResult) -> float:
    """Integrate ``chi`` over the forward (reduction) sweep.

    For a reversible thin-layer cell this equals 1 in dimensionless units,
    i.e. the charge passed equals ``n F A L c*`` (exhaustive electrolysis of
    everything in the gap).  Returned as a signed value.
    """
    n = result.current.shape[0]
    half = n // 2
    pot = result.potential[:half]
    cur = result.current[:half]
    return float(np.trapezoid(cur, pot))
