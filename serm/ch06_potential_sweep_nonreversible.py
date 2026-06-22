"""Chapter 6 helper: implicit finite-difference CV for non-reversible kinetics.

This module re-implements, in vectorised Python, the fully-implicit
finite-difference cyclic-voltammetry simulator of Honeychurch's *Simulating
Electrochemical Reactions in Mathematica* (SERM), Chapter 6 ("Potential sweep
methods -- non-reversible reactions") and the companion notebook
``Extra Notebooks/chapter6/ImplicitCVQuasi.nb``.

The electron transfer ``O + n e- <=> R`` obeys Butler--Volmer kinetics, so the
electrode is no longer at Nernstian equilibrium; the heterogeneous rate constant
``ks`` (entering through the dimensionless ``ksDim``) controls how reversible the
wave looks.  Three regimes emerge from a single solver:

* ``ksDim`` large  -> the Nernstian (reversible) wave of Chapter 5;
* ``ksDim`` small  -> the totally irreversible wave (peak shifts with scan rate);
* in between        -> the quasi-reversible wave with kinetic broadening.

Dimensionless scheme (as in SERM Ch. 5--6)
------------------------------------------
Time/potential are measured by the dimensionless sweep variable
``p = (nF/RT)(E - E0) = f (E - E0)``.  Because ``p`` advances linearly in time,
``p`` doubles as a dimensionless time ``z = sigma t`` with ``sigma = f v``.
Space is scaled as ``X = x sqrt(sigma / D)``.  The model diffusion number is
``DM = dtau / dX**2`` so a chosen ``DM`` fixes the spatial step
``dX = sqrt(dtau / DM)`` for a given dimensionless time step ``dtau``.

The dimensionless current returned is the surface gradient

.. math::
    \\chi_\\mathrm{grad} = \\left.\\frac{\\partial c_O}{\\partial X}\\right|_{X=0},

which relates to the laboratory current by
``i = n F A c^* \\sqrt{D \\sigma}\\, \\chi_\\mathrm{grad}``.  For a reduction the
forward (cathodic) branch is negative; the reversible cathodic peak of
``-\\chi_\\mathrm{grad}`` equals the Nicholson--Shain value ``0.4463``.

Butler--Volmer surface boundary condition
-----------------------------------------
With ``xi = exp(p)`` and a three-point one-sided surface derivative, eliminating
the unknown surface concentration ``c_{O,1}^{k+1}`` from the flux balance
(SERM Ch. 6, the ``Solve`` for ``c[O1]``) gives the factor

.. math::
    \\mathrm{tmp} = \\frac{\\xi^{\\alpha}}{3\\xi^{\\alpha} + k_s^*(1+\\xi)},
    \\qquad
    c_{O,1} = \\bigl(k_s^*\\xi^{1-\\alpha} + 4 c_2 - c_3\\bigr)\\,\\mathrm{tmp},

where ``ks*`` is the grid-scaled rate constant.  As ``ks* -> inf`` this collapses
to the Nernstian Dirichlet value ``c_{O,1} = xi/(1+xi)`` and Chapter 5 is
recovered; the first tridiagonal row is patched accordingly each step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .kinetics import (
    F, R, f_thermal, ks_star_sweep, bv_surface_factor, bv_surface_conc,
)

__all__ = [
    "F", "R", "f_thermal", "ks_dimensionless", "CVResult",
    "simulate_cv", "simulate_reversible_cv", "cottrell_surface_flux",
]


def ks_dimensionless(ks: float, v: float, D: float, temperature: float = 298.15) -> float:
    """Dimensionless standard rate constant ``ksDim = ks / sqrt(f v D)``.

    This is the grouping used in SERM Ch. 6 (``kDim = ks/Sqrt[f v D]``); it is
    the natural reversibility yardstick for a linear sweep: ``ksDim >> 1`` is
    reversible, ``ksDim << 1`` is irreversible.

    Parameters
    ----------
    ks : float
        Standard heterogeneous rate constant (cm/s).
    v : float
        Sweep rate (V/s).
    D : float
        Diffusion coefficient (cm^2/s).
    temperature : float
        Temperature (K).
    """
    return ks / np.sqrt(f_thermal(temperature) * v * D)


@dataclass
class CVResult:
    """Container for one simulated cyclic voltammogram.

    Attributes
    ----------
    p : numpy.ndarray
        Dimensionless potential ``f (E - E0)`` along the full forward+reverse
        sweep.
    chi : numpy.ndarray
        Dimensionless surface-gradient current ``dc_O/dX`` (reduction current is
        negative on the forward branch).
    n_forward : int
        Number of points on the forward (cathodic) sweep, so ``chi[:n_forward]``
        isolates the forward branch.
    """

    p: np.ndarray
    chi: np.ndarray
    n_forward: int

    @property
    def cathodic_peak(self) -> float:
        """Magnitude of the forward (cathodic) peak of ``-chi``."""
        return float(-self.chi[: self.n_forward].min())

    @property
    def cathodic_peak_potential(self) -> float:
        """Dimensionless potential ``p`` at the cathodic peak."""
        fwd = self.chi[: self.n_forward]
        return float(self.p[: self.n_forward][fwd.argmin()])


def simulate_cv(
    ks_dim: float,
    *,
    alpha: float = 0.5,
    DM: float = 4.0,
    dz: float = 0.02,
    upper: float = 11.6435,
    lower: float = -15.5766,
    space_factor: float = 6.0,
) -> CVResult:
    """Simulate a quasi-reversible cyclic voltammogram (fully implicit FD).

    Solves the dimensionless diffusion equation for the oxidised species with a
    Butler--Volmer surface boundary condition, sweeping ``p = f(E-E0)`` from
    ``upper`` down to ``lower`` and back.  Equal diffusion coefficients are
    assumed, so ``c_R = 1 - c_O`` and only ``c_O`` is propagated.

    Parameters
    ----------
    ks_dim : float
        Dimensionless standard rate constant ``ksDim`` (see
        :func:`ks_dimensionless`).  Large -> reversible, small -> irreversible.
    alpha : float
        Transfer coefficient.
    DM : float
        Model diffusion number ``dtau/dX**2`` (sets the spatial step).
    dz : float
        Dimensionless time/potential step ``dtau`` (``= f * dE``).
    upper, lower : float
        Dimensionless start and switching potentials ``f(E-E0)``.
    space_factor : float
        Domain length in units of ``sqrt(DM * n_steps)`` diffusion lengths; the
        far boundary is held at the bulk value ``c_O = 1``.

    Returns
    -------
    CVResult
    """
    span = 2.0 * (upper + abs(lower))        # total dimensionless sweep (= z span)
    n = int(round(span / dz))                # number of time steps
    tau = span / (n - 1)                     # dimensionless time step actually used
    dX = np.sqrt(tau / DM)

    ks_star = ks_star_sweep(ks_dim, span, DM, n)
    m = 1 + int(np.ceil(space_factor * np.sqrt(DM * (n - 1))))

    conc = np.ones(m)                        # c_O, bulk-filled, length m
    chi = np.zeros(n)
    p_axis = np.zeros(n)
    half = (n + 1) / 2.0
    y1 = 1.0 + 2.0 * DM
    z1 = -DM

    # Static parts of the tridiagonal system for the m-2 interior unknowns.
    ab = np.zeros((3, m - 2))                # banded (upper, diag, lower) for solve_banded
    ab[1, :] = 1.0 + 2.0 * DM
    ab[0, 1:] = -DM
    ab[2, :-1] = -DM

    for k in range(1, n + 1):
        # Triangular sweep in dimensionless potential.
        if k > half:
            p = upper - span + tau * (k - 1)         # reverse (anodic) branch
        else:
            p = upper - tau * (k - 1)                # forward (cathodic) branch
        p_axis[k - 1] = p
        xi = np.exp(p)

        tmp = bv_surface_factor(xi, ks_star, alpha)

        b = conc[1:m - 1].copy()
        b[0] += tmp * DM * ks_star * xi ** (1.0 - alpha)
        b[-1] += DM

        # Patch the first row for the Butler--Volmer boundary this step.
        ab[1, 0] = y1 - 4.0 * DM * tmp
        ab[0, 1] = z1 + DM * tmp

        interior = solve_banded((1, 1), ab, b)
        c_surface = bv_surface_conc(interior[0], interior[1], xi, ks_star, alpha, tmp)
        conc = np.concatenate(([c_surface], interior, [1.0]))

        chi[k - 1] = (3.0 * conc[0] - 4.0 * conc[1] + conc[2]) / (2.0 * dX)

    return CVResult(p=p_axis, chi=chi, n_forward=n // 2)


def simulate_reversible_cv(
    *,
    DM: float = 4.0,
    dz: float = 0.02,
    upper: float = 11.6435,
    lower: float = -15.5766,
    space_factor: float = 6.0,
) -> CVResult:
    """Reversible (Nernstian) CV: the Chapter 5 limit, with a Dirichlet surface.

    Imposes ``c_O(0) = xi/(1+xi)`` directly rather than through Butler--Volmer
    kinetics.  Used to confirm that :func:`simulate_cv` reproduces this curve as
    ``ks_dim -> inf``.
    """
    span = 2.0 * (upper + abs(lower))
    n = int(round(span / dz))
    tau = span / (n - 1)
    dX = np.sqrt(tau / DM)
    m = 1 + int(np.ceil(space_factor * np.sqrt(DM * (n - 1))))

    conc = np.ones(m)
    chi = np.zeros(n)
    p_axis = np.zeros(n)
    half = (n + 1) / 2.0

    ab = np.zeros((3, m - 2))
    ab[1, :] = 1.0 + 2.0 * DM
    ab[0, 1:] = -DM
    ab[2, :-1] = -DM

    for k in range(1, n + 1):
        if k > half:
            p = upper - span + tau * (k - 1)
        else:
            p = upper - tau * (k - 1)
        p_axis[k - 1] = p
        c_surface = 1.0 / (1.0 + np.exp(-p))      # xi/(1+xi)

        b = conc[1:m - 1].copy()
        b[0] += DM * c_surface
        b[-1] += DM
        interior = solve_banded((1, 1), ab, b)
        conc = np.concatenate(([c_surface], interior, [1.0]))
        chi[k - 1] = (3.0 * conc[0] - 4.0 * conc[1] + conc[2]) / (2.0 * dX)

    return CVResult(p=p_axis, chi=chi, n_forward=n // 2)


def cottrell_surface_flux(tau: np.ndarray) -> np.ndarray:
    """Dimensionless Cottrell surface flux ``1/sqrt(pi tau)`` for a step to c=0.

    Independent analytic reference used to certify that the FD grid and the
    surface-gradient operator are consistent (the gradient ``dc/dX`` of a
    fully-depleted-surface step must equal ``1/sqrt(pi tau)``).
    """
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(np.pi * tau)
