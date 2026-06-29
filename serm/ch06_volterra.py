"""Chapter 6 helpers: Volterra 2nd-kind CV and finite-difference variants.

This module supplements :mod:`serm.ch06_potential_sweep_nonreversible` with the
*other* solution routes Honeychurch develops in Chapter 6 ("Potential sweep
methods -- non-reversible reactions") and its companion notebooks
``ImplicitCVQuasiExp.nb`` (expanding space grid) and ``ImplicitCVQuasiRM.nb``
(Richtmyer time-stepping modification):

* :func:`volterra_irreversible_cv` -- the integral-equation route.  For a totally
  irreversible reduction the surface-concentration relation collapses to a
  **Volterra equation of the second kind**,

  .. math::
      \\chi(z) + g(z)\\int_0^z \\frac{\\chi(\\tau)}{\\sqrt{z-\\tau}}\\,d\\tau = g(z),

  solved here by the same discrete quadrature (kernel ``y^{-1/2}``, weights
  ``r_k`` / ``s_k``) used in Chapter 4.  The dimensionless current peak is the
  Nicholson--Shain totally-irreversible value ``sqrt(pi) chi_p = 0.4958`` --
  a closed-form anchor independent of any finite-difference grid.

* :func:`simulate_cv_richtmyer` -- a fully-implicit FD CV that, after four
  ordinary backward-Euler steps, switches to the **Richtmyer** five-point
  backward (BDF4-style) time discretisation: surface diagonal ``25/12 + 2 D_M``
  and right-hand side ``4 c^{k-1} - 3 c^{k-2} + (4/3) c^{k-3} - (1/4) c^{k-4}``.

* :func:`simulate_cv_expanding` -- the same Butler--Volmer CV on an
  **exponentially expanding space grid** (ratio ``a``), which packs nodes near
  the electrode and reaches the bulk in far fewer points.

All three reuse the Butler--Volmer surface elimination of
:mod:`serm.kinetics` (``bv_surface_factor`` / ``bv_surface_conc``) and reduce to
the validated reversible / irreversible limits.

Notes
-----
Re-implemented from the algorithms in the source notebooks; prose, figures and
numeric output are not copied.  Validation is by reduction to closed-form limits
(Nicholson--Shain 0.4463 reversible, 0.4958 sqrt(alpha) irreversible) -- never by
comparison to the book's bundled ``.dat`` data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .kinetics import (
    F, R, f_thermal, ks_star_sweep, bv_surface_factor, bv_surface_conc,
)
# Single source of truth for the expanding-grid node count (signature
# ``(n, a, DM)``), shared with Chapter 7 to avoid a duplicate definition.
from .ch07_quasireversible_ac_sw import expanding_grid_points

__all__ = [
    "VolterraResult", "volterra_irreversible_cv",
    "simulate_cv_richtmyer", "simulate_cv_expanding",
]


@dataclass
class VolterraResult:
    """One simulated voltammogram from the Volterra integral-equation route.

    Attributes
    ----------
    p : numpy.ndarray
        Dimensionless potential axis ``f (E - E0)`` (offset by the chosen start
        potential), monotonically decreasing along the forward sweep.
    chi : numpy.ndarray
        Dimensionless current function.  For the totally irreversible wave the
        peak of ``chi`` equals the Nicholson--Shain value ``0.4958`` (i.e.
        ``chi`` already carries the ``sqrt(pi)`` of ``sqrt(pi) chi``).
    """

    p: np.ndarray
    chi: np.ndarray

    @property
    def peak(self) -> float:
        """Magnitude of the cathodic peak of ``chi``."""
        return float(self.chi.max())

    @property
    def peak_potential(self) -> float:
        """Dimensionless potential at the cathodic peak."""
        return float(self.p[self.chi.argmax()])


def volterra_irreversible_cv(
    *,
    ks: float = 1.0e-6,
    v: float = 1.0,
    D: float = 1.0e-5,
    alpha: float = 0.5,
    de: float = 0.05,
    n: int = 1200,
    start_offset: float = 10.0,
    temperature: float = 298.15,
) -> VolterraResult:
    """Totally irreversible CV via the Volterra equation of the second kind.

    Solves the discretised integral equation by the Chapter-4 quadrature.  With
    ``d = alpha * de`` the step in ``b z`` (``b = alpha f v``), the recurrence
    for the slopes ``a[m]`` is

    .. math::
        a_m = \\frac{1}{g_m + h_1}\\Bigl(1
        - h_1\\!\\sum_{i=1}^{m-1} a_i\\bigl[(m-i+1)^{3/2}-(m-i)^{3/2}\\bigr]
        - g_m\\!\\sum_{i=1}^{m-1} a_i\\Bigr),

    with ``h_1 = (4/3) d^{3/2}`` and
    ``g_m = d sqrt(pi alpha f v D) e^{-m d} / k_i``, ``k_i = ks e^{-alpha s0}``.
    The current is the running sum ``chi_m = sqrt(pi) d sum_{j<=m} a_j``, which
    converges to the Nicholson--Shain irreversible peak ``0.4958`` as ``de`` is
    refined.

    Parameters
    ----------
    ks : float
        Standard heterogeneous rate constant (cm/s) -- only the irreversible
        grouping ``k_i`` enters, so any sufficiently small ``ks`` gives the same
        (scan-rate-independent) dimensionless wave.
    v, D : float
        Sweep rate (V/s) and diffusion coefficient (cm^2/s).
    alpha : float
        Transfer coefficient.
    de : float
        Dimensionless potential increment ``Delta e`` of the quadrature.
    n : int
        Number of potential steps.
    start_offset : float
        Dimensionless potential ``f(E_start - E0)`` at the start of the sweep
        (the ``10.`` of the source); sets the ``k_i`` offset.
    temperature : float
        Temperature (K).

    Returns
    -------
    VolterraResult
    """
    f = f_thermal(temperature)
    d = alpha * de
    ki = ks * np.exp(-alpha * start_offset)
    # Huber self-weight for the y^{-1/2} kernel on the step of width ``d``.
    h1 = (4.0 / 3.0) * d ** 1.5
    pref = d * np.sqrt(np.pi * alpha * f * v * D) / ki

    a = np.zeros(n + 1)               # 1-indexed slopes
    # Precompute the kernel weight increments w[k] = (k+1)^{3/2} - k^{3/2}.
    kk = np.arange(0, n + 1, dtype=float)
    wk = (kk + 1.0) ** 1.5 - kk ** 1.5

    for m in range(1, n + 1):
        gm = pref * np.exp(-m * d)
        if m == 1:
            s1 = 0.0
            s2 = 0.0
        else:
            ai = a[1:m]                          # a[1..m-1]
            # weights (m-i+1)^{3/2}-(m-i)^{3/2} for i=1..m-1  ==  wk[m-i]
            s1 = float(ai @ wk[m - np.arange(1, m)])
            s2 = float(ai.sum())
        a[m] = (1.0 / (gm + h1)) * (1.0 - h1 * s1 - gm * s2)

    # chi already carries the sqrt(pi) of the Nicholson-Shain function.
    chi = np.sqrt(np.pi) * np.cumsum(d * a[1:])
    p = start_offset - d * np.arange(1, n + 1)
    return VolterraResult(p=p, chi=chi)


def _triangular_xi(k: int, upper: float, span: float, tau: float, half: float):
    """Surface ratio ``xi`` and potential ``p`` at step ``k`` of a CV sweep."""
    if k > half:
        p = upper - span + tau * (k - 1)
    else:
        p = upper - tau * (k - 1)
    return np.exp(p), p


@dataclass
class CVResult:
    """Container mirroring :class:`serm.ch06_potential_sweep_nonreversible.CVResult`."""

    p: np.ndarray
    chi: np.ndarray
    n_forward: int

    @property
    def cathodic_peak(self) -> float:
        return float(-self.chi[: self.n_forward].min())

    @property
    def cathodic_peak_potential(self) -> float:
        fwd = self.chi[: self.n_forward]
        return float(self.p[: self.n_forward][fwd.argmin()])


def simulate_cv_richtmyer(
    ks_dim: float,
    *,
    alpha: float = 0.5,
    DM: float = 2.0,
    dz: float = 0.02,
    upper: float = 11.6435,
    lower: float = -15.5766,
    space_factor: float = 6.0,
) -> CVResult:
    """Butler--Volmer CV with the Richtmyer (BDF4) time modification.

    The first four steps use the ordinary fully-implicit backward-Euler row
    ``(1 + 2 D_M)``; from step five onward the time derivative is replaced by the
    five-point backward formula, giving surface diagonal ``25/12 + 2 D_M`` and a
    right-hand side built from the previous four concentration profiles,
    ``4 c^{k-1} - 3 c^{k-2} + (4/3) c^{k-3} - (1/4) c^{k-4}``.  This raises the
    temporal order while keeping the same tridiagonal structure and the same
    Butler--Volmer surface elimination.

    Parameters mirror
    :func:`serm.ch06_potential_sweep_nonreversible.simulate_cv`.
    """
    span = 2.0 * (upper + abs(lower))
    n = int(round(span / dz))
    tau = span / (n - 1)
    dX = np.sqrt(tau / DM)
    ks_star = ks_star_sweep(ks_dim, span, DM, n)
    m = 1 + int(np.ceil(space_factor * np.sqrt(DM * (n - 1))))
    half = (n + 1) / 2.0

    # Backward-Euler banded operator (steps 1..4).
    ab = np.zeros((3, m - 2))
    ab[1] = 1.0 + 2.0 * DM
    ab[0, 1:] = -DM
    ab[2, :-1] = -DM
    y1, z1 = 1.0 + 2.0 * DM, -DM

    # Richtmyer (BDF4) banded operator (steps >= 5).
    ab2 = np.zeros((3, m - 2))
    ab2[1] = 25.0 / 12.0 + 2.0 * DM
    ab2[0, 1:] = -DM
    ab2[2, :-1] = -DM
    y1a, z1a = 25.0 / 12.0 + 2.0 * DM, -DM

    conc = np.ones(m)
    history = [conc]
    chi = np.zeros(n)
    p_axis = np.zeros(n)
    p_axis[0] = upper
    chi[0] = (3.0 * conc[0] - 4.0 * conc[1] + conc[2]) / (2.0 * dX)

    for k in range(2, n + 1):
        xi, p = _triangular_xi(k, upper, span, tau, half)
        p_axis[k - 1] = p
        tmp = bv_surface_factor(xi, ks_star, alpha)

        if k <= 4:
            b = conc[1:m - 1].copy()
            b[0] += tmp * DM * ks_star * xi ** (1.0 - alpha)
            b[-1] += DM
            ab[1, 0] = y1 - 4.0 * DM * tmp
            ab[0, 1] = z1 + DM * tmp
            interior = solve_banded((1, 1), ab, b)
        else:
            c1, c2, c3, c4 = history[-1], history[-2], history[-3], history[-4]
            rhs = 4.0 * c1 - 3.0 * c2 + (4.0 / 3.0) * c3 - 0.25 * c4
            b = rhs[1:m - 1].copy()
            b[0] += tmp * DM * ks_star * xi ** (1.0 - alpha)
            b[-1] += DM
            ab2[1, 0] = y1a - 4.0 * DM * tmp
            ab2[0, 1] = z1a + DM * tmp
            interior = solve_banded((1, 1), ab2, b)

        c_surface = bv_surface_conc(interior[0], interior[1], xi, ks_star, alpha, tmp)
        conc = np.concatenate(([c_surface], interior, [1.0]))
        history.append(conc)
        chi[k - 1] = (3.0 * conc[0] - 4.0 * conc[1] + conc[2]) / (2.0 * dX)

    return CVResult(p=p_axis, chi=chi, n_forward=n // 2)


def simulate_cv_expanding(
    ks_dim: float,
    *,
    alpha: float = 0.5,
    DM: float = 2.0,
    a: float = 1.05,
    dz: float = 0.02,
    upper: float = 11.6435,
    lower: float = -15.5766,
) -> CVResult:
    """Butler--Volmer CV on an exponentially expanding space grid.

    Nodes grow geometrically away from the electrode with ratio ``a`` (SERM
    Section 3.4.3), so the bulk is reached in far fewer points than a uniform
    grid.  The interior diagonals carry the expanding-grid factors

    .. math::
        X_j = -D_M a^{4-2j},\\quad Y_j = 1 + (1+a) D_M a^{3-2j},\\quad
        Z_j = -D_M a^{3-2j},

    and the surface gradient uses the three-point expanding-grid stencil
    ``(2+a) a c_0 - (1+a)^2 c_1 + c_2`` with the appropriate prefactor.  The
    Butler--Volmer surface elimination and the first-row patch are unchanged.

    Parameters
    ----------
    a : float
        Grid expansion ratio (> 1).  ``a = 1.05`` matches the source notebook.

    Other parameters mirror
    :func:`serm.ch06_potential_sweep_nonreversible.simulate_cv`.
    """
    span = 2.0 * (upper + abs(lower))
    n = int(round(span / dz))
    tau = span / (n - 1)
    ks_star = ks_star_sweep(ks_dim, span, DM, n)
    half = (n + 1) / 2.0
    m = expanding_grid_points(n, a, DM)

    j = np.arange(2, m)                       # j = 2 .. m-1, length m-2
    x = -DM * a ** (4.0 - 2.0 * j)            # sub-diagonal coefficients
    y = 1.0 + (1.0 + a) * DM * a ** (3.0 - 2.0 * j)
    z = -DM * a ** (3.0 - 2.0 * j)            # super-diagonal coefficients
    y1, z1 = y[0], z[0]

    grad_pref = np.sqrt(DM * (n - 1) / (2.0 * a ** 2 * (1.0 + a) * span))

    def surface_grad(c: np.ndarray) -> float:
        return ((2.0 + a) * a * c[0] - (1.0 + a) ** 2 * c[1] + c[2]) * grad_pref

    conc = np.ones(m)
    chi = np.zeros(n)
    p_axis = np.zeros(n)
    p_axis[0] = upper
    chi[0] = surface_grad(conc)

    for k in range(2, n + 1):
        xi, p = _triangular_xi(k, upper, span, tau, half)
        p_axis[k - 1] = p
        tmp = bv_surface_factor(xi, ks_star, alpha)

        b = conc[1:m - 1].copy()
        b[0] += tmp * DM * ks_star * xi ** (1.0 - alpha)
        b[-1] += DM * a ** (5.0 - 2.0 * m)    # far-boundary (bulk) contribution

        ab = np.zeros((3, m - 2))
        ab[1] = y
        ab[0, 1:] = z[:-1]
        ab[2, :-1] = x[1:]
        ab[1, 0] = y1 - 4.0 * DM * tmp
        ab[0, 1] = z1 + DM * tmp

        interior = solve_banded((1, 1), ab, b)
        c_surface = bv_surface_conc(interior[0], interior[1], xi, ks_star, alpha, tmp)
        conc = np.concatenate(([c_surface], interior, [1.0]))
        chi[k - 1] = surface_grad(conc)

    return CVResult(p=p_axis, chi=chi, n_forward=n // 2)
