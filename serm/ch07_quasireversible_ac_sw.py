"""Chapter 7 helper: quasi-reversible AC / square-wave FD on an expanding grid.

This module re-implements, in vectorised Python, the three "Implicit ... on an
expanding space grid" simulators that accompany Honeychurch's *Simulating
Electrochemical Reactions in Mathematica* (SERM), Chapter 7:

* ``Extra Notebooks/chapter7/ImplicitACExp.nb`` -- AC voltammetry, a sinusoidal
  perturbation riding a linear DC ramp;
* ``Extra Notebooks/chapter7/ImplicitSWExp1.nb`` -- square-wave voltammetry with
  the square wave superimposed on a *linear ramp*;
* ``Extra Notebooks/chapter7/ImplicitSWExp2.nb`` -- square-wave voltammetry with
  the square wave superimposed on a *staircase* (true SWV).

All three share the same engine and differ only in the excitation waveform, so
this module factors the engine into one routine, :func:`_simulate_qr`, and
exposes three thin waveform-specific wrappers.

What is new relative to Chapter 6
---------------------------------
The Chapter 6 helper (:mod:`serm.ch06_potential_sweep_nonreversible`) solves the
same Butler--Volmer surface boundary, but on a *uniform* space grid.  Chapter 7's
extra notebooks introduce an **exponentially expanding space grid** -- nodes are
spaced ``x_j ~ a**j`` so that the mesh is fine next to the electrode (where the
gradient is steep) and coarse far away (where the concentration is flat).  This
lets a few hundred nodes span many diffusion lengths, which AC/SW simulations
need because they run for thousands of small time steps.

Expanding-grid discretisation
-----------------------------
With expansion factor ``a`` and model diffusion number ``DM = dtau/dx0**2`` (here
``dx0`` is the *first* spatial step), Honeychurch's ``makeDiagonals`` builds, for
interior node ``j = 2 .. m-1``, the three implicit diagonals

.. math::
    x_j = -DM\\, a^{4-2j}, \\quad
    y_j = 1 + (1+a)\\,DM\\, a^{3-2j}, \\quad
    z_j = -DM\\, a^{3-2j}

(the ``a**(4-2j)`` / ``a**(3-2j)`` powers are the metric factors of the
non-uniform Laplacian; in the source notebook they appear as
``SuperscriptBox["a", 4-2 j]`` etc.).  The far-field bulk feed enters the last
interior row as ``DM a^{5-2m}``.

The surface is closed with the **same** Butler--Volmer elimination as Chapter 6
(:func:`serm.kinetics.bv_surface_factor` / :func:`bv_surface_conc`): writing
``xi = exp[(nF/RT)(E-E0)]`` and ``ks_star`` for the grid-scaled rate constant,

.. math::
    \\mathrm{tmp} = \\frac{\\xi^{\\alpha}}{3\\xi^{\\alpha}+k_s^{*}(1+\\xi)},\\qquad
    c_{O,0} = (k_s^{*}\\xi^{1-\\alpha}+4c_1-c_2)\\,\\mathrm{tmp},

so the first implicit row is patched by ``y_1 -> y_1 - 4 DM\\,tmp`` and
``z_1 -> z_1 + DM\\,tmp`` each step.  As ``ks_star -> inf`` the elimination
collapses to the Nernstian Dirichlet value ``xi/(1+xi)`` and the reversible
Chapter 7 result is recovered (the validation used by the extras notebook).

The dimensionless current is the expanding-grid surface derivative

.. math::
    \\chi = \\bigl[(2+a)\\,a\\,c_0 - (1+a)^2 c_1 + c_2\\bigr]
            \\sqrt{\\frac{DM(n-1)}{2 a^2 (1+a)\\,\\mathrm{region}}}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .kinetics import F, R, f_thermal, bv_surface_factor

__all__ = [
    "F",
    "R",
    "f_thermal",
    "expanding_diagonals",
    "expanding_grid_points",
    "ks_star_expanding",
    "ACResult",
    "SWResult",
    "simulate_ac",
    "simulate_sw_ramp",
    "simulate_sw_staircase",
    "ac_dc_filter",
]


def expanding_diagonals(m: int, a: float, DM: float):
    """Three implicit diagonals of the expanding-grid Laplacian.

    Reproduces Honeychurch's ``makeDiagonals`` for interior nodes ``j = 2..m-1``:
    ``x_j = -DM a**(4-2j)`` (sub), ``y_j = 1 + (1+a) DM a**(3-2j)`` (main),
    ``z_j = -DM a**(3-2j)`` (super).

    Parameters
    ----------
    m : int
        Number of spatial grid points (nodes ``0..m-1``).
    a : float
        Grid expansion factor (``a > 1``); ``a = 1`` would be a uniform grid.
    DM : float
        Model diffusion number ``dtau / dx0**2`` based on the first step.

    Returns
    -------
    x, y, z : numpy.ndarray
        Sub-, main- and super-diagonal arrays, each of length ``m - 2``.
    """
    j = np.arange(2, m, dtype=float)            # j = 2..m-1, length m-2
    x = -DM * a ** (4.0 - 2.0 * j)
    y = 1.0 + (1.0 + a) * DM * a ** (3.0 - 2.0 * j)
    z = -DM * a ** (3.0 - 2.0 * j)
    return x, y, z


def expanding_grid_points(n: int, a: float, DM: float) -> int:
    """Number of spatial nodes ``m`` for an expanding grid spanning the layer.

    Solves Honeychurch's sizing rule
    ``sum_{j=1}^{m-1} a**(j-1) = 6 sqrt(DM (1+a)(n-1)/2)`` (the geometric series
    of node spacings must reach ~6 diffusion lengths) for ``m`` and rounds up.
    The closed form of the geometric sum gives
    ``m = 1 + ln[(a-1) R + 1] / ln a`` with ``R`` the right-hand side.

    Parameters
    ----------
    n : int
        Number of time steps.
    a : float
        Expansion factor (``a > 1``).
    DM : float
        Model diffusion number.

    Returns
    -------
    int
        ``m``, the number of spatial nodes.
    """
    rhs = 6.0 * np.sqrt(DM * (1.0 + a) * (n - 1) / 2.0)
    mm = 1.0 + np.log(rhs * (a - 1.0) + 1.0) / np.log(a)
    return int(np.ceil(mm))


def ks_star_expanding(ks: float, time: float, D: float, DM: float, n: int) -> float:
    """Grid-scaled standard rate constant ``ksStar`` for the expanding grid.

    ``ksStar = 2 ks sqrt(time / (D DM (n-1)))`` (Honeychurch's expanding-grid
    notebooks).  Here ``ks`` is the dimensional standard rate constant (cm/s),
    ``time`` the total experiment duration (s), ``D`` the diffusion coefficient
    (cm^2/s), ``DM`` the model diffusion number and ``n`` the number of steps.
    """
    return 2.0 * ks * np.sqrt(time / (D * DM * (n - 1)))


def _expanding_surface_derivative(c0, c1, c2, a: float, n: int, DM: float, region: float):
    """Dimensionless current from the 3-point expanding-grid surface derivative."""
    scale = np.sqrt(DM * (n - 1) / (2.0 * a ** 2 * (1.0 + a) * region))
    return ((2.0 + a) * a * c0 - (1.0 + a) ** 2 * c1 + c2) * scale


def _simulate_qr(
    eta_source,
    *,
    n: int,
    region: float,
    ks_star: float,
    alpha: float,
    a: float,
    DM: float,
):
    """Core expanding-grid quasi-reversible BV time-stepper.

    Parameters
    ----------
    eta_source : callable
        ``eta_source(k) -> float`` giving the *dimensionless* applied potential
        ``(nF/RT)(E-E0)`` at integer step ``k`` (``k = 0..n-1``).  Each waveform
        (AC sine, SW-on-ramp, SW-on-staircase) supplies its own.
    n : int
        Number of time steps.
    region : float
        Total dimensionless potential span (sets the current normalisation).
    ks_star : float
        Grid-scaled standard rate constant.
    alpha : float
        Transfer coefficient.
    a, DM : float
        Expansion factor and model diffusion number.

    Returns
    -------
    chi : numpy.ndarray, shape (n,)
        Dimensionless current at each step (``chi[0]`` is ``nan``: no flux at the
        initial condition).
    """
    m = expanding_grid_points(n, a, DM)
    x, y, z = expanding_diagonals(m, a, DM)
    y1, z1 = float(y[0]), float(z[0])
    Mi = m - 2

    # Banded matrix (l=1, u=1) for the interior unknowns conc[1..m-2].
    ab = np.zeros((3, Mi))
    ab[1, :] = y
    ab[0, 1:] = z[: Mi - 1]
    ab[2, :-1] = x[1:]

    bulk_feed = DM * a ** (5.0 - 2.0 * m)

    conc = np.ones(m)
    chi = np.full(n, np.nan)
    for k in range(1, n):
        xi = np.exp(eta_source(k))
        tmp = bv_surface_factor(xi, ks_star, alpha)

        b = conc[1:m - 1].copy()
        b[0] += tmp * DM * ks_star * xi ** (1.0 - alpha)
        b[-1] += bulk_feed

        ab[1, 0] = y1 - 4.0 * DM * tmp
        ab[0, 1] = z1 + DM * tmp

        interior = solve_banded((1, 1), ab, b)
        c0 = (ks_star * xi ** (1.0 - alpha) + 4.0 * interior[0] - interior[1]) * tmp
        conc = np.concatenate(([c0], interior, [1.0]))

        chi[k] = _expanding_surface_derivative(
            conc[0], conc[1], conc[2], a, n, DM, region
        )
    return chi


@dataclass
class ACResult:
    """Result of an expanding-grid quasi-reversible AC simulation."""

    eta_dc: np.ndarray
    """Smooth DC ramp ``upper - dtau*k`` (the AC-free potential axis)."""
    chi: np.ndarray
    """Dimensionless current at each step (``chi[0]`` is ``nan``)."""
    dtau: float
    """Dimensionless time/potential increment per step."""
    dE_dimless: float
    """Dimensionless AC amplitude ``f * amp_volt``."""
    n_cycles: float
    """Number of AC oscillations over the window (FFT fundamental bin)."""


def simulate_ac(
    ks: float = 1e3,
    *,
    alpha: float = 0.5,
    amp_volt: float = 0.005,
    Omega: float = 6.4 * np.pi,
    upper: float = 10.0,
    lower: float = -10.0,
    sweep_rate: float = 1.0,
    D: float = 1e-5,
    n: int = 2 ** 12,
    a: float = 1.2,
    DM: float = 2.0,
    temperature: float = 298.15,
) -> ACResult:
    """Quasi-reversible AC voltammetry on an expanding space grid.

    Re-implementation of ``ImplicitACExp.nb``.  A sinusoid of dimensionless
    amplitude ``f*amp_volt`` and dimensionless angular frequency ``Omega`` rides a
    linear DC ramp from ``upper`` to ``lower`` (in ``(nF/RT)(E-E0)``), with a
    Butler--Volmer surface boundary of standard rate constant ``ks``.

    Parameters
    ----------
    ks : float
        Standard heterogeneous rate constant (cm/s).  Large -> reversible.
    alpha : float
        Transfer coefficient.
    amp_volt : float
        AC amplitude in volts (dimensionless amplitude ``f*amp_volt``).
    Omega : float
        Dimensionless angular frequency.
    upper, lower : float
        Start/end dimensionless DC potential ``(nF/RT)(E-E0)``.
    sweep_rate : float
        DC sweep rate magnitude (V/s); sets the experiment time scale and hence
        ``ks_star``.
    D : float
        Diffusion coefficient (cm^2/s).
    n : int
        Number of time steps (a power of two suits the FFT analysis).
    a, DM : float
        Expansion factor and model diffusion number.
    temperature : float
        Temperature (K).

    Returns
    -------
    ACResult
    """
    f_RT = f_thermal(temperature)
    region = upper + abs(lower)
    dtau = region / (n - 1)
    sigma = f_RT * abs(sweep_rate)
    time = abs(region / sigma)
    ks_star = ks_star_expanding(ks, time, D, DM, n)
    dE = f_RT * amp_volt

    def eta_source(k: int) -> float:
        t = k * dtau
        return upper - t - dE * np.sin(Omega * t)

    chi = _simulate_qr(
        eta_source, n=n, region=region, ks_star=ks_star, alpha=alpha, a=a, DM=DM
    )
    eta_dc = upper - dtau * np.arange(n)
    n_cycles = Omega * (n * dtau) / (2.0 * np.pi)
    return ACResult(eta_dc=eta_dc, chi=chi, dtau=dtau, dE_dimless=dE, n_cycles=n_cycles)


@dataclass
class SWResult:
    """Result of an expanding-grid quasi-reversible square-wave simulation."""

    eta_sample: np.ndarray
    """DC (ramp/staircase) potential at each sampled square-wave cycle."""
    chi_forward: np.ndarray
    """Current sampled at the end of each forward (reductive) half-pulse."""
    chi_backward: np.ndarray
    """Current sampled at the end of each backward (oxidative) half-pulse."""
    chi_diff: np.ndarray
    """Difference current ``chi_forward - chi_backward`` (the SWV response)."""
    eta_full: np.ndarray
    """Full applied dimensionless waveform (DC + square wave) at every step."""
    chi_full: np.ndarray
    """Raw dimensionless current at every step (``chi_full[0]`` is ``nan``)."""
    n_cycles: int
    """Number of square-wave cycles."""
    tN: int
    """Time increments per half square-wave cycle."""


def _sw_solve(eta_source, *, n, region, ks_star, alpha, a, DM, tN):
    """Shared back-end for the two SW waveforms: solve, then sample fwd/back."""
    chi = _simulate_qr(
        eta_source, n=n, region=region, ks_star=ks_star, alpha=alpha, a=a, DM=DM
    )
    fwd = np.arange(tN, n, 2 * tN)
    bwd = np.arange(2 * tN, n, 2 * tN)
    npair = min(len(fwd), len(bwd))
    fwd, bwd = fwd[:npair], bwd[:npair]
    eta_full = np.array([eta_source(k) for k in range(n)])
    return chi, fwd, bwd, eta_full


def simulate_sw_ramp(
    ks: float = 1e3,
    *,
    alpha: float = 0.5,
    amp_sw_volt: float = 0.05,
    tp: float = 0.1,
    Omega: float = 6.4 * np.pi,
    upper: float = 10.0,
    rng: float = 20.0,
    D: float = 1e-5,
    a: float = 1.1,
    DM: float = 2.0,
    tN: int = 50,
    temperature: float = 298.15,
) -> SWResult:
    """Square-wave voltammetry on a *linear ramp* (``ImplicitSWExp1.nb``).

    The square wave of dimensionless half-amplitude ``f*amp_sw_volt`` is added to
    a linear DC ramp; ``tN`` time steps make up each half-cycle and ``Omega`` (with
    ``rng``) fixes the number of cycles.  Current is sampled at the end of each
    forward and reverse pulse, and the difference current is returned.

    Parameters
    ----------
    ks : float
        Standard rate constant (cm/s).
    alpha : float
        Transfer coefficient.
    amp_sw_volt : float
        Square-wave amplitude in volts.
    tp : float
        Square-wave period (s); sets the experiment time scale.
    Omega : float
        Dimensionless angular frequency (with ``rng`` sets the cycle count).
    upper : float
        Start dimensionless potential ``(nF/RT)(E-E0)``.
    rng : float
        Dimensionless potential span swept.
    D : float
        Diffusion coefficient (cm^2/s).
    a, DM : float
        Expansion factor and model diffusion number.
    tN : int
        Time increments per half square-wave cycle.
    temperature : float
        Temperature (K).

    Returns
    -------
    SWResult
    """
    f_RT = f_thermal(temperature)
    dE_sw = 2.0 * f_RT * amp_sw_volt
    n_cycles = int(round(0.5 * Omega * rng / np.pi))
    n = n_cycles * 2 * tN
    tau = rng / n
    time = n_cycles * tp
    ks_star = ks_star_expanding(ks, time, D, DM, n)

    def eta_source(k: int) -> float:
        unit = 1.0 if (-(k % (2 * tN)) + (tN - 1)) >= 0 else 0.0
        return upper - tau * k - dE_sw * (-0.5 + unit)

    chi, fwd, bwd, eta_full = _sw_solve(
        eta_source, n=n, region=rng, ks_star=ks_star, alpha=alpha,
        a=a, DM=DM, tN=tN,
    )
    eta_sample = upper - tau * fwd
    return SWResult(
        eta_sample=eta_sample,
        chi_forward=chi[fwd],
        chi_backward=chi[bwd],
        chi_diff=chi[fwd] - chi[bwd],
        eta_full=eta_full,
        chi_full=chi,
        n_cycles=n_cycles,
        tN=tN,
    )


def simulate_sw_staircase(
    ks: float = 1e3,
    *,
    alpha: float = 0.5,
    amp_sw_volt: float = 0.05,
    step_s_volt: float = 0.005,
    tp: float = 0.1,
    upper: float = 10.0,
    rng: float = 20.0,
    D: float = 1e-5,
    a: float = 1.1,
    DM: float = 2.0,
    tN: int = 50,
    temperature: float = 298.15,
) -> SWResult:
    """Square-wave voltammetry on a *staircase* -- true SWV (``ImplicitSWExp2.nb``).

    Differs from :func:`simulate_sw_ramp` only in the DC component: instead of a
    smooth ramp the base potential steps by ``f*step_s_volt`` once per
    square-wave cycle, holding constant within each cycle.  Each cycle is
    ``2 tN`` time steps, ``tN`` per half.

    Parameters
    ----------
    ks : float
        Standard rate constant (cm/s).
    alpha : float
        Transfer coefficient.
    amp_sw_volt : float
        Square-wave amplitude in volts.
    step_s_volt : float
        Staircase step in volts per square-wave cycle.
    tp : float
        Square-wave period (s).
    upper : float
        Start dimensionless potential.
    rng : float
        Approximate dimensionless span (rounded to a whole number of staircase
        steps).
    D : float
        Diffusion coefficient (cm^2/s).
    a, DM : float
        Expansion factor and model diffusion number.
    tN : int
        Time increments per half square-wave cycle.
    temperature : float
        Temperature (K).

    Returns
    -------
    SWResult
    """
    f_RT = f_thermal(temperature)
    dE_s = f_RT * step_s_volt
    dE_sw = f_RT * amp_sw_volt
    n_cycles = int(round(rng / dE_s))
    span = n_cycles * dE_s
    n = n_cycles * 2 * tN
    time = n_cycles * tp
    ks_star = ks_star_expanding(ks, time, D, DM, n)

    def staircase(k: int) -> float:
        return dE_s * ((k - (k % (2 * tN))) / (2 * tN))

    def eta_source(k: int) -> float:
        unit = 1.0 if (-(k % (2 * tN)) + (tN - 1)) >= 0 else 0.0
        return upper - staircase(k) - dE_sw * 2.0 * (-0.5 + unit)

    chi, fwd, bwd, eta_full = _sw_solve(
        eta_source, n=n, region=span, ks_star=ks_star, alpha=alpha,
        a=a, DM=DM, tN=tN,
    )
    eta_sample = upper - np.array([staircase(k) for k in fwd])
    return SWResult(
        eta_sample=eta_sample,
        chi_forward=chi[fwd],
        chi_backward=chi[bwd],
        chi_diff=chi[fwd] - chi[bwd],
        eta_full=eta_full,
        chi_full=chi,
        n_cycles=n_cycles,
        tN=tN,
    )


def ac_dc_filter(chi: np.ndarray, n_per_cycle: int, passes: int = 2):
    """Moving-average DC filter for an AC/SW voltammogram.

    Smooths out the AC/SW ripple by averaging over one full cycle, leaving the
    underlying DC (linear-sweep) response.  Honeychurch applies the
    :func:`serm.filters.moving_average` twice; this wrapper repeats it ``passes``
    times.  The output is shorter than the input (each pass drops
    ``n_per_cycle - 1`` points); the number of dropped points is returned so the
    caller can re-align the potential axis.

    Parameters
    ----------
    chi : numpy.ndarray
        Raw dimensionless current (the AC/SW voltammogram).  ``nan`` values are
        dropped first.
    n_per_cycle : int
        Window width = number of increments in one full AC/SW cycle.
    passes : int
        Number of moving-average passes (Honeychurch uses 2).

    Returns
    -------
    filtered : numpy.ndarray
        The DC-filtered current.
    overhang : int
        Total number of points lost (``len(input) - len(filtered)``); the filtered
        series is centred, so each end loses ``overhang / 2``.
    """
    from .filters import moving_average

    data = np.asarray(chi, dtype=float)
    data = data[~np.isnan(data)]
    n_in = len(data)
    for _ in range(passes):
        data = moving_average(data, n_per_cycle)
    overhang = n_in - len(data)
    return data, overhang
