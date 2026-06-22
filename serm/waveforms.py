"""Applied-potential waveform generators for the voltammetry chapters.

A small library of the excitation waveforms reused across Chapters 5--9 of
*Simulating Electrochemical Reactions in Python* so that each chapter does not
re-derive the same time/potential arrays:

* :func:`linear_sweep` -- a single-direction potential ramp (LSV).
* :func:`cyclic_sweep` -- the triangular forward/reverse ramp of cyclic
  voltammetry (CV), reversing at a switching potential.
* :func:`potential_step` -- a single step from an initial to a final potential
  (chronoamperometry / Cottrell experiment, Chapter 8).
* :func:`pulse_train` -- a staircase / pulse sequence (normal- and
  differential-pulse experiments, Chapter 8).
* :func:`ac_superposition` -- a DC waveform with a small sinusoidal AC
  perturbation added (AC voltammetry, Chapter 7).
* :func:`nernst_theta` and :func:`dimensionless_sweep_rate` -- the
  dimensionless helpers that connect an applied potential to the boundary
  condition used by the simulators.

Conventions
-----------
Potentials are measured relative to the formal potential ``E0`` unless an
``E0`` keyword is supplied, in which case the *overpotential* ``E - E0`` is what
enters the Nernstian / Butler--Volmer expressions.  Times are in seconds, sweep
rates ``v`` in V/s, potentials in V.  All array outputs are 1-D ``float``
``numpy.ndarray`` of equal length, so they can be zipped directly into a
simulation's time loop.

These generators describe only the *excitation*; the current response is
produced by the finite-difference simulators in each chapter.  The
dimensionless sweep parameter is grounded in SERM Ch. 5, where the sweep enters
through ``sigma = n F v / (R T)`` and a Nernstian surface boundary condition
(see ``Chapters/chapter5.nb``).
"""
from __future__ import annotations

import numpy as np

# Physical constants (SI).
F = 96485.33212      # Faraday constant, C/mol
R = 8.314462618      # gas constant, J/(mol K)


def linear_sweep(E_start, E_end, v, n_points):
    """Single linear potential ramp ``E(t) = E_start +/- v t``.

    Parameters
    ----------
    E_start, E_end : float
        Initial and final potential (V).
    v : float
        Sweep rate magnitude (V/s); the sign of the ramp is taken from
        ``sign(E_end - E_start)``.
    n_points : int
        Number of samples (>= 2).

    Returns
    -------
    t, E : numpy.ndarray, shape (n_points,)
        Time (s) and potential (V) arrays.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if v <= 0:
        raise ValueError("sweep rate v must be positive")
    duration = abs(E_end - E_start) / v
    t = np.linspace(0.0, duration, n_points)
    E = E_start + np.sign(E_end - E_start) * v * t
    return t, E


def cyclic_sweep(E_start, E_switch, v, n_points, E_end=None):
    """Triangular cyclic-voltammetry waveform.

    Sweeps linearly from ``E_start`` to the switching potential ``E_switch`` and
    back to ``E_end`` (default: ``E_start``) at constant rate ``v``.  This is the
    standard CV excitation of SERM Chapter 5.

    Parameters
    ----------
    E_start : float
        Initial potential (V).
    E_switch : float
        Switching (vertex) potential (V).
    v : float
        Sweep rate magnitude (V/s).
    n_points : int
        Total number of samples over the whole forward+reverse sweep (>= 3).
    E_end : float, optional
        Potential at the end of the reverse sweep; defaults to ``E_start``.

    Returns
    -------
    t, E : numpy.ndarray, shape (n_points,)
        Time (s) and potential (V) arrays. ``t`` is uniformly spaced; ``E`` is
        the triangular ramp.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    if v <= 0:
        raise ValueError("sweep rate v must be positive")
    if E_end is None:
        E_end = E_start
    d_fwd = abs(E_switch - E_start)
    d_rev = abs(E_end - E_switch)
    total = d_fwd + d_rev
    duration = total / v
    t = np.linspace(0.0, duration, n_points)
    t_switch = d_fwd / v
    E = np.where(
        t <= t_switch,
        E_start + np.sign(E_switch - E_start) * v * t,
        E_switch + np.sign(E_end - E_switch) * v * (t - t_switch),
    )
    return t, E


def potential_step(E_initial, E_final, t_total, n_points, t_step=0.0):
    """Single potential step (chronoamperometry / Cottrell experiment).

    ``E = E_initial`` for ``t < t_step`` and ``E = E_final`` for ``t >= t_step``.

    Parameters
    ----------
    E_initial, E_final : float
        Potentials before and after the step (V).
    t_total : float
        Total experiment duration (s).
    n_points : int
        Number of samples (>= 2).
    t_step : float, optional
        Time of the step (s); default 0 (step at the start).

    Returns
    -------
    t, E : numpy.ndarray, shape (n_points,)
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    t = np.linspace(0.0, t_total, n_points)
    E = np.where(t >= t_step, E_final, E_initial)
    return t, E


def pulse_train(base_levels, t_total, n_points):
    """Staircase / pulse-train waveform from a sequence of potential levels.

    The experiment time is split into ``len(base_levels)`` equal segments, each
    held at the corresponding level.  Use this for staircase CV and the
    base waveforms of normal-/differential-pulse voltammetry (Chapter 8).

    Parameters
    ----------
    base_levels : sequence of float
        Potential held during each successive equal-duration segment (V).
    t_total : float
        Total duration (s).
    n_points : int
        Number of samples (>= len(base_levels)).

    Returns
    -------
    t, E : numpy.ndarray, shape (n_points,)
    """
    base_levels = np.asarray(base_levels, dtype=float)
    n_seg = base_levels.size
    if n_seg == 0:
        raise ValueError("base_levels must be non-empty")
    if n_points < n_seg:
        raise ValueError("n_points must be >= number of levels")
    t = np.linspace(0.0, t_total, n_points)
    seg = np.minimum((t / t_total * n_seg).astype(int), n_seg - 1)
    E = base_levels[seg]
    return t, E


def ac_superposition(t, E_dc, amplitude, frequency, phase=0.0):
    """Add a sinusoidal AC perturbation to a DC waveform (AC voltammetry).

    ``E(t) = E_dc(t) + amplitude * sin(2 pi frequency t + phase)``.

    Parameters
    ----------
    t : array_like
        Time samples (s).
    E_dc : array_like
        DC waveform sampled at ``t`` (V) -- e.g. the output of
        :func:`linear_sweep` or :func:`cyclic_sweep`.
    amplitude : float
        AC amplitude (V).
    frequency : float
        AC frequency (Hz).
    phase : float, optional
        Phase offset (rad).

    Returns
    -------
    numpy.ndarray
        The superposed potential, same shape as ``t``.
    """
    t = np.asarray(t, dtype=float)
    E_dc = np.asarray(E_dc, dtype=float)
    return E_dc + amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def dimensionless_sweep_rate(v, n_electrons=1, temperature=298.15):
    """Dimensionless sweep parameter ``sigma = n F v / (R T)`` (units 1/s).

    From SERM Ch. 5: the CV problem is non-dimensionalised in time by ``sigma``,
    so the dimensionless potential axis is ``sigma * t`` (see
    ``Chapters/chapter5.nb``).

    Parameters
    ----------
    v : float
        Sweep rate (V/s).
    n_electrons : int
        Number of electrons transferred.
    temperature : float
        Temperature (K); default 298.15.

    Returns
    -------
    float
    """
    return n_electrons * F * v / (R * temperature)


def nernst_theta(E, E0=0.0, n_electrons=1, temperature=298.15):
    """Nernstian surface ratio ``theta = exp[n F (E - E0)/(R T)]``.

    For a reversible couple ``O + n e- <=> R`` the surface concentration ratio
    is ``c_O / c_R = exp[n F (E - E0)/(R T)]``; the diffusion-limited reversible
    voltammogram uses ``c_O(surface) = 1/(1 + 1/theta)`` (SERM Ch. 5 boundary
    condition ``xi[y]/(1 + xi[y])`` with ``xi = 1/theta``).

    Parameters
    ----------
    E : array_like
        Applied potential (V).
    E0 : float
        Formal potential (V).
    n_electrons : int
        Electrons transferred.
    temperature : float
        Temperature (K).

    Returns
    -------
    numpy.ndarray
        ``theta`` at each potential.
    """
    E = np.asarray(E, dtype=float)
    return np.exp(n_electrons * F * (E - E0) / (R * temperature))
