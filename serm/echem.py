"""Independent analytic reference results for validation.

Each chapter validates its finite-difference simulation against an *independent*
closed-form or limiting-case result.  Collecting those formulas here (rather
than re-deriving them inside every notebook) makes the validation cells short
and keeps a single, documented source of truth.  Every function cites the
standard form of the law it implements; the implementations are derived
independently of Honeychurch's numerical code, so they are a genuine cross-check
rather than a copy of the book's output.

Standard references for these results: A. J. Bard & L. R. Faulkner,
*Electrochemical Methods*, 2nd ed. (Wiley, 2001); the same expressions appear in
the relevant chapters of Honeychurch's SERM.

Constants are SI; potentials in V, currents in A, areas in cm^2 or m^2 as noted
in each docstring (be consistent with your own simulation's units when
comparing).
"""
from __future__ import annotations

import numpy as np

# Physical constants (SI).
F = 96485.33212        # Faraday constant, C/mol
R = 8.314462618        # gas constant, J/(mol K)


def cottrell_current(t, n, A, D, c_bulk):
    """Cottrell equation -- current after a diffusion-limited potential step.

    .. math::
        i(t) = \\frac{n F A \\sqrt{D}\\, c^*}{\\sqrt{\\pi t}}

    (Bard & Faulkner, 2nd ed., eq. 5.2.11.)  Validates Chapters 2 and 8.

    Parameters
    ----------
    t : array_like
        Time (s), ``> 0``.
    n : int
        Electrons transferred.
    A : float
        Electrode area (cm^2 if ``D`` is cm^2/s and ``c_bulk`` mol/cm^3).
    D : float
        Diffusion coefficient (cm^2/s).
    c_bulk : float
        Bulk concentration of the electroactive species (mol/cm^3).

    Returns
    -------
    numpy.ndarray
        Current (A). ``t = 0`` yields ``inf``.

    Notes
    -----
    Keep units self-consistent: with ``A`` in cm^2, ``D`` in cm^2/s and
    ``c_bulk`` in mol/cm^3 the result is in amperes.
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore"):
        return n * F * A * np.sqrt(D) * c_bulk / np.sqrt(np.pi * t)


def randles_sevcik_peak_current(n, A, D, c_bulk, v, temperature=298.15,
                                reversible=True, alpha=0.5):
    """Peak current of a (reversible) linear-sweep / cyclic voltammogram.

    Reversible (Randles--Sevcik):

    .. math::
        i_p = 0.4463\\, n F A c^* \\sqrt{\\frac{n F v D}{R T}}

    At 298.15 K this is the familiar ``i_p = 2.69e5 n^{3/2} A D^{1/2} c^* v^{1/2}``
    form (Bard & Faulkner, 2nd ed., eq. 6.2.19).  For an irreversible wave the
    coefficient ``0.4463`` is replaced by ``0.4958`` and ``n`` inside the square
    root by ``alpha n_a`` (eq. 6.3.8); set ``reversible=False`` for that case.

    Parameters
    ----------
    n : int
        Electrons transferred.
    A : float
        Electrode area (cm^2).
    D : float
        Diffusion coefficient (cm^2/s).
    c_bulk : float
        Bulk concentration (mol/cm^3).
    v : float
        Sweep rate (V/s).
    temperature : float
        Temperature (K).
    reversible : bool
        If False, use the totally irreversible coefficient.
    alpha : float
        Transfer coefficient (only used when ``reversible=False``).

    Returns
    -------
    float
        Peak current magnitude (A).
    """
    coeff = 0.4463 if reversible else 0.4958
    n_in_root = n if reversible else alpha * n
    return (coeff * n * F * A * c_bulk
            * np.sqrt(n_in_root * F * v * D / (R * temperature)))


def sand_transition_time(n, A, D, c_bulk, i_applied):
    """Sand equation -- transition time in chronopotentiometry.

    .. math::
        \\tau = \\frac{\\pi D (n F A c^*)^2}{4 i^2}

    i.e. ``i \\sqrt{\\tau} = \\tfrac12 n F A c^* \\sqrt{\\pi D}`` (Bard &
    Faulkner, 2nd ed., eq. 8.2.4).  Validates Chapter 9.

    Parameters
    ----------
    n : int
        Electrons transferred.
    A : float
        Electrode area (cm^2).
    D : float
        Diffusion coefficient (cm^2/s).
    c_bulk : float
        Bulk concentration (mol/cm^3).
    i_applied : float
        Magnitude of the applied constant current (A).

    Returns
    -------
    float
        Transition time tau (s).
    """
    return np.pi * D * (n * F * A * c_bulk) ** 2 / (4.0 * i_applied ** 2)


def levich_current(n, A, D, c_bulk, omega, nu):
    """Levich equation -- limiting current at a rotating disk electrode.

    .. math::
        i_L = 0.620\\, n F A D^{2/3} \\nu^{-1/6} \\omega^{1/2} c^*

    (Bard & Faulkner, 2nd ed., eq. 9.3.22; ``omega`` in rad/s.)  Validates
    Chapter 14.

    Parameters
    ----------
    n : int
        Electrons transferred.
    A : float
        Electrode area (cm^2).
    D : float
        Diffusion coefficient (cm^2/s).
    c_bulk : float
        Bulk concentration (mol/cm^3).
    omega : float
        Angular rotation rate (rad/s).
    nu : float
        Kinematic viscosity (cm^2/s).

    Returns
    -------
    float
        Levich limiting current (A).
    """
    return (0.620 * n * F * A * D ** (2.0 / 3.0)
            * nu ** (-1.0 / 6.0) * omega ** 0.5 * c_bulk)


def koutecky_levich_current(i_kinetic, n, A, D, c_bulk, omega, nu):
    """Koutecky--Levich combination of kinetic and mass-transport currents.

    .. math::
        \\frac{1}{i} = \\frac{1}{i_k} + \\frac{1}{i_L}

    where ``i_L`` is the Levich current (Bard & Faulkner, 2nd ed., eq. 9.3.39).
    Returns the measured current ``i`` for a given kinetic current ``i_k``.

    Parameters
    ----------
    i_kinetic : float
        Kinetically limited current ``i_k`` (A).
    n, A, D, c_bulk, omega, nu :
        As in :func:`levich_current`.

    Returns
    -------
    float
        Net current ``i`` (A).
    """
    i_L = levich_current(n, A, D, c_bulk, omega, nu)
    return 1.0 / (1.0 / i_kinetic + 1.0 / i_L)


def surface_wave_peak_current(n, A, gamma, v, temperature=298.15):
    """Peak current of a reversible surface-confined (adsorbed) redox couple.

    For a strongly adsorbed, reversibly reacting monolayer the CV peak current
    is

    .. math::
        i_p = \\frac{n^2 F^2 A \\Gamma^* v}{4 R T}

    (Bard & Faulkner, 2nd ed., eq. 14.3.11).  The peak is symmetric and grows
    *linearly* with sweep rate ``v`` (contrast the ``sqrt(v)`` of a diffusing
    species).  Validates Chapter 11.

    Parameters
    ----------
    n : int
        Electrons transferred.
    A : float
        Electrode area (cm^2).
    gamma : float
        Surface coverage ``Gamma*`` (mol/cm^2).
    v : float
        Sweep rate (V/s).
    temperature : float
        Temperature (K).

    Returns
    -------
    float
        Surface-wave peak current (A).
    """
    return (n ** 2 * F ** 2 * A * gamma * v) / (4.0 * R * temperature)
