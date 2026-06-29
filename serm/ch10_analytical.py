"""Analytical and semi-analytical companions to the thin-layer / thin-film
finite-difference solvers of SERM Chapter 10.

The implicit finite-difference solvers live in
:mod:`serm.ch10_thin_layers_and_films`.  This module adds the *analytical half*
of the chapter:

* :func:`theta2_kernel` -- the inverse-Laplace kernel
  :math:`\\mathcal{L}^{-1}\\{\\tanh\\sqrt{s}/\\sqrt{s}\\}(t)
  = 2\\sum_{n\\ge 0} e^{-\\pi^2 t (n+\\tfrac12)^2}`, i.e. an
  EllipticTheta-:math:`\\theta_2` series, with its fast-/slow-sweep limits.
* :func:`thin_film_convolution` -- the theta-function convolution that gives the
  reversible thin-film current directly from the Laplace solution, bypassing the
  spatial grid.  This is the "worked example" of Honeychurch's analytical
  section re-implemented in numpy.
* :func:`method_of_lines_cv` -- an NDSolve-equivalent solver: discretise space,
  integrate the resulting ODE system in time with a stiff solver
  (:func:`scipy.integrate.solve_ivp`, ``BDF``).  An independent cross-check of
  the implicit FD scheme.

Dimensionless conventions match :mod:`serm.ch10_thin_layers_and_films`:

``theta = nF (E - E0') / RT`` is the dimensionless potential; ``xi = exp(theta)``
is the surface ratio; the single controlling group is
``L_param = L^2 sigma / D`` (large => fast sweep / semi-infinite-like, small =>
slow sweep / true thin-layer).  The dimensionless current is
``chi = i / (n F A c* L sigma)``.

Derivation (verified symbolically in the chapter notebook).  In Laplace space
Fick's second law on ``x in [0, 1]`` with ``c(x,0)=1`` is
``Cbar'' - s Cbar + 1 = 0`` so ``Cbar = 1/s + C1 cosh(sqrt(s) x)
+ C2 sinh(sqrt(s) x)``.  The impermeable-wall condition ``Cbar'(1)=0`` forces
``C2 = -C1 tanh(sqrt(s))``, and the surface flux is
``-Cbar'(0) = C1 sqrt(s) tanh(sqrt(s))``.  Eliminating ``C1`` with the reversible
surface value ``Cbar(0) = [xi/(1+xi)]/s`` gives the current transfer function

    i(s) = -[xi/(1+xi)] * tanh(sqrt(s)) / sqrt(s).

Inverting the kernel ``tanh(sqrt(s))/sqrt(s)`` term by term yields the theta
series above, and the time-domain current is the convolution implemented in
:func:`thin_film_convolution`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve


# --------------------------------------------------------------------------- #
# 1.  The theta-function inverse-Laplace kernel                               #
# --------------------------------------------------------------------------- #
def theta2_kernel(t: np.ndarray, n_terms: int = 600) -> np.ndarray:
    """Inverse-Laplace kernel ``L^{-1}{ tanh(sqrt s)/sqrt s }(t)``.

    Equal to the EllipticTheta-:math:`\\theta_2` series

    .. math::

        k(t) = 2 \\sum_{n=0}^{\\infty} \\exp\\!\\big[-\\pi^2 t (n+\\tfrac12)^2\\big].

    Limits (both used as validation targets):

    * **fast sweep** (small ``t``): ``k(t) -> 1/sqrt(pi t)`` (the Cottrell /
      semi-infinite kernel);
    * **slow sweep** (large ``t``): ``k(t) -> 2 exp(-pi^2 t / 4)`` (the leading
      ``n = 0`` term).

    Parameters
    ----------
    t : array_like
        Dimensionless time (``>= 0``).  Non-positive entries return ``0``.
    n_terms : int
        Number of series terms.  600 is far more than enough except at very
        small ``t`` where the Cottrell form should be used instead.

    Returns
    -------
    numpy.ndarray
        ``k(t)`` with the same shape as ``t``.
    """
    t = np.asarray(t, dtype=float)
    n = np.arange(n_terms).reshape((-1,) + (1,) * t.ndim)
    tt = np.maximum(t, 0.0)
    series = 2.0 * np.exp(-(np.pi ** 2) * tt[None, ...] * (n + 0.5) ** 2).sum(axis=0)
    return np.where(t > 0.0, series, 0.0)


def cottrell_kernel(t: np.ndarray) -> np.ndarray:
    """Fast-sweep limit of :func:`theta2_kernel`: ``1/sqrt(pi t)``.

    This is the semi-infinite (Cottrell) kernel; using it in
    :func:`thin_film_convolution` recovers the ordinary semi-infinite linear
    sweep voltammogram, the ``L_param -> infinity`` limit of the thin film.
    """
    t = np.asarray(t, dtype=float)
    return np.where(t > 0.0, 1.0 / np.sqrt(np.pi * np.maximum(t, np.finfo(float).tiny)), 0.0)


# --------------------------------------------------------------------------- #
# 2.  Theta-function convolution (semi-analytical thin-film current)          #
# --------------------------------------------------------------------------- #
@dataclass
class CVCurve:
    """A potential / current pair for a thin-layer or thin-film voltammogram.

    Attributes
    ----------
    theta : numpy.ndarray
        Dimensionless potential ``nF (E - E0')/RT`` along the triangular sweep.
    current : numpy.ndarray
        Dimensionless current ``chi``.
    """

    theta: np.ndarray
    current: np.ndarray


def thin_film_convolution(
    L_param: float = 1.0,
    theta_start: float = 10.0,
    theta_min: float = -10.0,
    n_half: int = 6000,
    kernel: str = "theta",
) -> CVCurve:
    """Reversible thin-film voltammogram by theta-function convolution.

    Evaluates the inverse transform of the Laplace current

    .. math:: i(s) = -\\frac{\\xi}{1+\\xi}\\,\\frac{\\tanh\\sqrt s}{\\sqrt s}

    as a convolution of the surface-coverage rate
    ``d/dt[xi/(1+xi)] = xi/(1+xi)^2 * dtheta/dt`` with :func:`theta2_kernel`.
    No spatial grid is used.

    The mapping between the diffusion time ``t`` of the kernel and the sweep is
    ``theta = theta_start - L_param * t`` on the forward branch (mirrored on the
    reverse), because ``dtheta/dt = -sigma L^2/D = -L_param``.  The current
    normalisation that matches :mod:`serm.ch10_thin_layers_and_films`
    (``chi = i/(nFAc*L sigma)``) introduces an overall factor ``1/L_param``.

    Parameters
    ----------
    L_param : float
        Dimensionless group ``L^2 sigma / D``.  Large => fast sweep.
    theta_start, theta_min : float
        Sweep extremes in dimensionless potential.
    n_half : int
        Samples per sweep branch.  The convolution converges to the
        finite-difference result as this is refined.
    kernel : {"theta", "cottrell"}
        ``"theta"`` uses the exact thin-film kernel; ``"cottrell"`` uses its
        fast-sweep limit ``1/sqrt(pi t)`` (i.e. forces semi-infinite diffusion).

    Returns
    -------
    CVCurve
    """
    span = theta_start - theta_min
    t_half = span / L_param
    t = np.linspace(0.0, 2.0 * t_half, 2 * n_half + 1)
    dt = t[1] - t[0]

    theta = np.where(
        t <= t_half,
        theta_start - L_param * t,
        theta_min + L_param * (t - t_half),
    )
    xi = np.exp(theta)
    coverage = xi / (1.0 + xi)               # xi/(1+xi) = oxidised fraction
    rate = np.gradient(coverage, dt)         # d/dt of surface coverage

    if kernel == "theta":
        K = theta2_kernel(t)
    elif kernel == "cottrell":
        K = cottrell_kernel(t)
    else:                                    # pragma: no cover - guard
        raise ValueError(f"unknown kernel {kernel!r}")

    chi = -fftconvolve(rate, K)[: t.size] * dt / L_param
    return CVCurve(theta, chi)


# --------------------------------------------------------------------------- #
# 3.  Method of lines (NDSolve-equivalent) cross-check                        #
# --------------------------------------------------------------------------- #
def method_of_lines_cv(
    L_param: float = 0.3,
    geometry: str = "layer",
    m: int = 81,
    theta_start: float = 10.0,
    theta_min: float = -10.0,
    n_out: int = 4001,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    skip_fraction: float = 0.02,
) -> CVCurve:
    """Reversible thin-layer / thin-film CV by the method of lines.

    Discretises ``x in [0, 1]`` on ``m`` nodes and integrates the resulting
    stiff ODE system ``dc_j/dt = (c_{j+1} - 2 c_j + c_{j-1})/dx^2`` in time with
    :func:`scipy.integrate.solve_ivp` (``method="BDF"``).  This mirrors
    Honeychurch's ``NDSolve`` approach and is an *independent* discretisation
    from the implicit tridiagonal solver, so agreement is a genuine cross-check.

    The electrode boundary is the **reversible** (Nernstian) Dirichlet value
    ``c(0, t) = 1/(1 + xi(t))``.  The far boundary is

    * ``geometry="film"``  -- impermeable wall, ``c_{m-1}=(4 c_{m-2}-c_{m-3})/3``
      (zero three-point flux), and the current is the single-electrode flux;
    * ``geometry="layer"`` -- a second reactive electrode
      ``c(1, t) = 1/(1+xi(t))``, and the current sums the flux at both faces.

    The first ``skip_fraction`` of the time record is dropped to remove the
    ``NDSolve`` start-up transient created by the discontinuous initial drop of
    the surface boundary value (Honeychurch flags the same discontinuity).

    Parameters
    ----------
    L_param : float
        Dimensionless group ``L^2 sigma / D``.
    geometry : {"layer", "film"}
        Two reactive electrodes vs one electrode plus an impermeable wall.
    m : int
        Number of spatial nodes.
    theta_start, theta_min : float
        Sweep extremes in dimensionless potential.
    n_out : int
        Number of output times.
    rtol, atol : float
        Tolerances passed to ``solve_ivp``.
    skip_fraction : float
        Fraction of the leading time record discarded (start-up transient).

    Returns
    -------
    CVCurve
    """
    if geometry not in ("layer", "film"):
        raise ValueError(f"geometry must be 'layer' or 'film', got {geometry!r}")

    x = np.linspace(0.0, 1.0, m)
    dx = x[1] - x[0]
    span = theta_start - theta_min
    t_half = span / L_param

    def theta_of(t):
        return np.where(t <= t_half, theta_start - L_param * t,
                        theta_min + L_param * (t - t_half))

    def csurf(t):
        return 1.0 / (1.0 + np.exp(theta_of(t)))

    def rhs(t, c):
        full = np.empty(m)
        full[1:-1] = c
        full[0] = csurf(t)
        if geometry == "film":
            full[-1] = (4.0 * c[-1] - c[-2]) / 3.0      # zero-flux wall
        else:
            full[-1] = csurf(t)                          # second electrode
        return (full[2:] - 2.0 * full[1:-1] + full[:-2]) / (dx * dx)

    t_eval = np.linspace(0.0, 2.0 * t_half, n_out)
    sol = solve_ivp(
        rhs, (0.0, 2.0 * t_half), np.ones(m - 2),
        method="BDF", t_eval=t_eval, rtol=rtol, atol=atol,
    )

    c0 = csurf(sol.t)
    grad0 = (-3.0 * c0 + 4.0 * sol.y[0] - sol.y[1]) / (2.0 * dx)
    if geometry == "film":
        chi = grad0 / L_param
    else:
        grad1 = (-3.0 * c0 + 4.0 * sol.y[-1] - sol.y[-2]) / (2.0 * dx)
        chi = (grad0 + grad1) / L_param

    keep = sol.t > skip_fraction * (2.0 * t_half)
    return CVCurve(theta_of(sol.t)[keep], chi[keep])
