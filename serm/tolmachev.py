r"""Unified convective-diffusion formalism of Tolmachev, Wang & Scherson (1996).

New material for *Simulating Electrochemical Reactions in Python*, going beyond
Honeychurch's single rotating-disk chapter to a **single mathematical
framework** that treats the rotating disk, rotating ring-disk, channel, and
tube electrodes at once.

Reference
---------
Y. V. Tolmachev, Z. Wang & D. A. Scherson, *"In Situ Spectroscopy in the
Presence of Convective Flow under Steady-State Conditions: A Unified
Mathematical Formalism,"* J. Electrochem. Soc. **143**, 3539-3548 (1996).
The inverse-Laplace convolution kernels (the "R" relations) are the transforms
tabulated by W. G. Sutton, Proc. R. Soc. **182**, 48 (1943).

The unified idea
----------------
For each geometry the steady convective-diffusion equation, after the Levich
near-wall linearisation of the axial velocity and the geometry-specific change
of variables in Table I of the paper, collapses to **one** dimensionless
equation (paper Eq. I-5):

.. math::
    Y\,\frac{\partial \theta}{\partial X} = \frac{\partial^2 \theta}{\partial Y^2},

with :math:`\theta \to 0` as :math:`X \to 0` and as :math:`Y \to \infty`.
Here :math:`X` is a dimensionless coordinate along the flow and :math:`Y` a
dimensionless coordinate normal to the wall; the geometry only enters through
the scale factors that define :math:`X` and :math:`Y` (see
:func:`geometry_scales`).

A Laplace transform in :math:`X` (variable :math:`s`) turns Eq. I-5 into the
Airy equation :math:`\bar\theta_{YY} = s\,Y\,\bar\theta`, whose bounded solution
is

.. math::
    \bar\theta(s, Y) = A_1(s)\,\mathrm{Ai}(s^{1/3} Y).

Matching at :math:`Y = 0` and using :math:`\mathrm{Ai}(0) =
1/[3^{2/3}\Gamma(2/3)]`, :math:`\mathrm{Ai}'(0) = -1/[3^{1/3}\Gamma(1/3)]`
gives the surface flux/concentration reciprocity in Laplace space
(paper Eqs. L1-L4, Table II) and, after inversion (Sutton), the real-space
fractional-integral relations R1-R4:

* **R3** :math:`\;\theta(X,0) = -\dfrac{1}{3^{1/3}\Gamma(2/3)}
  \displaystyle\int_0^X \theta_Y|_{0}(x)\,(X-x)^{-2/3}\,dx`
* **R4** :math:`\;\theta_Y|_{0}(X) = -\dfrac{3^{1/3}}{\Gamma(1/3)}
  \displaystyle\int_0^X (X-x)^{-1/3}\,d\theta(x,0)`

These are the two workhorses used throughout this module.

What this module provides
-------------------------
* :func:`airy_profile` -- the bounded Airy solution
  :math:`\bar\theta(s,Y)/\bar\theta(s,0)` in Laplace space.
* :func:`flux_prefactor` -- the constant :math:`3^{1/3}\Gamma(2/3)/\Gamma(1/3)
  = -\mathrm{Ai}'(0)/\mathrm{Ai}(0)` relating surface flux and concentration.
* :func:`surface_conc_from_flux` / :func:`flux_from_surface_conc` -- the R3/R4
  convolution (Abel / fractional-integral) relations, built on the
  cell-integrated Riemann-Liouville kernels of :mod:`serm.semiintegration`.
* :func:`solve_first_order_surface` -- the Abel integral equation for the
  first-order irreversible surface reaction (the user's convolution notebook).
* :func:`levich_flux_coefficient`, :func:`channel_prefactor`,
  :func:`tubular_prefactor` -- closed-form limiting-current anchors *derived
  from the unified formalism* (the Airy ``Ai'(0)/Ai(0)`` flux constant and the
  Levich :math:`X^{-1/3}` mean-flux integral).  These are then compared against
  genuinely external anchors (the textbook Levich constant 0.620, the B&F
  Table 11.6.1 channel/tube values 1.47/1.61).
* :func:`collection_efficiency` / :func:`F_collection` -- a thin re-export of
  the Albery--Bruckenstein RRDE closed form already in :mod:`serm.rrde` (B&F
  eqs. 9.4.16/9.4.17).  This is the *electrochemical* collection efficiency; it
  is **not** an independent Tolmachev-formalism result (see the docstrings).
  The paper's own ring quantity is the distinct *spectroscopic* collection
  efficiency of Table III (its Eqs. 57/61/64), a different physical quantity.
* :func:`geometry_scales` -- the Table I dimensionless-variable scale factors
  for the four geometries.

All interfacial reactions are assumed first-order and irreversible with no
coupled homogeneous chemistry, exactly the regime the paper is valid in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import airy, gamma

from . import semiintegration

__all__ = [
    "AI0",
    "AIP0",
    "flux_prefactor",
    "airy_profile",
    "R3_kernel",
    "surface_conc_from_flux",
    "flux_from_surface_conc",
    "FirstOrderSurface",
    "solve_first_order_surface",
    "levich_flux_coefficient",
    "channel_prefactor",
    "tubular_prefactor",
    "F_collection",
    "collection_efficiency",
    "geometry_scales",
]

# --------------------------------------------------------------------------- #
# Airy anchors (paper, discussion around Eqs. 3-7).                            #
# --------------------------------------------------------------------------- #
#: :math:`\mathrm{Ai}(0) = 3^{-2/3}/\Gamma(2/3)`.
AI0: float = 1.0 / (3.0 ** (2.0 / 3.0) * gamma(2.0 / 3.0))
#: :math:`\mathrm{Ai}'(0) = -3^{-1/3}/\Gamma(1/3)`.
AIP0: float = -1.0 / (3.0 ** (1.0 / 3.0) * gamma(1.0 / 3.0))


def flux_prefactor() -> float:
    r"""Surface flux/concentration prefactor :math:`3^{1/3}\Gamma(2/3)/\Gamma(1/3)`.

    This is :math:`-\mathrm{Ai}'(0)/\mathrm{Ai}(0)`, the constant appearing in
    paper Eq. 5, :math:`\theta_Y|_0 = -3^{1/3}[\Gamma(2/3)/\Gamma(1/3)]
    \bar\theta(s,0)\,s^{1/3}` in Laplace space.  Its reciprocal times
    :math:`X^{-1/3}` is the Levich diffusion-limited flux (Eq. 26a).

    Returns
    -------
    float
        :math:`3^{1/3}\Gamma(2/3)/\Gamma(1/3) \approx 0.72901`.
    """
    return 3.0 ** (1.0 / 3.0) * gamma(2.0 / 3.0) / gamma(1.0 / 3.0)


def airy_profile(s: NDArray[np.float64] | float, Y: NDArray[np.float64] | float
                 ) -> NDArray[np.float64]:
    r"""Laplace-space concentration ratio :math:`\bar\theta(s,Y)/\bar\theta(s,0)`.

    From paper Eq. 3-4, the bounded solution is
    :math:`\bar\theta(s,Y)=A_1(s)\mathrm{Ai}(s^{1/3}Y)`; dividing out the
    surface value gives :math:`\mathrm{Ai}(s^{1/3}Y)/\mathrm{Ai}(0)`, which
    decays monotonically from 1 at the wall to 0 in the bulk.

    Parameters
    ----------
    s : float or ndarray
        Laplace variable (:math:`> 0`).
    Y : float or ndarray
        Dimensionless normal coordinate (:math:`\ge 0`).

    Returns
    -------
    ndarray
        :math:`\bar\theta(s,Y)/\bar\theta(s,0)`.
    """
    s = np.asarray(s, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ai, _, _, _ = airy(np.cbrt(s) * Y)
    return ai / AI0


# --------------------------------------------------------------------------- #
# R3 / R4 real-space convolution (Sutton inverse transforms).                 #
# --------------------------------------------------------------------------- #
def R3_kernel(n: int, dX: float) -> NDArray[np.float64]:
    r"""Cell-integrated :math:`(X-x)^{-2/3}` kernel for the R3 relation.

    R3 (paper Table II) is
    :math:`\theta(X,0)=-\frac{1}{3^{1/3}\Gamma(2/3)}\int_0^X
    \theta_Y|_0\,(X-x)^{-2/3}dx`, i.e. a Riemann-Liouville fractional integral
    of order :math:`1/3` times :math:`\Gamma(1/3)`.  We reuse the
    cell-integrated RL weights of :func:`serm.semiintegration.rl_kernel` (order
    :math:`q=1/3`) so the integrable singularity is handled exactly, then
    fold in the R3 constant.

    Returns
    -------
    ndarray, shape (n,)
        Convolution weights realising ``-1/(3^(1/3) Gamma(2/3))`` times the
        :math:`(X-x)^{-2/3}` fractional integral, so
        ``theta_surface = kernel (*) flux_surface``.
    """
    # rl_kernel(q=1/3) gives weights for (1/Gamma(1/3)) int f (X-x)^{-2/3} dx.
    # R3 wants -(1/(3^(1/3) Gamma(2/3))) int flux (X-x)^{-2/3} dx.
    # Ratio of constants = -Gamma(1/3)/(3^(1/3) Gamma(2/3)).
    rl = semiintegration.rl_kernel(n, dX, q=1.0 / 3.0)
    const = -gamma(1.0 / 3.0) / (3.0 ** (1.0 / 3.0) * gamma(2.0 / 3.0))
    return const * rl


def surface_conc_from_flux(flux_surface: NDArray[np.float64], dX: float
                           ) -> NDArray[np.float64]:
    r"""Surface concentration :math:`\theta(X,0)` from the wall flux (R3).

    Evaluates paper Table II R3,
    :math:`\theta(X,0)=-\frac{1}{3^{1/3}\Gamma(2/3)}\int_0^X
    \theta_Y|_0(x)(X-x)^{-2/3}dx`, on a uniform :math:`X` grid by discrete
    convolution with :func:`R3_kernel`.

    Parameters
    ----------
    flux_surface : ndarray
        Wall flux :math:`\theta_Y|_0(X)` sampled on ``X = 0, dX, 2 dX, ...``.
    dX : float
        Uniform step in :math:`X`.

    Returns
    -------
    ndarray
        Surface concentration :math:`\theta(X,0)` on the same grid.
    """
    flux_surface = np.asarray(flux_surface, dtype=float)
    n = flux_surface.size
    k = R3_kernel(n, dX)
    return np.convolve(flux_surface, k)[:n]


def flux_from_surface_conc(surface_conc: NDArray[np.float64], dX: float
                           ) -> NDArray[np.float64]:
    r"""Wall flux :math:`\theta_Y|_0(X)` from the surface concentration (R4).

    Evaluates paper Table II R4,
    :math:`\theta_Y|_0(X)=-\frac{3^{1/3}}{\Gamma(1/3)}\int_0^X
    (X-x)^{-1/3}\,d\theta(x,0)` (a Stieltjes integral).  For a smooth
    :math:`\theta(X,0)` starting at 0 we integrate by parts to the equivalent
    :math:`(X-x)^{-2/3}` convolution against :math:`d\theta/dx`.  R3 and R4 are
    exact inverses of one another for the surface pair (the Beta identity of
    paper Eq. 46).  Here we implement **R4 directly** as the fractional integral
    of order :math:`2/3` of :math:`d\theta/dx` (we do *not* reuse the R3 kernel;
    the R3 relation is implemented separately in :func:`surface_conc_from_flux`).

    Under diffusion-limiting conditions (:math:`\theta(X,0)=1` for :math:`X>0`)
    the Stieltjes measure is a step at the origin and this reduces to the
    Levich form :math:`-(3^{1/3}/\Gamma(1/3))X^{-1/3}` (paper Eq. 26a); see
    :func:`levich_flux_coefficient`.

    Parameters
    ----------
    surface_conc : ndarray
        :math:`\theta(X,0)` sampled on ``X = 0, dX, 2 dX, ...`` (with
        ``surface_conc[0]`` the value just off the leading edge).
    dX : float
        Uniform step in :math:`X`.

    Returns
    -------
    ndarray
        Wall flux :math:`\theta_Y|_0(X)` on the same grid.
    """
    theta = np.asarray(surface_conc, dtype=float)
    n = theta.size
    # d(theta)/dx by a forward/one-sided difference; theta(0)=0 at the edge.
    dtheta = np.empty(n)
    dtheta[0] = theta[0] / dX
    dtheta[1:] = (theta[1:] - theta[:-1]) / dX
    # R4 = -(3^(1/3)/Gamma(1/3)) int (X-x)^{-1/3} dtheta
    #    = -(3^(1/3)/Gamma(1/3)) * Gamma(2/3) * [RL fractional integral order 2/3 of dtheta]
    rl = semiintegration.rl_kernel(n, dX, q=2.0 / 3.0)  # (1/Gamma(2/3)) int f (X-x)^{-1/3}
    const = -(3.0 ** (1.0 / 3.0) / gamma(1.0 / 3.0)) * gamma(2.0 / 3.0)
    return const * np.convolve(dtheta, rl)[:n]


# --------------------------------------------------------------------------- #
# First-order irreversible surface reaction (the user's convolution notebook).#
# --------------------------------------------------------------------------- #
@dataclass
class FirstOrderSurface:
    """Solution of the first-order irreversible surface problem.

    Attributes
    ----------
    X : ndarray
        Dimensionless flow coordinate grid.
    theta0 : ndarray
        Surface concentration :math:`\\theta(X,0)`.
    flux : ndarray
        Wall flux :math:`\\theta_Y|_0(X)` (:math:`= (\\sigma/\\alpha)[\\theta(X,0)-1]`
        style closure; here ``flux = -alpha^{-1}[theta0 - 1]`` per Eq. 16).
    sigma : float
        Dimensionless rate parameter :math:`\\sigma`.
    """

    X: NDArray[np.float64]
    theta0: NDArray[np.float64]
    flux: NDArray[np.float64]
    sigma: float


def solve_first_order_surface(sigma: float, X_max: float = 1.0, n: int = 2001
                              ) -> FirstOrderSurface:
    r"""Solve the Abel integral equation for a first-order surface reaction.

    The paper closes the R-relations with the first-order irreversible surface
    boundary condition (Eq. 16),
    :math:`\theta_Y|_0 = \frac{1}{\alpha}\sigma^{1/3}[\theta(X,0)-1]`,
    with :math:`\alpha = 3^{-1/3}\Gamma(1/3)/\Gamma(2/3)`.  Combining with R3
    (which expresses :math:`\theta(X,0)` as the :math:`(X-x)^{-2/3}` fractional
    integral of the flux) gives, for the surface concentration deficit
    :math:`u(X)=1-\theta(X,0)`, the Abel-type Volterra equation of the second
    kind written in the user's convolution notebook,

    .. math::
        \theta(X,0) = K \int_0^X [\,1-\theta(x,0)\,]\,(X-x)^{-2/3}\,dx,
        \qquad K = \frac{\sigma^{1/3}}{3^{1/3}\Gamma(2/3)}\,\frac{1}{\alpha}
                 = \frac{\sigma^{1/3}\,\Gamma(2/3)^{-1}\Gamma(1/3)^{-1}}
                        {\,\cdots}.

    We solve it by the same cell-integrated fractional-integral quadrature used
    for R3, marching the surface concentration outward in :math:`X`.  The
    resulting wall flux follows from Eq. 16.

    Parameters
    ----------
    sigma : float
        Dimensionless rate parameter :math:`\sigma` (:math:`\to\infty`
        recovers the diffusion-limited Levich flux; :math:`\to 0` the
        reaction-limited case).
    X_max : float
        Upper end of the flow coordinate.
    n : int
        Number of grid points (uniform in :math:`X`).

    Returns
    -------
    FirstOrderSurface
    """
    X = np.linspace(0.0, X_max, n)
    dX = X[1] - X[0]

    alpha = 3.0 ** (-1.0 / 3.0) * gamma(1.0 / 3.0) / gamma(2.0 / 3.0)
    # R3:  theta0 = c3 * conv(flux, w) , c3 = -1/(3^(1/3)Gamma(2/3)),
    #      w = cell-integrated (X-x)^{-2/3} weights (times Gamma(1/3)).
    # BC (Eq.16): flux = (sigma^{1/3}/alpha) (theta0 - 1).
    # => theta0(X) = c3 * (sigma^{1/3}/alpha) * conv(theta0 - 1, w)
    #             = -K * conv(1 - theta0, w),  K = c3 * sigma^{1/3}/alpha (<0 times...)
    # Assemble the Riemann-Liouville order-1/3 weights (from rl_kernel):
    rl = semiintegration.rl_kernel(n, dX, q=1.0 / 3.0)  # (1/Gamma(1/3)) int f (X-x)^{-2/3}
    # w_j realises int f (X-x)^{-2/3} dx  ->  multiply rl by Gamma(1/3):
    w = gamma(1.0 / 3.0) * rl
    c3 = -1.0 / (3.0 ** (1.0 / 3.0) * gamma(2.0 / 3.0))
    g = sigma ** (1.0 / 3.0) / alpha            # flux = g (theta0 - 1)

    # theta0[i] = c3 * g * sum_{j=0..i} (theta0[i-j]-1) w[j]
    # solve the diagonal (j=0) term explicitly (Volterra 2nd kind march).
    theta0 = np.zeros(n)
    coef = c3 * g
    w0 = w[0]
    for i in range(1, n):
        # convolution over j=1..i uses already-known theta0[i-j]
        acc = 0.0
        # (theta0[i-j]-1) for j=1..i
        for j in range(1, i + 1):
            acc += (theta0[i - j] - 1.0) * w[j]
        # unknown j=0 term: (theta0[i]-1) w0
        # theta0[i] = coef*(acc + (theta0[i]-1) w0)
        # theta0[i](1 - coef w0) = coef*acc - coef*w0
        theta0[i] = (coef * acc - coef * w0) / (1.0 - coef * w0)

    flux = g * (theta0 - 1.0)
    return FirstOrderSurface(X=X, theta0=theta0, flux=flux, sigma=sigma)


# --------------------------------------------------------------------------- #
# Closed-form limiting-current anchors derived from the formalism.            #
# (Levich / channel / tube: the Airy flux constant + the X^{-1/3} integral.)  #
# --------------------------------------------------------------------------- #
def levich_flux_coefficient() -> float:
    r"""Levich diffusion-limited flux coefficient :math:`3^{1/3}/\Gamma(1/3)`.

    Under diffusion-limiting conditions :math:`\theta(0<X\le1,0)=1`, so R4
    (paper Eq. 26a) gives the wall flux
    :math:`\theta_Y|_0 = -\frac{3^{1/3}}{\Gamma(1/3)}X^{-1/3}`.  The returned
    magnitude :math:`3^{1/3}/\Gamma(1/3)` is the coefficient of :math:`X^{-1/3}`.
    Integrating over :math:`0<X<1` gives the total Levich current; for the
    rotating disk (:math:`X=(r/r_0)^3`) this reproduces the classical Levich
    :math:`0.620\,nFA D^{2/3}\nu^{-1/6}\omega^{1/2}c^*`.

    Returns
    -------
    float
        :math:`3^{1/3}/\Gamma(1/3) \approx 0.53836`.
    """
    return 3.0 ** (1.0 / 3.0) / gamma(1.0 / 3.0)


#: Levich (Leveque) constant :math:`C_0 = (3/2)/[9^{1/3}\Gamma(4/3)]`.
#  Equivalently 3^{1/3}/Gamma(1/3) * (3/2) via Gamma(4/3)=Gamma(1/3)/3.
LEVEQUE_C0: float = 1.5 / (9.0 ** (1.0 / 3.0) * gamma(4.0 / 3.0))


def channel_prefactor() -> float:
    r"""Channel limiting-current prefactor (:math:`\approx 1.47`).

    Re-derived from the unified formalism: the average wall flux over an
    electrode of length :math:`l` under limiting conditions is
    :math:`C_0\,C_b D^{2/3} s^{1/3} l^{-1/3}` with
    :math:`C_0=(3/2)/[9^{1/3}\Gamma(4/3)]` (the integral of the Levich
    :math:`X^{-1/3}` flux, since :math:`\int_0^1 X^{-1/3}dX = 3/2`).  For the
    channel geometry (Table I: :math:`s = 6 v_0/h`, area :math:`A=wl`, volume
    flow :math:`Q=v_0 w h`) this rearranges to
    :math:`C_0\,6^{1/3}\,nFc(DA/h)^{2/3}Q^{1/3}`, i.e. prefactor
    :math:`C_0\,6^{1/3}`.

    The genuine external anchor is the **B&F Table 11.6.1** value 1.47; agreement
    with :func:`serm.convdiff2d.channel_prefactor` is *definitional* (both evaluate
    the same closed form :math:`C_0\,6^{1/3}`), not an independent cross-check.

    Returns
    -------
    float
        :math:`C_0\,6^{1/3} \approx 1.4674` (B&F Table 11.6.1: 1.47).
    """
    return LEVEQUE_C0 * 6.0 ** (1.0 / 3.0)


def tubular_prefactor() -> float:
    r"""Tubular limiting-current prefactor (:math:`\approx 1.61`).

    Re-derived from the unified formalism: for the tube geometry (Table I:
    near-wall shear :math:`s = 4 v_0/R`, perimeter :math:`2\pi R`, volume flow
    :math:`Q=\pi v_0 R^2/2`... the :math:`\pi` powers cancel), the average
    Levich flux integrates to :math:`2\,C_0\,nFc(DA/R)^{2/3}Q^{1/3}`.

    The genuine external anchor is the **B&F Table 11.6.1** value 1.61; agreement
    with :func:`serm.convdiff2d.tubular_prefactor` is *definitional* (same closed
    form :math:`2C_0`), not an independent cross-check.  Note the analytic
    constant is :math:`2C_0 = 1.6151`, which rounds to 1.62; B&F tabulate the
    rounded 1.61 -- a real ~0.3% offset kept honest here.

    Returns
    -------
    float
        :math:`2 C_0 \approx 1.6151` (B&F Table 11.6.1: rounded 1.61).
    """
    return 2.0 * LEVEQUE_C0


def F_collection(theta: float | NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Albery--Bruckenstein auxiliary function :math:`F(\theta)` (B&F eq. 9.4.17).

    .. math::
        F(\theta) = \frac{\sqrt 3}{4\pi}
            \ln\!\frac{(1+\theta^{1/3})^3}{1+\theta}
          + \frac{3}{2\pi}\arctan\frac{2\theta^{1/3}-1}{\sqrt 3} + \frac14 .

    .. warning::
        This is byte-for-byte :func:`serm.rrde.F_albery` (B&F eq. 9.4.17); it is
        re-exposed here only for convenience.  It is **not** an independent
        re-derivation of that closed form.

    The same functional :math:`F` does appear in the paper's Eq. 59 for the RDE
    *surface concentration* beyond the electrode edge: one can check numerically
    that the paper's Eq. 59 surface value equals
    :math:`1 - F(\rho^3 - 1)` for :math:`\rho > 1`.  That connection is between
    :math:`F` and the paper's **spectroscopic / surface-concentration** profile,
    *not* a Tolmachev-formalism derivation of the electrochemical RRDE
    collection efficiency :math:`N` (which is what
    :func:`collection_efficiency` returns).
    """
    theta = np.asarray(theta, dtype=float)
    cbrt = np.cbrt(theta)
    return (
        np.sqrt(3.0) / (4.0 * np.pi) * np.log((1.0 + cbrt) ** 3 / (1.0 + theta))
        + 3.0 / (2.0 * np.pi) * np.arctan((2.0 * cbrt - 1.0) / np.sqrt(3.0))
        + 0.25
    )


def collection_efficiency(r1: float, r2: float, r3: float) -> float:
    r"""Electrochemical RRDE collection efficiency :math:`N` (B&F eq. 9.4.16).

    .. warning::
        This is a **thin re-export** of the Albery--Bruckenstein closed form
        already implemented in :func:`serm.rrde.collection_efficiency_closed_form`
        (Bard & Faulkner eq. 9.4.16, 2nd ed. p. 352).  It evaluates the *same*
        expression via the *same* :func:`F_collection`, so any comparison against
        :mod:`serm.rrde` is **exact by construction and is NOT an independent
        check** of the Tolmachev formalism.  It is provided here only so the
        chapter can display the electrochemical :math:`N` alongside the paper's
        (distinct) spectroscopic ring quantity.  Do not attribute :math:`N=0.555`
        to "an independent route from the unified formalism."

    .. math::
        N = 1 - F(\alpha/\beta) + \beta^{2/3}[1-F(\alpha)]
          - (1+\alpha+\beta)^{2/3}\{1-F[(\alpha/\beta)(1+\alpha+\beta)]\},

    with :math:`\alpha=(r_2/r_1)^3-1`, :math:`\beta=(r_3^3-r_2^3)/r_1^3`.  For
    :math:`(r_1,r_2,r_3)=(0.187,0.200,0.332)` cm this gives
    :math:`N \approx 0.555`.

    Parameters
    ----------
    r1, r2, r3 : float
        Disk radius, ring inner radius, ring outer radius (:math:`0<r_1\le r_2<r_3`).

    Returns
    -------
    float
        Collection efficiency :math:`N \in [0,1]`.
    """
    if not (0.0 < r1 <= r2 < r3):
        raise ValueError("radii must satisfy 0 < r1 <= r2 < r3")
    alpha = (r2 / r1) ** 3 - 1.0
    beta = (r3 ** 3 - r2 ** 3) / r1 ** 3
    return float(
        1.0
        - F_collection(alpha / beta)
        + beta ** (2.0 / 3.0) * (1.0 - F_collection(alpha))
        - (1.0 + alpha + beta) ** (2.0 / 3.0)
        * (1.0 - F_collection((alpha / beta) * (1.0 + alpha + beta)))
    )


# --------------------------------------------------------------------------- #
# Table I geometry scale factors.                                             #
# --------------------------------------------------------------------------- #
@dataclass
class GeometryScales:
    """Dimensionless-variable scale factors for one geometry (paper Table I).

    Attributes
    ----------
    name : str
        Geometry label.
    X_of : callable
        Map from the physical along-flow position to :math:`X`.
    description : str
        Human-readable summary of the Table I entries.
    """

    name: str
    description: str


def geometry_scales(geometry: str) -> GeometryScales:
    r"""Return the Table I dimensionless-variable definitions for a geometry.

    The four geometries reduce to the single Eq. I-5 through these changes of
    variable (paper Table I, Eqs. I-2/I-3):

    * ``"rotating_disk"``: :math:`X=(r/r_0)^3=\rho^3`,
      :math:`Y=(r/r_0)(3a\nu/D)^{1/3}(\Omega/\nu)^{1/2}(\dots)`; the governing
      operator is :math:`a y\,\Omega(\Omega/\nu)^{1/2}[r\,\partial_r-y\,\partial_y]c
      = D\,\partial_{yy}c`.
    * ``"channel"``: :math:`X=x/l`, :math:`Y=(2v_0h^2/D l)^{1/3}(y/h)`;
      operator :math:`2v_0(y/h)\partial_x c = D\,\partial_{yy}c`.
    * ``"tube"``: :math:`X=x/l`, :math:`Y=(2v_0R^2/Dl)^{1/3}(1-r/R)`;
      operator :math:`2v_0(1-r/R)\partial_x c = D\,\partial_{rr}c`.

    Parameters
    ----------
    geometry : str
        One of ``"rotating_disk"``, ``"channel"``, ``"tube"``.

    Returns
    -------
    GeometryScales
    """
    table = {
        "rotating_disk": GeometryScales(
            "rotating_disk",
            "X = (r/r0)^3 = rho^3;  Y = (r/r0)(3 a nu/D)^(1/3)(Omega/nu)^(1/2) zeta;"
            " governing a y Omega (Omega/nu)^(1/2)[r d_r - y d_y]c = D c_yy.",
        ),
        "channel": GeometryScales(
            "channel",
            "X = x/l;  Y = (2 v0 h^2/(D l))^(1/3)(y/h);"
            " governing 2 v0 (y/h) c_x = D c_yy.",
        ),
        "tube": GeometryScales(
            "tube",
            "X = x/l;  Y = (2 v0 R^2/(D l))^(1/3)(1 - r/R);"
            " governing 2 v0 (1 - r/R) c_x = D c_rr.",
        ),
    }
    if geometry not in table:
        raise ValueError(f"unknown geometry {geometry!r}; choose from {list(table)}")
    return table[geometry]
