"""Semi-integration and fractional integro-differentiation (SERM Appendix 2).

Semi-integration is the convolution operation at the heart of *convolutive* (or
*semi-integral*) voltammetry: the semi-integral ``m(t)`` of a faradaic current
``i(t)`` removes the diffusional ``t^{-1/2}`` tail, turning a peak-shaped
diffusion voltammogram into a sigmoidal, steady-state-like wave whose plateau is
proportional to the bulk concentration.  Appendix 2 of Honeychurch's SERM
implements this; the underlying calculus is Oldham & Spanier's fractional
*semi-integration* (K. B. Oldham and J. Spanier, *The Fractional Calculus*,
Academic Press, 1974; see also K. B. Oldham, *Anal. Chem.* on semi-integral
electroanalysis).

Definitions
-----------
The Riemann--Liouville fractional integral of order ``q > 0`` is the convolution

.. math::
    \\bigl(D^{-q} f\\bigr)(t)
        = \\frac{1}{\\Gamma(q)} \\int_0^{t} (t-\\tau)^{q-1} f(\\tau)\\,d\\tau .

The *semi-integral* is the special case ``q = 1/2``; the *semi-derivative* is the
fractional derivative of order ``1/2`` (``q = -1/2`` in the sign convention used
here, where a positive ``order`` argument means differentiation).

Discretisation
--------------
Two equivalent discretisations are provided:

* :func:`semi_integrate` uses a **Riemann--Liouville convolution kernel**
  ``k_j`` obtained by integrating ``(t)^{q-1}/\\Gamma(q)`` analytically over each
  grid cell, which avoids the ``\\tau = 0`` singularity of a naive midpoint rule.
  This mirrors the convolution style of :mod:`serm.filters` (a kernel convolved
  against the data).
* :func:`fractional_integrodifferentiate` uses the **Grünwald--Letnikov** (G--L)
  form, the difference-quotient definition of the fractional operator,

  .. math::
      \\bigl(D^{q} f\\bigr)(t_i) \\approx \\Delta t^{-q}
          \\sum_{j=0}^{i} w_j\\, f(t_{i-j}),
      \\qquad
      w_0 = 1,\\quad w_j = w_{j-1}\\Bigl(1 - \\frac{q+1}{j}\\Bigr),

  valid for any real order ``q`` (``q < 0`` integrates, ``q > 0``
  differentiates).  The recurrence for the weights is the binomial-coefficient
  ratio ``w_j = (-1)^j \\binom{q}{j}``.  This is the algorithm of Oldham &
  Spanier's semi-integration and the one used in convolutive voltammetry.

Validated behaviour (see :func:`semiintegration_selfcheck`)
-----------------------------------------------------------
* The semi-integral of a Cottrell current ``i \\propto t^{-1/2}`` is *constant*
  (the limiting plateau): analytically ``D^{-1/2}[\\,t^{-1/2}/\\sqrt{\\pi}\\,] = 1``.
* The semi-derivative of a diffusion-limited reversible LSV current is a
  *symmetric* peak.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma

__all__ = [
    "rl_kernel",
    "semi_integrate",
    "gl_weights",
    "fractional_integrodifferentiate",
    "semi_derivative",
    "semiintegration_selfcheck",
]


def rl_kernel(n: int, dt: float, q: float = 0.5) -> np.ndarray:
    """Riemann--Liouville convolution kernel for a fractional integral of order ``q``.

    The continuous kernel is ``g(t) = t**(q-1) / Gamma(q)``.  To avoid the
    integrable singularity at ``t = 0`` we use the *cell-integrated* weights

    .. math::
        k_j = \\frac{1}{\\Gamma(q+1)}\\,
              \\bigl[((j+1)\\,\\Delta t)^{q} - (j\\,\\Delta t)^{q}\\bigr],

    i.e. the exact integral of ``g`` over the ``j``-th step (``\\int t^{q-1}
    = t^{q}/q`` and ``q\\,\\Gamma(q) = \\Gamma(q+1)``).  Convolving a sampled
    function with these weights then approximates ``D^{-q} f`` without ever
    evaluating the singular kernel at ``t = 0``.

    Parameters
    ----------
    n : int
        Number of kernel taps (same length as the data to be transformed).
    dt : float
        Uniform time step.
    q : float
        Integration order (``0.5`` for the semi-integral).  Must be ``> 0``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Convolution weights ``k_0 .. k_{n-1}``.
    """
    if q <= 0:
        raise ValueError("rl_kernel order q must be > 0 (a fractional integral)")
    j = np.arange(n + 1, dtype=float)
    edges = (j * dt) ** q
    return (edges[1:] - edges[:-1]) / gamma(q + 1.0)


def semi_integrate(y, dt: float, q: float = 0.5) -> np.ndarray:
    """Semi-integral (Riemann--Liouville fractional integral, order ``q``).

    Computes ``D^{-q} y`` on a uniform grid by convolving ``y`` with the
    cell-integrated Riemann--Liouville kernel of :func:`rl_kernel` (causal,
    lower terminal at ``t = 0``).  The default ``q = 0.5`` is the semi-integral
    used in convolutive voltammetry.

    Parameters
    ----------
    y : array_like, 1-D
        Sampled function (e.g. a faradaic current) on a uniform time grid
        starting at ``t = 0``.
    dt : float
        Uniform sampling step.
    q : float
        Integration order (``> 0``); ``0.5`` for the semi-integral.

    Returns
    -------
    numpy.ndarray, shape (len(y),)
        The fractional integral ``D^{-q} y`` at each sample.

    Notes
    -----
    The transform is a *lower triangular* (causal) convolution: output sample
    ``i`` uses inputs ``0 .. i`` only.  We evaluate it with a full convolution
    truncated to the first ``len(y)`` taps, which is the discrete form of the
    convolution integral.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    k = rl_kernel(n, dt, q)
    # Causal convolution: (k * y)[i] = sum_{j<=i} k[j] y[i-j].
    full = np.convolve(y, k)
    return full[:n]


def gl_weights(q: float, n: int) -> np.ndarray:
    """Grünwald--Letnikov weights ``w_j = (-1)^j C(q, j)`` for order ``q``.

    Generated by the stable recurrence ``w_0 = 1``,
    ``w_j = w_{j-1} (1 - (q+1)/j)``.  Positive ``q`` differentiates, negative
    ``q`` integrates.

    Parameters
    ----------
    q : float
        Differintegration order.
    n : int
        Number of weights.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for j in range(1, n):
        w[j] = w[j - 1] * (1.0 - (q + 1.0) / j)
    return w


def fractional_integrodifferentiate(y, dt: float, order: float) -> np.ndarray:
    """Fractional differintegral of arbitrary real ``order`` (Grünwald--Letnikov).

    Evaluates ``D^{order} y`` on a uniform grid using the Grünwald--Letnikov
    difference quotient

    .. math::
        \\bigl(D^{order} y\\bigr)_i = \\Delta t^{-order}
            \\sum_{j=0}^{i} w_j\\, y_{i-j},

    with weights from :func:`gl_weights`.  Sign convention: ``order > 0`` is a
    fractional *derivative* (``order = 0.5`` -> semi-derivative); ``order < 0``
    is a fractional *integral* (``order = -0.5`` -> semi-integral); ``order = 1``
    and ``order = -1`` recover the ordinary first derivative and integral
    (Oldham & Spanier, *The Fractional Calculus*).

    Parameters
    ----------
    y : array_like, 1-D
        Sampled function on a uniform grid starting at ``t = 0``.
    dt : float
        Uniform sampling step.
    order : float
        Differintegration order (any real number).

    Returns
    -------
    numpy.ndarray, shape (len(y),)
        ``D^{order} y`` at each sample.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    w = gl_weights(order, n)
    # Causal convolution of the weights with y, scaled by dt**(-order).
    full = np.convolve(y, w)[:n]
    return dt ** (-order) * full


def semi_derivative(y, dt: float) -> np.ndarray:
    """Semi-derivative (fractional derivative of order 1/2) via Grünwald--Letnikov.

    Convenience wrapper for ``fractional_integrodifferentiate(y, dt, 0.5)``.
    Applied to a diffusion-limited LSV current it yields a symmetric peak whose
    height is proportional to concentration (convolutive voltammetry).
    """
    return fractional_integrodifferentiate(y, dt, 0.5)


def _cottrell_plateau(n: int) -> tuple[float, float]:
    """Return ``(plateau_mean, plateau_flatness_std)`` of the semi-integral.

    Helper for :func:`semiintegration_selfcheck`: builds an ``n``-sample Cottrell
    current ``1/\\sqrt{\\pi t}`` on ``t in [0, 4]`` and returns the mean and
    standard deviation of the second half of its RL semi-integral.
    """
    t = np.linspace(0.0, 4.0, n)
    dt = t[1] - t[0]
    i_cott = np.zeros_like(t)
    i_cott[1:] = 1.0 / np.sqrt(np.pi * t[1:])  # t=0 singularity dropped
    m = semi_integrate(i_cott, dt, q=0.5)
    plateau = m[m.size // 2:]
    return float(plateau.mean()), float(plateau.std())


def semiintegration_selfcheck(tol_flatness: float = 2e-3,
                              tol_plateau: float = 1.2e-2,
                              tol_symmetry: float = 0.1) -> None:
    """Assert the two reference results of semi-integration.

    Validation tiers (per the project validation policy):

    * **Closed-form analytic check (semi-integral):** the semi-integral of a
      Cottrell current ``i = 1/\\sqrt{\\pi t}`` is the constant ``1``
      (``D^{-1/2}[t^{-1/2}/\\sqrt{\\pi}] = \\Gamma(1/2)/\\Gamma(1)\\,/\\sqrt{\\pi}
      = 1``).  Three things are checked: (a) the plateau is *flat* (its standard
      deviation over the second half is below ``tol_flatness``); (b) the plateau
      value equals 1 within ``tol_plateau`` (the residual error comes from the
      ``t = 0`` singularity of the Cottrell current and is grid-limited); and
      (c) it *converges* toward 1 under grid refinement -- a self-consistency
      check that the residual is discretisation error, not bias.  Both
      :func:`semi_integrate` (RL kernel) and
      :func:`fractional_integrodifferentiate` (Grünwald--Letnikov) are exercised
      and cross-checked against each other.
    * **Symmetry / self-consistency (semi-derivative):** the semi-derivative of a
      diffusion-limited reversible LSV current is a symmetric peak; we require
      the half-width on the two sides of the maximum to agree within
      ``tol_symmetry`` (relative).

    Raises
    ------
    AssertionError
        If any reference result is not met.
    """
    # --- semi-integral of a Cottrell current -> constant plateau = 1 ---
    t = np.linspace(0.0, 4.0, 4000)
    dt = t[1] - t[0]
    i_cott = np.zeros_like(t)
    i_cott[1:] = 1.0 / np.sqrt(np.pi * t[1:])  # t=0 singularity dropped

    m_rl = semi_integrate(i_cott, dt, q=0.5)
    m_gl = fractional_integrodifferentiate(i_cott, dt, order=-0.5)
    plateau_rl = m_rl[m_rl.size // 2:]
    plateau_gl = m_gl[m_gl.size // 2:]

    # (a) flatness: the plateau is genuinely constant.
    assert plateau_rl.std() < tol_flatness, (
        f"RL plateau not flat: std {plateau_rl.std():.2e} (tol {tol_flatness:.1e})"
    )
    # (b) plateau value equals the analytic constant 1 within grid tolerance.
    err_rl = abs(plateau_rl.mean() - 1.0)
    err_gl = abs(plateau_gl.mean() - 1.0)
    assert err_rl < tol_plateau, (
        f"RL semi-integral plateau off from 1 by {err_rl:.2e} (tol {tol_plateau:.1e})"
    )
    assert err_gl < tol_plateau, (
        f"GL semi-integral plateau off from 1 by {err_gl:.2e} (tol {tol_plateau:.1e})"
    )
    # The two discretisations must agree on the plateau.
    assert abs(plateau_rl.mean() - plateau_gl.mean()) < tol_plateau, (
        "RL and GL plateaus disagree"
    )
    # (c) convergence toward 1 under refinement (error must shrink).
    err_coarse = abs(_cottrell_plateau(4000)[0] - 1.0)
    err_fine = abs(_cottrell_plateau(16000)[0] - 1.0)
    assert err_fine < err_coarse, (
        f"plateau error did not shrink on refinement: {err_coarse:.2e} -> {err_fine:.2e}"
    )

    # --- semi-derivative of a reversible LSV current -> symmetric peak ---
    # Build a genuine reversible diffusion-limited LSV current on a grid where
    # the dimensionless potential ``p`` advances linearly in time.  The fraction
    # of O reduced (the cumulative surface depletion) follows the reversible
    # sigmoid ``theta = 1/(1+exp(-p))``, which rises smoothly from ~0; its
    # semi-derivative is the diffusion-limited LSV *current* ``i_lsv`` -- a
    # peak whose maximum is the Nicholson--Shain reversible value 0.4463, with
    # the characteristic asymmetric ``t^{-1/2}`` diffusion tail.  Oldham's
    # semi-integral result: a further semi-derivative removes that diffusion
    # tail and returns a *symmetric* peak (here the logistic bell ``theta'``).
    p = np.linspace(-15.0, 15.0, 6000)
    dp = p[1] - p[0]
    theta = 1.0 / (1.0 + np.exp(-p))             # cumulative depletion, starts ~0
    i_lsv = semi_derivative(theta, dp)           # diffusion-limited LSV current
    peak = semi_derivative(i_lsv, dp)            # -> symmetric peak

    imax = int(np.argmax(peak))
    half = peak[imax] / 2.0
    left_idx = imax
    while left_idx > 0 and peak[left_idx] > half:
        left_idx -= 1
    right_idx = imax
    while right_idx < peak.size - 1 and peak[right_idx] > half:
        right_idx += 1
    w_left = imax - left_idx
    w_right = right_idx - imax
    if w_left == 0 or w_right == 0:
        raise AssertionError("semi-derivative peak has degenerate half-width")
    asym = abs(w_left - w_right) / max(w_left, w_right)
    assert asym < tol_symmetry, (
        f"semi-derivative peak asymmetry {asym:.3f} exceeds tol {tol_symmetry:.2f}"
    )


if __name__ == "__main__":  # pragma: no cover
    semiintegration_selfcheck()
    print("serm.semiintegration: semi-integral plateau = 1 and "
          "semi-derivative peak symmetric -- OK.")
