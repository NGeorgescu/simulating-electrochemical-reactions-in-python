r"""Rotating ring--disk electrode (RRDE) collection efficiency.

New material for *Simulating Electrochemical Reactions in Python*, going
beyond Honeychurch's rotating-disk chapter to the **two-electrode** rotating
ring--disk geometry of Bard & Faulkner (B&F), *Electrochemical Methods*,
2nd ed., section 9.4 / 9.4.2 (pp. 350--352).

The physics
-----------
A disk of radius ``r1`` generates species R; an annular ring (inner radius
``r2``, outer radius ``r3``) downstream collects it.  Radial convection sweeps
R outward while axial diffusion carries it back to the wall.  The steady-state
ring convective--diffusion equation for the collected species R is B&F eq.
(9.4.9):

.. math::
    r\,\frac{\partial C_R}{\partial r}
        - y\,\frac{\partial C_R}{\partial y}
        = \frac{D_R}{B'}\,\frac{1}{y}\,
          \frac{\partial^2 C_R}{\partial y^2},

where ``y`` is the axial distance from the electrode plane, ``r`` is the radial
coordinate, and ``B' = 0.51\,\omega^{3/2}\nu^{-1/2}`` is the Cochran
wall-shear coefficient.  Radial diffusion has been dropped, so the equation is
**parabolic in r**: it can be marched outward in ``r`` on a wall-graded ``y``
grid.

Boundary conditions at the wall ``y = 0`` (B&F eqs. 9.4.11--9.4.13):

* disk, ``0 <= r < r1``: imposed flux
  ``(dCR/dy)_{y=0} = -i_D / (pi r1^2 n F D_R)``  (eq. 9.4.11);
* gap, ``r1 <= r < r2``: zero flux ``(dCR/dy)_{y=0} = 0``  (eq. 9.4.12);
* ring, ``r2 <= r < r3``: Dirichlet ``C_R(y=0) = 0``  (eq. 9.4.13, limiting
  collection).

The ring current is the integrated wall flux over the ring (eq. 9.4.14):

.. math::
    i_R = n F D_R\, 2\pi \int_{r_2}^{r_3}
          \Bigl(\frac{\partial C_R}{\partial y}\Bigr)_{y=0} r\,dr,

and the **collection efficiency** is ``N = -i_R / i_D`` (eq. 9.4.15).

Invariance
----------
Because the equation is linear and homogeneous in ``C_R`` and the only physical
parameter is the group ``D_R/B'`` (a length scale in ``y``), the collection
efficiency ``N`` depends **only** on the geometry ``(r1, r2, r3)`` and is
independent of ``omega``, bulk concentration, ``n``, ``F`` and ``D_R`` (B&F:
"depends only on r1, r2, and r3 and is independent of omega, C_O^*, D_O, D_R,
etc.", p. 351).  The solver scales ``y`` by ``(D_R/B')^{1/3}`` so the coefficient
becomes unity.  Because the solver is fully nondimensionalised, ``omega``,
``D_R`` and ``C*`` literally never enter its signature, so the invariance is
exact *by construction* -- it is a structural sanity check that cannot fail,
not a falsifiable validation.  The real independent cross-check is the geometry
sweep (closed form vs PDE march across several geometries) in the companion
notebook.

The Albery--Bruckenstein closed form
------------------------------------
B&F give the analytic collection efficiency (eqs. 9.4.16/9.4.17, p. 352):

.. math::
    N = 1 - F(\alpha/\beta) + \beta^{2/3}\,[1 - F(\alpha)]
        - (1+\alpha+\beta)^{2/3}\,
          \bigl\{1 - F[(\alpha/\beta)(1+\alpha+\beta)]\bigr\},

with

.. math::
    F(\theta) = \frac{\sqrt 3}{4\pi}
        \ln\!\left[\frac{(1+\theta^{1/3})^3}{1+\theta}\right]
        + \frac{3}{2\pi}\arctan\!\left(\frac{2\theta^{1/3}-1}{\sqrt 3}\right)
        + \frac14,

``\alpha = (r2/r1)^3 - 1`` and ``\beta = (r3^3 - r2^3)/r1^3`` (eq. 9.4.8).
For ``(r1, r2, r3) = (0.187, 0.200, 0.332)`` cm this gives ``N = 0.555`` --
"55.5% of the product generated at the disk is collected at the ring" (B&F
p. 352).  That number is the tier-1 validation anchor reproduced independently
by the PDE march in :func:`collection_efficiency_pde`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "F_albery",
    "collection_efficiency_closed_form",
    "shielding_factor",
    "ring_disk_current_ratio",
    "PDEResult",
    "collection_efficiency_pde",
]


# --------------------------------------------------------------------------- #
# Albery--Bruckenstein closed form (B&F eqs. 9.4.16/9.4.17)                    #
# --------------------------------------------------------------------------- #
def F_albery(theta: float | NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Albery--Bruckenstein auxiliary function ``F(theta)`` (B&F eq. 9.4.17).

    .. math::
        F(\theta) = \frac{\sqrt 3}{4\pi}
            \ln\!\left[\frac{(1+\theta^{1/3})^3}{1+\theta}\right]
            + \frac{3}{2\pi}\arctan\!\left(\frac{2\theta^{1/3}-1}{\sqrt 3}\right)
            + \frac14 .

    Parameters
    ----------
    theta : float or ndarray
        Non-negative dimensionless argument.

    Returns
    -------
    ndarray
        ``F(theta)``.
    """
    theta = np.asarray(theta, dtype=float)
    cbrt = np.cbrt(theta)
    return (
        np.sqrt(3.0) / (4.0 * np.pi) * np.log((1.0 + cbrt) ** 3 / (1.0 + theta))
        + 3.0 / (2.0 * np.pi) * np.arctan((2.0 * cbrt - 1.0) / np.sqrt(3.0))
        + 0.25
    )


def _alpha_beta(r1: float, r2: float, r3: float) -> tuple[float, float]:
    """Return the Albery geometry parameters ``(alpha, beta)``.

    ``alpha = (r2/r1)^3 - 1`` and ``beta = (r3^3 - r2^3)/r1^3`` (B&F eq. 9.4.8).
    """
    if not (0.0 < r1 <= r2 < r3):
        raise ValueError("radii must satisfy 0 < r1 <= r2 < r3")
    alpha = (r2 / r1) ** 3 - 1.0
    beta = (r3**3 - r2**3) / r1**3
    return alpha, beta


def collection_efficiency_closed_form(r1: float, r2: float, r3: float) -> float:
    r"""Collection efficiency ``N`` from the Albery--Bruckenstein formula.

    Evaluates B&F eq. 9.4.16 (p. 352).  Depends only on geometry.

    Parameters
    ----------
    r1, r2, r3 : float
        Disk radius, ring inner radius, ring outer radius (any consistent
        length unit), with ``0 < r1 <= r2 < r3``.

    Returns
    -------
    float
        Collection efficiency ``N`` in ``[0, 1]``.

    Examples
    --------
    >>> round(collection_efficiency_closed_form(0.187, 0.200, 0.332), 3)
    0.555
    """
    alpha, beta = _alpha_beta(r1, r2, r3)
    return float(
        1.0
        - F_albery(alpha / beta)
        + beta ** (2.0 / 3.0) * (1.0 - F_albery(alpha))
        - (1.0 + alpha + beta) ** (2.0 / 3.0)
        * (1.0 - F_albery((alpha / beta) * (1.0 + alpha + beta)))
    )


def ring_disk_current_ratio(r1: float, r2: float, r3: float) -> float:
    r"""Limiting ring/disk current ratio ``i_R/i_D = beta^{2/3}`` (B&F eq. 9.4.8).

    Valid for a ring under mass-transport-limited conditions when the
    *disk is inactive* (``i_D = 0``); see B&F eqs. 9.4.5--9.4.8, p. 350.  This is
    a distinct quantity from the collection efficiency ``N``.
    """
    _, beta = _alpha_beta(r1, r2, r3)
    return float(beta ** (2.0 / 3.0))


def shielding_factor(r1: float, r2: float, r3: float) -> float:
    r"""Shielding factor ``1 - N beta^{-2/3}`` (B&F eq. 9.4.20, p. 353).

    When the disk current is at its limiting value, the ring limiting current is
    reduced from its disk-off value by this factor:
    ``i_{R,l} = i_{R,l}^0 (1 - N beta^{-2/3})``.
    """
    _, beta = _alpha_beta(r1, r2, r3)
    n = collection_efficiency_closed_form(r1, r2, r3)
    return float(1.0 - n * beta ** (-2.0 / 3.0))


# --------------------------------------------------------------------------- #
# Independent PDE march of B&F eq. 9.4.9                                       #
# --------------------------------------------------------------------------- #
@dataclass
class PDEResult:
    """Result of the parabolic ``r``-march of B&F eq. 9.4.9.

    Attributes
    ----------
    N : float
        Collection efficiency ``N = -i_R / i_D`` from integrating the ring wall
        flux (eq. 9.4.14).
    r : ndarray
        Radial march stations (scaled units, ``r1 = 1``).
    y : ndarray
        Axial grid (scaled by ``(D_R/B')^{1/3}``), wall-graded.
    wall_flux : ndarray
        Wall axial gradient ``(dCR/dy)_{y=0}`` at each ``r`` station, in the
        same scaled units, with the disk flux normalised to ``-1``.
    C : ndarray
        Concentration field ``C[i, j]`` at radial station ``r[i]`` and axial
        node ``y[j]`` (scaled so the disk flux magnitude is 1).
    """

    N: float
    r: NDArray[np.float64]
    y: NDArray[np.float64]
    wall_flux: NDArray[np.float64]
    C: NDArray[np.float64]


def _graded_y_grid(ny: int, y_max: float, stretch: float) -> NDArray[np.float64]:
    """Return a monotone wall-graded axial grid on ``[0, y_max]``.

    Uses a geometric (exponential) stretch so nodes cluster near the wall
    ``y = 0`` where the boundary layer is thin.  ``stretch`` controls the
    clustering (``> 1`` packs nodes at the wall; ``-> 1`` recovers uniform).
    """
    if stretch <= 1.0:
        return np.linspace(0.0, y_max, ny)
    s = np.linspace(0.0, 1.0, ny)
    return y_max * (np.expm1(stretch * s) / np.expm1(stretch))


def collection_efficiency_pde(
    r1: float,
    r2: float,
    r3: float,
    *,
    ny: int = 1000,
    nr: int = 12000,
    y_max: float = 7.0,
    stretch: float = 6.5,
    r_start: float = 1e-5,
) -> PDEResult:
    r"""Collection efficiency by an **independent** parabolic march of eq. 9.4.9.

    Marches the steady-state ring convective--diffusion equation (B&F eq. 9.4.9)

    .. math::
        r\,\partial_r C - y\,\partial_y C
            = \frac{1}{y}\,\partial_{yy} C

    outward in ``r`` on a wall-graded ``y`` grid.  The axial coordinate is scaled
    by ``(D_R/B')^{1/3}`` so the right-hand-side coefficient is unity, which makes
    the result manifestly independent of ``omega``, ``D_R`` and concentration --
    exactly the invariance B&F state for ``N``.  Radii are scaled by ``r1`` so the
    disk edge is at ``r = 1``.

    Each radial step is treated implicitly in ``y`` (backward Euler in the
    parabolic marching variable ``ln r``), giving an unconditionally stable
    tridiagonal solve per station.  Wall boundary conditions switch with the
    region:

    * disk (``r < r1``): Neumann, scaled wall flux ``= -1`` (eq. 9.4.11);
    * gap (``r1 <= r < r2``): Neumann, zero flux (eq. 9.4.12);
    * ring (``r2 <= r < r3``): Dirichlet ``C = 0`` (eq. 9.4.13).

    The ring current follows from integrating the wall flux (eq. 9.4.14) and
    ``N = -i_R / i_D`` (eq. 9.4.15).

    Parameters
    ----------
    r1, r2, r3 : float
        Geometry (any consistent length unit); only ratios matter.
    ny : int, optional
        Number of axial nodes.
    nr : int, optional
        Number of radial march stations from ``r_start`` to ``r3``.
    y_max : float, optional
        Outer axial extent in scaled units (boundary-layer thickness ~ 1).
    stretch : float, optional
        Wall-grading strength of the axial grid.
    r_start : float, optional
        Inner radial start (avoids the ``r=0`` singularity of ``dC = .../r``).

    Returns
    -------
    PDEResult
        Collection efficiency and the marched field/flux.
    """
    if not (0.0 < r1 <= r2 < r3):
        raise ValueError("radii must satisfy 0 < r1 <= r2 < r3")

    # Scale lengths by r1: disk edge -> 1.
    R1, R2, R3 = 1.0, r2 / r1, r3 / r1

    y = _graded_y_grid(ny, y_max, stretch)
    # March in ln(r) so r dC/dr = dC/d(ln r); uniform in ln r.
    lnr = np.linspace(np.log(r_start), np.log(R3), nr)
    r = np.exp(lnr)

    # Second-difference operator on the non-uniform y grid:
    #   C''_j ~ a_j C_{j-1} + b_j C_j + c_j C_{j+1}.
    dy_m = y[1:-1] - y[:-2]      # h_{j-1}
    dy_p = y[2:] - y[1:-1]       # h_j
    a = 2.0 / (dy_m * (dy_m + dy_p))           # coeff of C_{j-1}
    c = 2.0 / (dy_p * (dy_m + dy_p))           # coeff of C_{j+1}
    b = -(a + c)                               # coeff of C_j
    # RHS coefficient of eq. 9.4.9: (1/y) C'' ; interior nodes j = 1..ny-2.
    inv_y = 1.0 / y[1:-1]

    C = np.zeros((nr, ny))
    wall_flux = np.zeros(nr)

    # Backward-Euler step in the marching variable s = ln r.
    #
    # B&F eq. 9.4.9 is  r dC/dr - y dC/dy = (1/y) C''.  With s = ln r
    # (so r dC/dr = dC/ds) this becomes the parabolic march
    #
    #     dC/ds = y dC/dy + (1/y) C'' .
    #
    # The advective term has coefficient +y > 0 multiplying dC/dy.  The axial
    # convection physically sweeps R toward larger y as r grows (the boundary
    # layer thickens), so the upwind direction for this term is *forward*
    # (toward node j+1); a forward one-sided difference keeps the implicit
    # operator an M-matrix and the march unconditionally stable.  A central
    # difference here makes the scheme blow up.
    h_p = np.empty(ny)
    h_p[:-1] = y[1:] - y[:-1]       # forward spacing y[j+1] - y[j]
    h_p[-1] = h_p[-2]
    # forward-difference coefficients for dC/dy at interior nodes:
    #   dC/dy ~ (C_{j+1} - C_j)/h_p[j]
    fwd_0 = -1.0 / h_p[1:-1]        # coeff of C_j
    fwd_p = 1.0 / h_p[1:-1]         # coeff of C_{j+1}

    conv = y[1:-1]                  # coefficient +y of dC/dy
    diff = inv_y                    # coefficient 1/y of C''

    # Raw RHS coefficients of  dC/ds = y C'_y + (1/y) C'' at interior nodes.
    L_int = diff * a                                # coeff of C_{j-1} (no advection)
    D_int = conv * fwd_0 + diff * b                 # coeff of C_j
    U_int = conv * fwd_p + diff * c                 # coeff of C_{j+1}

    C_prev = np.zeros(ny)            # C = 0 in the bulk feed at r_start
    for i in range(nr):
        region_disk = r[i] < R1
        region_ring = (r[i] >= R2) & (r[i] < R3)

        if i == 0:
            C[i] = C_prev
            wall_flux[i] = -1.0 if region_disk else 0.0
            continue

        ds = lnr[i] - lnr[i - 1]
        n = ny
        lower = np.zeros(n)
        diag = np.ones(n)
        upper = np.zeros(n)
        rhs = C_prev.copy()

        j = slice(1, n - 1)
        lower[j] = -ds * L_int
        diag[j] = 1.0 - ds * D_int
        upper[j] = -ds * U_int

        # --- wall BC at j = 0 ---
        if region_ring:
            # Dirichlet C = 0 (eq. 9.4.13)
            diag[0] = 1.0
            upper[0] = 0.0
            rhs[0] = 0.0
        else:
            # Neumann (C1 - C0)/h0 = g: g = -1 on disk (eq. 9.4.11), 0 in gap.
            g = -1.0 if region_disk else 0.0
            h0 = y[1] - y[0]
            diag[0] = -1.0 / h0
            upper[0] = 1.0 / h0
            rhs[0] = g

        # --- outer BC at j = n-1: C = 0 (bulk) ---
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs[-1] = 0.0

        C_new = _thomas(lower, diag, upper, rhs)
        C[i] = C_new
        C_prev = C_new

        # wall flux (dC/dy at y=0): one-sided 3-point on the graded grid.
        h1 = y[1] - y[0]
        h2 = y[2] - y[0]
        wall_flux[i] = (
            -(h1 + h2) / (h1 * h2) * C_new[0]
            + h2 / (h1 * (h2 - h1)) * C_new[1]
            - h1 / (h2 * (h2 - h1)) * C_new[2]
        )

    # i_D: disk flux (-1) integrated over disk area, scaled units.
    #   i_D ~ 2 pi int_0^{R1} (dC/dy)_disk r dr = 2 pi (-1) R1^2 / 2 = -pi R1^2.
    i_D = np.pi * (-1.0) * R1**2

    # i_R: integrate wall flux over the ring (eq. 9.4.14), 2 pi int r flux dr.
    ring_mask = (r >= R2) & (r <= R3)
    integrand = 2.0 * np.pi * wall_flux * r
    i_R = np.trapezoid(integrand[ring_mask], r[ring_mask])

    N = -i_R / i_D
    return PDEResult(N=float(N), r=r, y=y, wall_flux=wall_flux, C=C)


def _thomas(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal system by the Thomas algorithm.

    ``lower[i]`` multiplies ``x[i-1]``, ``diag[i]`` multiplies ``x[i]`` and
    ``upper[i]`` multiplies ``x[i+1]`` (``lower[0]`` and ``upper[-1]`` unused).
    """
    n = diag.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
