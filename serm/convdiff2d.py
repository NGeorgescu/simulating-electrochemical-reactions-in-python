"""Steady 2-D convective-diffusion mass transport at channel and tubular electrodes.

This module solves the genuinely two-dimensional steady convective-diffusion
problem for an electrode embedded in the wall of a laminar flow cell -- the case
the rotating-disk electrode (Chapter 14) *cannot* be reduced to, because the RDE
collapses to a 1-D similarity problem while a wall-mounted band electrode retains
an explicit dependence on both the axial coordinate ``x`` (along the flow) and the
transverse coordinate ``y`` (normal to the wall).

Physics
=======
A planar band electrode of length ``L`` sits flush in one wall of a rectangular
channel; species O is reduced at the diffusion-limited rate (``C = 0`` on the
electrode).  The steady convective-diffusion equation for the (dimensional)
concentration ``C(x, y)`` is

    D (d2C/dx2 + d2C/dy2) = v_x(y) dC/dx ,                              (1)

with ``v_x`` the axial fluid velocity.  Within the thin diffusion layer that
grows over the electrode the velocity is well approximated by the **Leveque**
linear-shear profile (the leading term of the parabolic Poiseuille profile near
a no-slip wall),

    v_x(y) ~= s * y ,                                                   (2)

where ``s = (dv_x/dy)|_{wall}`` is the wall shear rate.  For laminar flow:

* channel of height ``b`` and mean velocity ``Um``: parabolic profile
  ``v_x = 6 Um (y/b)(1 - y/b)``  =>  ``s = 6 Um / b``;
* tube of radius ``r`` and mean velocity ``Um``: ``v_x = 2 Um (1-(rho/r)^2)``,
  so near the wall (``y = r - rho``)  ``s = 4 Um / r``.

Closed-form anchor (Leveque)
============================
Neglecting axial diffusion (``d2C/dx2``, valid because the diffusion layer is thin
and the Peclet number large), Eq. (1)-(2) admit the similarity solution

    C/Cb = (1/Gamma(4/3)) integral_0^eta exp(-t^3) dt ,
        eta = y (s / (9 D x))^(1/3).

The local wall flux integrated over an electrode of axial length ``L`` gives an
average flux ``N_avg = C0 * Cb * D^(2/3) s^(1/3) L^(-1/3)`` with the *exact*
geometry-independent constant

    C0 = (3/2) / (9^(1/3) Gamma(4/3)) = 0.80755...

Total limiting current ``i = n F * (perimeter) * integral_0^L N dx``.  Re-expressing
in Bard & Faulkner's variables reproduces **Table 11.6.1** (B&F 2nd ed., p. 444):

    channel : i = 1.47 nFC (D A / b)^(2/3) v^(1/3)      (planar, parallel flow)
    tubular : i = 1.61 nFC (D A / r)^(2/3) v^(1/3)

where ``A`` is the electrode area and ``v`` the volume flow rate.  The dimensionless
prefactors 1.47 and 1.61 follow from ``C0`` together with the geometric factors
(``s = 6Um/b`` vs ``4Um/r``; flat width ``w`` vs perimeter ``2 pi r``); see
:func:`channel_prefactor` and :func:`tubular_prefactor`.

Numerical method
================
We discretise Eq. (1) (keeping axial diffusion, so the solve is genuinely 2-D and
not merely a re-statement of the Leveque ODE) by finite differences on a
**structured grid graded toward the wall**, where the entire concentration
variation lives in a thin Leveque layer of thickness ``~ (9 D x L^2 / s)^(1/3)``.
Convection ``v_x dC/dx`` is upwinded (flow is in +x, so a backward difference),
diffusion is central, and the resulting sparse linear system ``A c = rhs`` is
solved once with :func:`scipy.sparse.linalg.spsolve`.  The diffusion-limited
current is recovered by integrating the wall-normal gradient over the electrode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Exact Leveque constant and Table 11.6.1 prefactors
# ---------------------------------------------------------------------------
#: Geometry-independent Leveque constant ``C0 = (3/2)/(9^(1/3) Gamma(4/3))``.
LEVEQUE_C0: float = 1.5 / (9.0 ** (1.0 / 3.0) * gamma(4.0 / 3.0))


def channel_prefactor() -> float:
    """Dimensionless prefactor for a channel (planar parallel-flow) electrode.

    Returns the constant multiplying ``nFC (D A / b)^(2/3) v^(1/3)`` in B&F
    Table 11.6.1; should equal 1.47 to three significant figures.

    Derivation: with ``s = 6 Um / b``, electrode area ``A = w L``, volume flow
    ``v = Um w b``, the Leveque total current
    ``i = nF w L * C0 Cb D^(2/3) s^(1/3) L^(-1/3)`` rearranges to
    ``C0 * 6^(1/3) * nFC (D A/b)^(2/3) v^(1/3)``.
    """
    return LEVEQUE_C0 * 6.0 ** (1.0 / 3.0)


def tubular_prefactor() -> float:
    """Dimensionless prefactor for a tubular electrode.

    Returns the constant multiplying ``nFC (D A / r)^(2/3) v^(1/3)`` in B&F
    Table 11.6.1; should equal 1.61 to three significant figures.

    Derivation: with ``s = 4 Um / r``, perimeter ``2 pi r`` (area
    ``A = 2 pi r L``), volume flow ``v = Um pi r^2``, the Leveque total current
    ``i = nF (2 pi r) * C0 Cb D^(2/3) s^(1/3) L^(2/3)`` rearranges to
    ``2 * C0 * nFC (D A/r)^(2/3) v^(1/3)`` (the pi powers cancel exactly).
    """
    return 2.0 * LEVEQUE_C0


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def graded_wall_grid(y_max: float, n: int, beta: float = 1.04) -> np.ndarray:
    """Return ``n+1`` node positions on ``[0, y_max]`` geometrically graded to the wall.

    Spacing grows by the factor ``beta`` away from ``y = 0`` so the thin Leveque
    diffusion layer (where ``C`` varies) is well resolved.  ``beta = 1`` gives a
    uniform grid.
    """
    if beta == 1.0:
        return np.linspace(0.0, y_max, n + 1)
    # geometric series of cell widths h, h*beta, ... summing to y_max
    powers = beta ** np.arange(n)
    widths = powers / powers.sum() * y_max
    nodes = np.concatenate(([0.0], np.cumsum(widths)))
    return nodes


@dataclass
class ChannelProblem:
    """Dimensionless statement of the Leveque convective-diffusion problem.

    The PDE ``D(C_xx + C_yy) = s y C_x`` is solved on ``x in [0, L]``,
    ``y in [0, y_max]`` with ``C = 0`` on the electrode (``y = 0``), ``C = Cb`` at
    the inlet (``x = 0``) and the outer edge (``y = y_max``), and a zero axial
    second-derivative (convective outflow) condition at ``x = L``.

    All numbers default to ``D = s = Cb = L = 1`` so the recovered current is
    directly the dimensionless prefactor; only the grid parameters matter for
    accuracy.
    """

    D: float = 1.0          # diffusion coefficient
    s: float = 1.0          # wall shear rate dv_x/dy|_wall
    Cb: float = 1.0         # bulk / inlet concentration
    L: float = 1.0          # electrode (axial) length
    nx: int = 200           # axial cells
    ny: int = 160           # transverse cells
    y_max: float | None = None   # outer-edge distance (default: 6x Leveque layer)
    beta: float = 1.05      # wall-grading factor (transverse)
    x_beta: float = 1.0     # leading-edge grading factor (axial); 1.0 = uniform
    geometry: str = "channel"    # "channel" or "tubular" (sets the prefactor target)

    def leveque_layer(self) -> float:
        """Characteristic Leveque diffusion-layer thickness at the trailing edge."""
        return (9.0 * self.D * self.L ** 2 / self.s) ** (1.0 / 3.0)


def solve_channel(prob: ChannelProblem) -> dict:
    """Solve the 2-D convective-diffusion problem and return the limiting current.

    Returns a dict with the concentration field ``C`` (shape ``(ny+1, nx+1)``),
    the grids ``x``, ``y``, the dimensionless ``prefactor`` recovered from the
    integrated wall flux, and intermediate quantities.

    The dimensionless prefactor is defined so that for ``D = s = Cb = L = 1`` the
    integrated average flux equals ``C0`` (the Leveque constant); we therefore
    report ``prefactor = N_avg / (Cb D^(2/3) s^(1/3) L^(-1/3))`` which should
    converge to :data:`LEVEQUE_C0`.
    """
    D, s, Cb, L = prob.D, prob.s, prob.Cb, prob.L
    y_max = prob.y_max if prob.y_max is not None else 6.0 * prob.leveque_layer()

    # axial grid: optionally graded toward the leading edge (x = 0), where the
    # Leveque flux has an integrable x^(-1/3) cusp that a uniform grid undersamples.
    x = graded_wall_grid(L, prob.nx, prob.x_beta)
    y = graded_wall_grid(y_max, prob.ny, prob.beta)
    nx1, ny1 = x.size, y.size
    N = nx1 * ny1

    def idx(i, j):  # i over x (0..nx), j over y (0..ny)
        return j * nx1 + i

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(N)

    # transverse non-uniform second-derivative weights (3-point)
    # for node j: hm = y_j - y_{j-1}, hp = y_{j+1} - y_j
    hm = np.empty(ny1)
    hp = np.empty(ny1)
    hm[1:] = np.diff(y)
    hp[:-1] = np.diff(y)

    # axial non-uniform spacings: gm = x_i - x_{i-1}, gp = x_{i+1} - x_i
    gm = np.empty(nx1)
    gp = np.empty(nx1)
    gm[1:] = np.diff(x)
    gp[:-1] = np.diff(x)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    for j in range(ny1):
        for i in range(nx1):
            k = idx(i, j)
            # --- Dirichlet boundaries ---
            if j == 0:
                # wall: electrode (C=0) for x>0, but at the leading corner i=0
                # the inlet condition (C=Cb) wins to avoid an over-constrained
                # singular corner; electrode covers the whole wall x in (0, L].
                add(k, k, 1.0)
                rhs[k] = 0.0 if i > 0 else Cb
                continue
            if j == ny1 - 1:        # outer edge: bulk
                add(k, k, 1.0)
                rhs[k] = Cb
                continue
            if i == 0:              # inlet: bulk
                add(k, k, 1.0)
                rhs[k] = Cb
                continue

            # --- interior + outflow column (i == nx) ---
            vx = s * y[j]

            # transverse diffusion D * C_yy  (non-uniform central)
            am = D * 2.0 / (hm[j] * (hm[j] + hp[j]))      # coef of C_{i,j-1}
            ap = D * 2.0 / (hp[j] * (hm[j] + hp[j]))      # coef of C_{i,j+1}
            ac = -(am + ap)                                # coef of C_{i,j}
            add(k, idx(i, j - 1), am)
            add(k, idx(i, j + 1), ap)

            if i == prob.nx:
                # convective outflow: no axial diffusion, backward-difference
                # convection only (standard convective outlet for flow in +x).
                conv = vx / gm[i]
                add(k, k, ac - conv)
                add(k, idx(i - 1, j), conv)
                continue

            # axial diffusion D * C_xx  (non-uniform central 3-point)
            axm = D * 2.0 / (gm[i] * (gm[i] + gp[i]))   # coef of C_{i-1,j}
            axp = D * 2.0 / (gp[i] * (gm[i] + gp[i]))   # coef of C_{i+1,j}
            add(k, idx(i - 1, j), axm)
            add(k, idx(i + 1, j), axp)
            ac_x = -(axm + axp)

            # axial convection -vx * C_x, upwind (flow +x) -> backward difference
            conv = vx / gm[i]
            add(k, idx(i - 1, j), conv)      # -vx*(C_i - C_{i-1})/gm -> +conv on i-1
            add(k, k, ac + ac_x - conv)

    A = sp.csr_array((data, (rows, cols)), shape=(N, N))
    c = spsolve(A.tocsc(), rhs)
    C = c.reshape(ny1, nx1)  # C[j, i]

    # --- wall flux: D * dC/dy at y=0, one-sided 2nd-order over the graded grid ---
    h1 = y[1] - y[0]
    h2 = y[2] - y[0]
    # 3-point one-sided derivative at node 0 for non-uniform spacing
    c0 = -(h1 + h2) / (h1 * h2)
    c1 = h2 / (h1 * (h2 - h1))
    c2 = -h1 / (h2 * (h2 - h1))
    dCdy_wall = c0 * C[0, :] + c1 * C[1, :] + c2 * C[2, :]
    flux = D * dCdy_wall   # local wall flux N(x) (positive into electrode)

    # --- integrate the wall flux over the electrode ---------------------------
    # The concentration is discontinuous at the leading corner (x=0, y=0): the
    # inlet holds C=Cb while the electrode imposes C=0.  Finite differences cannot
    # resolve this corner, so the first few flux nodes are corrupted (they
    # *over*-estimate the true x^(-1/3) cusp).  Away from the corner the numerical
    # flux matches the Leveque similarity flux N(x) = K x^(-1/3) to <0.1%.  We
    # therefore (i) fit K to a clean mid-electrode window and (ii) replace the
    # corrupted head integral [0, x_cut] by its exact analytic value K*(3/2)
    # x_cut^(2/3), integrating the trusted numerical flux over the remainder.
    trapz = getattr(np, "trapz", None) or np.trapezoid

    # clean window: x in [0.2 L, 0.8 L]
    win = (x >= 0.2 * L) & (x <= 0.8 * L)
    K = float(np.median(flux[win] * x[win] ** (1.0 / 3.0)))  # N = K x^(-1/3)

    # x_cut: smallest node beyond which the numerical flux tracks K x^(-1/3) to
    # better than 1% (the corrupted head ends here).
    ratio = flux[1:] / (K * x[1:] ** (-1.0 / 3.0))
    good = np.where(np.abs(ratio - 1.0) < 0.01)[0]
    i_cut = int(good[0]) + 1 if good.size else 1
    x_cut = x[i_cut]

    head = K * 1.5 * x_cut ** (2.0 / 3.0)          # exact int_0^{x_cut} K x^-1/3 dx
    tail = trapz(flux[i_cut:], x[i_cut:])
    total_flux = head + tail
    N_avg = total_flux / L

    prefactor = N_avg / (Cb * D ** (2.0 / 3.0) * s ** (1.0 / 3.0) * L ** (-1.0 / 3.0))

    return {
        "C": C,
        "x": x,
        "y": y,
        "flux": flux,
        "total_flux": total_flux,
        "N_avg": N_avg,
        "prefactor": prefactor,
        "y_max": y_max,
        "leveque_layer": prob.leveque_layer(),
        "K": K,
        "x_cut": x_cut,
        "i_cut": i_cut,
    }


def limiting_current_scaling(
    prob: ChannelProblem, flow_factors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the integrated current as the wall shear ``s`` is scaled.

    Because ``s`` is proportional to the mean velocity (and hence the volume flow
    rate), the total wall flux should scale as ``s^(1/3) ~ (flow rate)^(1/3)``.
    Returns ``(s_values, total_flux_values)`` for a log-log slope check.
    """
    s_vals = prob.s * np.asarray(flow_factors, dtype=float)
    fluxes = np.empty_like(s_vals)
    for k, sv in enumerate(s_vals):
        p = ChannelProblem(**{**prob.__dict__, "s": sv})
        fluxes[k] = solve_channel(p)["total_flux"]
    return s_vals, fluxes
