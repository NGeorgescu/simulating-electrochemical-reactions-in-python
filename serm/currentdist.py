r"""Primary current distribution on a disk electrode (Newman 1966).

This module solves an **ohmic** (potential-field) boundary-value problem that is
physically distinct from the mass-transport problems elsewhere in ``serm``.  A
disk electrode of radius ``a`` sits flush in a coplanar insulating plane, in
contact with an electrolyte of uniform conductivity ``kappa`` filling the
half-space ``z > 0``.  With electrode kinetics ignored, the potential obeys
Laplace's equation

.. math::
    \nabla^2 \Phi = 0 \qquad (z > 0),

subject to the mixed boundary condition on the plane ``z = 0``

.. math::
    \Phi = \Phi_0      \quad (r < a, \text{ the disk, an equipotential}), \\
    \partial\Phi/\partial z = 0  \quad (r > a, \text{ the insulator}),

and ``\Phi \to 0`` at infinity.  This is the **primary current distribution**.

Closed form (Newman 1966)
--------------------------
Newman separates the problem in *oblate spheroidal* (rotational elliptic)
coordinates :math:`(\xi, \eta)` defined by [Newman 1966, Eq. 1, p. 501]

.. math::
    z = a\,\xi\eta, \qquad r = a\sqrt{(1+\xi^2)(1-\eta^2)} .

The ``n = 0`` separated solution that satisfies all the boundary conditions is
[Eq. 6]

.. math::
    \Phi/\Phi_0 = 1 - \tfrac{2}{\pi}\,\tan^{-1}\xi .

Differentiating gives the edge-singular current density at the disk [Eq. 7],

.. math::
    i(r) = \frac{2\kappa\Phi_0}{\pi\sqrt{a^2 - r^2}} ,

whose integral over the disk is the total current [Eq. 8]

.. math::
    I = 2\pi\int_0^a i\,r\,dr = 4\kappa a \Phi_0 ,

so the disk (access / spreading) resistance is [Eq. 9]

.. math::
    R = \Phi_0 / I = \frac{1}{4\kappa a} .

Newman quotes the worked number ``R = 114.7`` ohm for a ``0.5`` cm diameter disk
(``a = 0.25`` cm) in 0.1 M CuSO4 with ``kappa = 0.00872`` (ohm-cm)^{-1}
[Newman 1966, p. 502 and Table I, p. 501].  Newman (1970), *Electrochemical
Systems*, re-derives the same ``R = 1/(4\kappa a)`` as a corollary.

Independent numerical check
---------------------------
:func:`solve_laplace_fd` solves the *same* Laplace problem from scratch on a
graded cylindrical ``(r, z)`` finite-difference grid -- with no reference to the
spheroidal separation -- and recovers ``R`` by integrating the numerical current
density over the disk.  Agreement of the two to a tight tolerance is the
tier-1 validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Closed-form results (Newman 1966)
# ---------------------------------------------------------------------------
def disk_resistance(kappa: float, a: float) -> float:
    r"""Primary disk resistance ``R = 1/(4 kappa a)`` [Newman 1966, Eq. 9].

    Parameters
    ----------
    kappa : float
        Electrolyte conductivity (e.g. ohm^-1 cm^-1).
    a : float
        Disk radius (e.g. cm).

    Returns
    -------
    float
        Resistance in ohm (consistent units).
    """
    return 1.0 / (4.0 * kappa * a)


def total_current(kappa: float, a: float, phi0: float) -> float:
    r"""Total current to the disk ``I = 4 kappa a Phi0`` [Newman 1966, Eq. 8]."""
    return 4.0 * kappa * a * phi0


def current_density(r, kappa: float, a: float, phi0: float):
    r"""Primary current density ``i(r) = 2 kappa Phi0 / (pi sqrt(a^2 - r^2))``.

    [Newman 1966, Eq. 7.]  Edge-singular: ``i -> inf`` as ``r -> a``.  Defined
    only on the disk ``r < a``; ``r >= a`` returns ``inf``.

    Parameters
    ----------
    r : array_like
        Radial coordinate(s) on the disk, ``0 <= r < a``.
    """
    r = np.asarray(r, dtype=float)
    out = np.full(r.shape, np.inf)
    inside = r < a
    out[inside] = (2.0 * kappa * phi0) / (np.pi * np.sqrt(a * a - r[inside] ** 2))
    return out if out.shape else float(out)


def potential_spheroidal(xi, phi0: float):
    r"""Closed-form potential ``Phi/Phi0 = 1 - (2/pi) arctan(xi)`` [Eq. 6].

    Parameters
    ----------
    xi : array_like
        Oblate-spheroidal radial coordinate ``xi >= 0`` (``xi = 0`` on the disk
        surface, ``xi -> inf`` far away).
    """
    xi = np.asarray(xi, dtype=float)
    return phi0 * (1.0 - (2.0 / np.pi) * np.arctan(xi))


def cylindrical_to_xi(r, z, a: float):
    r"""Oblate-spheroidal ``xi`` for cylindrical ``(r, z)`` over a disk radius ``a``.

    Inverts Newman Eq. 1.  Solving
    ``z = a xi eta`` and ``r = a sqrt((1+xi^2)(1-eta^2))`` for ``xi`` gives the
    nonnegative root of ``xi^2`` of the quadratic

    .. math::
        a^2(\xi^2)^2 + (a^2 - r^2 - z^2)\,\xi^2 - z^2 = 0 ,

    i.e. ``xi = sqrt[ (p + sqrt(p^2 + 4 z^2/a^2)) / 2 ]`` with
    ``p = (r^2 + z^2)/a^2 - 1``.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    p = (r * r + z * z) / (a * a) - 1.0
    xi2 = 0.5 * (p + np.sqrt(p * p + 4.0 * z * z / (a * a)))
    xi2 = np.clip(xi2, 0.0, None)
    return np.sqrt(xi2)


def potential_field(r, z, a: float, phi0: float):
    r"""Closed-form potential at cylindrical ``(r, z)`` via the spheroidal map."""
    return potential_spheroidal(cylindrical_to_xi(r, z, a), phi0)


# ---------------------------------------------------------------------------
# Independent numerical solve: Laplace FD on a graded (r, z) grid
# ---------------------------------------------------------------------------
@dataclass
class DiskSolution:
    """Result of :func:`solve_laplace_fd`.

    Attributes
    ----------
    r, z : np.ndarray
        1-D graded grid coordinates (length ``nr`` and ``nz``).
    phi : np.ndarray
        Potential on the grid, shape ``(nr, nz)`` (``phi[i, k]`` at
        ``r[i], z[k]``).
    I : float
        Total current to the disk, from integrating the numerical current
        density over the disk face.
    R : float
        Numerical resistance ``Phi0 / I``.
    r_disk : np.ndarray
        Disk-face radial nodes (``r < a``) where the current density is sampled.
    i_disk : np.ndarray
        Numerical current density ``i = -kappa dPhi/dz`` at ``r_disk`` on ``z=0``.
    """

    r: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    I: float
    R: float
    r_disk: np.ndarray
    i_disk: np.ndarray


def _graded_axis(a: float, far: float, n_in: int, n_out: int, ratio: float):
    """Build a 1-D graded coordinate clustered near the disk edge ``r = a``.

    Uniform-ish inside ``[0, a]`` (``n_in`` cells) then geometrically expanding
    out to ``far`` (``n_out`` cells, growth factor ``ratio``).  The edge ``a`` is
    always a node, so the singularity sits exactly on a cell boundary.
    """
    inside = np.linspace(0.0, a, n_in + 1)
    steps = ratio ** np.arange(n_out)
    outside = a + (far - a) * np.cumsum(steps) / np.sum(steps)
    return np.unique(np.concatenate([inside, outside]))


def solve_laplace_fd(
    a: float = 0.25,
    kappa: float = 0.00872,
    phi0: float = 1.0,
    far: float | None = None,
    n_in: int = 120,
    n_out: int = 140,
    ratio: float = 1.06,
) -> DiskSolution:
    r"""Solve Laplace's equation for the disk electrode by finite differences.

    Axisymmetric Laplace operator in cylindrical ``(r, z)``,

    .. math::
        \frac{1}{r}\partial_r(r\,\partial_r\Phi) + \partial_{zz}\Phi = 0 ,

    discretised by a finite-volume (conservative) scheme on a graded grid so the
    mixed disk/insulator boundary on ``z = 0`` is imposed without reference to
    the spheroidal closed form.  ``R = Phi0 / I`` is then read off by integrating
    the numerical current density over the disk -- an *independent* recovery of
    Newman's ``R = 1/(4 kappa a)``.

    Parameters
    ----------
    a : float
        Disk radius.
    kappa : float
        Conductivity (only rescales ``i`` and ``R``; the potential field is
        independent of it).
    phi0 : float
        Disk potential (Dirichlet value on ``r < a``, ``z = 0``).
    far : float, optional
        Outer truncation radius of the (quarter-disk) domain.  Defaults to
        ``40 a`` -- far enough that the ``Phi ~ 2 kappa Phi0 a/(pi rho)`` decay
        [Newman 1966, Eq. 10] is small at the boundary.
    n_in : int
        Cells across ``[0, a]`` in each direction.
    n_out, ratio : int, float
        Number and geometric growth of the expanding cells out to ``far``.

    Returns
    -------
    DiskSolution
    """
    if far is None:
        far = 40.0 * a

    r = _graded_axis(a, far, n_in, n_out, ratio)
    z = _graded_axis(a, far, n_in, n_out, ratio)
    nr, nz = r.size, z.size
    N = nr * nz

    def idx(i, k):
        return i * nz + k

    # Cell faces (midpoints) for the conservative finite-volume stencil.
    rf = np.empty(nr + 1)
    rf[1:-1] = 0.5 * (r[:-1] + r[1:])
    rf[0] = r[0]
    rf[-1] = r[-1]
    zf = np.empty(nz + 1)
    zf[1:-1] = 0.5 * (z[:-1] + z[1:])
    zf[0] = z[0]
    zf[-1] = z[-1]

    dr = np.diff(r)
    dz = np.diff(z)
    # Control-volume widths (distance between bounding faces).
    hr = np.diff(rf)
    hz = np.diff(zf)

    # Face radius for the cylindrical 1/r d/dr(r d/dr) term; r-flux through the
    # face at rf[i] carries area weight proportional to rf[i].
    a_disk_node = np.argmin(np.abs(r - a))  # index of the edge node r = a

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(N)

    def add(row, col, val):
        rows.append(row)
        cols.append(col)
        data.append(val)

    for i in range(nr):
        for k in range(nz):
            row = idx(i, k)

            # --- Dirichlet on the disk face: r < a at z = 0 ---
            if k == 0 and r[i] < a - 1e-12:
                add(row, row, 1.0)
                rhs[row] = phi0
                continue

            # --- Far-field Dirichlet Phi = 0 on outer r and outer z faces ---
            if i == nr - 1 or k == nz - 1:
                add(row, row, 1.0)
                rhs[row] = 0.0
                continue

            # Interior / Neumann-on-z=0 control volume.  Sum of conservative
            # fluxes through the four faces = 0.  Face coefficient =
            # (face area)/(node spacing); cylindrical r-weight uses face radius.
            diag = 0.0

            # r-direction faces
            if i > 0:
                w = rf[i] * hz[k] / dr[i - 1]
                add(row, idx(i - 1, k), w)
                diag -= w
            # i == 0 is the axis r = 0: rf[0] = 0 so no flux (regularity).
            if i < nr - 1:
                w = rf[i + 1] * hz[k] / dr[i]
                add(row, idx(i + 1, k), w)
                diag -= w

            # z-direction faces (r-weight = cell-centre radius * width)
            rc = max(r[i], 1e-30)
            if k > 0:
                w = rc * hr[i] / dz[k - 1]
                add(row, idx(i, k - 1), w)
                diag -= w
            else:
                # z = 0 with r >= a: insulating Neumann (no z-flux below).
                pass
            if k < nz - 1:
                w = rc * hr[i] / dz[k]
                add(row, idx(i, k + 1), w)
                diag -= w

            add(row, row, diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    phi_flat = spla.spsolve(A.tocsc(), rhs)
    phi = phi_flat.reshape(nr, nz)

    # Current density on the disk face: i = -kappa dPhi/dz at z = 0, computed
    # with a one-sided three-point stencil on the (nonuniform) z grid.
    z0, z1, z2 = z[0], z[1], z[2]
    c0 = (2.0 * z0 - z1 - z2) / ((z0 - z1) * (z0 - z2))
    c1 = (z0 - z2) / ((z1 - z0) * (z1 - z2))
    c2 = (z0 - z1) / ((z2 - z0) * (z2 - z1))
    dphidz0 = c0 * phi[:, 0] + c1 * phi[:, 1] + c2 * phi[:, 2]
    i_face = -kappa * dphidz0  # current density entering electrolyte at z=0

    disk_mask = r < a - 1e-12
    r_disk = r[disk_mask]
    i_disk = i_face[disk_mask]

    # Total current I = integral over disk of i * 2 pi r dr.  Integrate on the
    # nodes that lie on the disk (trapezoid in r, with the edge node included).
    r_int = r[r <= a + 1e-12]
    i_int = i_face[r <= a + 1e-12]
    # The edge node value is the singular one; the FD value there is finite but
    # under-resolves the singularity.  Use trapezoid up to the last interior
    # node and treat the integral as a robust lower-bracket estimate.
    integrand = i_int * 2.0 * np.pi * r_int
    I = np.trapezoid(integrand, r_int)
    R = phi0 / I if I != 0 else np.inf

    return DiskSolution(
        r=r, z=z, phi=phi, I=I, R=R, r_disk=r_disk, i_disk=i_disk
    )
