"""Chapter 15 -- sparse finite-difference solvers on an expanding space grid.

Python re-implementation of the algorithms in Michael Honeychurch,
*Simulating Electrochemical Reactions in Mathematica* (SERM), Chapter 15,
"Finite difference simulations using sparse arrays", and the companion notebooks
``sparseTri.nb``, ``sparseECRxnExp.nb`` and ``sparseSquareRxnExp.nb``.

The chapter's thesis is purely numerical-linear-algebra: the *physics* (fully
implicit finite differences of Fick's second law on an exponentially expanding
grid) is unchanged from Chapters 3 and 13, but the linear system solved at every
time step is assembled and solved as a **sparse** matrix instead of a dense one.
For a single diffusing species the system is tridiagonal and a banded/Thomas
solver wins; the sparse representation pays off when several species are coupled
(an EC mechanism, a square scheme), where the matrix is banded-block but no
longer tridiagonal, so a generic dense ``solve`` would waste almost all of its
work on structural zeros.

Two ports live here, each provided in a dense and a sparse flavour so the chapter
can cross-check them against one another to machine precision:

* :func:`simulate_cv_single` -- quasireversible ``O + e- <=> R`` cyclic
  voltammetry, the tridiagonal single-species problem of ``sparseTri.nb``.
* :func:`simulate_cv_ec` -- the coupled ``O + e- <=> R``, ``R <=> P`` EC
  mechanism of ``sparseECRxnExp.nb`` (three species, block-tridiagonal matrix).

Expanding-grid discretisation (SERM Section 3.4)
------------------------------------------------
With expansion factor ``a`` the dimensionless interior node ``j`` (1-based,
``j = 2 .. m-1``) of a single species obeys the fully implicit equation

    -D_M a^(4-2j) c_{j-1} + [1 + (1+a) D_M a^(3-2j)] c_j - D_M a^(3-2j) c_{j+1}
        = c_j^{old},

i.e. a tridiagonal row with sub-diagonal ``-D_M a^(4-2j)``, main diagonal
``1 + (1+a) D_M a^(3-2j)`` and super-diagonal ``-D_M a^(3-2j)`` (cf.
``makeDiagonals`` and ``mat3`` in the source notebooks).  A homogeneous
first-order reaction with dimensionless rate constant ``k`` adds ``+k`` to the
main diagonal of the consumed species and ``-k`` off-diagonal coupling to its
partner (``sparseECRxnExp.nb``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import spsolve

from .kinetics import (
    F, R, ks_star_sweep, bv_surface_factor, bv_surface_conc,
)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class CVParams:
    """Dimensionless parameters for an expanding-grid CV simulation.

    The defaults reproduce the quasireversible single-species case of
    ``sparseTri.nb`` (2 mV steps, ``a = 1.05``, ``D_M = 2``).  ``ks_dim`` is the
    dimensional standard rate constant ``k^o`` (cm/s); it is converted to the
    dimensionless ``ksStar`` exactly as in the source notebook.
    """

    alpha: float = 0.5            # transfer coefficient
    upper_limit: float = 11.6435  # initial (n F /RT)(E - E0)
    lower_limit: float = -15.5766 # switching (n F /RT)(E - E0)
    a: float = 1.05               # grid expansion factor
    D_M: float = 2.0              # model diffusion coefficient
    ks_dim: float = 0.05          # standard rate constant k^o (cm/s)
    dE_mV: float = 2.0            # potential step per time increment (mV)
    temperature: float = 298.15   # K

    @property
    def script_f(self) -> float:
        """``F / (R T)`` (1/V)."""
        return F / (R * self.temperature)

    @property
    def sweep_span(self) -> float:
        """Total dimensionless sweep length ``T = 2(upper + |lower|)`` (SERM)."""
        return 2.0 * (self.upper_limit + abs(self.lower_limit))

    @property
    def n_time(self) -> int:
        """Number of time/potential increments for the requested step size."""
        return round(self.sweep_span / (self.dE_mV * 1e-3 * self.script_f))

    @property
    def tau(self) -> float:
        """Dimensionless time increment ``T / (n - 1)``."""
        return self.sweep_span / (self.n_time - 1)

    @property
    def m_space(self) -> int:
        """Number of spatial nodes ``m = 1 + ceil(6 sqrt(D_M (n-1)))`` (SERM)."""
        return 1 + int(np.ceil(6.0 * np.sqrt(self.D_M * (self.n_time - 1))))

    @property
    def ks_star(self) -> float:
        """Dimensionless rate constant ``2 k^o sqrt(T / (D_M (n-1)))`` (SERM)."""
        return ks_star_sweep(self.ks_dim, self.sweep_span, self.D_M, self.n_time)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def potential_xi(k, p: CVParams):
    """Surface potential factor ``xi = exp[(nF/RT)(E - E0)]`` at increment ``k``.

    Triangular sweep: ramp down from ``upper_limit`` until the vertex at
    ``k = (n+1)/2``, then ramp back up (cf. the ``If`` in every ``solve*``
    function in the source notebooks).
    """
    if k > (p.n_time + 1) / 2:
        return np.exp(p.upper_limit - p.sweep_span + p.tau * (k - 1))
    return np.exp(p.upper_limit - p.tau * (k - 1))


def potential_axis(p: CVParams):
    """Dimensionless potential ``(nF/RT)(E - E0)`` at each of the ``n`` points."""
    return np.array(
        [
            p.upper_limit - p.sweep_span + p.tau * (k - 1)
            if k > (p.n_time + 1) / 2
            else p.upper_limit - p.tau * (k - 1)
            for k in range(1, p.n_time + 1)
        ]
    )


def _interior_diagonals(p: CVParams):
    """Sub/main/super diagonals for the ``m-2`` interior nodes ``j = 2..m-1``."""
    j = np.arange(2, p.m_space)
    sub = -p.D_M * p.a ** (4 - 2 * j)        # A[j, j-1]
    sup = -p.D_M * p.a ** (3 - 2 * j)        # A[j, j+1]
    main = 1.0 + (1.0 + p.a) * p.D_M * p.a ** (3 - 2 * j)
    return sub, main, sup


# ---------------------------------------------------------------------------
# Single-species quasireversible CV  (sparseTri.nb)
# ---------------------------------------------------------------------------
def simulate_cv_single(p: CVParams, backend="sparse"):
    """Fully implicit quasireversible ``O + e- <=> R`` CV on an expanding grid.

    The interior is the tridiagonal system of :func:`_interior_diagonals`; the
    surface node is eliminated through the Butler--Volmer / equal-and-opposite
    flux boundary condition, which injects the time-dependent factor ``tmp`` into
    the first row of the matrix and the first/last entries of the right-hand side
    (a direct port of ``solveCV1`` / ``solveCV3``).

    Parameters
    ----------
    p : CVParams
        Dimensionless simulation parameters.
    backend : {"sparse", "dense"}
        ``"sparse"`` assembles a ``scipy.sparse`` matrix and solves with
        :func:`scipy.sparse.linalg.spsolve`; ``"dense"`` uses a full numpy array
        and :func:`scipy.linalg.solve`.  The two must agree to ~1e-13.

    Returns
    -------
    profiles : ndarray, shape (n, m)
        Concentration of ``O`` at every node and time increment (row 0 = initial
        condition).  The surface value is ``profiles[:, 0]``.
    """
    m = p.m_space
    M = m - 2
    sub, main, sup = _interior_diagonals(p)
    y1, z1 = main[0], sup[0]
    tail = p.D_M * p.a ** (5 - 2 * m)         # bulk-boundary feed into b[-1]

    if backend == "dense":
        A = np.zeros((M, M))
        idx = np.arange(M)
        A[idx, idx] = main
        A[idx[1:], idx[:-1]] = sub[1:]
        A[idx[:-1], idx[1:]] = sup[:-1]
    elif backend == "sparse":
        A = sp.lil_matrix((M, M))
        A.setdiag(main)
        A.setdiag(sub[1:], -1)
        A.setdiag(sup[:-1], 1)
    else:
        raise ValueError("backend must be 'sparse' or 'dense'")

    cur = np.ones(m)
    profiles = [cur]
    for k in range(2, p.n_time + 1):
        xi = potential_xi(k, p)
        tmp = bv_surface_factor(xi, p.ks_star, p.alpha)
        b = cur[1:M + 1].copy()
        b[0] += tmp * p.D_M * p.ks_star * xi ** (1.0 - p.alpha)
        b[-1] += tail
        A[0, 0] = y1 - 4.0 * p.D_M * tmp
        A[0, 1] = z1 + p.D_M * tmp
        if backend == "dense":
            inner = dense_solve(A, b)
        else:
            inner = spsolve(A.tocsc(), b)
        c0 = bv_surface_conc(inner[0], inner[1], xi, p.ks_star, p.alpha, tmp)
        cur = np.concatenate(([c0], inner, [1.0]))
        profiles.append(cur)
    return np.array(profiles)


def cv_current_single(profiles, p: CVParams):
    """Dimensionless current from the three-point surface gradient (SERM cv1).

    ``i ~ [(2+a) a c_0 - (1+a)^2 c_1 + c_2] * sqrt(D_M (n-1)/(2 a^2 (1+a) T))``.
    """
    scale = np.sqrt(
        p.D_M * (p.n_time - 1) / (2.0 * p.a ** 2 * (1.0 + p.a) * p.sweep_span)
    )
    c0, c1, c2 = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    return ((2.0 + p.a) * p.a * c0 - (1.0 + p.a) ** 2 * c1 + c2) * scale


# ---------------------------------------------------------------------------
# Coupled EC mechanism  (sparseECRxnExp.nb): O + e- <=> R, R <=> P
# ---------------------------------------------------------------------------
@dataclass
class ECParams:
    """Dimensionless parameters for the coupled EC CV (``sparseECRxnExp.nb``)."""

    alpha: float = 0.5
    upper_limit: float = 10.0
    lower_limit: float = -10.0
    a: float = 1.1
    ks_dim: float = 1.0           # dimensionless standard rate constant k^o
    dE_mV: float = 1.0            # 1 mV steps
    kf_dim: float = 50.0          # dimensional forward homogeneous rate (1/s units)
    kb_dim: float = 10.0          # dimensional backward homogeneous rate
    temperature: float = 298.0

    @property
    def script_f(self):
        return F / (R * self.temperature)

    @property
    def sweep_span(self):
        return 2.0 * (self.upper_limit + abs(self.lower_limit))

    @property
    def n_time(self):
        return round(self.sweep_span / (self.dE_mV * 1e-3 * self.script_f))

    @property
    def tau(self):
        return self.sweep_span / (self.n_time - 1)

    @property
    def k_plus(self):
        """Dimensionless forward homogeneous rate ``tau * kf_dim``."""
        return self.tau * self.kf_dim

    @property
    def k_minus(self):
        """Dimensionless backward homogeneous rate ``tau * kb_dim``."""
        return self.tau * self.kb_dim

    @property
    def D_M(self):
        """``D_M = 5 max(k+1, k-1, 0.4)`` (source notebook)."""
        return 5.0 * max(self.k_plus, self.k_minus, 0.4)

    @property
    def m_space(self):
        return 1 + int(np.ceil(6.0 * np.sqrt(self.D_M * (self.n_time - 1))))

    @property
    def ks_star(self):
        return ks_star_sweep(self.ks_dim, self.sweep_span, self.D_M, self.n_time)


def _ec_index(j, species):
    """Row/column index of species (0=O, 1=R, 2=P) at node ``j`` (1-based)."""
    return 3 * (j - 1) + species


def build_ec_matrix(p: ECParams):
    """Assemble the constant part of the block-tridiagonal EC matrix (sparse).

    Returns ``(A, L)`` where ``A`` is a ``scipy.sparse`` CSC matrix of size
    ``L = 3(m-1)`` with the three surface boundary-condition rows and all
    interior diffusion/reaction rows filled.  The two time-dependent surface
    entries (``A[0, idx(1,O)]`` and ``A[0, idx(1,R)]``) carry placeholder values
    of 1.0 so the sparsity pattern already contains those slots; the time loop
    overwrites them without changing the structure.
    """
    m = p.m_space
    L = 3 * (m - 1)
    a, D_M = p.a, p.D_M
    A = sp.lil_matrix((L, L))
    ix = _ec_index
    for j in range(2, m):
        cm = -D_M * a ** (4 - 2 * j)
        cd = 1.0 + (1.0 + a) * D_M * a ** (3 - 2 * j)
        cp = -D_M * a ** (3 - 2 * j)
        # O: pure diffusion
        r = ix(j, 0)
        A[r, ix(j - 1, 0)] = cm
        A[r, ix(j, 0)] = cd
        if j + 1 <= m - 1:
            A[r, ix(j + 1, 0)] = cp
        # R: diffusion + consumed by forward reaction, fed by P
        r = ix(j, 1)
        A[r, ix(j - 1, 1)] = cm
        A[r, ix(j, 1)] = cd + p.k_plus
        if j + 1 <= m - 1:
            A[r, ix(j + 1, 1)] = cp
        A[r, ix(j, 2)] = -p.k_minus
        # P: diffusion + consumed by backward reaction, fed by R
        r = ix(j, 2)
        A[r, ix(j - 1, 2)] = cm
        A[r, ix(j, 2)] = cd + p.k_minus
        if j + 1 <= m - 1:
            A[r, ix(j + 1, 2)] = cp
        A[r, ix(j, 1)] = -p.k_plus
    # Surface boundary conditions (rows 0, 1, 2).
    A[0, ix(1, 0)] = 1.0     # placeholder (Butler-Volmer, time dependent)
    A[0, ix(1, 1)] = 1.0     # placeholder
    A[0, ix(2, 0)] = -(1.0 + a) ** 2
    A[0, ix(3, 0)] = 1.0
    A[1, ix(1, 0)] = a * (2.0 + a)
    A[1, ix(1, 1)] = a * (2.0 + a)
    A[1, ix(2, 0)] = -(1.0 + a) ** 2
    A[1, ix(2, 1)] = -(1.0 + a) ** 2
    A[1, ix(3, 0)] = 1.0
    A[1, ix(3, 1)] = 1.0
    A[2, ix(1, 2)] = a * (2.0 + a)
    A[2, ix(2, 2)] = -(1.0 + a) ** 2
    A[2, ix(3, 2)] = 1.0
    return A.tocsc(), L


def simulate_cv_ec(p: ECParams, backend="sparse"):
    """Coupled EC cyclic voltammogram (``O + e- <=> R``, ``R <=> P``).

    Solves the block-tridiagonal system :func:`build_ec_matrix` at every time
    step, updating only the two potential-dependent surface entries.  Port of
    ``solveSparseEC`` from ``sparseECRxnExp.nb``.

    Parameters
    ----------
    p : ECParams
    backend : {"sparse", "dense"}
        ``"sparse"`` keeps the assembled matrix sparse and uses ``spsolve``;
        ``"dense"`` densifies it once and uses ``scipy.linalg.solve`` (only
        practical for the small illustrative grids used in the cross-check).

    Returns
    -------
    profiles : ndarray, shape (n, L)
        Stacked ``[cO, cR, cP]`` for nodes ``j = 1 .. m-1`` at each time
        increment.  Surface O/R/P are columns 0, 1, 2.
    """
    A_sp, L = build_ec_matrix(p)
    m = p.m_space
    a, D_M, alpha = p.a, p.D_M, p.alpha
    tail = D_M * a ** (5 - 2 * m)
    ix = _ec_index

    if backend == "dense":
        work = A_sp.toarray()
    elif backend == "sparse":
        work = A_sp.tolil()
    else:
        raise ValueError("backend must be 'sparse' or 'dense'")

    cur = np.tile([1.0, 0.0, 0.0], m - 1)
    profiles = [cur]
    for k in range(2, p.n_time + 1):
        xi = (
            np.exp(p.upper_limit - p.sweep_span + p.tau * (k - 1))
            if k > (p.n_time + 1) / 2
            else np.exp(p.upper_limit - p.tau * (k - 1))
        )
        b = cur.copy()
        b[0] = b[1] = b[2] = 0.0
        b[L - 3] += tail
        work[0, ix(1, 0)] = a * (2.0 + a) + p.ks_star * xi ** (-alpha)
        work[0, ix(1, 1)] = -p.ks_star * xi ** (1.0 - alpha)
        if backend == "dense":
            cur = dense_solve(work, b)
        else:
            cur = spsolve(work.tocsc(), b)
        profiles.append(cur)
    return np.array(profiles)


def cv_current_ec(profiles, p: ECParams):
    """Dimensionless current of the EC voltammogram from the O surface gradient.

    Uses the O concentrations at the first three nodes (columns 0, 3, 6 of the
    stacked profile), matching the ``cv1`` expression in ``sparseECRxnExp.nb``.
    """
    scale = np.sqrt(
        p.D_M * (p.n_time - 1) / (2.0 * p.a ** 2 * (1.0 + p.a) * p.sweep_span)
    )
    c0, c1, c2 = profiles[:, 0], profiles[:, 3], profiles[:, 6]
    return ((2.0 + p.a) * p.a * c0 - (1.0 + p.a) ** 2 * c1 + c2) * scale
