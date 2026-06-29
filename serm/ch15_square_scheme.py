"""Chapter 15 -- sparse / dense square-scheme cyclic voltammetry.

Companion to :mod:`serm.ch15_sparse_finite_differences`.  Here we port the
*square-scheme* simulator of Honeychurch, *Simulating Electrochemical Reactions
in Mathematica* (SERM), Chapter 15, Section "A square scheme with an expanding
space grid", whose full implementation lives in the standalone notebook
``Extra Notebooks/chapter15/sparseSquareRxnExp.nb``.

The square scheme couples two one-electron redox couples,

    O + e- <=> R        (formal potential offset ``dE_OR``)
    A + e- <=> B        (formal potential offset ``dE_AB``)

through homogeneous isomerisations on each oxidation state and a bimolecular
cross reaction::

    O <=>(k+1, k-1) A          (first order, linear)
    R <=>(k+2, k-2) B          (first order, linear)
    O + B <=>(k_cf, k_cb) R + A   (cross reaction, *second order* -> nonlinear)

Because the cross reaction is second order, the implicit finite-difference
equations are nonlinear: the matrix entries that carry ``k_cf`` / ``k_cb``
depend on the *unknown* concentrations.  Following the source notebook we solve
this by **fixed-point (Picard) iteration** -- at each time step the cross-reaction
block is evaluated at the previous iterate's concentrations, the resulting linear
system is solved, and the iteration repeats until the mean absolute change falls
below a tolerance.

Discretisation.  Identical expanding-grid stencil to
:mod:`serm.ch15_sparse_finite_differences` (SERM Section 3.4): for an interior
node ``j`` (1-based, ``j = 2 .. m-1``) the sub/main/super coefficients are
``-D_M a^(4-2j)``, ``1 + (1+a) D_M a^(3-2j)``, ``-D_M a^(3-2j)``.  Unknowns are
interleaved ``[cO(j), cR(j), cA(j), cB(j)]`` per node, so the constant part of
the matrix is block-tridiagonal with ``4 x 4`` blocks.  The four surface rows are
the two Butler--Volmer conditions (one per couple) and the two flux-conservation
conditions (``O+R`` and ``A+B``), matching ``bc`` in the source notebook.

Two backends are exposed -- ``"sparse"`` (``scipy.sparse`` + ``spsolve``) and
``"dense"`` (full array + ``scipy.linalg.solve``) -- so the chapter can
cross-check them to machine precision and time the sparse advantage.  The dense
backend here plays the role of the "dense Chapter-13 square-scheme reference"
called for by the port: it solves the *same* linearised equations with a generic
dense solver, so agreement to ~1e-10 validates the sparse assembly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import spsolve

from .kinetics import F, R


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class SquareParams:
    """Dimensionless parameters for an expanding-grid square-scheme CV.

    The defaults follow ``sparseSquareRxnExp.nb`` in spirit (two fast,
    near-reversible electron transfers, a fast cross reaction) but are sized for
    a tractable in-notebook cross-check rather than a publication-resolution
    sweep.  All homogeneous rate constants are supplied in the dimensionless
    ``k = tau * k_dim`` convention of the source notebook; the cross-reaction
    constants ``k_cf`` / ``k_cb`` are dimensionless second-order rates.
    """

    alpha: float = 0.5              # transfer coefficient (both couples)
    upper_limit: float = 12.0       # initial (F/RT)(E - E0_OR)
    lower_limit: float = -8.0       # switching (F/RT)(E - E0_OR)
    a: float = 1.1                  # grid expansion factor
    D_M: float = 5.0                # model diffusion coefficient
    dE_mV: float = 2.0              # potential step per increment (mV)
    temperature: float = 298.0      # K

    ks_dim1: float = 1.0e4          # dimensionless k^o, couple O/R
    ks_dim2: float = 1.0e4          # dimensionless k^o, couple A/B

    dE_OR: float = 0.0              # formal-potential offset of couple O/R (RT/F)
    dE_AB: float = 0.5              # formal-potential offset of couple A/B (RT/F)
    K_OA: float = 1.0e2             # equilibrium constant O <-> A
    K_RB: float = 1.0e2             # equilibrium constant R <-> B

    k_m1: float = 1.0               # dimensionless k-1 (A -> O), before tau scaling
    k_m2: float = 1.0               # dimensionless k-2 (B -> R), before tau scaling
    k_cb: float = 1.0e3             # dimensionless cross-reaction backward rate

    @property
    def script_f(self) -> float:
        return F / (R * self.temperature)

    @property
    def sweep_span(self) -> float:
        """Total dimensionless sweep length ``T = 2(|upper| + |lower|)``."""
        return 2.0 * (abs(self.upper_limit) + abs(self.lower_limit))

    @property
    def n_time(self) -> int:
        return round(self.sweep_span / (self.dE_mV * 1e-3 * self.script_f))

    @property
    def tau(self) -> float:
        return self.sweep_span / (self.n_time - 1)

    @property
    def m_space(self) -> int:
        return 1 + int(np.ceil(6.0 * np.sqrt(self.D_M * (self.n_time - 1))))

    # --- dimensionless homogeneous rate constants (k = tau * k_dim) ----------
    @property
    def k_minus1(self) -> float:
        return self.tau * self.k_m1

    @property
    def k_plus1(self) -> float:
        return self.k_minus1 * self.K_OA

    @property
    def k_minus2(self) -> float:
        return self.tau * self.k_m2

    @property
    def k_plus2(self) -> float:
        return self.k_minus2 * self.K_RB

    @property
    def K_c(self) -> float:
        """Cross-reaction equilibrium constant ``K_OA / K_RB``."""
        return self.K_OA / self.K_RB

    @property
    def k_cf(self) -> float:
        """Cross-reaction forward rate ``k_cb * K_c``."""
        return self.k_cb * self.K_c

    @property
    def ks_star1(self) -> float:
        return 2.0 * self.ks_dim1 * np.sqrt(
            self.sweep_span / (self.D_M * (self.n_time - 1))
        )

    @property
    def ks_star2(self) -> float:
        return 2.0 * self.ks_dim2 * np.sqrt(
            self.sweep_span / (self.D_M * (self.n_time - 1))
        )

    @property
    def cOi(self) -> float:
        """Initial bulk fraction of O: ``1/(1 + K_OA)``."""
        return 1.0 / (1.0 + self.K_OA)

    @property
    def cAi(self) -> float:
        """Initial bulk fraction of A: ``K_OA/(1 + K_OA)``."""
        return self.K_OA / (1.0 + self.K_OA)


# Species offsets within a node block [O, R, A, B].
_O, _R, _A, _B = 0, 1, 2, 3


def _sq_index(j: int, species: int) -> int:
    """Row/column index of ``species`` (0=O,1=R,2=A,3=B) at node ``j`` (1-based)."""
    return 4 * (j - 1) + species


def build_square_matrix(p: SquareParams):
    """Constant (concentration-independent) part of the block-tridiagonal matrix.

    Returns ``(A, L)`` with ``A`` a ``scipy.sparse.lil_matrix`` of size
    ``L = 4 (m-1)`` containing the four surface rows and every interior
    diffusion + linear-isomerisation row.  The four potential-dependent surface
    entries carry placeholder 1.0 so the sparsity pattern already owns those
    slots; the time loop overwrites them.  The nonlinear cross-reaction entries
    are *not* placed here -- they are added per iteration in
    :func:`simulate_square_cv`.
    """
    m = p.m_space
    L = 4 * (m - 1)
    a, D_M = p.a, p.D_M
    A = sp.lil_matrix((L, L))
    ix = _sq_index

    for j in range(2, m):
        cm = -D_M * a ** (4 - 2 * j)
        cd = 1.0 + (1.0 + a) * D_M * a ** (3 - 2 * j)
        cp = -D_M * a ** (3 - 2 * j)
        last = j + 1 <= m - 1
        # O: diffusion + isomerisation O<->A (loses k+1, gains k-1 from A)
        r = ix(j, _O)
        A[r, ix(j - 1, _O)] = cm
        A[r, ix(j, _O)] = cd + p.k_plus1
        if last:
            A[r, ix(j + 1, _O)] = cp
        A[r, ix(j, _A)] = -p.k_minus1
        # R: diffusion + isomerisation R<->B
        r = ix(j, _R)
        A[r, ix(j - 1, _R)] = cm
        A[r, ix(j, _R)] = cd + p.k_plus2
        if last:
            A[r, ix(j + 1, _R)] = cp
        A[r, ix(j, _B)] = -p.k_minus2
        # A: diffusion + isomerisation A<->O
        r = ix(j, _A)
        A[r, ix(j - 1, _A)] = cm
        A[r, ix(j, _A)] = cd + p.k_minus1
        if last:
            A[r, ix(j + 1, _A)] = cp
        A[r, ix(j, _O)] = -p.k_plus1
        # B: diffusion + isomerisation B<->R
        r = ix(j, _B)
        A[r, ix(j - 1, _B)] = cm
        A[r, ix(j, _B)] = cd + p.k_minus2
        if last:
            A[r, ix(j + 1, _B)] = cp
        A[r, ix(j, _R)] = -p.k_plus2

    # --- Surface rows (0..3) -------------------------------------------------
    # Row 0  (couple O/R Butler-Volmer):  theta cO1 + theta cR1 - (1+a)^2 cO2 + cO3
    A[0, ix(1, _O)] = 1.0     # placeholder (potential dependent)
    A[0, ix(1, _R)] = 1.0     # placeholder
    A[0, ix(2, _O)] = -(1.0 + a) ** 2
    A[0, ix(3, _O)] = 1.0
    # Row 1  (couple O/R flux conservation): a(2+a)(cO1+cR1) - (1+a)^2(cO2+cR2) + cO3+cR3
    A[1, ix(1, _O)] = a * (2.0 + a)
    A[1, ix(1, _R)] = a * (2.0 + a)
    A[1, ix(2, _O)] = -(1.0 + a) ** 2
    A[1, ix(2, _R)] = -(1.0 + a) ** 2
    A[1, ix(3, _O)] = 1.0
    A[1, ix(3, _R)] = 1.0
    # Row 2  (couple A/B Butler-Volmer):  theta cA1 + theta cB1 - (1+a)^2 cA2 + cA3
    A[2, ix(1, _A)] = 1.0     # placeholder
    A[2, ix(1, _B)] = 1.0     # placeholder
    A[2, ix(2, _A)] = -(1.0 + a) ** 2
    A[2, ix(3, _A)] = 1.0
    # Row 3  (couple A/B flux conservation)
    A[3, ix(1, _A)] = a * (2.0 + a)
    A[3, ix(1, _B)] = a * (2.0 + a)
    A[3, ix(2, _A)] = -(1.0 + a) ** 2
    A[3, ix(2, _B)] = -(1.0 + a) ** 2
    A[3, ix(3, _A)] = 1.0
    A[3, ix(3, _B)] = 1.0
    return A, L


def _cross_reaction_entries(p: SquareParams, old: np.ndarray, L: int):
    """Sparse coordinate lists for the nonlinear cross-reaction block.

    The cross reaction ``O + B <=>(k_cf, k_cb) R + A`` linearised about the
    iterate ``old`` contributes, for every interior node block starting at row
    ``r`` (``r = 4, 8, ... L-4``; 0-based), eight entries transcribed from the
    ``rules`` table of the source notebook (see inline comments for the exact
    eight).  Returns ``(rows, cols, vals)`` ready for ``scipy.sparse.coo_matrix``.
    """
    kcf, kcb = p.k_cf, p.k_cb
    rows, cols, vals = [], [], []
    # Source `rules` (1-based): old[[j+1..j+4]] = cO,cR,cA,cB at the node, and
    # the row/col indices {j+1..j+4} -> 0-based O=r, R=r+1, A=r+2, B=r+3.
    #   (O,O)+=kcf*cB   (O,A)+=-kcb*cR
    #   (R,R)+=kcb*cA   (R,B)+=kcf*cO
    #   (A,A)+=kcb*cR   (A,O)+=-kcf*cB
    #   (B,B)+=kcf*cO   (B,R)+=-kcb*cA
    for r in range(4, L - 3, 4):
        cO, cR, cA, cB = old[r], old[r + 1], old[r + 2], old[r + 3]
        rows += [r, r];         cols += [r, r + 2];     vals += [kcf * cB, -kcb * cR]
        rows += [r + 1, r + 1]; cols += [r + 1, r + 3]; vals += [kcb * cA, kcf * cO]
        rows += [r + 2, r + 2]; cols += [r + 2, r];     vals += [kcb * cR, -kcf * cB]
        rows += [r + 3, r + 3]; cols += [r + 3, r + 1]; vals += [kcf * cO, -kcb * cA]
    return rows, cols, vals


def simulate_square_cv(p: SquareParams, backend: str = "sparse",
                       tol: float = 1e-7, max_iter: int = 50):
    """Square-scheme cyclic voltammogram on an expanding grid.

    Picard-iterates the nonlinear cross-reaction block at each time step until
    the mean absolute concentration change drops below ``tol``.  Port of
    ``solveSparseSquare`` from ``sparseSquareRxnExp.nb``.

    Parameters
    ----------
    p : SquareParams
    backend : {"sparse", "dense"}
        ``"sparse"`` assembles a COO cross-reaction patch on top of the constant
        sparse matrix and solves with :func:`scipy.sparse.linalg.spsolve`;
        ``"dense"`` works on a full array and uses :func:`scipy.linalg.solve`.
    tol : float
        Picard convergence tolerance on ``mean|new - old|``.
    max_iter : int
        Iteration cap per time step.

    Returns
    -------
    profiles : ndarray, shape (n, L)
        Interleaved ``[cO, cR, cA, cB]`` for nodes ``j = 1 .. m-1`` at each time
        increment (row 0 = initial condition).
    iters : ndarray, shape (n-1,)
        Picard iteration count actually used at each advanced step.
    """
    A_const, L = build_square_matrix(p)
    m = p.m_space
    a, D_M, alpha = p.a, p.D_M, p.alpha
    tail = D_M * a ** (5 - 2 * m)
    ix = _sq_index

    if backend == "dense":
        base = A_const.toarray()
    elif backend == "sparse":
        base = A_const.tocsr()
    else:
        raise ValueError("backend must be 'sparse' or 'dense'")

    # initial condition: bulk equilibrium O/A split, R = B = 0
    cur = np.tile([p.cOi, 0.0, p.cAi, 0.0], m - 1)
    profiles = [cur.copy()]
    iters = []

    for k in range(2, p.n_time + 1):
        # two surface potentials, offset by the two formal potentials
        if k > (p.n_time + 1) / 2:
            eta1 = p.upper_limit - p.dE_OR - p.sweep_span + p.tau * (k - 1)
            eta2 = p.upper_limit - p.dE_AB - p.sweep_span + p.tau * (k - 1)
        else:
            eta1 = p.upper_limit - p.dE_OR - p.tau * (k - 1)
            eta2 = p.upper_limit - p.dE_AB - p.tau * (k - 1)
        xi1, xi2 = np.exp(eta1), np.exp(eta2)

        # right-hand side: previous step, zero the four surface rows, bulk feed
        b = cur.copy()
        b[0:4] = 0.0
        b[L - 4] += p.cOi * tail     # O bulk feed at node m-1
        b[L - 2] += p.cAi * tail     # A bulk feed at node m-1

        # potential-dependent surface entries (couple 1 in rows 0, couple 2 row 2)
        s1d = a * (2.0 + a) + p.ks_star1 * xi1 ** (-alpha)
        s1o = -p.ks_star1 * xi1 ** (1.0 - alpha)
        s2d = a * (2.0 + a) + p.ks_star2 * xi2 ** (-alpha)
        s2o = -p.ks_star2 * xi2 ** (1.0 - alpha)

        old = cur.copy()
        for it in range(1, max_iter + 1):
            rows, cols, vals = _cross_reaction_entries(p, old, L)
            if backend == "dense":
                work = base.copy()
                work[0, ix(1, _O)] = s1d
                work[0, ix(1, _R)] = s1o
                work[2, ix(1, _A)] = s2d
                work[2, ix(1, _B)] = s2o
                work[rows, cols] += vals
                new = dense_solve(work, b)
            else:
                patch = sp.coo_matrix((vals, (rows, cols)), shape=(L, L))
                work = (base + patch).tolil()
                work[0, ix(1, _O)] = s1d
                work[0, ix(1, _R)] = s1o
                work[2, ix(1, _A)] = s2d
                work[2, ix(1, _B)] = s2o
                new = spsolve(work.tocsc(), b)
            resid = np.abs(new - old).sum() / L
            old = new
            if resid < tol:
                break
        cur = old
        profiles.append(cur.copy())
        iters.append(it)

    return np.array(profiles), np.array(iters)


def square_cv_current(profiles: np.ndarray, p: SquareParams) -> np.ndarray:
    """Total dimensionless current from both couples' surface gradients.

    Sums the O-couple current (from cO at the first three nodes) and the A-couple
    current (from cA), matching ``cv1 + cv2`` in ``sparseSquareRxnExp.nb``.
    """
    scale = np.sqrt(
        p.D_M * (p.n_time - 1) / (2.0 * p.a ** 2 * (1.0 + p.a) * p.sweep_span)
    )
    a = p.a
    # node O concentrations at j = 1, 2, 3  -> columns 0, 4, 8
    cv1 = ((2.0 + a) * a * profiles[:, 0]
           - (1.0 + a) ** 2 * profiles[:, 4] + profiles[:, 8]) * scale
    # node A concentrations at j = 1, 2, 3  -> columns 2, 6, 10
    cv2 = ((2.0 + a) * a * profiles[:, 2]
           - (1.0 + a) ** 2 * profiles[:, 6] + profiles[:, 10]) * scale
    return cv1 + cv2
