"""Chapter 13 square-scheme simulators (uniform grid, three solver strategies).

Honeychurch's Chapter 13 closes with a *square scheme*: two one-electron couples
that are linked by chemical interconversion of both the oxidised and the reduced
partner, plus a homogeneous *cross* reaction.  Writing ``O, R`` for the first
couple and ``A, B`` for the second,

    O + e-  <=>  R          (formal potential E0_OR,  surface ratio xi1)
    A + e-  <=>  B          (formal potential E0_AB,  surface ratio xi2)

    O  <=>(k+1, k-1)  A     (oxidised forms interconvert)
    R  <=>(k+2, k-2)  B     (reduced forms interconvert)

    O + B  <=>(kcf, kcb)  R + A     (the diagonal *cross* reaction)

Four solution species (O, R, A, B) couple at every spatial node, so the implicit
diffusion operator becomes block-tridiagonal with ``4 x 4`` blocks (Rudolph's
trick again -- cf. :mod:`serm.ch13_coupled_chemical_reactions`).  The first four
reactions are first order and enter linearly through a constant kinetics block.
The cross reaction is *bimolecular*, hence the discrete equations are non-linear
in the unknown ``(k+1)``-level concentrations.

Three strategies handle that non-linearity, in increasing fidelity:

* :func:`simulate_square_cross_ignored` -- drop the cross reaction
  (``kcf = kcb = 0``).  The system is linear; one banded solve per step.  This is
  the reference the other two must reduce to when the cross terms vanish, and it
  is what the chapter shows first.
* :func:`simulate_square_linearized` -- keep the cross reaction but *linearise*
  it about the previous time level, ``c^{k+1} = c^{k} + dc`` with the
  second-order ``dc_O dc_B`` term dropped (Honeychurch's linear approximation,
  ``SquareRxnExp2.nb``).  Still one banded solve per step, but the cross terms
  are carried explicitly in both the matrix and the right-hand side.
* :func:`simulate_square_newton` -- solve the full non-linear step by
  Newton--Raphson, assembling the analytic ``4 x 4`` Jacobian blocks of the
  residual and iterating the banded solve to a concentration tolerance
  (``SquareRxnExp3.nb``).

Discretisation and non-dimensionalisation match
:mod:`serm.ch13_coupled_chemical_reactions`: a *uniform* grid, distance scaled by
a diffusion length, time/potential scaled by ``sigma`` (potential axis in
``RT/nF`` about each formal potential), and ``D_M = D dt/dx^2``.  Dimensionless
homogeneous rate constants are the bare-rate times the dimensionless step,
``kh = k * tau`` (the ``khf = tau*k`` convention of the source notebooks).

The electrode boundary is taken Nernstian for *both* couples, reusing the
three-point net-flux elimination of
:mod:`serm.ch13_coupled_chemical_reactions`.  With the cross reaction switched off
and the second couple made inert (``A`` absent, no interconversion) the simulator
reduces to the single-couple reversible CV of Chapter 5 -- the no-reaction
validation used in the accompanying notebook.

Species index convention throughout: ``0 = O, 1 = R, 2 = A, 3 = B``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .ch05_potential_sweep_reversible import surface_ratio, space_points
from .kinetics import triangular_sweep_potential

# Species indices.
O, R, A, B = 0, 1, 2, 3
NSP = 4


@dataclass
class SquareCVResult:
    """Result of a square-scheme CV simulation.

    Attributes
    ----------
    c : numpy.ndarray, shape (n, m, 4)
        Dimensionless concentrations ``c[k, j, i]`` for species ``i`` in
        ``(O, R, A, B)`` order, spatial node ``j`` (0 = surface), step ``k``.
    current : numpy.ndarray, shape (n,)
        Total dimensionless current ``sqrt(pi) chi`` (the sum of the two
        couples' surface fluxes of the oxidised partner, ``O`` and ``A``).
    potential : numpy.ndarray, shape (n,)
        Dimensionless potential ``nF(E - E0_OR)/RT`` at each step.
    n, m : int
        Number of potential steps and spatial nodes.
    D_M, tau, T : float
        Model diffusion coefficient, dimensionless step, total sweep length.
    iterations : numpy.ndarray or None
        Newton iteration count per step (only for the Newton solver).
    """

    c: np.ndarray
    current: np.ndarray
    potential: np.ndarray
    n: int
    m: int
    D_M: float
    tau: float
    T: float
    iterations: np.ndarray | None = None


@dataclass
class SquareParams:
    """Dimensionless parameters defining a square-scheme experiment.

    The two formal-potential offsets ``dE_OR`` and ``dE_AB`` are measured in the
    same ``RT/nF`` units as the sweep; the surface ratios are
    ``xi1 = exp[(E - E0_OR) nF/RT]`` and ``xi2 = exp[(E - E0_AB) nF/RT]`` with
    ``E0_AB = E0_OR + dE_AB`` (so a non-zero ``dE_AB`` separates the two waves).

    Homogeneous rate constants are *dimensionless per step* already (``kh = k
    tau``); pass them directly.  ``k_plus1/k_minus1`` interconvert ``O<->A``,
    ``k_plus2/k_minus2`` interconvert ``R<->B``, and ``kcf/kcb`` are the forward
    and backward cross-reaction constants ``O + B <-> R + A``.
    """

    upper_limit: float = 12.0
    lower_limit: float = 12.0
    dE_AB: float = 0.0          # E0_AB - E0_OR, in RT/nF units
    n: int = 401
    D_M: float = 0.9
    k_plus1: float = 0.0
    k_minus1: float = 0.0
    k_plus2: float = 0.0
    k_minus2: float = 0.0
    kcf: float = 0.0
    kcb: float = 0.0
    # Initial bulk composition (fraction of total in each species).  Default:
    # all O (only the first couple's oxidised form present).
    bulk: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


def _const_kinetics_block(p: SquareParams) -> np.ndarray:
    """Constant (first-order) homogeneous-kinetics block added to each diagonal.

    Encodes ``O<->A`` and ``R<->B`` interconversion only (the cross reaction is
    handled separately by each solver).  Row ``i`` is the net loss of species
    ``i``; consumption sits on the diagonal, production off-diagonal.
    """
    kp1, km1, kp2, km2 = p.k_plus1, p.k_minus1, p.k_plus2, p.k_minus2
    K = np.zeros((NSP, NSP))
    # O -> A forward (kp1), A -> O backward (km1)
    K[O, O] += kp1
    K[O, A] += -km1
    K[A, A] += km1
    K[A, O] += -kp1
    # R -> B forward (kp2), B -> R backward (km2)
    K[R, R] += kp2
    K[R, B] += -km2
    K[B, B] += km2
    K[B, R] += -kp2
    return K


def _potential_factors(p: SquareParams):
    """Per-step surface ratios ``(xi1, xi2)`` for the two couples."""
    k_all = np.arange(1, p.n + 1)
    tau = p.T / (p.n - 1)
    frac1 = surface_ratio(k_all, tau, p.T, p.upper_limit)
    xi1 = frac1 / (1.0 - frac1)
    # second couple: same sweep, shifted formal potential by dE_AB
    frac2 = surface_ratio(k_all, tau, p.T, p.upper_limit - p.dE_AB)
    xi2 = frac2 / (1.0 - frac2)
    return xi1, xi2


def _assemble_banded(D_M, Ydiag, c_old_rhs, xi1, xi2, bulk):
    """Build and solve the banded 4-species block-tridiagonal step.

    Parameters
    ----------
    D_M : float
        Model diffusion coefficient.
    Ydiag : ndarray, shape (m, 4, 4)
        The main-diagonal ``4x4`` block for every interior node (already
        including ``(1+2 D_M) I`` and all kinetics for that node).  Surface and
        bulk rows are overwritten here, so ``Ydiag[0]`` / ``Ydiag[m-1]`` are
        ignored.
    c_old_rhs : ndarray, shape (m, 4)
        Right-hand side for interior nodes (old concentrations, possibly with
        cross-reaction source terms folded in by the caller).
    xi1, xi2 : float
        Surface ratios for couples (O,R) and (A,B) at this step.
    bulk : ndarray, shape (4,)
        Bulk concentrations held at node ``m-1``.

    Returns
    -------
    ndarray, shape (m, 4)
        New concentrations.
    """
    m = c_old_rhs.shape[0]
    s = NSP
    N = m * s
    off = -D_M * np.eye(s)
    bw = 2 * s - 1
    ab = np.zeros((2 * bw + 1, N))
    u = bw
    b = np.zeros(N)

    def put(i, j, val):
        ab[u + i - j, j] = val

    # --- Surface node 0: Nernst ratios for both couples + two net-flux rows. ---
    # O: c_O(0) - xi1 c_R(0) = 0
    put(O, O, 1.0)
    put(O, R, -xi1)
    # A: c_A(0) - xi2 c_B(0) = 0
    put(A, A, 1.0)
    put(A, B, -xi2)
    # R row: total reductive flux balance for couple 1:
    #   (c_O1 - c_O0) + (c_R1 - c_R0) = 0
    put(R, O, -1.0)
    put(R, R, -1.0)
    put(R, s + O, 1.0)
    put(R, s + R, 1.0)
    # B row: total reductive flux balance for couple 2:
    #   (c_A1 - c_A0) + (c_B1 - c_B0) = 0
    put(B, A, -1.0)
    put(B, B, -1.0)
    put(B, s + A, 1.0)
    put(B, s + B, 1.0)
    # b[0:4] already zero.

    # --- Interior nodes. ---
    for jnode in range(1, m - 1):
        r = jnode * s
        Yj = Ydiag[jnode]
        for ia in range(s):
            for ib in range(s):
                if Yj[ia, ib]:
                    put(r + ia, r + ib, Yj[ia, ib])
                if off[ia, ib]:
                    put(r + ia, r - s + ib, off[ia, ib])
                    put(r + ia, r + s + ib, off[ia, ib])
        b[r:r + s] = c_old_rhs[jnode]

    # --- Bulk node m-1: Dirichlet. ---
    r = (m - 1) * s
    for ia in range(s):
        put(r + ia, r + ia, 1.0)
        b[r + ia] = bulk[ia]

    return solve_banded((bw, bw), ab, b).reshape(m, s)


def _make_result(p, c, xi_unused=None, iterations=None):
    """Build a :class:`SquareCVResult` (current = summed surface flux of O and A)."""
    n, m = p.n, c.shape[1]
    tau = p.T / (n - 1)
    scale = np.sqrt(p.D_M * (n - 1)) / np.sqrt(4.0 * p.T)
    cO, cA = c[:, :, O], c[:, :, A]
    gO = 3.0 * cO[:, 0] - 4.0 * cO[:, 1] + cO[:, 2]
    gA = 3.0 * cA[:, 0] - 4.0 * cA[:, 1] + cA[:, 2]
    current = (gO + gA) * scale
    potential = triangular_sweep_potential(n, tau, p.T, p.upper_limit)
    return SquareCVResult(c=c, current=current, potential=potential,
                          n=n, m=m, D_M=p.D_M, tau=tau, T=p.T,
                          iterations=iterations)


def _init_grid(p: SquareParams):
    """Set up ``T``, ``m`` and the initial concentration array."""
    p.T = 2.0 * (p.upper_limit + abs(p.lower_limit))
    if p.n % 2 == 0:
        p.n += 1
    m = space_points(p.D_M, p.n)
    c = np.zeros((p.n, m, NSP))
    c[0, :] = p.bulk
    # Consistent surface initial condition: at the very first step the Nernstian
    # surface ratios of both couples are imposed, so the surface node of column 0
    # carries the equilibrated split rather than the raw bulk value (matches the
    # IC of serm.ch13_coupled_chemical_reactions).
    xi1_all, xi2_all = _potential_factors(p)
    xi1_0, xi2_0 = xi1_all[0], xi2_all[0]
    tot1 = p.bulk[O] + p.bulk[R]
    tot2 = p.bulk[A] + p.bulk[B]
    c[0, 0, O] = tot1 * xi1_0 / (1.0 + xi1_0)
    c[0, 0, R] = tot1 / (1.0 + xi1_0)
    c[0, 0, A] = tot2 * xi2_0 / (1.0 + xi2_0)
    c[0, 0, B] = tot2 / (1.0 + xi2_0)
    return m, c


def simulate_square_cross_ignored(p: SquareParams) -> SquareCVResult:
    """Square scheme with the cross reaction ignored (``kcf = kcb = 0``).

    Linear in the unknowns: the constant kinetics block (interconversion of the
    two couples) is added once to every interior diagonal and the step is a
    single banded solve.  Validation tier: reduction to a validated limit -- with
    all chemistry off and only couple 1 present this is the Chapter 5 reversible
    CV (see notebook).

    Returns
    -------
    SquareCVResult
    """
    m, c = _init_grid(p)
    xi1, xi2 = _potential_factors(p)
    K = _const_kinetics_block(p)
    A_block = (1.0 + 2.0 * p.D_M) * np.eye(NSP) + K
    Ydiag = np.broadcast_to(A_block, (m, NSP, NSP)).copy()
    for k in range(1, p.n):
        c[k] = _assemble_banded(p.D_M, Ydiag, c[k - 1], xi1[k], xi2[k], p.bulk)
    return _make_result(p, c)


def _cross_var_block(cnode: np.ndarray, kcf: float, kcb: float) -> np.ndarray:
    """Linearised cross-reaction contribution to the diagonal block (one node).

    From Honeychurch's linear approximation (``SquareRxnExp2.nb`` ``yVar``): with
    ``r = kcf c_O c_B - kcb c_R c_A`` linearised about the previous level, the
    net-loss rows pick up the partial-rate terms evaluated at the old
    concentrations ``cnode = (c_O, c_R, c_A, c_B)``.
    """
    cO, cR, cA, cB = cnode
    # Row signs: O and B lose at rate +r; R and A gain (lose -r).
    row_OB = np.array([kcf * cB, -kcb * cA, -kcb * cR, kcf * cO])
    row_RA = -row_OB
    Yv = np.empty((NSP, NSP))
    Yv[O] = row_OB
    Yv[R] = row_RA
    Yv[A] = row_RA
    Yv[B] = row_OB
    return Yv


def _cross_rhs(cnode: np.ndarray, kcf: float, kcb: float) -> np.ndarray:
    """RHS bilinear correction for the linearised cross reaction (one node).

    The ``vect`` pure-function of ``SquareRxnExp2.nb``: moves the explicit
    bilinear rate ``r = kcf c_O c_B - kcb c_R c_A`` (evaluated at the old level)
    to the right-hand side so that, combined with the linearised matrix block, the
    scheme is consistent to first order in ``dc``.
    """
    cO, cR, cA, cB = cnode
    r = kcf * cO * cB - kcb * cR * cA
    return np.array([cO + r, cR - r, cA - r, cB + r])


def simulate_square_linearized(p: SquareParams) -> SquareCVResult:
    """Square scheme with the cross reaction kept but linearised about ``c^k``.

    One banded solve per step: the diagonal block carries the constant
    interconversion kinetics plus the linearised cross-reaction Jacobian
    evaluated at the previous level, and the right-hand side carries the explicit
    bilinear rate (Honeychurch's ``SquareRxnExp2.nb``).  Validation tier:
    self-consistency / agreement with the Newton solver as ``tau -> 0`` (notebook)
    and reduction to :func:`simulate_square_cross_ignored` when ``kcf = kcb = 0``.

    Returns
    -------
    SquareCVResult
    """
    m, c = _init_grid(p)
    xi1, xi2 = _potential_factors(p)
    K = _const_kinetics_block(p)
    A_const = (1.0 + 2.0 * p.D_M) * np.eye(NSP) + K
    kcf, kcb = p.kcf, p.kcb
    for k in range(1, p.n):
        cprev = c[k - 1]
        Ydiag = np.empty((m, NSP, NSP))
        rhs = np.empty((m, NSP))
        for j in range(m):
            Ydiag[j] = A_const + _cross_var_block(cprev[j], kcf, kcb)
            rhs[j] = _cross_rhs(cprev[j], kcf, kcb)
        c[k] = _assemble_banded(p.D_M, Ydiag, rhs, xi1[k], xi2[k], p.bulk)
    return _make_result(p, c)


def _residual_and_jacobian(cnew, cold, D_M, K, kcf, kcb):
    """Interior residual ``F`` and its diagonal Jacobian block for one node.

    For the implicit reaction-diffusion balance at an interior node, the residual
    (excluding the diffusion coupling to neighbours, which is linear and lives on
    the off-diagonal blocks) is

        F_diag(c) = (1+2 D_M) c + K c + g(c) - c_old,

    with the bilinear cross term ``g`` whose entries are
    ``+r`` for O,B and ``-r`` for R,A, ``r = kcf c_O c_B - kcb c_R c_A``.  The
    diagonal Jacobian block is ``(1+2 D_M) I + K + dg/dc``.  Returns
    ``(Fdiag, Jdiag)`` (the off-diagonal ``-D_M I`` blocks are constant and added
    by the assembler).
    """
    cO, cR, cA, cB = cnew
    r = kcf * cO * cB - kcb * cR * cA
    g = np.array([r, -r, -r, r])
    base = (1.0 + 2.0 * D_M) * np.eye(NSP) + K
    Fdiag = base @ cnew + g
    # dg/dc:  dr/dc = (kcf cB, -kcb cA, -kcb cR, kcf cO)
    dr = np.array([kcf * cB, -kcb * cA, -kcb * cR, kcf * cO])
    dg = np.empty((NSP, NSP))
    dg[O] = dr
    dg[R] = -dr
    dg[A] = -dr
    dg[B] = dr
    Jdiag = base + dg
    return Fdiag, Jdiag


def simulate_square_newton(p: SquareParams, tol: float = 1e-10,
                           max_iter: int = 30) -> SquareCVResult:
    """Square scheme solved exactly per step by Newton--Raphson (full Jacobian).

    Each time step solves the non-linear block-tridiagonal system
    ``F(c^{k+1}) = 0`` by Newton iteration: assemble the analytic ``4 x 4``
    Jacobian diagonal blocks (:func:`_residual_and_jacobian`) plus the constant
    ``-D_M I`` diffusion off-diagonals, solve the banded linear system for the
    update ``dc``, and repeat until ``max|dc|`` falls below ``tol``.  Port of the
    ``While`` loop in ``SquareRxnExp3.nb``, but using a Newton *correction* form
    (solve for the increment) rather than re-solving for the absolute level.

    Validation tier: agreement with :func:`simulate_square_linearized` as
    ``tau -> 0`` (both converge to the same continuum limit); per-step mass
    balance (total of all four species conserved); reduction to
    :func:`simulate_square_cross_ignored` when ``kcf = kcb = 0`` (Newton
    converges in one step).

    Parameters
    ----------
    p : SquareParams
    tol : float
        Convergence tolerance on ``max|dc|`` per Newton step.
    max_iter : int
        Safety cap on Newton iterations per time step.

    Returns
    -------
    SquareCVResult
        ``.iterations`` holds the Newton iteration count for each time step.
    """
    m, c = _init_grid(p)
    xi1, xi2 = _potential_factors(p)
    K = _const_kinetics_block(p)
    kcf, kcb = p.kcf, p.kcb
    D_M = p.D_M
    iters = np.zeros(p.n, dtype=int)

    for k in range(1, p.n):
        cold = c[k - 1]
        guess = cold.copy()                      # initial guess: previous level
        for it in range(max_iter):
            # Build the banded Newton system  J . dc = -F.
            Ydiag = np.empty((m, NSP, NSP))
            Fint = np.empty((m, NSP))
            for j in range(1, m - 1):
                Fdiag, Jdiag = _residual_and_jacobian(
                    guess[j], cold[j], D_M, K, kcf, kcb)
                Ydiag[j] = Jdiag
                # full interior residual incl. diffusion coupling to neighbours
                Fint[j] = (Fdiag
                           - D_M * guess[j - 1]
                           - D_M * guess[j + 1]
                           - cold[j])
            # Solve J dc = -F with the SAME banded assembler: feed -F as the
            # "rhs" and a zero-Nernst surface so dc satisfies the linear BCs.
            dc = _solve_newton_update(D_M, Ydiag, Fint, guess,
                                      xi1[k], xi2[k], p.bulk, m)
            guess = guess + dc
            if np.max(np.abs(dc)) < tol:
                break
        iters[k] = it + 1
        c[k] = guess
    return _make_result(p, c, iterations=iters)


def _solve_newton_update(D_M, Ydiag, Fint, guess, xi1, xi2, bulk, m):
    """Solve the banded Newton system ``J dc = -F`` for the increment ``dc``.

    The boundary rows are the *linearised* surface/bulk conditions, whose
    residual at the current guess must also be driven to zero, so ``dc`` carries
    the boundary residuals on its right-hand side.
    """
    s = NSP
    N = m * s
    off = -D_M * np.eye(s)
    bw = 2 * s - 1
    ab = np.zeros((2 * bw + 1, N))
    u = bw
    rhs = np.zeros(N)

    def put(i, j, val):
        ab[u + i - j, j] = val

    # Surface rows (same structure as _assemble_banded), residual on RHS.
    put(O, O, 1.0);  put(O, R, -xi1)
    put(A, A, 1.0);  put(A, B, -xi2)
    put(R, O, -1.0); put(R, R, -1.0); put(R, s + O, 1.0); put(R, s + R, 1.0)
    put(B, A, -1.0); put(B, B, -1.0); put(B, s + A, 1.0); put(B, s + B, 1.0)
    g0 = guess[0]
    g1 = guess[1]
    # residual of each boundary eq at current guess; dc must cancel it
    rhs[O] = -(g0[O] - xi1 * g0[R])
    rhs[A] = -(g0[A] - xi2 * g0[B])
    rhs[R] = -((g1[O] - g0[O]) + (g1[R] - g0[R]))
    rhs[B] = -((g1[A] - g0[A]) + (g1[B] - g0[B]))

    for jnode in range(1, m - 1):
        r = jnode * s
        Yj = Ydiag[jnode]
        for ia in range(s):
            for ib in range(s):
                if Yj[ia, ib]:
                    put(r + ia, r + ib, Yj[ia, ib])
                if off[ia, ib]:
                    put(r + ia, r - s + ib, off[ia, ib])
                    put(r + ia, r + s + ib, off[ia, ib])
        rhs[r:r + s] = -Fint[jnode]

    r = (m - 1) * s
    for ia in range(s):
        put(r + ia, r + ia, 1.0)
        rhs[r + ia] = -(guess[m - 1, ia] - bulk[ia])

    return solve_banded((bw, bw), ab, rhs).reshape(m, s)


def total_mass(c: np.ndarray) -> np.ndarray:
    """Spatially-averaged total concentration of all four species per step.

    ``mean_j sum_i c[k, j, i]`` -- conserved (equal to the bulk total) when the
    chemistry only interconverts species, a per-step mass-balance diagnostic.
    """
    return c.sum(axis=2).mean(axis=1)
