"""Chapter 13 helpers: electrode reactions coupled to homogeneous chemistry.

This module re-implements, in vectorised numpy / scipy, the block-implicit
finite-difference machinery that Honeychurch develops in *Simulating
Electrochemical Reactions in Mathematica* (SERM), Chapter 13
(``Chapters/chapter13.nb`` and the standalone notebook
``Extra Notebooks/chapter13/coupledECRxn.nb``).  The chapter shows how to extend
the implicit diffusion solver of Chapter 6 to systems in which the
electrogenerated species reacts in solution.

Two mechanisms are provided:

* **EC** -- ``O + e- <=> R`` followed by a reversible first-order chemical step
  ``R <=>(k_f, k_b) P``.  Three solution species (O, R, P) couple at every node.
* **EC'** (catalytic) -- ``O + e- <=> R`` with a pseudo-first-order regeneration
  ``R + Z ->(k) O`` (Z in large excess).  Two species (O, R) couple; the
  chemical step turns the wave sigmoidal and, in the pure-kinetic limit, gives a
  scan-rate-independent plateau.

Method.  Following Rudolph's trick (used throughout the chapter), the implicit
diffusion operator is kept *tridiagonal* even though each node now carries a
small vector of concentrations: the scalar tridiagonal coefficients
``(-D_M, 1+2 D_M, -D_M)`` are promoted to ``s x s`` blocks (``s`` = number of
species), and the homogeneous kinetics enter only through a constant block ``K``
added to the main-diagonal block.  Writing the unknowns interleaved
``[c_O(1), c_R(1), ..., c_O(j), c_R(j), ...]`` turns the block-tridiagonal
system into an ordinary *banded* system that
:func:`scipy.linalg.solve_banded` handles directly -- so we never form a dense
matrix and the cost stays ``O(m s^2)`` per step.

Non-dimensionalisation matches Chapter 5 / :mod:`serm.ch05_potential_sweep_reversible`:
distance is scaled by a diffusion length, time/potential by
``sigma = n F v /(R T)`` (so the potential axis is in units of ``RT/nF`` about
``E0``), concentration by the bulk value of O.  The dimensionless homogeneous
rate constant is ``k_h = k * tau`` (rate constant times the dimensionless time
step), exactly the ``khf = tau * k`` definition in ``coupledECRxn.nb``.

The electrode boundary is taken Nernstian (reversible electron transfer), reusing
:func:`serm.ch05_potential_sweep_reversible.surface_ratio`.  This keeps the
surface condition identical to Chapter 5, so that switching the chemistry off
(``k = 0``) recovers the Chapter 5 reversible voltammogram exactly -- the basis
of the no-reaction validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded

from .ch05_potential_sweep_reversible import surface_ratio, space_points
from .kinetics import triangular_sweep_potential, ks_star_sweep
from .boundary import bv_dirichlet_surface


@dataclass
class CoupledCVResult:
    """Result of a coupled-reaction CV simulation.

    Attributes
    ----------
    c : numpy.ndarray, shape (n, m, s)
        Dimensionless concentrations: ``c[k, j, i]`` is species ``i`` at spatial
        node ``j`` and potential step ``k``.  Species order is ``O, R`` (and
        ``P`` for EC).  Node 0 is the electrode surface, node ``m-1`` the bulk.
    current : numpy.ndarray, shape (n,)
        Dimensionless current ``sqrt(pi) * chi`` at each step (cathodic
        positive), from the three-point surface gradient of O.
    potential : numpy.ndarray, shape (n,)
        Dimensionless potential ``nF(E - E0)/RT`` at each step.
    n, m, s : int
        Number of potential steps, spatial nodes, and species.
    D_M : float
        Model diffusion coefficient ``D dt / dx**2``.
    tau : float
        Dimensionless time/potential step.
    T : float
        Total dimensionless sweep length.
    """

    c: np.ndarray
    current: np.ndarray
    potential: np.ndarray
    n: int
    m: int
    s: int
    D_M: float
    tau: float
    T: float


def _potential_axis(n: int, tau: float, T: float, upper_limit: float) -> np.ndarray:
    """Dimensionless potential at each (1-based) step, like Chapter 5's ``cv2``."""
    return triangular_sweep_potential(n, tau, T, upper_limit)


def _solve_block_tridiag(D_M, K, c_old, xi, bulk):
    """Advance one implicit step of an ``s``-species coupled diffusion system.

    All ``m`` spatial nodes are unknowns.  Interior nodes ``j = 1 .. m-2``
    satisfy the implicit reaction-diffusion balance

        -D_M c_{j-1} + (A + K) c_j - D_M c_{j+1} = c_j^old ,   A = (1+2 D_M) I,

    where ``K`` is the homogeneous-kinetics block.  The two boundaries are *not*
    simple Dirichlet conditions, because the chemistry changes the surface
    concentrations: instead they are the reversible-electrode conditions

      * node 0 (surface), species O:  ``c_O(0) - xi c_R(0) = 0``  (Nernst ratio
        ``xi = c_O/c_R = exp[nF(E-E0)/RT]``);
      * node 0 (surface), species R:  zero *net* faradaic flux, i.e. the O lost
        equals the R gained, ``(c_O(1)-c_O(0)) + (c_R(1)-c_R(0)) = 0``;
      * node 0, any further species (P): zero flux, ``c_P(0) - c_P(1) = 0``;
      * node ``m-1`` (bulk): Dirichlet ``c = bulk``.

    Writing the unknowns interleaved ``[O,R,(P)]`` per node turns the whole thing
    into a banded system (bandwidth ``2 s - 1``) solved with
    :func:`scipy.linalg.solve_banded` (which pivots).  Cost ``O(m s^2)`` per step.

    Parameters
    ----------
    D_M : float
        Model diffusion coefficient.
    K : ndarray, shape (s, s)
        Constant homogeneous-kinetics block added to each interior main diagonal.
    c_old : ndarray, shape (m, s)
        Concentrations at the previous step (the RHS for interior nodes).
    xi : float
        Nernstian surface ratio ``c_O/c_R`` at the new step.
    bulk : ndarray, shape (s,)
        Bulk concentrations (held fixed at node ``m-1``).

    Returns
    -------
    ndarray, shape (m, s)
        Concentrations at the new step.
    """
    m, s = c_old.shape
    N = m * s
    A = (1.0 + 2.0 * D_M) * np.eye(s) + K
    off = -D_M * np.eye(s)

    bw = 2 * s - 1
    ab = np.zeros((2 * bw + 1, N))
    u = bw
    b = np.zeros(N)

    def put(i, j, val):
        ab[u + i - j, j] = val

    # --- Surface node 0 (rows 0 .. s-1): reversible-electrode conditions. ---
    # O: c_O(0) - xi c_R(0) = 0
    put(0, 0, 1.0)
    put(0, 1, -xi)
    b[0] = 0.0
    # R: zero net flux  (c_O1 - c_O0) + (c_R1 - c_R0) = 0
    put(1, 0, -1.0)            # -c_O(0)
    put(1, 1, -1.0)            # -c_R(0)
    put(1, s + 0, 1.0)         # +c_O(1)
    put(1, s + 1, 1.0)         # +c_R(1)
    b[1] = 0.0
    # further species P: zero flux  c_P(0) - c_P(1) = 0
    for a in range(2, s):
        put(a, a, 1.0)
        put(a, s + a, -1.0)
        b[a] = 0.0

    # --- Interior nodes 1 .. m-2: reaction-diffusion. ---
    for jnode in range(1, m - 1):
        r = jnode * s
        for a in range(s):
            for bb in range(s):
                if A[a, bb]:
                    put(r + a, r + bb, A[a, bb])
                put(r + a, r - s + bb, off[a, bb])   # sub-diagonal block
                put(r + a, r + s + bb, off[a, bb])   # super-diagonal block
        b[r:r + s] = c_old[jnode]

    # --- Bulk node m-1: Dirichlet. ---
    r = (m - 1) * s
    for a in range(s):
        put(r + a, r + a, 1.0)
        b[r + a] = bulk[a]

    sol = solve_banded((bw, bw), ab, b).reshape(m, s)
    return sol


def simulate_coupled_cv(mechanism, k_dim, *, n=401, D_M=0.45,
                        upper_limit=8.0, lower_limit=8.0, k_back=0.0):
    """Simulate a reversible CV with a coupled homogeneous reaction.

    Parameters
    ----------
    mechanism : {"E", "EC", "EC'"}
        ``"E"``   -- no chemistry (recovers the Chapter 5 reversible CV);
        ``"EC"``  -- following reversible step ``R <=>(k_dim, k_back) P``;
        ``"EC'"`` -- catalytic regeneration ``R + Z ->(k_dim) O``.
    k_dim : float
        *Dimensional* homogeneous rate constant (1/s).  The dimensionless
        constant used internally is ``k_h = k_dim * (1/sigma) * tau`` where the
        time per step in seconds is ``(1/sigma) * tau`` -- see Notes.
    n : int
        Number of potential steps (forced odd so the vertex lands on a node).
    D_M : float
        Model diffusion coefficient ``D dt / dx**2``.
    upper_limit, lower_limit : float
        Dimensionless potential limits about ``E0`` (units ``RT/nF``).
    k_back : float
        Dimensional backward rate constant (1/s) for the EC step (ignored for
        ``"E"`` and ``"EC'"``).

    Returns
    -------
    CoupledCVResult

    Notes
    -----
    Dimensionless time is ``T_dimless = sigma * t`` so one step ``tau`` (in
    ``RT/nF`` potential units) corresponds to ``tau / sigma`` seconds.  The
    dimensionless homogeneous rate constant is therefore
    ``k_h = k_dim * tau / sigma``.  We expose the *combined* dimensionless group

        lambda = k_dim / sigma           (catalytic kinetic parameter)

    directly through :func:`catalytic_kinetic_parameter`; here ``k_dim`` and the
    sweep enter only as the product ``k_h = lambda * tau``.  Because the whole
    problem is dimensionless, we pass ``k_dim / sigma`` in as ``k_dim`` with
    ``sigma = 1`` -- i.e. *callers supply the already sigma-scaled rate*.  See
    the chapter notebook for the bookkeeping.
    """
    if n % 2 == 0:
        n += 1
    T = 2.0 * (upper_limit + abs(lower_limit))
    tau = T / (n - 1)
    m = space_points(D_M, n)

    s = 2 if mechanism in ("E", "EC'") else 3

    # Dimensionless homogeneous rate constants (k_h = k * tau in dimensionless
    # time, with k already expressed per unit dimensionless time).
    kf = k_dim * tau
    kb = k_back * tau

    if mechanism == "E":
        K = np.zeros((s, s))
    elif mechanism == "EC":
        # species order (O, R, P).  R <-> P first-order exchange.
        K = np.array([[0.0, 0.0, 0.0],
                      [0.0, kf, -kb],
                      [0.0, -kf, kb]])
    elif mechanism == "EC'":
        # species order (O, R).  R -> O regeneration: R consumed, O produced.
        K = np.array([[0.0, -kf],
                      [0.0, kf]])
    else:
        raise ValueError("mechanism must be 'E', 'EC', or 'EC''")

    bulk = np.zeros(s)
    bulk[0] = 1.0                      # only O present in bulk

    # Nernstian surface ratio xi = c_O/c_R = exp[nF(E-E0)/RT] at each step.
    # surface_ratio() returns the O *fraction* xi/(1+xi); invert to get xi.
    k_all = np.arange(1, n + 1)
    frac = surface_ratio(k_all, tau, T, upper_limit)
    xi = frac / (1.0 - frac)

    c = np.zeros((n, m, s))
    c[0, :, 0] = 1.0                   # initial: O everywhere
    c[0, 0, 0] = frac[0]              # consistent surface IC (R = 1 - frac)
    c[0, 0, 1] = 1.0 - frac[0]

    for k in range(1, n):
        c[k] = _solve_block_tridiag(D_M, K, c[k - 1], xi[k], bulk)

    # Dimensionless current: three-point surface gradient of O (Chapter 5 form).
    cO = c[:, :, 0]
    grad = 3.0 * cO[:, 0] - 4.0 * cO[:, 1] + cO[:, 2]
    current = grad * math.sqrt(D_M * (n - 1)) / math.sqrt(4.0 * T)
    potential = _potential_axis(n, tau, T, upper_limit)

    return CoupledCVResult(c=c, current=current, potential=potential,
                           n=n, m=m, s=s, D_M=D_M, tau=tau, T=T)


def catalytic_kinetic_parameter(k_dim, sigma):
    """Catalytic kinetic parameter ``lambda = k / sigma`` (dimensionless).

    With ``sigma = n F v /(R T)`` the group ``lambda = k_cat / sigma = k_cat R T
    /(n F v)`` measures how many chemical relaxation times fit into one
    ``RT/nF`` of potential sweep.  Small ``lambda`` -> diffusion-controlled
    (peaked) wave; large ``lambda`` -> pure-kinetic (sigmoidal, scan-rate
    independent) plateau.
    """
    return k_dim / sigma


def catalytic_plateau_ratio(lam):
    """Closed-form pure-kinetic catalytic plateau, in the ``sqrt(pi) chi`` scale.

    In the pure-kinetic (large-``lambda``) limit the catalytic mechanism reaches
    a steady reaction-layer balance: R is consumed as fast as it is produced
    within a thin layer of thickness ``~ sqrt(D/k)`` at the electrode, and the
    surface flux is the classic reaction-layer result ``j = c* sqrt(D k)``
    (Bard & Faulkner, 2nd ed., Sec. 12.3; Saveant, *Elements of Molecular and
    Biomolecular Electrochemistry*).

    In the dimensionless current ``sqrt(pi) chi`` used throughout Chapters 5 and
    13 -- where distance is scaled by ``sqrt(D/sigma)`` and the flux by
    ``c* sqrt(D sigma)`` -- that steady flux becomes simply

    .. math::
        (\\sqrt{\\pi}\\,\\chi)_{\\text{plateau}} = \\sqrt{\\lambda},
        \\qquad \\lambda = k/\\sigma .

    The constant ``1`` is confirmed independently in the chapter notebook by
    refining the spatial grid: the simulated plateau converges monotonically to
    ``sqrt(lambda)`` as the reaction layer is resolved.  This is the catalytic
    validation reference.

    Parameters
    ----------
    lam : float or array_like
        Kinetic parameter ``lambda = k / sigma``.

    Returns
    -------
    float or numpy.ndarray
        Plateau value of ``sqrt(pi) chi``.
    """
    return np.sqrt(np.asarray(lam, dtype=float))


# ---------------------------------------------------------------------------
# Quasi-reversible EC mechanism (Butler--Volmer surface, uniform grid)
# ---------------------------------------------------------------------------
def _solve_block_tridiag_bv(D_M, K, c_old, xi, bulk, ks_star, alpha):
    """One implicit EC step with a *quasi-reversible* Butler--Volmer surface.

    Identical interior reaction-diffusion blocks to :func:`_solve_block_tridiag`,
    but the surface node O/R rows implement the discrete Butler--Volmer flux
    balance instead of the Nernst ratio.  With the one-sided three-point surface
    gradient (uniform grid) and ``ks_star`` scaled as in
    :func:`serm.kinetics.ks_star_sweep`, the surface equations are (cf. the
    expanding-grid surface rows of :mod:`serm.ch15_sparse_finite_differences`
    in the ``a -> 1`` limit):

      * O:  ``(3 + ks_star xi**-alpha) c_O0 - ks_star xi**(1-alpha) c_R0
             - 4 c_O1 + c_O2 = 0``  (faradaic flux = BV rate);
      * R:  zero *net* flux ``3(c_O0+c_R0) - 4(c_O1+c_R1) + (c_O2+c_R2) = 0``;
      * P (if present): zero flux ``c_P0 - c_P1 = 0``.

    As ``ks_star -> inf`` the O row collapses to the Nernst ratio ``c_O0 = xi c_R0``
    (divide by ``ks_star``: ``xi**-alpha c_O0 - xi**(1-alpha) c_R0 = 0`` i.e.
    ``c_O0 = xi c_R0``), recovering :func:`_solve_block_tridiag`.

    Parameters mirror :func:`_solve_block_tridiag`, with the added dimensionless
    rate constant ``ks_star`` and transfer coefficient ``alpha``.
    """
    m, s = c_old.shape
    N = m * s
    Amat = (1.0 + 2.0 * D_M) * np.eye(s) + K
    off = -D_M * np.eye(s)

    bw = 2 * s - 1
    ab = np.zeros((2 * bw + 1, N))
    u = bw
    b = np.zeros(N)

    def put(i, j, val):
        ab[u + i - j, j] = val

    # --- Surface node 0: Butler-Volmer flux balance. ---
    # The faradaic flux (two-point gradient c_O1 - c_O0) equals the BV rate
    #   (c_O1 - c_O0) = -ks_loc (xi^-alpha c_O0 - xi^(1-alpha) c_R0),
    # i.e.  (1 + ks_loc xi^-alpha) c_O0 - ks_loc xi^(1-alpha) c_R0 - c_O1 = 0.
    # As ks_loc -> inf this collapses (divide by ks_loc) to the Nernst ratio
    # c_O0 = xi c_R0 -- the reversible row used by _solve_block_tridiag.
    put(0, 0, 1.0 + ks_star * xi ** (-alpha))
    put(0, 1, -ks_star * xi ** (1.0 - alpha))
    put(0, s + 0, -1.0)
    b[0] = 0.0
    # R row: zero *net* faradaic flux (same two-point form as the reversible
    # solver): (c_O1 - c_O0) + (c_R1 - c_R0) = 0.
    put(1, 0, -1.0); put(1, 1, -1.0)
    put(1, s + 0, 1.0); put(1, s + 1, 1.0)
    b[1] = 0.0
    # further species P: zero flux  c_P(0) - c_P(1) = 0
    for ia in range(2, s):
        put(ia, ia, 1.0)
        put(ia, s + ia, -1.0)
        b[ia] = 0.0

    # --- Interior nodes 1 .. m-2: reaction-diffusion. ---
    for jnode in range(1, m - 1):
        r = jnode * s
        for ia in range(s):
            for ib in range(s):
                if Amat[ia, ib]:
                    put(r + ia, r + ib, Amat[ia, ib])
                put(r + ia, r - s + ib, off[ia, ib])
                put(r + ia, r + s + ib, off[ia, ib])
        b[r:r + s] = c_old[jnode]

    # --- Bulk node m-1: Dirichlet. ---
    r = (m - 1) * s
    for ia in range(s):
        put(r + ia, r + ia, 1.0)
        b[r + ia] = bulk[ia]

    return solve_banded((bw, bw), ab, b).reshape(m, s)


def simulate_coupled_cv_quasirev(mechanism, k_dim, ks_dim, *, alpha=0.5,
                                 n=401, D_M=0.45, upper_limit=8.0,
                                 lower_limit=8.0, k_back=0.0):
    """Coupled-reaction CV with a *quasi-reversible* Butler--Volmer electrode.

    Same mechanisms and non-dimensionalisation as :func:`simulate_coupled_cv`,
    but the electrode kinetics are finite: the surface obeys the Butler--Volmer
    condition (see :func:`_solve_block_tridiag_bv`) with dimensionless standard
    rate constant ``ks_star = 2 ks_dim sqrt(T /(D_M (n-1)))``
    (:func:`serm.kinetics.ks_star_sweep`) and transfer coefficient ``alpha``.

    As ``ks_dim -> inf`` the result must coincide with the reversible
    :func:`simulate_coupled_cv` (the reduction-to-validated-limit check used in
    the chapter notebook); as ``ks_dim`` falls the wave broadens and the peaks
    pull apart, exactly as for the uncoupled quasi-reversible CV of Chapter 6.

    Parameters
    ----------
    mechanism : {"E", "EC", "EC'"}
        As in :func:`simulate_coupled_cv`.
    k_dim : float
        Sigma-scaled homogeneous (chemical) rate constant.
    ks_dim : float
        Dimensional standard heterogeneous rate constant ``k^o`` (cm/s).
    alpha : float
        Transfer coefficient.
    n, D_M, upper_limit, lower_limit, k_back :
        As in :func:`simulate_coupled_cv`.

    Returns
    -------
    CoupledCVResult
    """
    if n % 2 == 0:
        n += 1
    T = 2.0 * (upper_limit + abs(lower_limit))
    tau = T / (n - 1)
    m = space_points(D_M, n)
    s = 2 if mechanism in ("E", "EC'") else 3

    kf = k_dim * tau
    kb = k_back * tau
    if mechanism == "E":
        K = np.zeros((s, s))
    elif mechanism == "EC":
        K = np.array([[0.0, 0.0, 0.0],
                      [0.0, kf, -kb],
                      [0.0, -kf, kb]])
    elif mechanism == "EC'":
        K = np.array([[0.0, -kf],
                      [0.0, kf]])
    else:
        raise ValueError("mechanism must be 'E', 'EC', or 'EC''")

    ks_star = ks_star_sweep(ks_dim, T, D_M, n)

    bulk = np.zeros(s)
    bulk[0] = 1.0

    k_all = np.arange(1, n + 1)
    frac = surface_ratio(k_all, tau, T, upper_limit)
    xi = frac / (1.0 - frac)

    c = np.zeros((n, m, s))
    c[0, :, 0] = 1.0
    # consistent quasi-reversible surface IC: at the start the surface is close to
    # equilibrium; seed it with the Nernstian split (negligible for large initial
    # overpotential, and harmless because the first solved step overwrites it).
    o0 = bv_dirichlet_surface(xi[0])
    c[0, 0, 0] = float(o0)
    c[0, 0, 1] = 1.0 - float(o0)

    for k in range(1, n):
        c[k] = _solve_block_tridiag_bv(D_M, K, c[k - 1], xi[k], bulk,
                                       ks_star, alpha)

    cO = c[:, :, 0]
    grad = 3.0 * cO[:, 0] - 4.0 * cO[:, 1] + cO[:, 2]
    current = grad * math.sqrt(D_M * (n - 1)) / math.sqrt(4.0 * T)
    potential = _potential_axis(n, tau, T, upper_limit)

    return CoupledCVResult(c=c, current=current, potential=potential,
                           n=n, m=m, s=s, D_M=D_M, tau=tau, T=T)
