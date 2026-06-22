"""Monte Carlo / random-walk machinery for Chapter 12.

A Python-native re-implementation of the random-walk simulations in Michael
Honeychurch's *Simulating Electrochemical Reactions in Mathematica* (SERM),
Chapter 12 ("Monte Carlo simulations").  The original notebook builds diffusion
and electrochemistry out of nothing but coin-flips: a molecule takes a unit step
left or right (1-D), and an ensemble of such walkers reproduces Fickian
diffusion, a reversible voltammogram, and a Cottrell transient.

The Wolfram code is procedural and scalar (``FoldList``, ``NestWhileList``,
``While`` loops over one walker at a time).  Here the same physics is expressed
with vectorised numpy: a whole ensemble of walkers is advanced together as a
2-D array of steps, which is both far faster and idiomatic.

All randomness goes through an explicit ``numpy.random.Generator`` so every
result is reproducible from a seed.

References
----------
The algorithms trace to ``Chapters/chapter12.nb`` (SERM).  The continuum results
they reproduce (Gaussian spreading, mean-squared displacement ``<x^2> = 2 D t``,
the first-passage / Cottrell ``t^{-1/2}`` flux) are standard; see e.g.
S. Chandrasekhar, *Rev. Mod. Phys.* **15**, 1 (1943), and A. J. Bard &
L. R. Faulkner, *Electrochemical Methods*, 2nd ed. (Wiley, 2001).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# 1-D random walk (Section 12.2 of SERM)
# --------------------------------------------------------------------------- #
def walk_steps(n_walkers, n_steps, rng):
    """Draw a block of unit steps for an ensemble of 1-D walkers.

    Each step is ``+1`` or ``-1`` with equal probability -- the Wolfram
    ``rw := 2 Random[Integer, {0,1}] - 1``.  Vectorised over the whole ensemble.

    Parameters
    ----------
    n_walkers : int
        Number of independent walkers.
    n_steps : int
        Number of time increments.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    ndarray, shape (n_walkers, n_steps), dtype int8
        ``+1``/``-1`` steps.
    """
    return (2 * rng.integers(0, 2, size=(n_walkers, n_steps), dtype=np.int8)
            - 1).astype(np.int8)


def walk_positions(n_walkers, n_steps, rng):
    """Cumulative positions of an ensemble of 1-D walkers starting at the origin.

    Vectorised analogue of ``FoldList[Plus, 0, Table[rw, {t}]]`` applied to many
    walkers.  Column ``k`` is the position after ``k`` steps; column 0 is the
    origin.

    Returns
    -------
    ndarray, shape (n_walkers, n_steps + 1)
        Integer positions; ``[:, 0]`` is 0.
    """
    steps = walk_steps(n_walkers, n_steps, rng)
    pos = np.zeros((n_walkers, n_steps + 1), dtype=np.int64)
    np.cumsum(steps, axis=1, out=pos[:, 1:])
    return pos


def position_histogram(final_positions):
    """Fraction of walkers at each occupied lattice site.

    Port of SERM ``frequencies3`` (``Split[Sort[...]]`` then divide each group
    length by the walker count): returns occupied positions and the fraction of
    walkers there, i.e. an empirical probability mass function.

    Parameters
    ----------
    final_positions : array_like of int
        Final position of each walker.

    Returns
    -------
    positions : ndarray of int
        Occupied lattice sites, sorted ascending.
    fraction : ndarray of float
        Fraction of walkers at each site (sums to 1).
    """
    final_positions = np.asarray(final_positions)
    positions, counts = np.unique(final_positions, return_counts=True)
    fraction = counts / final_positions.size
    return positions, fraction


def gaussian_pmf(x, t, d=1.0, tau=1.0):
    """Continuum random-walk probability of finding a walker at site ``x``.

    The lattice random walk of unit step ``d`` per time increment ``tau`` tends,
    after ``t`` steps, to the Gaussian

    .. math::
        P(x) = \\sqrt{\\frac{2}{\\pi}\\,\\frac{\\tau}{t\\,d^2}}\\;
               \\exp\\!\\left(-\\frac{x^2\\,\\tau}{2\\,t\\,d^2}\\right).

    The factor of 2 in the prefactor (relative to a naive ``1/sqrt(2 pi sigma^2)``)
    accounts for the lattice parity: after an even number of steps only even
    sites are reachable, so the *occupied* sites carry twice the smooth density.
    For ``d = tau = 1`` this is the curve SERM fits in Section 12.2,
    ``Sqrt[2/(Pi t)] Exp[-x^2/(2 t)]``.

    Parameters
    ----------
    x : array_like
        Lattice position(s).
    t : float
        Number of time increments (steps).
    d : float
        Step length.
    tau : float
        Time per step.

    Returns
    -------
    ndarray
        Probability mass at each ``x`` (for the reachable-parity sites).
    """
    x = np.asarray(x, dtype=float)
    var = t * d ** 2 / tau
    return np.sqrt(2.0 / (np.pi * var)) * np.exp(-x ** 2 / (2.0 * var))


# --------------------------------------------------------------------------- #
# n-dimensional lattice walk (Sections 12.2.1, 12.2.2 of SERM)
# --------------------------------------------------------------------------- #
def lattice_walk_nd(n_steps, dim, rng):
    """Single n-dimensional simple-cubic lattice random walk from the origin.

    Generalises the SERM ``twoDStep`` / ``threeDStep`` construction: at each
    step pick one of the ``2*dim`` unit moves ``(+-e_i)`` with equal
    probability.

    Parameters
    ----------
    n_steps : int
        Number of steps.
    dim : int
        Spatial dimension (1, 2 or 3 in the chapter).
    rng : numpy.random.Generator

    Returns
    -------
    ndarray, shape (n_steps + 1, dim)
        The trajectory, starting at the origin.
    """
    moves = np.zeros((2 * dim, dim), dtype=np.int64)
    for axis in range(dim):
        moves[2 * axis, axis] = 1
        moves[2 * axis + 1, axis] = -1
    choice = rng.integers(0, 2 * dim, size=n_steps)
    steps = moves[choice]
    traj = np.zeros((n_steps + 1, dim), dtype=np.int64)
    np.cumsum(steps, axis=0, out=traj[1:])
    return traj


# --------------------------------------------------------------------------- #
# Electrochemical Monte Carlo: reversible LSV (Section 12.3 of SERM)
# --------------------------------------------------------------------------- #
@dataclass
class LSVResult:
    """Result of :func:`monte_carlo_lsv`."""
    step_index: np.ndarray      # time-step index at which each transfer occurred
    potential: np.ndarray       # dimensionless potential at each transfer
    charge: np.ndarray          # +1 (reduction) or -1 (oxidation) per transfer


def _surface_crossings(traj):
    """Step indices (1-based, as in SERM) at which a walker is at the surface x=0.

    ``traj`` is a single walker's trajectory including the start point; SERM
    indexes the NestList output 1-based and treats element ``k+1`` as "after
    ``k`` steps".  We return the number of steps taken when the walker sits on
    the surface, matching ``Position[walk, 0]`` minus the start offset.
    """
    # positions after step k live at traj[k]; k = 0 is the start (never the
    # surface here because walkers start at x >= 1).
    return np.nonzero(traj == 0)[0]


def monte_carlo_lsv(n_walkers, n_steps, incr, initial, start_max, rng):
    """Monte Carlo reversible linear-sweep voltammogram (SERM Section 12.3).

    Each walker starts a random integer distance ``1..start_max`` from a planar
    electrode and takes unit steps.  Every time it lands on the surface (``x=0``)
    the electrode tests it: the (cathodic) dimensionless potential after ``k``
    steps is ``E_k = initial - k*incr`` and the reduction probability is the
    Nernstian ``xi/(1+xi)`` with ``xi = exp(E_k)``.

    Following SERM's faster vectorised variant (``sortData``), we run the full
    walk first, find every surface crossing, draw the oxidation *state* the
    walker would have after each crossing from ``sign(xi/(1+xi) - u)``, and
    record an electron transfer wherever the state *changes* between consecutive
    crossings (a sign flip means one electron crossed the interface).  The sign
    of the change gives the charge: ``+1`` for a reduction, ``-1`` for an
    oxidation.

    Parameters
    ----------
    n_walkers : int
        Number of walkers (statistics scale as ``sqrt(n_walkers)``).
    n_steps : int
        Steps per walker == number of potential increments in the sweep.
    incr : float
        Dimensionless potential increment per step.
    initial : float
        Initial dimensionless potential (sweep goes ``initial -> initial -
        n_steps*incr``).
    start_max : int
        Maximum starting distance from the electrode (start drawn uniformly in
        ``1..start_max``).
    rng : numpy.random.Generator

    Returns
    -------
    LSVResult
        One entry per recorded electron transfer.
    """
    starts = rng.integers(1, start_max + 1, size=n_walkers)
    steps = (2 * rng.integers(0, 2, size=(n_walkers, n_steps), dtype=np.int8)
             - 1)
    pos = np.empty((n_walkers, n_steps + 1), dtype=np.int64)
    pos[:, 0] = starts
    np.cumsum(steps, axis=1, out=pos[:, 1:])
    pos[:, 1:] += starts[:, None]

    step_idx_all = []
    pot_all = []
    charge_all = []
    for w in range(n_walkers):
        cross = _surface_crossings(pos[w])  # step counts where x == 0
        if cross.size == 0:
            continue
        E = initial - cross * incr
        xi = np.exp(E)
        p_ox = xi / (1.0 + xi)                       # P(oxidized after test)
        u = rng.random(cross.size)
        # state after each crossing: +1 oxidized, -1 reduced
        state = np.sign(p_ox - u)
        state[state == 0] = 1.0
        # walker arrives oxidized (state = +1); prepend that initial state.
        full_state = np.concatenate(([1.0], state))
        change = full_state[1:] - full_state[:-1]    # nonzero => transfer
        tr = np.nonzero(change)[0]
        if tr.size == 0:
            continue
        # change = -2 => +1 -> -1 (reduction, +1 charge);
        # change = +2 => -1 -> +1 (oxidation, -1 charge).
        q = -np.sign(change[tr])
        step_idx_all.append(cross[tr])
        pot_all.append(E[tr])
        charge_all.append(q)

    if step_idx_all:
        step_index = np.concatenate(step_idx_all)
        potential = np.concatenate(pot_all)
        charge = np.concatenate(charge_all)
    else:
        step_index = np.array([], dtype=np.int64)
        potential = np.array([], dtype=float)
        charge = np.array([], dtype=float)
    return LSVResult(step_index, potential, charge)


def bin_voltammogram(result, n_steps, incr, initial):
    """Net charge per potential step -> a pseudo-voltammogram.

    Sums the recorded ``+-1`` charges into the discrete potential bins
    ``E_k = initial - k*incr`` for ``k = 1..n_steps``, reproducing SERM's
    "sort, split, add the charges" pipeline.

    Returns
    -------
    potential : ndarray
        Bin potentials (ascending in ``k`` -> descending in ``E``).
    net_charge : ndarray
        Net charge in each bin (reduction positive).
    """
    k = np.arange(1, n_steps + 1)
    potential = initial - k * incr
    net = np.zeros(n_steps, dtype=float)
    if result.step_index.size:
        np.add.at(net, result.step_index - 1, result.charge)
    return potential, net


# --------------------------------------------------------------------------- #
# Electrochemical Monte Carlo: chronoamperometry (Section 12.3.2 of SERM)
# --------------------------------------------------------------------------- #
def first_passage_times(n_walkers, n_steps, start_max, rng):
    """First-passage step count for each walker to reach the surface x=0.

    Port of SERM ``ca2``: run a full walk, then take the first index at which
    the position is zero (``First[Position[..., 0]]``).  Walkers that never
    reach the surface within ``n_steps`` are reported as ``n_steps + 2`` (SERM's
    sentinel) and filtered out by the caller.

    Parameters
    ----------
    n_walkers, n_steps : int
    start_max : int
        Walkers start uniformly in ``1..start_max`` from the surface.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray, shape (n_walkers,)
        First-passage step count (or the sentinel ``n_steps + 2``).
    """
    starts = rng.integers(1, start_max + 1, size=n_walkers)
    steps = (2 * rng.integers(0, 2, size=(n_walkers, n_steps), dtype=np.int8)
             - 1)
    pos = np.empty((n_walkers, n_steps + 1), dtype=np.int64)
    pos[:, 0] = starts
    np.cumsum(steps, axis=1, out=pos[:, 1:])
    pos[:, 1:] += starts[:, None]

    at_surface = (pos == 0)
    ever = at_surface.any(axis=1)
    fpt = np.full(n_walkers, n_steps + 2, dtype=np.int64)
    fpt[ever] = at_surface[ever].argmax(axis=1)
    return fpt


def arrivals_per_step(fpt, n_steps):
    """Histogram of first-passage times -> arrival flux per time step.

    The number of walkers first reaching the surface at step ``k`` is the
    discrete diffusion-limited current at that time (each arrival is one
    reduction).  Sentinel values (never reached) are dropped.

    Returns
    -------
    t : ndarray
        Step indices ``1..n_steps``.
    counts : ndarray
        Walkers arriving at each step.
    """
    valid = fpt[(fpt >= 1) & (fpt <= n_steps)]
    counts = np.bincount(valid, minlength=n_steps + 1)[1:n_steps + 1]
    t = np.arange(1, n_steps + 1)
    return t, counts
