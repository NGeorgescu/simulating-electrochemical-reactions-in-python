"""Quasi-reversible Butler--Volmer surface boundary for implicit FD simulations.

Chapters 6, 7, 8 and 13 of *Simulating Electrochemical Reactions in Python*
(after Honeychurch's SERM) all replace the Nernstian Dirichlet surface condition
of the reversible chapters with a *quasi-reversible* Butler--Volmer (BV)
condition for the couple ``O + n e- <=> R``.  Rather than re-inline the algebra in
every chapter, this module packages the finite-difference surface-concentration
*elimination* once, in a form that a sweep, step, AC or coupled-reaction solver
can all call.

The physical surface condition
-------------------------------
At a planar electrode the faradaic flux of ``O`` equals the net BV rate::

    -D dc_O/dx |_0 = k_f c_O(0) - k_b c_R(0),

with the BV rate constants (relative to the formal potential ``E0``)::

    k_f = k_s exp[-alpha   n F (E - E0) / R T],
    k_b = k_s exp[(1-alpha) n F (E - E0) / R T].

In the dimensionless variables of SERM the surface gradient is approximated by a
one-sided three-point stencil ``(3 c0 - 4 c1 + c2)/(2 dX)`` (``c0`` the surface
node, ``c1, c2`` the first two interior nodes), and the rate constant enters
through the grid-scaled ``ks_star``.  Writing ``xi = exp[n F (E - E0)/R T]`` and
using ``c_R = 1 - c_O`` (equal diffusion coefficients), the discrete flux balance
is linear in the unknown surface concentration ``c0`` and can be solved for it::

    c0 = (ks_star * xi**(1 - alpha) + 4 c1 - c2) * tmp,
    tmp = xi**alpha / (3 xi**alpha + ks_star (1 + xi)).

This is the elimination Honeychurch performs symbolically in
``Extra Notebooks/chapter6/ImplicitCVQuasi.nb``; the two pieces ``tmp`` and
``c0`` are :func:`serm.kinetics.bv_surface_factor` and
:func:`serm.kinetics.bv_surface_conc`.  This module re-exports them under
intent-revealing names and adds:

* :func:`bv_dirichlet_surface` -- the Nernstian limiting surface value
  ``xi/(1+xi)`` (what the BV condition collapses to as ``ks_star -> inf``);
* :func:`bv_row_patch` -- the per-step modification of the first tridiagonal row
  (diagonal and super-diagonal entries, and the RHS increment) that injects the
  BV surface elimination into an implicit solver, so chapters do not re-derive
  the patch coefficients;
* :func:`bv_limits_selfcheck` -- a runnable assertion that the two limits
  (``ks_star -> inf`` -> Nernstian, ``ks_star -> 0`` -> blocking / zero faradaic
  flux) are reproduced to tight tolerance.

Limiting behaviour (validated in :func:`bv_limits_selfcheck`)
------------------------------------------------------------
* ``ks_star -> inf`` (reversible): ``tmp -> xi**alpha / (ks_star (1+xi))`` so the
  ``ks_star`` factors cancel in ``c0`` and ``c0 -> xi/(1+xi)``, the Nernstian
  Dirichlet value used by the reversible chapters -- *independent of the
  interior concentrations*.
* ``ks_star -> 0`` (totally irreversible at fixed ``E``): ``tmp -> 1/3`` and
  ``c0 -> (4 c1 - c2)/3``, which makes the surface gradient ``3 c0 - 4 c1 + c2``
  vanish: with no kinetic driving force the electrode passes no faradaic current
  (a blocking electrode).  The finite current of an irreversible *sweep* comes
  from the ``ks_star * xi**(1-alpha)`` source term carried at finite ``ks_star``.
"""
from __future__ import annotations

import numpy as np

from .kinetics import (
    F,
    R,
    f_thermal,
    ks_star_sweep,
    bv_surface_factor,
    bv_surface_conc,
)

__all__ = [
    "F",
    "R",
    "f_thermal",
    "ks_star_sweep",
    "bv_surface_factor",
    "bv_surface_conc",
    "bv_dirichlet_surface",
    "bv_surface_state",
    "bv_row_patch",
    "bv_limits_selfcheck",
]


def bv_dirichlet_surface(xi):
    """Nernstian (reversible) surface concentration of ``O``: ``xi/(1+xi)``.

    This is the Dirichlet value the quasi-reversible BV condition collapses to as
    ``ks_star -> inf`` (see module docstring).  With ``xi = exp[nF(E-E0)/RT]`` a
    reduction (``E << E0``, ``xi -> 0``) depletes ``O`` at the surface, and an
    oxidation (``E >> E0``, ``xi -> inf``) restores it to the bulk value 1.

    Parameters
    ----------
    xi : float or array_like
        Surface ratio ``exp[nF(E - E0)/RT]``.

    Returns
    -------
    float or numpy.ndarray
        ``xi / (1 + xi)``.
    """
    xi = np.asarray(xi, dtype=float)
    return xi / (1.0 + xi)


def bv_surface_state(c1, c2, xi, ks_star: float, alpha: float):
    """Eliminated surface concentration and its elimination factor in one call.

    Convenience wrapper returning ``(c0, tmp)`` so a solver can reuse ``tmp`` for
    both the surface-concentration update and the first-row patch without
    recomputing it.

    Parameters
    ----------
    c1, c2 : float or array_like
        Concentration of ``O`` at the first and second interior nodes.
    xi : float or array_like
        Surface ratio ``exp[nF(E - E0)/RT]``.
    ks_star : float
        Dimensionless (grid-scaled) standard rate constant.
    alpha : float
        Transfer coefficient.

    Returns
    -------
    c0 : float or numpy.ndarray
        Eliminated surface concentration of ``O``.
    tmp : float or numpy.ndarray
        Elimination factor :func:`serm.kinetics.bv_surface_factor`.
    """
    tmp = bv_surface_factor(xi, ks_star, alpha)
    c0 = bv_surface_conc(c1, c2, xi, ks_star, alpha, tmp)
    return c0, tmp


def bv_row_patch(xi, ks_star: float, alpha: float, DM: float, tmp=None):
    """First-row tridiagonal patch + RHS increment for the BV surface boundary.

    For an implicit (backward-Euler / Crank--Nicolson-style) diffusion step whose
    first interior unknown ``c1`` has the *base* diagonal ``1 + 2 DM`` and
    super-diagonal ``-DM`` (the Laplacian stencil with ``DM = dtau/dX**2``),
    substituting the eliminated surface concentration ``c0 = (ks_star
    xi**(1-alpha) + 4 c1 - c2) tmp`` into the first interior equation modifies
    that row.  This helper returns the three quantities a solver needs to apply
    that substitution, matching the patch used in
    :mod:`serm.ch06_potential_sweep_nonreversible`::

        diag[0]      += -4 * DM * tmp        # from the +4 c1 term in c0
        super[0]     += +1 * DM * tmp        # from the -c2 term in c0
        rhs[0]       += DM * tmp * ks_star * xi**(1 - alpha)   # source term

    Parameters
    ----------
    xi : float
        Surface ratio ``exp[nF(E - E0)/RT]`` at this step.
    ks_star : float
        Dimensionless standard rate constant.
    alpha : float
        Transfer coefficient.
    DM : float
        Model diffusion number ``dtau/dX**2``.
    tmp : float, optional
        Precomputed :func:`serm.kinetics.bv_surface_factor`; computed if omitted.

    Returns
    -------
    diag_delta : float
        Increment to add to the first row's main-diagonal entry.
    super_delta : float
        Increment to add to the first row's super-diagonal entry.
    rhs_delta : float
        Increment to add to the first RHS entry.

    Notes
    -----
    The base diagonal/super-diagonal themselves are *not* applied here; only the
    deltas due to the surface elimination are returned, so the helper composes
    with whatever base stencil the caller has built.
    """
    if tmp is None:
        tmp = bv_surface_factor(xi, ks_star, alpha)
    diag_delta = -4.0 * DM * tmp
    super_delta = DM * tmp
    rhs_delta = DM * tmp * ks_star * xi ** (1.0 - alpha)
    return diag_delta, super_delta, rhs_delta


def bv_limits_selfcheck(alpha: float = 0.5, tol: float = 1e-6) -> None:
    """Assert the BV surface helper reproduces both kinetic limits.

    Validation tiers used (per the project validation policy):

    * **Reduction to a validated limit (reversible):** as ``ks_star -> inf`` the
      eliminated surface concentration must equal the Nernstian Dirichlet value
      ``xi/(1+xi)`` (:func:`bv_dirichlet_surface`) to ``tol``, for arbitrary
      interior concentrations -- the reversible boundary the Chapter 5 solver is
      already validated against.
    * **Reduction to a known limit (irreversible):** as ``ks_star -> 0`` the
      elimination factor ``tmp -> 1/3`` and the surface flux ``3 c0 - 4 c1 + c2``
      must vanish (blocking electrode -- no faradaic current without finite
      kinetics).

    Raises
    ------
    AssertionError
        If either limit is not met to the requested tolerance.
    """
    xis = np.array([1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 1e2, 1e3])
    # Use non-trivial, unequal interior concentrations so the reversible limit
    # is a real test that c0 stops depending on c1, c2.
    c1, c2 = 0.73, 0.41

    # --- reversible limit: ks_star -> inf -> Nernstian Dirichlet ---
    ks_big = 1e12
    c0_big, _ = bv_surface_state(c1, c2, xis, ks_big, alpha)
    nernst = bv_dirichlet_surface(xis)
    err_rev = np.max(np.abs(c0_big - nernst))
    assert err_rev < tol, (
        f"reversible (ks->inf) limit off by {err_rev:.2e} (tol {tol:.1e})"
    )

    # --- irreversible limit: ks_star -> 0 -> tmp=1/3, zero faradaic flux ---
    ks_small = 1e-12
    c0_small, tmp_small = bv_surface_state(c1, c2, xis, ks_small, alpha)
    err_tmp = np.max(np.abs(tmp_small - 1.0 / 3.0))
    assert err_tmp < tol, (
        f"irreversible tmp limit off by {err_tmp:.2e} (tol {tol:.1e})"
    )
    flux = 3.0 * c0_small - 4.0 * c1 + c2
    err_flux = np.max(np.abs(flux))
    assert err_flux < tol, (
        f"irreversible zero-flux limit off by {err_flux:.2e} (tol {tol:.1e})"
    )


if __name__ == "__main__":  # pragma: no cover
    bv_limits_selfcheck()
    print("serm.boundary: BV quasi-reversible limits OK "
          "(ks->inf Nernstian, ks->0 blocking).")
