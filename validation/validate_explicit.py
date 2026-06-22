"""Validate the explicit FD solver against the analytical Cottrell response.

Run: python validation/validate_explicit.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import serm
from serm.grids import make_grid, space_points, dx_dimensionless


def run(D_M=0.45, n=2000):
    m = space_points(D_M, n)
    c = make_grid(m, n)
    serm.explicit_solve(c, D_M)

    i_sim = serm.electrode_current(c, D_M)
    i_cot = serm.cottrell_current(n)

    # Compare over a window away from the very-short-time region, where the
    # explicit scheme is least accurate (steep gradient, coarse early steps).
    k = np.arange(n)
    tau = k / (n - 1)
    mask = (tau >= 0.05) & (tau <= 0.95)
    rel = np.abs(i_sim[mask] - i_cot[mask]) / i_cot[mask]
    return dict(
        D_M=D_M, n=n, m=m,
        dx=dx_dimensionless(D_M, n),
        max_rel=float(np.nanmax(rel)),
        mean_rel=float(np.nanmean(rel)),
    )


if __name__ == "__main__":
    for D_M, n in [(0.4, 200), (0.45, 1000), (0.45, 4000)]:
        r = run(D_M, n)
        print(f"D_M={r['D_M']}, n={r['n']}, m={r['m']}: "
              f"mean rel err={r['mean_rel']:.4e}, max rel err={r['max_rel']:.4e}")
