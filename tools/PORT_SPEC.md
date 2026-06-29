# PORT_SPEC — contract for chapter porting agents

This document is the working contract for agents porting individual chapters of
Honeychurch's *Simulating Electrochemical Reactions in Mathematica* (SERM) to
this Python project. Read it before touching any file. It defines the validation
policy, the file-ownership rule, how to use the shared infrastructure built in
the FOUNDATION phase, and the data-redistribution rule.

The source book is **read-only**; never modify it. Convert any `.nb` to readable
text with:

```bash
.venv/bin/python tools/nb_extract.py "<path to .nb>"
```

---

## 1. Validation policy (strongest first)

Every new method gets **at least one assert-backed check**. State in the
notebook prose which tier you used. Tiers, strongest first:

1. **Closed-form / limiting analytic check** via `serm.echem` (e.g. Cottrell,
   Randles–Ševčík, Sand, Levich, surface-wave peak) or another independently
   derived closed form.
2. **Reduction to a validated limit.** Examples already in the package:
   `ks -> infinity` recovers the Nernstian reversible result (validated in
   `serm.boundary.bv_limits_selfcheck`); a no-reaction / no-kinetics limit
   recovers the simple CV; `ks -> 0` recovers the blocking / fully-irreversible
   limit.
3. **Convergence / self-consistency** (grid refinement, sparse-vs-dense), or a
   round-trip identity. For timing / scaling demos assert only **relative**
   ordering (e.g. sparse faster than dense), **never absolute times**.

Rules:

- Do **not** loosen a validation just to make it pass. If you change the
  science, keep or add a check.
- Do **not** copy Honeychurch's prose, figures, or numerical output. Re-derive
  and re-implement; regenerate all figures in matplotlib.
- Where the book overlays DigiSim / `.dat` data, substitute an independent
  analytic reference or cited published working-curve points, or validate by
  reduction to a known limit.

---

## 2. File-ownership rule (hard boundary)

Each chapter agent owns a narrow slice and must stay inside it. For chapter
`NN`, you **may**:

- Edit **only** your own `notebooks/NN_*.ipynb`.
- Create `notebooks/extras/NN_*.ipynb` for supplementary material.
- Add a chapter-specific helper module `serm/chNN_*.py`.

You **must NOT**:

- Edit shared `serm/` modules (`__init__.py`, `kinetics.py`, `grids.py`,
  `tridiagonal.py`, `waveforms.py`, `echem.py`, `plotting.py`, `filters.py`,
  `boundary.py`, `semiintegration.py`).
- Edit another chapter's notebook or `serm/chNN_*.py`.
- Edit `README.md`, `CONTRIBUTING.md`, CI (`.github/`), `requirements.txt`, or
  this `PORT_SPEC.md`.

If you find you need a change to a shared module, **stop and flag it** for the
foundation owner rather than editing it yourself.

### Notebook conventions

- Build / modify notebooks with `nbformat`; kernelspec name `python3`.
- Each notebook must self-add the repo root to `sys.path` in its first code
  cell and use the inline backend:

  ```python
  import os, sys
  sys.path.insert(0, os.path.abspath('..'))
  %matplotlib inline
  ```

- Execute headless until clean, then confirm **0 error outputs AND 0
  stderr/warning stream outputs**:

  ```bash
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 notebooks/NN_*.ipynb
  ```

---

## 3. Using the shared FOUNDATION infrastructure

### Quasi-reversible Butler–Volmer surface boundary — `serm.boundary`

Replaces the Nernstian Dirichlet boundary for a quasi-reversible
`O + n e- <-> R` couple. Usable for sweep, step, AC and coupled-reaction
simulations. Key entry points (see module docstring for the full derivation):

```python
from serm.boundary import (
    bv_dirichlet_surface,   # Nernstian limit xi/(1+xi)
    bv_surface_state,       # (c0, tmp): eliminated surface conc + factor
    bv_row_patch,           # (diag_delta, super_delta, rhs_delta) for the
                            # first implicit-FD row
    bv_limits_selfcheck,    # asserts ks->inf Nernstian and ks->0 blocking
)
from serm.kinetics import ks_star_sweep, bv_surface_factor, bv_surface_conc
```

- `xi = exp[n F (E - E0) / R T]` is the surface ratio; `ks_star` is the
  grid-scaled dimensionless rate constant (`serm.kinetics.ks_star_sweep` for a
  sweep); `alpha` is the transfer coefficient.
- For an implicit step with base first-row diagonal `1 + 2*DM` and
  super-diagonal `-DM`, get the per-step boundary patch from
  `bv_row_patch(xi, ks_star, alpha, DM)` and recover the surface node with
  `bv_surface_state(c1, c2, xi, ks_star, alpha)`.
- Validation already wired in: `bv_limits_selfcheck()` asserts the
  `ks_star -> inf` (Nernstian) and `ks_star -> 0` (blocking / irreversible)
  limits. Call it (or reduce to one of these limits) as your chapter's
  reduction-to-validated-limit check.

### Semi-integration / fractional calculus — `serm.semiintegration`

Appendix-2 semi-integration and general fractional integro-differentiation:

```python
from serm.semiintegration import (
    semi_integrate,                    # Riemann–Liouville semi-integral (q=1/2)
    fractional_integrodifferentiate,   # Grünwald–Letnikov, any real order
    semi_derivative,                   # order +1/2 convenience wrapper
    semiintegration_selfcheck,         # asserts the two reference results
)
```

- `semi_integrate(y, dt, q=0.5)` — RL convolution semi-integral. The
  semi-integral of a Cottrell current `~ t^-1/2` is a constant plateau.
- `fractional_integrodifferentiate(y, dt, order)` — `order > 0` differentiates,
  `order < 0` integrates; `order = ±0.5` is the semi-derivative / semi-integral.
- `semi_derivative` of a diffusion-limited reversible LSV current is a symmetric
  peak.
- `semiintegration_selfcheck()` asserts both reference results; reuse it (or its
  pattern) as your validation.

---

## 4. No data redistribution

Do **not** add or rely on the book's bundled `.dat` / experimental data files,
and do not commit them to the repository. Where you need a reference curve,
either compute an independent analytic reference (preferably via `serm.echem`),
cite published working-curve points, or validate by reduction to a known limit.
Cite any external numbers you use.
