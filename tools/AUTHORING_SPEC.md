# Authoring specification — *Simulating Electrochemical Reactions in Python*

This is the contract every chapter agent must follow. It exists so that 16
chapters can be written in parallel without colliding and without drifting in
style or quality. Read it in full before writing a chapter. The reference for
"good" is the pilot, `notebooks/02_explicit_finite_differences.ipynb`; match its
voice and structure.

---

## 1. What you are writing

You are adapting one chapter of Michael Honeychurch's *Simulating
Electrochemical Reactions in Mathematica* (SERM) into a **Python-native**
notebook. You are **not** transliterating Wolfram line-by-line. You are teaching
the method in idiomatic Python and re-deriving / re-validating the results
independently.

The source `.nb` files are the reference for the science and the algorithms.
There is **no Wolfram kernel** here. Recover the original code with the
extractor (Section 6), study it, then re-implement in numpy/scipy/matplotlib.

## 2. House style and voice

- **Textbook narrative.** Markdown cells teach: state the physics, show the
  derivation, explain why the method works and where it breaks. Write prose, not
  bullet-point stubs. The pilot's markdown cells are the calibration.
- **Clean code cells.** Functions/classes with type hints and docstrings;
  vectorized numpy. No line-by-line Wolfram transliteration, no dead code, no
  commented-out experiments left in.
- **numpy/scipy/matplotlib are the engine.** Use `sympy` **only** for genuinely
  symbolic work (deriving a finite-difference stencil, a closed form, a Laplace
  inversion). Never use sympy as the numerical solver.
- **Math in LaTeX** inside markdown (`$...$`, `$$...$$`), matching the pilot.
- **Attribution.** The chapter intro must name Honeychurch / SERM and the source
  chapter, as the pilot's first markdown cell does.
- **No emojis.** Plain technical prose.

## 3. Required notebook structure

Every chapter notebook must contain, in this order:

1. **Title + intro** (markdown): chapter title, one-paragraph attribution to
   SERM and the source chapter, and a sentence on what the chapter delivers.
2. **Theory / derivation** (markdown, possibly several cells): the
   non-dimensionalisation, the governing equation, the discretisation or
   analytic solution. Show the derivation; do not assert results without a path
   to them.
3. **Implementation** (code): the solver(s). Import shared machinery from
   `serm` (Section 4); only write chapter-specific code for what `serm` does not
   provide.
4. **Figures** (code): concentration profiles, voltammograms/transients, and a
   3-D surface or animation where it aids understanding. Use `serm.plotting`
   helpers. All plots are regenerated in matplotlib — never reuse cached Wolfram
   graphics.
5. **Validation with `assert`** (code + markdown): at least one cell that
   compares the simulation to an independent reference and `assert`s the error
   is within tolerance, printing a `PASS:` line. See Section 5.
6. **Summary** (markdown): what was shown, the key numerical result, and a
   pointer to where the method is extended in later chapters.

## 4. Importing and reusing `serm`

The shared package is at the repo root. The standard preamble (copy from the
pilot) makes it importable from `notebooks/`:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy as np
import matplotlib.pyplot as plt
import serm
```

**Do not re-implement what `serm` provides.** Use it:

- `serm.tridiagonal` — `tridiag_solve` (bare Thomas, no pivoting, matches the
  original `TridiagSolver`) and **`tridiag_solve_banded`** (wraps
  `scipy.linalg.solve_banded`, pivots — **this is the recommended default** for
  implicit/Crank–Nicolson chapters).
- `serm.filters` — `moving_average`, `convolution_filter`, `gaussian_kernel`
  (port of `Filters.m`) for smoothing noisy/experimental data.
- `serm.grids` — `make_grid`, `space_points`, `dx_dimensionless` (FD grid and
  dimensionless sizing).
- `serm.plotting` — `plot_profiles`, `plot_surface` (replaces `ListPlot3D`),
  `plot_current`, `animate_profiles` (a `FuncAnimation` helper that embeds in
  headless-executed notebooks via `HTML(anim.to_jshtml())`).
- `serm.waveforms` — excitation waveforms: `linear_sweep`, `cyclic_sweep`,
  `potential_step`, `pulse_train`, `ac_superposition`, and the helpers
  `dimensionless_sweep_rate` (`sigma = nFv/RT`) and `nernst_theta`. Voltammetry
  chapters (5–9) must use these rather than re-deriving the waveform.
- `serm.echem` — independent analytic references for validation:
  `cottrell_current`, `randles_sevcik_peak_current`, `sand_transition_time`,
  `levich_current`, `koutecky_levich_current`, `surface_wave_peak_current`.
  Import these in your validation cell.

If you need a helper specific to your chapter, you may add a new module
`serm/chNN_*.py` (e.g. `serm/ch13_kinetics.py`) and import it. You must **not**
edit the shared modules listed above, nor any other chapter's files.

## 5. Validation policy

Every chapter must validate its simulation and `assert` the result. Choose the
strongest strategy available and **state in markdown which one you used and
why**:

1. **Preferred — independent closed-form / limiting analytic check.** Compare
   against an exact result from `serm.echem` (Cottrell, Randles–Sevcik peak,
   Sand, Levich/Koutecky–Levich, surface-wave peak) or another closed form you
   derive. This is the gold standard: the analytic result is computed
   independently of the FD code, so agreement is a real cross-check. Assert the
   relative error is below a stated tolerance over a window where the method is
   expected to be accurate (exclude singular edges, e.g. `tau -> 0`, as the
   pilot does).
2. **Convergence / self-consistency** (when no closed form exists). Refine the
   grid (and/or time step) and show the error or the solution change decreasing
   at the expected order; `assert` the error shrinks as the grid is refined.
3. **Dense-vs-sparse / two-implementation cross-check** (e.g. Chapter 15).
   Run a sparse implementation and a dense one on the same problem and `assert`
   they agree to numerical tolerance.

Do **not** validate by copying Honeychurch's printed numbers. Re-derive or
re-compute the reference yourself. Every physics formula stated in prose or a
docstring must trace to something you read in the source `.nb` or a result you
compute and check in the notebook.

## 6. Recovering the Wolfram source (translation reference)

Use the extractor to turn a box-format `.nb` into readable text:

```bash
SRC="/home/nsg/Dropbox/Files/Research - Books/Electrochemistry/Honeychurch - Simulating Electrochemical Reactions in Mathematica/Chapters"
.venv/bin/python tools/nb_extract.py "$SRC/chapterNN.nb" -o /tmp/chNN.txt
```

It classifies cells, drops cached graphics/output, and emits clean prose and
best-effort Wolfram source for code cells. It is a *reading aid* — the Wolfram
it prints is approximate (operator spacing, `[[...]]` indexing, block-tridiagonal
matrix literals may need interpretation). Read it to understand the algorithm,
then implement cleanly in Python. Ignore cached `Graphics`/`Output` blobs;
regenerate every plot in matplotlib.

## 7. File ownership (no collisions)

Each chapter agent may write **only**:

- `notebooks/NN_<slug>.ipynb` — your chapter notebook (the one file you own).
- `serm/chNN_*.py` — optional chapter-specific helper module(s).
- files under `validation/` named `validate_chNN_*.py` if you want a standalone
  script (optional).

You must **not** edit: the shared `serm` modules (`tridiagonal`, `filters`,
`grids`, `plotting`, `waveforms`, `echem`, `__init__.py`), `README.md`, anything
under `tools/`, or any other chapter's notebook/helper. The source book
directory is **read-only — never modify it.**

## 8. Kernelspec and execution

Build the notebook programmatically with `nbformat` and set the kernelspec so
headless execution resolves to the project venv:

```python
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3 (serm venv)",
    "language": "python",
}
```

Then execute it headless and confirm it runs clean (0 error outputs):

```bash
MPLBACKEND=Agg /home/nsg/Dropbox/Files/Python/serm-python/.venv/bin/jupyter \
    nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 \
    notebooks/NN_<slug>.ipynb
```

After execution, programmatically verify there are **no** cells with
`output_type == "error"` before declaring the chapter done. A chapter that does
not execute clean is not finished.

## 9. Environment (verified this session)

- Python `3.14.5` in `/home/nsg/Dropbox/Files/Python/serm-python/.venv`.
- `numpy 2.4.6`, `scipy 1.18.0`, `matplotlib 3.11.0` (plus `sympy`, `nbformat`,
  `jupyterlab`, `nbclient`, `ipykernel` — see `requirements.txt`).
- Always invoke the venv interpreter explicitly
  (`/home/nsg/Dropbox/Files/Python/serm-python/.venv/bin/python`); the system
  Python is PEP-668 locked.
