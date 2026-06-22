# Simulating Electrochemical Reactions in Python

A Python-native adaptation of Michael Honeychurch's
**_Simulating Electrochemical Reactions in Mathematica_** (SERM). The original
Mathematica notebooks are the reference for the science and the numerical
algorithms; this project re-implements them as idiomatic Python using
**numpy / scipy / matplotlib** (with **sympy** reserved for genuinely symbolic
work).

> **Attribution.** All physics, algorithms, and the structure of the worked
> examples are due to Michael Honeychurch, *Simulating Electrochemical Reactions
> in Mathematica*. This repository is an independent Python re-implementation for
> study; the `.nb`/`.m` source files distributed with the book are the
> authoritative reference. The Python code here was validated independently
> against analytical results (e.g. the Cottrell equation) rather than by copying
> the book's numbers.

## Status

**Complete.** All 16 chapters, both appendices (A: Python refresher, B: generated
`serm` reference), and the shared `serm` package are implemented. Every notebook
executes fresh and headless with **zero error outputs**, and each chapter carries
in-notebook validation `assert`s against an analytical or independently-computed
reference (Cottrell, Randles–Sevcik, Nicholson–Shain, Sand, Levich, etc.) that
pass on execution. See the [table of contents](#table-of-contents-python-native)
for per-chapter validation methods.

## What's here

```
serm-python/
  README.md
  requirements.txt
  tools/
    nb_extract.py        # box-format .nb -> readable text (translation aid)
    build_notebook.py    # builds the Chapter 2 notebook with nbformat
    AUTHORING_SPEC.md    # the contract every chapter agent follows
    _extracted_ExplicitFD.txt   # extracted reference from ExplicitFD.nb
  serm/
    __init__.py          # explicit FD solver, electrode current, Cottrell ref
    tridiagonal.py       # Thomas algorithm + scipy.linalg.solve_banded wrapper
    filters.py           # MovingAve + Gaussian ConvolutionFilter (port of Filters.m)
    grids.py             # FD grid construction (port of makeGrid in ExplicitFD.nb)
    plotting.py          # matplotlib helpers (profiles, 3-D surface, animation)
    waveforms.py         # potential excitation waveforms (sweep/step/pulse/AC)
    echem.py             # analytic reference results for validation
    ch05_*.py ch06_*.py ch10_*.py ch12_*.py ch13_*.py ch14_*.py ch15_*.py
                         # chapter-specific solver modules
  notebooks/
    01_solving_pdes.ipynb ... 16_processing_experimental_data.ipynb
    A_appendix_a_python_refresher.ipynb
    appendix_b_serm_reference.ipynb        # auto-generated serm API reference
  validation/
    validate_explicit.py # standalone Cottrell-comparison script
```

The top-level `serm` package re-exports the explicit FD solver plus the
`tridiagonal`, `filters`, `grids`, `plotting`, `waveforms`, and `echem`
submodules. Note there are two distinct `cottrell_current` functions: the
top-level `serm.cottrell_current(n)` is the *dimensionless* reference from the
Chapter 2 pilot, while `serm.echem.cottrell_current(t, n, A, D, c_bulk)` is the
*dimensional* form. `import serm` resolves when the project root is on
`sys.path`; the notebooks do this with `sys.path.insert(0, "..")` near the top of
each notebook (the package is not pip-installed).

### Ported Mathematica sources

- `serm/tridiagonal.py` — port of `Electrochem/Tridiagonal.m` (Honeychurch,
  2002). `tridiag_solve` is the bare Thomas algorithm (no pivoting, matching the
  original `TridiagSolver`); `tridiag_solve_banded` wraps
  `scipy.linalg.solve_banded` as the recommended, pivoting production path.
- `serm/filters.py` — port of `Electrochem/Filters.m`. `moving_average`
  reproduces `MovingAve`; `convolution_filter` reproduces `ConvolutionFilter`
  with the same Gaussian kernel `exp(-k^2/100)` over `-len..len`, normalised.
- `serm/grids.py` and the solver in `serm/__init__.py` — ported from the code
  cells of `Extra Notebooks/chapter2/ExplicitFD.nb`.

## Install and run

This repository was developed against Python 3.14 with a project-local virtual
environment (the host's system Python is externally managed / PEP 668).

```bash
cd serm-python
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# register the kernel referenced by the notebooks (display name "Python 3 (serm venv)")
.venv/bin/python -m ipykernel install --user --name serm-venv \
    --display-name "Python 3 (serm venv)"

# sanity-check the package imports (run from the project root)
.venv/bin/python -c "import serm, serm.tridiagonal, serm.filters, serm.grids, \
    serm.plotting, serm.waveforms, serm.echem"

# execute every notebook end-to-end (headless); the kernel name "python3"
# resolves to this venv because the command is the venv's own jupyter
for nb in notebooks/*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 "$nb"
done

# or open the suite interactively in JupyterLab
.venv/bin/jupyter lab
```

The notebooks declare the kernel `python3`; running them through the **venv's**
`jupyter` (`.venv/bin/jupyter`) is what binds that name to this interpreter. A
`serm-venv` kernel (display name "Python 3 (serm venv)") is also registered for
interactive use in JupyterLab.

## The pilot in brief (Chapter 2)

The notebook derives and implements the **explicit forward-difference** solution
of Fick's second law for a potential step into the diffusion-limited regime of
`O + e- <-> R`. Key points, all reproduced from `ExplicitFD.nb`:

- Dimensionless model diffusion coefficient `D_M = dt/dx^2`, with grid sizing
  `m = 1 + ceil(6*sqrt(D_M*(n-1)))`.
- Explicit update
  `c[j,k] = D_M*c[j-1,k-1] + (1-2*D_M)*c[j,k-1] + D_M*c[j+1,k-1]`.
- **Stability limit `D_M <= 0.5`** (the notebook demonstrates the blow-up at
  `D_M = 0.52`).
- **Validation against the Cottrell equation** `i = 1/sqrt(pi*tau)`: the
  simulated diffusion-limited current matches the analytical response to a mean
  relative error of about `9e-4` (~0.09 %) at `n = 2000`, converging as the grid
  is refined.

The `tools/nb_extract.py` helper turns the book's box-format Mathematica 5.2
notebooks into readable text so the original code can be studied without a
Wolfram kernel (none is installed here).

## Table of contents (Python-native)

The book is re-organised into a Python-native sequence. The Mathematica
introduction (source Chapter 0) is replaced by a Python primer, and the
Mathematica stylesheet appendix is dropped; a generated `serm` reference is added
as Appendix B. Every notebook below executes headless with **zero error
outputs**; the "Validation" column names the in-notebook `assert`-based check
that holds on execution.

| Ch. | Title | Status | Validation method (in-notebook asserts) |
|----:|-------|--------|------------------------------------------|
| 01 | Solving partial differential equations | done | sympy symbolic checks: separation-of-variables residual ≡ 0; Fourier coefficient A₁ = 4/π |
| 02 | Explicit finite differences | done | Cottrell match (mean rel err < 5e-3); demonstrates D_M > 0.5 instability |
| 03 | Speed and accuracy: implicit & Crank–Nicolson | done | convergence orders: backward-Euler ≈ 1, Crank–Nicolson ≈ 2 |
| 04 | Other numerical methods: Runge–Kutta, Volterra, SOR | done | RK/Volterra peak vs Randles–Sevcik constant (< 5e-3); RK2 error decreases on refinement |
| 05 | Potential sweep — reversible CV | done | peak current vs Randles–Sevcik (< 5e-3) |
| 06 | Potential sweep — quasi/non-reversible | done | Cottrell certification + reversible limit vs Nicholson–Shain 0.4463 |
| 07 | AC voltammetry | done | fundamental-harmonic peak ≈ 1/4 located at E⁰ |
| 08 | Potential steps and pulses | done | dimensionless & physical Cottrell match (< 1e-2); first-order convergence |
| 09 | Chronopotentiometry | done | Sand product/constant; wave-shape RMSE bound |
| 10 | Thin layers and thin films | done | thin-layer peak height, voltammogram symmetry, absence of diffusional tail |
| 11 | Strongly adsorbed molecules | done | surface-wave peak ψ ≈ 1/4 at E⁰, symmetric |
| 12 | Monte Carlo simulations | done | MSD = 2 D t within statistical tolerance; Cottrell t^(−1/2) slope |
| 13 | Coupled chemical reactions | done | no-reaction limit matches Ch. 5 to machine precision; monotone grid convergence |
| 14 | Rotating disk electrode voltammetry | done | Levich magnitude (< 5e-3) and Levich-plot linearity R² > 0.9999 |
| 15 | Finite differences with sparse arrays | done | sparse solver matches dense FD to < 1e-11 |
| 16 | Processing experimental data | done | smoothing reduces RMS; Savitzky–Golay preserves peak position/height |
| App. A | Python for electrochemical simulation | done | numpy-vs-list equivalence asserts throughout |
| App. B | The `serm` package reference (generated) | done | auto-rendered signatures/docstrings + one runnable example per module |

The source Appendix 1 (Mathematica stylesheet) is dropped as
Mathematica-specific.

## How this was built and how to validate

- **Source of truth.** The book's Mathematica 5.2 notebooks (box format,
  `BoxData[RowBox[...]]`) are the reference for the physics and algorithms. No
  Wolfram kernel is used; `tools/nb_extract.py` recovers the original Wolfram
  code as readable text, which is then **re-implemented** in idiomatic
  numpy/scipy/matplotlib (sympy only for genuine symbolic derivation). Cached
  Mathematica graphics are ignored and all plots are regenerated in matplotlib.
- **Independent validation, not number-copying.** Each chapter validates against
  an analytical result or an independently-computed quantity via `assert`s that
  run as part of the notebook (Cottrell, Randles–Sevcik, Nicholson–Shain, Sand,
  Levich, Koutecký–Levich, surface-wave theory, convergence-order studies, and
  cross-chapter limit checks). Honeychurch's printed numbers are **not** assumed.
- **How to re-validate everything.** Run the headless execution loop in
  [Install and run](#install-and-run); a clean run with no `CellExecutionError`
  means every chapter's asserts held. Appendix B re-imports the package and
  re-renders the API reference, so executing it confirms the documented
  signatures and the one-example-per-module smoke tests still pass.

Chapter authors: follow [`tools/AUTHORING_SPEC.md`](tools/AUTHORING_SPEC.md).
