# Simulating Electrochemical Reactions in ~~Mathematica~~ Python

A Python-native adaptation of Michael Honeychurch's
**_Simulating Electrochemical Reactions in Mathematica_** (SERM). This project
re-implements the book's electrochemistry and numerical methods as idiomatic
Python: a complete, runnable course in **digital simulation of electrochemical
reactions** — diffusion-controlled and kinetically-controlled mass transport,
**cyclic voltammetry**, chronoamperometry, chronopotentiometry, AC voltammetry,
adsorbed-species and thin-film responses, and **rotating disk electrode**
voltammetry — built with the **finite difference method** and related
**numerical methods** on top of **NumPy**, **SciPy**, and **matplotlib**, with
**SymPy** reserved for genuinely symbolic work. Every worked example lives in a
**Jupyter** notebook, and each chapter is checked against the closed-form
results of **electroanalytical chemistry** (Cottrell, Randles–Sevcik,
Nicholson–Shain, Sand, Levich, Butler–Volmer kinetics, and a **Monte Carlo**
random-walk model) so the **electrochemical simulation** code is verified, not
just plausible.

> **Attribution.** All physics, algorithms, and the structure of the worked
> examples are due to Michael Honeychurch, *Simulating Electrochemical Reactions
> in Mathematica*. This repository is an independent Python re-implementation for
> study; the original Mathematica notebooks distributed with the book are the
> authoritative reference for the science and the numerical algorithms. The
> Python code here was **validated independently against published analytic
> results** (closed-form electrochemistry equations such as the Cottrell
> equation), **not** by copying the book's printed numbers.

## Badges

![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Status

**Complete.** All 16 chapters, both appendices (A: Python refresher, B: generated
`serm` reference), and the shared `serm` package are implemented. Every notebook
executes fresh and headless with **zero error outputs**, and each chapter carries
in-notebook validation `assert`s against an analytical or independently-computed
reference (Cottrell, Randles–Sevcik, Nicholson–Shain, Sand, Levich, etc.) that
pass on execution. See the [table of contents](#table-of-contents-python-native)
for per-chapter validation methods.

## Installation

This repository was developed against **Python 3.14** with a project-local
virtual environment (the host's system Python is externally managed / PEP 668).
The `.venv/` directory is gitignored and fully rebuildable from
`requirements.txt`.

```bash
git clone https://github.com/NGeorgescu/simulating-electrochemical-reactions-in-python.git
cd simulating-electrochemical-reactions-in-python

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): numpy, scipy, matplotlib, sympy,
nbformat, jupyterlab, nbclient, ipykernel.

## Usage

Open the notebook suite interactively in JupyterLab:

```bash
.venv/bin/jupyter lab
```

To register the kernel the notebooks reference (display name
"Python 3 (serm venv)") for interactive use:

```bash
.venv/bin/python -m ipykernel install --user --name serm-venv \
    --display-name "Python 3 (serm venv)"
```

To sanity-check that the `serm` package imports (run from the project root):

```bash
.venv/bin/python -c "import serm, serm.tridiagonal, serm.filters, serm.grids, \
    serm.plotting, serm.waveforms, serm.echem"
```

To execute every notebook end-to-end (headless). Running them through the
**venv's** `jupyter` is what binds the notebooks' declared kernel name to this
interpreter:

```bash
for nb in notebooks/*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 "$nb"
done
```

A clean run with no `CellExecutionError` means every chapter's `assert`-based
validation held.

## What's inside

```
simulating-electrochemical-reactions-in-python/
  README.md
  LICENSE                    # MIT
  requirements.txt
  tools/
    nb_extract.py            # box-format .nb -> readable text (translation aid)
    build_notebook.py        # builds the Chapter 2 notebook with nbformat
    AUTHORING_SPEC.md        # the contract every chapter agent follows
    _extracted_ExplicitFD.txt   # extracted reference from ExplicitFD.nb
  serm/
    __init__.py              # explicit FD solver, electrode current, Cottrell ref
    tridiagonal.py           # Thomas algorithm + scipy.linalg.solve_banded wrapper
    filters.py               # MovingAve + Gaussian ConvolutionFilter (port of Filters.m)
    grids.py                 # FD grid construction (port of makeGrid in ExplicitFD.nb)
    plotting.py              # matplotlib helpers (profiles, 3-D surface, animation)
    waveforms.py             # potential excitation waveforms (sweep/step/pulse/AC)
    echem.py                 # analytic reference results for validation
    kinetics.py
    ch05_*.py ch06_*.py ch10_*.py ch12_*.py ch13_*.py ch14_*.py ch15_*.py
                             # chapter-specific solver modules
  notebooks/
    01_solving_pdes.ipynb ... 16_processing_experimental_data.ipynb
    A_appendix_a_python_refresher.ipynb
    appendix_b_serm_reference.ipynb        # auto-generated serm API reference
  validation/
    validate_explicit.py     # standalone Cottrell-comparison script
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
| 01 | [Solving partial differential equations](notebooks/01_solving_pdes.ipynb) | done | sympy symbolic checks: separation-of-variables residual ≡ 0; Fourier coefficient A₁ = 4/π |
| 02 | [Explicit finite differences](notebooks/02_explicit_finite_differences.ipynb) | done | Cottrell match (mean rel err < 5e-3); demonstrates D_M > 0.5 instability |
| 03 | [Speed and accuracy: implicit & Crank–Nicolson](notebooks/03_speed_and_accuracy.ipynb) | done | convergence orders: backward-Euler ≈ 1, Crank–Nicolson ≈ 2 |
| 04 | [Other numerical methods: Runge–Kutta, Volterra, SOR](notebooks/04_other_numerical_methods.ipynb) | done | RK/Volterra peak vs Randles–Sevcik constant (< 5e-3); RK2 error decreases on refinement |
| 05 | [Potential sweep — reversible CV](notebooks/05_potential_sweep_reversible.ipynb) | done | peak current vs Randles–Sevcik (< 5e-3) |
| 06 | [Potential sweep — quasi/non-reversible](notebooks/06_potential_sweep_nonreversible.ipynb) | done | Cottrell certification + reversible limit vs Nicholson–Shain 0.4463 |
| 07 | [AC voltammetry](notebooks/07_ac_voltammetry.ipynb) | done | fundamental-harmonic peak ≈ 1/4 located at E⁰ |
| 08 | [Potential steps and pulses](notebooks/08_potential_steps_and_pulses.ipynb) | done | dimensionless & physical Cottrell match (< 1e-2); first-order convergence |
| 09 | [Chronopotentiometry](notebooks/09_chronopotentiometry.ipynb) | done | Sand product/constant; wave-shape RMSE bound |
| 10 | [Thin layers and thin films](notebooks/10_thin_layers_and_films.ipynb) | done | thin-layer peak height, voltammogram symmetry, absence of diffusional tail |
| 11 | [Strongly adsorbed molecules](notebooks/11_adsorbed_species.ipynb) | done | surface-wave peak ψ ≈ 1/4 at E⁰, symmetric |
| 12 | [Monte Carlo simulations](notebooks/12_monte_carlo.ipynb) | done | MSD = 2 D t within statistical tolerance; Cottrell t^(−1/2) slope |
| 13 | [Coupled chemical reactions](notebooks/13_coupled_chemical_reactions.ipynb) | done | no-reaction limit matches Ch. 5 to machine precision; monotone grid convergence |
| 14 | [Rotating disk electrode voltammetry](notebooks/14_rotating_disk_electrode.ipynb) | done | Levich magnitude (< 5e-3) and Levich-plot linearity R² > 0.9999 |
| 15 | [Finite differences with sparse arrays](notebooks/15_sparse_finite_differences.ipynb) | done | sparse solver matches dense FD to < 1e-11 |
| 16 | [Processing experimental data](notebooks/16_processing_experimental_data.ipynb) | done | smoothing reduces RMS; Savitzky–Golay preserves peak position/height |
| App. A | [Python for electrochemical simulation](notebooks/A_appendix_a_python_refresher.ipynb) | done | numpy-vs-list equivalence asserts throughout |
| App. B | [The `serm` package reference (generated)](notebooks/appendix_b_serm_reference.ipynb) | done | auto-rendered signatures/docstrings + one runnable example per module |

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
  [Usage](#usage); a clean run with no `CellExecutionError`
  means every chapter's asserts held. Appendix B re-imports the package and
  re-renders the API reference, so executing it confirms the documented
  signatures and the one-example-per-module smoke tests still pass.

Chapter authors: follow [`tools/AUTHORING_SPEC.md`](tools/AUTHORING_SPEC.md).

## License

Released under the MIT License. See [`LICENSE`](LICENSE). The MIT license applies
to this independent Python re-implementation only; the original book and its
Mathematica notebooks remain the work of Michael Honeychurch.
</content>
</invoke>
