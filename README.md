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

![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

All 16 chapters and both appendices are complete and self-validating: every
notebook runs end-to-end and checks itself against a closed-form or
independently-computed result. Jump in via the
[table of contents](#table-of-contents).

## Installation

```bash
git clone https://github.com/NGeorgescu/simulating-electrochemical-reactions-in-python.git
cd simulating-electrochemical-reactions-in-python

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Launch JupyterLab and open any chapter:

```bash
.venv/bin/jupyter lab
```

To run every notebook end-to-end (headless):

```bash
for nb in notebooks/*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

A clean run means every chapter's validation `assert`s held.

## What's inside

```
simulating-electrochemical-reactions-in-python/
  README.md
  LICENSE                    # MIT
  requirements.txt
  tools/
    nb_extract.py            # box-format .nb -> readable text (translation aid)
    build_notebook.py        # builds the Chapter 2 notebook with nbformat
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

## Table of contents

The book is re-organised into a Python-native sequence: the original Mathematica
introduction becomes a Python primer (Appendix A), and a generated `serm`
reference is added as Appendix B. Each chapter validates itself against a
closed-form or independently-computed result; the right-hand column names that
check.

| Ch. | Title | Validation method (in-notebook asserts) |
|----:|-------|------------------------------------------|
| 01 | [Solving partial differential equations](notebooks/01_solving_pdes.ipynb) | sympy symbolic checks: separation-of-variables residual ≡ 0; Fourier coefficient A₁ = 4/π |
| 02 | [Explicit finite differences](notebooks/02_explicit_finite_differences.ipynb) | Cottrell match (mean rel err < 5e-3); demonstrates D_M > 0.5 instability |
| 03 | [Speed and accuracy: implicit & Crank–Nicolson](notebooks/03_speed_and_accuracy.ipynb) | convergence orders: backward-Euler ≈ 1, Crank–Nicolson ≈ 2 |
| 04 | [Other numerical methods: Runge–Kutta, Volterra, SOR](notebooks/04_other_numerical_methods.ipynb) | RK/Volterra peak vs Randles–Sevcik constant (< 5e-3); RK2 error decreases on refinement |
| 05 | [Potential sweep — reversible CV](notebooks/05_potential_sweep_reversible.ipynb) | peak current vs Randles–Sevcik (< 5e-3) |
| 06 | [Potential sweep — quasi/non-reversible](notebooks/06_potential_sweep_nonreversible.ipynb) | Cottrell certification + reversible limit vs Nicholson–Shain 0.4463 |
| 07 | [AC voltammetry](notebooks/07_ac_voltammetry.ipynb) | fundamental-harmonic peak ≈ 1/4 located at E⁰ |
| 08 | [Potential steps and pulses](notebooks/08_potential_steps_and_pulses.ipynb) | dimensionless & physical Cottrell match (< 1e-2); first-order convergence |
| 09 | [Chronopotentiometry](notebooks/09_chronopotentiometry.ipynb) | Sand product/constant; wave-shape RMSE bound |
| 10 | [Thin layers and thin films](notebooks/10_thin_layers_and_films.ipynb) | thin-layer peak height, voltammogram symmetry, absence of diffusional tail |
| 11 | [Strongly adsorbed molecules](notebooks/11_adsorbed_species.ipynb) | surface-wave peak ψ ≈ 1/4 at E⁰, symmetric |
| 12 | [Monte Carlo simulations](notebooks/12_monte_carlo.ipynb) | MSD = 2 D t within statistical tolerance; Cottrell t^(−1/2) slope |
| 13 | [Coupled chemical reactions](notebooks/13_coupled_chemical_reactions.ipynb) | no-reaction limit matches Ch. 5 to machine precision; monotone grid convergence |
| 14 | [Rotating disk electrode voltammetry](notebooks/14_rotating_disk_electrode.ipynb) | Levich magnitude (< 5e-3) and Levich-plot linearity R² > 0.9999 |
| 15 | [Finite differences with sparse arrays](notebooks/15_sparse_finite_differences.ipynb) | sparse solver matches dense FD to < 1e-11 |
| 16 | [Processing experimental data](notebooks/16_processing_experimental_data.ipynb) | smoothing reduces RMS; Savitzky–Golay preserves peak position/height |
| App. A | [Python for electrochemical simulation](notebooks/A_appendix_a_python_refresher.ipynb) | numpy-vs-list equivalence asserts throughout |
| App. B | [The `serm` package reference (generated)](notebooks/appendix_b_serm_reference.ipynb) | auto-rendered signatures/docstrings + one runnable example per module |

## License

Released under the MIT License. See [`LICENSE`](LICENSE). The MIT license applies
to this independent Python re-implementation only; the original book and its
Mathematica notebooks remain the work of Michael Honeychurch.
</content>
</invoke>
