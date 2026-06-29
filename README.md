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

- **`notebooks/`** — the 18 chapter and appendix notebooks; this is the book.
- **`serm/`** — the shared package the notebooks import: finite-difference
  solvers (`tridiagonal`, `grids`), potential `waveforms`, `filters`, `plotting`
  helpers, `kinetics`, and an `echem` library of closed-form references used for
  validation.
- **`tools/nb_extract.py`** — converts the original Mathematica `.nb` files to
  plain text so the source algorithms can be read without Mathematica.

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
| 08 | [Potential steps and pulses](notebooks/08_potential_steps_and_pulses.ipynb) | Cottrell `1/√t` *shape* check (amplitude is matched at one time first, so it is self-consistent, not an independent prefactor match) < 1e-2; first-order convergence |
| 09 | [Chronopotentiometry](notebooks/09_chronopotentiometry.ipynb) | Sand product/constant; wave-shape RMSE bound |
| 10 | [Thin layers and thin films](notebooks/10_thin_layers_and_films.ipynb) | thin-layer peak height, voltammogram symmetry, absence of diffusional tail |
| 11 | [Strongly adsorbed molecules](notebooks/11_adsorbed_species.ipynb) | surface-wave peak ψ ≈ 1/4 at E⁰, symmetric |
| 12 | [Monte Carlo simulations](notebooks/12_monte_carlo.ipynb) | MSD = 2 D t within statistical tolerance; Cottrell t^(−1/2) slope |
| 13 | [Coupled chemical reactions](notebooks/13_coupled_chemical_reactions.ipynb) | sim-vs-sim self-consistency: no-reaction limit reproduces the Ch. 5 simulation to machine precision; monotone grid convergence to √λ |
| 14 | [Rotating disk electrode voltammetry](notebooks/14_rotating_disk_electrode.ipynb) | independent Levich magnitude vs `echem.levich_current` (< 5e-3) and Levich-plot linearity R² > 0.9999; the tight Koutecky–Levich assert is an algebraic identity from the same fit, not an independent check |
| 15 | [Finite differences with sparse arrays](notebooks/15_sparse_finite_differences.ipynb) | sparse solver matches dense FD to < 1e-11 |
| 16 | [Processing experimental data](notebooks/16_processing_experimental_data.ipynb) | smoothing reduces RMS; Savitzky–Golay preserves peak position/height |
| App. A | [Python for electrochemical simulation](notebooks/A_appendix_a_python_refresher.ipynb) | numpy-vs-list equivalence asserts throughout |
| App. A2 | [Semi-integration and fractional calculus](notebooks/A2_semiintegration.ipynb) | semi-integral plateau matches analytic value (< 2%); sigmoid round-trip correlation > 0.999; fractional power-rule identity (< 5e-3) |
| App. B | [The `serm` package reference (generated)](notebooks/appendix_b_serm_reference.ipynb) | auto-rendered signatures/docstrings + one runnable example per module |

## Additional methods

Beyond the main chapters, [`notebooks/extras/`](notebooks/extras/) collects
supplementary notebooks that port further algorithms and variants from the book,
grouped by the chapter they extend. Each validates itself in the same
assert-backed way and links back to its parent chapter.

**Chapter 1 — Solving PDEs**
- [Numerical inversion of Laplace transforms (Gaver–Stehfest)](notebooks/extras/01_stehfest_inversion.ipynb)

**Chapter 3 — Speed and accuracy**
- [Implicit solver on an expanding (non-uniform) grid](notebooks/extras/03_expanding_implicit_grid.ipynb)
- [Richtmyer higher-order time stepping](notebooks/extras/03_richtmyer.ipynb)

**Chapter 4 — Other numerical methods**
- [The block (extended) Thomas algorithm](notebooks/extras/04_block_thomas.ipynb)
- [Volterra equations of the second kind](notebooks/extras/04_volterra_second_kind.ipynb)

**Chapter 5 — Potential sweep, reversible**
- [Reversible CV by the method of lines](notebooks/extras/05_method_of_lines.ipynb)
- [Spherical-diffusion correction](notebooks/extras/05_spherical_diffusion.ipynb)

**Chapter 6 — Potential sweep, non-reversible**
- [Volterra 2nd-kind and FD variants for non-reversible CV](notebooks/extras/06_volterra_nonreversible.ipynb)

**Chapter 7 — AC voltammetry**
- [Quasi-reversible AC & square-wave voltammetry on an expanding grid](notebooks/extras/07_quasireversible_ac_sw.ipynb)

**Chapter 8 — Potential steps and pulses**
- [Staircase voltammetry](notebooks/extras/08_staircase_voltammetry.ipynb)

**Chapter 9 — Chronopotentiometry**
- [Current reversal and successive electron transfers](notebooks/extras/09_chronopot_reversal_and_EE.ipynb)

**Chapter 10 — Thin layers and thin films**
- [The analytical thin-layer / thin-film response](notebooks/extras/10_analytical_thin_layer.ipynb)

**Chapter 11 — Adsorbed species**
- [Marcus theory of electron transfer for adsorbed species](notebooks/extras/11_marcus_theory.ipynb)

**Chapter 13 — Coupled chemical reactions**
- [The square scheme: three ways to handle the cross reaction](notebooks/extras/13_square_scheme.ipynb)

**Chapter 14 — Rotating disk electrode**
- [The von Kármán rotating-disk velocity profile](notebooks/extras/14_velocity_profile.ipynb)

**Chapter 15 — Sparse finite differences**
- [FIRM: the Richtmyer / BDF4 sparse time scheme](notebooks/extras/15_firm_sparse.ipynb)
- [The square scheme on an expanding grid, sparse vs. dense](notebooks/extras/15_square_scheme_sparse.ipynb)

**Appendix A2 — Semi-integration**
- [Fractional differintegration of arbitrary order](notebooks/extras/A2_fractional_orders.ipynb)
- [Semi-integral voltammetry: peak → wave](notebooks/extras/A2_semiintegral_lsv.ipynb)

## License

Released under the MIT License. See [`LICENSE`](LICENSE). The MIT license applies
to this independent Python re-implementation only; the original book and its
Mathematica notebooks remain the work of Michael Honeychurch.
</content>
</invoke>
