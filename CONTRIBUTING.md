# Contributing

This repository is a Python study port of Michael Honeychurch's *Simulating
Electrochemical Reactions in Mathematica*. Contributions — fixes, clearer
explanations, tighter validations, new worked examples — are welcome.

## Ground rules

- **Every notebook must execute cleanly, end to end.** CI runs all notebooks on
  each push and pull request (`.github/workflows/ci.yml`) and fails on any error
  or failed `assert`. A pull request needs to keep CI green.
- **Keep the validation honest.** Each chapter checks itself with `assert`s
  against a closed-form or independently-computed reference. If you change the
  science, keep (or add) such a check — don't loosen a validation just to make it
  pass.
- **Reuse the package.** Shared code lives in the `serm/` package (solvers,
  waveforms, filters, plotting, and the `echem` library of analytic references).
  Import from it rather than re-implementing.

## Setup

```bash
git clone https://github.com/NGeorgescu/simulating-electrochemical-reactions-in-python.git
cd simulating-electrochemical-reactions-in-python

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

## Running the checks locally

Reproduce what CI does:

```bash
for nb in notebooks/*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

A clean run means every chapter's `assert`s held.

## Reporting issues

Open a GitHub issue with the chapter/notebook, what you expected, and the error
or the failing `assert`. The notebooks render directly on GitHub, so feel free to
link the specific cell.
