"""Build the chapter-4 extra notebooks (block Thomas, Volterra second-kind).

Run with the project venv:
    .venv/bin/python tools/_build_ch04_extras.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

KERNEL = {
    "display_name": "Python 3 (serm venv)",
    "language": "python",
    "name": "python3",
}

SETUP = '''import os, sys

# Walk up from the notebook's working directory to the repo root (the directory
# that contains the ``serm`` package); works whether this notebook is run from
# notebooks/ or notebooks/extras/.
_d = os.getcwd()
while not os.path.isdir(os.path.join(_d, "serm")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

# %matplotlib inline embeds figures and makes plt.show() a harmless no-op
# under headless (Agg) execution.
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import serm
from serm import echem
from serm.tridiagonal import tridiag_solve

np.set_printoptions(precision=6, suppress=True)
'''


def write(nb, path):
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    with open(path, "w") as fh:
        nbf.write(nb, fh)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Notebook 1: block (extended) Thomas algorithm
# ---------------------------------------------------------------------------
def build_block_thomas():
    nb = new_notebook()
    c = nb.cells

    c.append(new_markdown_cell(
        "# Chapter 4 (extra) — The block (extended) Thomas algorithm\n"
        "\n"
        "The scalar Thomas algorithm of Chapter 4 solves a tridiagonal system\n"
        "$A\\,u = b$ in $O(n)$ work by an LU sweep without pivoting. When several\n"
        "species are coupled at every grid node — for example $O$, $R$ and an\n"
        "intermediate in a homogeneous chemical reaction, or the real and\n"
        "imaginary parts of an AC problem — the unknown at each node is itself a\n"
        "*vector* and the matrix becomes **block-tridiagonal**: each scalar entry\n"
        "$x_i, y_i, z_i$ is replaced by a $p\\times p$ block $\\mathbf{X}_i,\n"
        "\\mathbf{Y}_i, \\mathbf{Z}_i$, and each $b_i$ by a length-$p$ vector.\n"
        "\n"
        "Honeychurch (SERM §4.3, *Extending the Thomas algorithm*) notes that\n"
        "Rudolph's general simulator (the basis of DigiSim) uses exactly this\n"
        "block form. The derivation is the scalar one with every division\n"
        "replaced by a matrix inverse and every product by a matrix product. This\n"
        "notebook re-implements it idiomatically in NumPy as a 3-D-array engine\n"
        "(`X`, `Y`, `Z` of shape `(m, p, p)`), which the later coupled chapters\n"
        "build on.\n"
    ))

    c.append(new_code_cell(SETUP))

    c.append(new_markdown_cell(
        "## 1. Block LU without pivoting\n"
        "\n"
        "Write the block system as $\\mathbf{M}\\,\\mathbf{u}=\\mathbf{b}$ with\n"
        "block sub-, main- and super-diagonals $\\mathbf{X}_i,\\mathbf{Y}_i,\n"
        "\\mathbf{Z}_i$. Factor $\\mathbf{M}=\\mathbf{L}\\,\\mathbf{U}$ where\n"
        "$\\mathbf{U}$ has identity blocks on its diagonal and $\\mathbf{Z}_i$\n"
        "above it, and $\\mathbf{L}$ has the pivot blocks\n"
        "$\\boldsymbol{\\alpha}_i$ on its diagonal and $\\mathbf{X}_i$ below.\n"
        "Matching blocks gives the recurrences\n"
        "\n"
        "$$\\boldsymbol{\\alpha}_1=\\mathbf{Y}_1,\\qquad\n"
        "\\boldsymbol{\\alpha}_i=\\mathbf{Y}_i-\\mathbf{X}_i\\,\n"
        "\\boldsymbol{\\alpha}_{i-1}^{-1}\\,\\mathbf{Z}_{i-1}.$$\n"
        "\n"
        "**Forward substitution** ($\\mathbf{L}\\,\\mathbf{f}=\\mathbf{b}$):\n"
        "\n"
        "$$\\mathbf{f}_1=\\boldsymbol{\\alpha}_1^{-1}\\mathbf{b}_1,\\qquad\n"
        "\\mathbf{f}_i=\\boldsymbol{\\alpha}_i^{-1}\n"
        "\\bigl(\\mathbf{b}_i-\\mathbf{X}_i\\,\\mathbf{f}_{i-1}\\bigr).$$\n"
        "\n"
        "**Back substitution** ($\\mathbf{U}\\,\\mathbf{u}=\\mathbf{f}$):\n"
        "\n"
        "$$\\mathbf{u}_m=\\mathbf{f}_m,\\qquad\n"
        "\\mathbf{u}_i=\\mathbf{f}_i-\\boldsymbol{\\alpha}_i^{-1}\\,\n"
        "\\mathbf{Z}_i\\,\\mathbf{u}_{i+1}.$$\n"
        "\n"
        "For $p=1$ every block is a scalar and these collapse to the ordinary\n"
        "Thomas recurrences — the property we exploit for validation. Rather than\n"
        "forming each $\\boldsymbol{\\alpha}_i^{-1}$ explicitly we solve the small\n"
        "$p\\times p$ systems with `numpy.linalg.solve`, which is both faster and\n"
        "better conditioned than an explicit inverse.\n"
    ))

    c.append(new_code_cell(
        'def block_tridiag_solve(\n'
        '    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, b: np.ndarray\n'
        ') -> np.ndarray:\n'
        '    """Solve a block-tridiagonal system ``M u = b`` (extended Thomas).\n'
        '\n'
        '    Parameters\n'
        '    ----------\n'
        '    X : ndarray, shape (m-1, p, p)\n'
        '        Block sub-diagonal: ``X[i]`` is the block ``M[i+1, i]``.\n'
        '    Y : ndarray, shape (m, p, p)\n'
        '        Block main diagonal.\n'
        '    Z : ndarray, shape (m-1, p, p)\n'
        '        Block super-diagonal: ``Z[i]`` is the block ``M[i, i+1]``.\n'
        '    b : ndarray, shape (m, p)\n'
        '        Right-hand side, one length-``p`` vector per block row.\n'
        '\n'
        '    Returns\n'
        '    -------\n'
        '    ndarray, shape (m, p)\n'
        '        Solution blocks ``u``.\n'
        '\n'
        '    Notes\n'
        '    -----\n'
        '    No pivoting (matching the scalar Thomas algorithm); intended for the\n'
        '    diagonally dominant blocks that arise from implicit finite-difference\n'
        '    diffusion-reaction problems. For ``p == 1`` this reduces exactly to\n'
        '    the scalar Thomas algorithm.\n'
        '    """\n'
        '    Y = np.asarray(Y, dtype=float)\n'
        '    b = np.asarray(b, dtype=float)\n'
        '    m, p, _ = Y.shape\n'
        '    if X.shape[0] != m - 1 or Z.shape[0] != m - 1:\n'
        '        raise ValueError("X and Z must have m-1 blocks")\n'
        '    if b.shape != (m, p):\n'
        '        raise ValueError("b must have shape (m, p)")\n'
        '\n'
        '    alpha = np.empty((m, p, p))\n'
        '    f = np.empty((m, p))\n'
        '    alpha[0] = Y[0]\n'
        '    f[0] = np.linalg.solve(alpha[0], b[0])\n'
        '    for i in range(1, m):\n'
        '        # alpha_i = Y_i - X_i . alpha_{i-1}^{-1} . Z_{i-1}\n'
        '        w = np.linalg.solve(alpha[i - 1], Z[i - 1])   # alpha_{i-1}^{-1} Z_{i-1}\n'
        '        alpha[i] = Y[i] - X[i - 1] @ w\n'
        '        f[i] = np.linalg.solve(alpha[i], b[i] - X[i - 1] @ f[i - 1])\n'
        '\n'
        '    u = np.empty((m, p))\n'
        '    u[-1] = f[-1]\n'
        '    for i in range(m - 2, -1, -1):\n'
        '        # u_i = f_i - alpha_i^{-1} . Z_i . u_{i+1}\n'
        '        u[i] = f[i] - np.linalg.solve(alpha[i], Z[i] @ u[i + 1])\n'
        '    return u\n'
    ))

    c.append(new_markdown_cell(
        "## 2. A worked $2\\times2$-block example\n"
        "\n"
        "A coupled pair $O\\rightleftharpoons R$ diffusing with a first-order\n"
        "homogeneous interconversion gives, after an implicit (backward-Euler)\n"
        "discretisation, exactly such a block system. Here we build a small\n"
        "deterministic block-tridiagonal matrix with $p=2$, $m=6$ and check the\n"
        "engine against an explicit dense solve assembled from the same blocks.\n"
    ))

    c.append(new_code_cell(
        'rng = np.random.default_rng(4)\n'
        'm, p = 6, 2\n'
        '\n'
        '# Build diagonally dominant random blocks so the no-pivot solve is stable.\n'
        'def diag_dominant_block(p):\n'
        '    A = rng.standard_normal((p, p))\n'
        '    A += p * np.eye(p)                      # push weight onto the diagonal\n'
        '    return A\n'
        '\n'
        'Y = np.stack([3.0 * diag_dominant_block(p) for _ in range(m)])\n'
        'X = np.stack([0.4 * rng.standard_normal((p, p)) for _ in range(m - 1)])\n'
        'Z = np.stack([0.4 * rng.standard_normal((p, p)) for _ in range(m - 1)])\n'
        'b = rng.standard_normal((m, p))\n'
        '\n'
        'u_block = block_tridiag_solve(X, Y, Z, b)\n'
        '\n'
        '# Assemble the equivalent dense (m*p) x (m*p) matrix for a reference solve.\n'
        'M = np.zeros((m * p, m * p))\n'
        'for i in range(m):\n'
        '    M[i*p:(i+1)*p, i*p:(i+1)*p] = Y[i]\n'
        'for i in range(m - 1):\n'
        '    M[(i+1)*p:(i+2)*p, i*p:(i+1)*p] = X[i]        # sub-diagonal block\n'
        '    M[i*p:(i+1)*p, (i+1)*p:(i+2)*p] = Z[i]        # super-diagonal block\n'
        'u_dense = np.linalg.solve(M, b.reshape(-1)).reshape(m, p)\n'
        '\n'
        'err = np.max(np.abs(u_block - u_dense))\n'
        'print(f"max |block-Thomas - dense solve| = {err:.2e}")\n'
    ))

    c.append(new_markdown_cell(
        "## 3. Reduction to the scalar Thomas algorithm ($p=1$)\n"
        "\n"
        "The strongest internal check is the reduction-to-validated-limit: with\n"
        "$1\\times1$ blocks the engine must reproduce, bit-for-bit up to rounding,\n"
        "the already-validated scalar `serm.tridiagonal.tridiag_solve`. We use the\n"
        "same constant-diagonal test system as the scalar module's own port\n"
        "(sub/super $=-1$, main $=2$).\n"
    ))

    c.append(new_code_cell(
        'n = 50\n'
        'x = -np.ones(n - 1)          # scalar sub-diagonal\n'
        'y = 2.0 * np.ones(n)         # scalar main diagonal\n'
        'z = -np.ones(n - 1)          # scalar super-diagonal\n'
        'b_scalar = np.ones(n)\n'
        'b_scalar[-1] = 1.5\n'
        '\n'
        'u_scalar = tridiag_solve(x, y, z, b_scalar)\n'
        '\n'
        '# Same system promoted to 1x1 blocks.\n'
        'X1 = x.reshape(n - 1, 1, 1)\n'
        'Y1 = y.reshape(n, 1, 1)\n'
        'Z1 = z.reshape(n - 1, 1, 1)\n'
        'b1 = b_scalar.reshape(n, 1)\n'
        'u_1block = block_tridiag_solve(X1, Y1, Z1, b1).ravel()\n'
        '\n'
        'reduction_err = np.max(np.abs(u_1block - u_scalar))\n'
        'print(f"max |block(p=1) - scalar Thomas| = {reduction_err:.2e}")\n'
    ))

    c.append(new_markdown_cell(
        "## 4. Validation\n"
        "\n"
        "Two assert-backed checks, strongest first per the project validation\n"
        "policy:\n"
        "\n"
        "1. **Reduction to a validated limit (tier 2).** For $p=1$ the block\n"
        "   engine reproduces `serm.tridiagonal.tridiag_solve` (itself a port of\n"
        "   the SERM scalar solver) to machine precision.\n"
        "2. **Two-implementation cross-check (tier 3).** For a $p=2$ block system\n"
        "   the engine agrees with an independent dense `numpy.linalg.solve` of\n"
        "   the assembled matrix to machine precision.\n"
    ))

    c.append(new_code_cell(
        '# --- Validation 1: reduction to the scalar Thomas algorithm (tier 2) ---\n'
        'assert reduction_err < 1e-10, "block(p=1) disagrees with scalar Thomas"\n'
        'print(f"PASS: block engine with p=1 reproduces scalar Thomas "\n'
        '      f"(max err {reduction_err:.2e}).")\n'
        '\n'
        '# --- Validation 2: block solve vs. dense reference (tier 3) ---\n'
        'assert err < 1e-9, "block solve disagrees with dense reference"\n'
        'print(f"PASS: 2x2-block solve matches dense numpy.linalg.solve "\n'
        '      f"(max err {err:.2e}).")\n'
    ))

    c.append(new_markdown_cell(
        "## 5. Summary\n"
        "\n"
        "The block (extended) Thomas algorithm generalises the scalar LU sweep to\n"
        "block-tridiagonal systems by replacing scalar division with a small\n"
        "$p\\times p$ solve. The NumPy implementation above (`block_tridiag_solve`)\n"
        "is the engine that the coupled-reaction (Chapter 13) and multi-species\n"
        "problems require: each grid node carries a length-$p$ concentration\n"
        "vector and the implicit step is a single block solve. It was validated by\n"
        "reduction to the already-validated scalar solver ($p=1$) and by an\n"
        "independent dense cross-check ($p=2$).\n"
    ))

    write(nb, "notebooks/extras/04_block_thomas.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2: Volterra equations of the second kind -> irreversible &
# quasi-reversible cyclic voltammetry (Huber / Nicholson-Olmstead)
# ---------------------------------------------------------------------------
def build_volterra_second_kind():
    nb = new_notebook()
    c = nb.cells

    c.append(new_markdown_cell(
        "# Chapter 4 (extra) — Volterra equations of the second kind\n"
        "\n"
        "The main Chapter 4 notebook solves a *first-kind* Volterra equation\n"
        "$\\int_0^t K(t-z)f(z)\\,dz = g(t)$ and obtains the **reversible** cyclic\n"
        "voltammogram. When electron transfer is *not* fast the surface boundary\n"
        "condition becomes Butler-Volmer rather than Nernstian, and the integral\n"
        "equation picks up the unknown $f$ on *both* sides — a **second-kind**\n"
        "Volterra equation,\n"
        "\n"
        "$$f(t) = g(t) + \\int_0^t K(t-z)\\,f(z)\\,dz .$$\n"
        "\n"
        "Honeychurch (SERM §4.7) notes that, after applying Huber's piecewise\n"
        "approximation, the second-kind discrete equation differs from the\n"
        "first-kind one *only in the leading right-hand-side term*. The kernel\n"
        "weights are the same $k^{3/2}-(k-1)^{3/2}$ Huber weights. This notebook\n"
        "re-implements the second-kind recurrence for the two classic cases of\n"
        "Nicholson & Olmstead (1972): a **totally irreversible** wave and a\n"
        "**quasi-reversible** wave, and validates each against an independent\n"
        "closed-form peak coefficient.\n"
    ))

    c.append(new_code_cell(SETUP))

    c.append(new_markdown_cell(
        "## 1. The Huber weights\n"
        "\n"
        "For the planar diffusion kernel $K(y)=1/\\sqrt{\\pi y}$, integrating the\n"
        "piecewise-constant $f$ over each step of width $d$ produces the Huber\n"
        "weight for a lag of $k$ steps,\n"
        "\n"
        "$$w_k = k^{3/2}-(k-1)^{3/2},$$\n"
        "\n"
        "with the leading (self) coefficient $h_1 = \\tfrac{4}{3}\\,d^{3/2}$\n"
        "(the same $\\tfrac43 d^{3/2}$ that appears in the first-kind solver of the\n"
        "main notebook). Below, the convolution sum\n"
        "$S^{(1)}_m=\\sum_{i<m} w_{m-i}\\,a_i$ is what couples step $m$ to its\n"
        "history; we evaluate it with a vectorised dot product.\n"
    ))

    c.append(new_code_cell(
        'def huber_lag_weights(m: int) -> np.ndarray:\n'
        '    """Huber lag weights ``w_k = k**1.5 - (k-1)**1.5`` for k = m-1 .. 1.\n'
        '\n'
        '    Returns the weights already ordered to multiply the history\n'
        '    ``a[0], a[1], ..., a[m-2]`` (i.e. lag decreasing from ``m-1`` to 1).\n'
        '    """\n'
        '    ii = np.arange(1, m)                 # i = 1 .. m-1\n'
        '    lag = m - ii                         # k = m-i\n'
        '    return lag ** 1.5 - (lag - 1) ** 1.5\n'
    ))

    c.append(new_markdown_cell(
        "## 2. Totally irreversible cyclic voltammetry\n"
        "\n"
        "For a totally irreversible reduction $O + n e^- \\to R$ only the forward\n"
        "Butler-Volmer term survives. Following Nicholson & Olmstead via SERM\n"
        "§4.7, working on the dimensionless potential axis\n"
        "$\\alpha\\,\\tfrac{nF}{RT}(E-E_0)$ with step $\\Delta e$ and a rate group\n"
        "\n"
        "$$g_m = \\frac{\\sqrt{\\pi\\,\\alpha\\,(nF/RT)\\,v\\,D}}{k_i}\\,\n"
        "e^{-m\\alpha\\Delta e},\\qquad k_i = k_s\\,e^{-\\alpha p_0},$$\n"
        "\n"
        "the second-kind recurrence for the interval slopes $a_m$ is\n"
        "\n"
        "$$a_m = \\frac{1}{g_m\\alpha\\Delta e + h_1}\n"
        "\\Bigl(1 - h_1\\!\\sum_{i<m} w_{m-i}\\,a_i\n"
        "- g_m\\alpha\\Delta e\\!\\sum_{i<m} a_i\\Bigr),$$\n"
        "\n"
        "and the dimensionless current function is the running sum\n"
        "$\\chi_m = \\sqrt{\\pi}\\,\\alpha\\Delta e\\sum_{i\\le m} a_i$, with the\n"
        "Huber self-weight $h_1 = \\tfrac43 (\\alpha\\Delta e)^{3/2}$ (the kernel\n"
        "step width on this axis is $\\alpha\\Delta e$). The forward peak of $\\chi$\n"
        "is the **Nicholson-Shain totally-irreversible coefficient $0.4958$** —\n"
        "our validation target.\n"
    ))

    c.append(new_code_cell(
        'def irreversible_cv(\n'
        '    n: int = 2000, de: float = 0.02, ks: float = 1e-5,\n'
        '    alpha: float = 0.5, D: float = 1e-5, v: float = 1.0, p0: float = 10.0,\n'
        ') -> tuple[np.ndarray, np.ndarray]:\n'
        '    """Totally irreversible CV via the second-kind Volterra equation.\n'
        '\n'
        '    Parameters\n'
        '    ----------\n'
        '    n : int\n'
        '        Number of potential steps.\n'
        '    de : float\n'
        '        Dimensionless potential step on the ``alpha*(nF/RT)(E-E0)`` axis.\n'
        '    ks : float\n'
        '        Standard heterogeneous rate constant (cm/s).\n'
        '    alpha : float\n'
        '        Transfer coefficient.\n'
        '    D : float\n'
        '        Diffusion coefficient (cm^2/s).\n'
        '    v : float\n'
        '        Sweep rate (V/s).\n'
        '    p0 : float\n'
        '        Dimensionless start potential (units anodic of E0).\n'
        '\n'
        '    Returns\n'
        '    -------\n'
        '    (potential, chi) : tuple of ndarray, shape (n,)\n'
        '        Dimensionless potential ``alpha*(nF/RT)(E-E0)`` and current\n'
        '        function ``chi``.\n'
        '    """\n'
        '    F, R, T = echem.F, echem.R, 298.15\n'
        '    ki = ks * np.exp(-alpha * p0)\n'
        '    # Huber self-weight on the irreversible axis: the step width is the\n'
        '    # dimensionless increment alpha*de, hence (alpha*de)**1.5.\n'
        '    h1 = (4.0 / 3.0) * (alpha * de) ** 1.5\n'
        '    pref = np.sqrt(np.pi * alpha * (F / (R * T)) * v * D) / ki\n'
        '    a = np.zeros(n)\n'
        '    for m in range(1, n + 1):\n'
        '        g = pref * np.exp(-m * alpha * de)\n'
        '        if m > 1:\n'
        '            w = huber_lag_weights(m)\n'
        '            s1 = np.dot(a[:m - 1], w)        # history convolution\n'
        '            s2 = a[:m - 1].sum()\n'
        '        else:\n'
        '            s1 = s2 = 0.0\n'
        '        gade = g * alpha * de\n'
        '        a[m - 1] = (1.0 - h1 * s1 - gade * s2) / (gade + h1)\n'
        '    chi = np.sqrt(np.pi) * alpha * de * np.cumsum(a)\n'
        '    potential = p0 - alpha * de * np.arange(1, n + 1)\n'
        '    return potential, chi\n'
    ))

    c.append(new_code_cell(
        'pot_irr, chi_irr = irreversible_cv()\n'
        'ip_irr = chi_irr.max()\n'
        'print(f"irreversible peak chi = {ip_irr:.5f}  "\n'
        '      f"(Nicholson-Shain coefficient 0.4958)")\n'
        '\n'
        'fig, ax = plt.subplots(figsize=(5.6, 4))\n'
        'ax.plot(pot_irr, chi_irr, "b-", lw=1.3)\n'
        'ax.axhline(0.4958, color="0.5", ls=":", lw=1.0,\n'
        '           label=r"$\\chi_p = 0.4958$ (irreversible)")\n'
        'ax.invert_xaxis()\n'
        'ax.set_xlabel(r"$\\alpha\\,\\frac{nF}{RT}(E - E_0)$")\n'
        'ax.set_ylabel(r"current function $\\chi$")\n'
        'ax.set_title("Totally irreversible CV (second-kind Volterra)")\n'
        'ax.legend(fontsize=8)\n'
        'fig.tight_layout()\n'
        'plt.show()\n'
    ))

    c.append(new_markdown_cell(
        "## 3. Quasi-reversible cyclic voltammetry\n"
        "\n"
        "For a quasi-reversible couple both Butler-Volmer terms survive and the\n"
        "wave shape depends on the dimensionless kinetic parameter\n"
        "\n"
        "$$\\psi = \\frac{k_s}{\\sqrt{\\pi\\,(nF/RT)\\,v\\,D}} .$$\n"
        "\n"
        "Working on the reversible potential axis $\\tfrac{nF}{RT}(E-E_0)$ with\n"
        "step $\\Delta e$, $g_m = e^{-m\\Delta e}$, $c_1 = e^{\\alpha p_0}/\\psi$\n"
        "and $c_2 = e^{p_0}$, the second-kind recurrence (SERM §4.7, after\n"
        "Nicholson & Olmstead 1972) is\n"
        "\n"
        "$$a_m = \\frac{1 - (1+c_2 g_m)\\,h_1\\sum_{i<m} w_{m-i}a_i\n"
        "- c_1 g_m^{\\alpha}\\Delta e\\sum_{i<m} a_i}\n"
        "{c_1 g_m^{\\alpha}\\Delta e + h_1 + c_2 g_m h_1},\n"
        "\\qquad h_1=\\tfrac43\\Delta e^{3/2}.$$\n"
        "\n"
        "As $k_s\\to\\infty$, $\\psi\\to\\infty$ so $c_1\\to0$ and the equation\n"
        "collapses to the *first-kind* reversible solver: the peak must approach\n"
        "the **Randles-Sevcik constant $0.4463$**. That is the\n"
        "reduction-to-validated-limit check for this solver.\n"
    ))

    c.append(new_code_cell(
        'def quasireversible_cv(\n'
        '    ks: float, n: int = 1500, de: float = 0.02,\n'
        '    alpha: float = 0.5, D: float = 1e-5, v: float = 1.0, p0: float = 10.0,\n'
        ') -> tuple[np.ndarray, np.ndarray]:\n'
        '    """Quasi-reversible CV via the second-kind Volterra equation.\n'
        '\n'
        '    Same parameters as :func:`irreversible_cv` except the potential axis\n'
        '    is the reversible ``(nF/RT)(E-E0)`` (no ``alpha`` factor). As\n'
        '    ``ks -> inf`` the result reduces to the Nernstian reversible CV.\n'
        '\n'
        '    Returns\n'
        '    -------\n'
        '    (potential, chi) : tuple of ndarray, shape (n,)\n'
        '    """\n'
        '    F, R, T = echem.F, echem.R, 298.15\n'
        '    psi = ks / np.sqrt(np.pi * (F / (R * T)) * v * D)\n'
        '    h1 = (4.0 / 3.0) * de ** 1.5\n'
        '    c1 = np.exp(alpha * p0) / psi\n'
        '    c2 = np.exp(p0)\n'
        '    a = np.zeros(n)\n'
        '    for m in range(1, n + 1):\n'
        '        g = np.exp(-m * de)\n'
        '        if m > 1:\n'
        '            w = huber_lag_weights(m)\n'
        '            s1 = np.dot(a[:m - 1], w)\n'
        '            s2 = a[:m - 1].sum()\n'
        '        else:\n'
        '            s1 = s2 = 0.0\n'
        '        c1g = c1 * g ** alpha * de\n'
        '        denom = c1g + h1 + c2 * g * h1\n'
        '        a[m - 1] = (1.0 - (1.0 + c2 * g) * h1 * s1 - c1g * s2) / denom\n'
        '    chi = np.sqrt(np.pi) * de * np.cumsum(a)\n'
        '    potential = p0 - de * np.arange(1, n + 1)\n'
        '    return potential, chi\n'
    ))

    c.append(new_code_cell(
        'ks_values = [1e-3, 1e-2, 1e-1, 1.0, 1e3]\n'
        'fig, ax = plt.subplots(figsize=(5.8, 4.2))\n'
        'peaks = {}\n'
        'for ks in ks_values:\n'
        '    pot_q, chi_q = quasireversible_cv(ks)\n'
        '    peaks[ks] = chi_q.max()\n'
        '    ax.plot(pot_q, chi_q, lw=1.2,\n'
        '            label=fr"$k_s={ks:g}$,  $\\chi_p={chi_q.max():.3f}$")\n'
        'ax.axhline(0.4463, color="0.5", ls=":", lw=1.0,\n'
        '           label=r"$\\chi_p=0.4463$ (reversible limit)")\n'
        'ax.invert_xaxis()\n'
        'ax.set_xlabel(r"$\\frac{nF}{RT}(E - E_0)$")\n'
        'ax.set_ylabel(r"current function $\\chi$")\n'
        'ax.set_title("Quasi-reversible CV vs. rate constant (second-kind Volterra)")\n'
        'ax.legend(fontsize=7)\n'
        'fig.tight_layout()\n'
        'plt.show()\n'
        '\n'
        'print("ks -> infinity peak:", f"{peaks[1e3]:.5f}",\n'
        '      "(Randles-Sevcik 0.4463)")\n'
    ))

    c.append(new_markdown_cell(
        "## 4. Validation\n"
        "\n"
        "Both checks are **tier 1 (independent closed-form coefficient)**, with\n"
        "the quasi-reversible one doubling as a **tier 2 reduction-to-validated\n"
        "limit** ($k_s\\to\\infty$ recovers the Nernstian reversible CV already\n"
        "validated in the main Chapter 4 notebook).\n"
        "\n"
        "1. The totally-irreversible current function peaks at the\n"
        "   **Nicholson-Shain coefficient $0.4958$** (Bard & Faulkner,\n"
        "   *Electrochemical Methods*, 2nd ed., irreversible LSV).\n"
        "2. As $k_s\\to\\infty$ the quasi-reversible solver reduces to the\n"
        "   reversible wave, peaking at the **Randles-Sevcik constant $0.4463$**,\n"
        "   cross-checked dimensionally against\n"
        "   `serm.echem.randles_sevcik_peak_current`. For decreasing $k_s$ the\n"
        "   peak must fall monotonically below the reversible value (the\n"
        "   signature of sluggish kinetics).\n"
    ))

    c.append(new_code_cell(
        '# --- Validation 1: irreversible peak == 0.4958 (tier 1, closed form) ---\n'
        'target_irr = 0.4958\n'
        'rel_irr = abs(ip_irr - target_irr) / target_irr\n'
        'print(f"irreversible peak = {ip_irr:.5f}, target = {target_irr}, "\n'
        '      f"rel. err = {rel_irr:.2e}")\n'
        'assert rel_irr < 5e-3, "irreversible peak does not match 0.4958"\n'
        'print("PASS: second-kind Volterra reproduces the irreversible 0.4958 peak.")\n'
    ))

    c.append(new_code_cell(
        '# --- Validation 2: ks->inf reduces to reversible Randles-Sevcik (tier 1/2) ---\n'
        'pot_rev, chi_rev = quasireversible_cv(ks=1e4)\n'
        'ip_rev = chi_rev.max()\n'
        'target_rev = 0.4463\n'
        'rel_rev = abs(ip_rev - target_rev) / target_rev\n'
        'print(f"ks->inf peak = {ip_rev:.5f}, Randles-Sevcik = {target_rev}, "\n'
        '      f"rel. err = {rel_rev:.2e}")\n'
        'assert rel_rev < 5e-3, "ks->inf peak does not match Randles-Sevcik 0.4463"\n'
        '\n'
        '# Dimensional cross-check against serm.echem (independent closed form).\n'
        'F, R = echem.F, echem.R\n'
        'n_e, A, D, c_bulk, v, Tk = 1, 1.0, 1e-5, 1e-6, 1.0, 298.15\n'
        'sigma = n_e * F * v / (R * Tk)\n'
        'ip_dim = n_e * F * A * c_bulk * np.sqrt(D * sigma) * ip_rev\n'
        'ip_closed = echem.randles_sevcik_peak_current(n_e, A, D, c_bulk, v,\n'
        '                                              temperature=Tk)\n'
        'rel_dim = abs(ip_dim - ip_closed) / ip_closed\n'
        'print(f"dimensional i_p (Volterra) = {ip_dim:.6e} A,  "\n'
        '      f"closed form = {ip_closed:.6e} A,  rel. err = {rel_dim:.2e}")\n'
        'assert rel_dim < 5e-3, "dimensional reversible peak disagrees with serm.echem"\n'
        '\n'
        '# Monotonic suppression of the peak as kinetics slow down.\n'
        'ordered = [peaks[k] for k in sorted(peaks)]   # increasing ks\n'
        'assert all(ordered[i] <= ordered[i + 1] + 1e-3 for i in range(len(ordered) - 1)), \\\n'
        '    "peak should grow monotonically toward the reversible limit as ks increases"\n'
        'assert ordered[0] < target_rev, "slowest-kinetics peak should sit below reversible"\n'
        'print("PASS: quasi-reversible solver reduces to Randles-Sevcik 0.4463 as "\n'
        '      "ks->inf; peak grows monotonically with ks.")\n'
    ))

    c.append(new_markdown_cell(
        "## 5. Summary\n"
        "\n"
        "The second-kind Volterra equation extends the integral-equation method to\n"
        "non-Nernstian electron transfer. A single change to the leading\n"
        "right-hand-side term of the first-kind recurrence yields:\n"
        "\n"
        "- a **totally irreversible** CV whose current function peaks at the\n"
        "  Nicholson-Shain coefficient $0.4958$; and\n"
        "- a **quasi-reversible** family parameterised by $\\psi$, which recovers\n"
        "  the reversible Randles-Sevcik peak $0.4463$ as $k_s\\to\\infty$ and\n"
        "  shows the expected peak suppression and cathodic shift as kinetics slow.\n"
        "\n"
        "Both were validated against independent closed-form peak coefficients,\n"
        "with the quasi-reversible $k_s\\to\\infty$ limit also serving as a\n"
        "reduction to the already-validated reversible solver.\n"
    ))

    write(nb, "notebooks/extras/04_volterra_second_kind.ipynb")


if __name__ == "__main__":
    build_block_thomas()
    build_volterra_second_kind()
