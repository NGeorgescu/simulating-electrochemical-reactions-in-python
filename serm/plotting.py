"""Shared matplotlib helpers for the SERM Python notebooks.

Replacements for the Wolfram ``ListPlot``/``ListPlot3D``/``Animate`` calls used
in ``ExplicitFD.nb``: concentration profiles, the concentration surface, the
current transient, and an animation of the evolving profile.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_profiles(c, taus, x, ax=None):
    """Plot concentration vs. distance at several dimensionless times.

    Parameters
    ----------
    c : ndarray, shape (m, n)
        Concentration grid (space x time).
    taus : sequence of float
        Dimensionless times (0..1) at which to draw a profile.
    x : ndarray, shape (m,)
        Dimensionless distance coordinate.
    ax : matplotlib axis, optional
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    n = c.shape[1]
    for tau in taus:
        k = min(int(round(tau * (n - 1))), n - 1)
        ax.plot(x, c[:, k], label=f"$\\tau$ = {tau:.2f}")
    ax.set_xlabel("dimensionless distance $x$")
    ax.set_ylabel("dimensionless concentration $c$")
    ax.set_title("Concentration profiles of O")
    ax.legend(fontsize=8)
    return ax


def plot_surface(c, x, ax=None):
    """3-D surface of concentration over distance and time (replaces ListPlot3D).

    Parameters
    ----------
    c : ndarray, shape (m, n)
    x : ndarray, shape (m,)
        Dimensionless distance.
    """
    m, n = c.shape
    tau = np.linspace(0.0, 1.0, n)
    X, T = np.meshgrid(x, tau, indexing="ij")  # both shape (m, n)
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, c, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("distance $x$")
    ax.set_ylabel("time $\\tau$")
    ax.set_zlabel("concentration $c$")
    ax.set_title("Concentration $c(x, \\tau)$")
    return ax


def plot_current(tau, i_sim, i_cottrell=None, ax=None):
    """Plot the dimensionless current transient, optionally with Cottrell."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(tau, i_sim, "r-", lw=1.5, label="explicit FD")
    if i_cottrell is not None:
        ax.plot(tau, i_cottrell, "k--", lw=1, label="Cottrell $1/\\sqrt{\\pi\\tau}$")
    ax.set_xlabel("dimensionless time $\\tau$")
    ax.set_ylabel("dimensionless current $|i|$")
    ax.set_title("Diffusion-limited current transient")
    ax.set_ylim(0, min(np.nanmax(i_sim[1:]) * 1.2, 6))
    ax.legend(fontsize=8)
    return ax


def animate_profiles(c, x, step=4, interval=60):
    """Return a FuncAnimation of the evolving concentration profile.

    Parameters
    ----------
    c : ndarray, shape (m, n)
    x : ndarray, shape (m,)
    step : int
        Plot every ``step``-th time slice.
    interval : int
        Delay between frames (ms).
    """
    m, n = c.shape
    frames = range(0, n, step)
    fig, ax = plt.subplots(figsize=(5, 4))
    (line,) = ax.plot(x, c[:, 0], "r-", lw=1.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("dimensionless distance $x$")
    ax.set_ylabel("dimensionless concentration $c$")
    title = ax.set_title("$\\tau$ = 0.000")

    def update(k):
        line.set_ydata(c[:, k])
        title.set_text(f"$\\tau$ = {k / (n - 1):.3f}")
        return line, title

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=False
    )
    plt.close(fig)  # prevent duplicate static display in notebooks
    return anim
