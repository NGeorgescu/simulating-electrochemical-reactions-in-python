"""Smoothing filters for noisy simulated data.

Python port of ``Electrochem/Filters.m`` by Mike Honeychurch (2002), from
*Simulating Electrochemical Reactions in Mathematica* (SERM).

The original package defines two functions:

* ``MovingAve[list, n]`` -- an ``n``-point moving average implemented as
  ``ListCorrelate[Table[1., {n}], list] / n``.
* ``ConvolutionFilter[data, len]`` -- a Gaussian smoothing filter with kernel
  ``Table[Exp[-k^2/100], {k, -len, len}]`` normalised to unit sum, applied via
  ``ListConvolve[kern, data, {-1-len, 1+len}]``.

We reproduce the *same* kernels here using numpy.
"""
from __future__ import annotations

import numpy as np


def moving_average(data, n):
    """``n``-point moving average.

    Port of ``MovingAve[list, n]``.  The original uses
    ``ListCorrelate[Table[1., {n}], list] / n``, which is a *valid*-mode
    correlation with a flat kernel: the output has ``len(data) - n + 1``
    points, each the mean of ``n`` consecutive input points.

    Parameters
    ----------
    data : array_like, 1-D
    n : int
        Window width (positive).

    Returns
    -------
    numpy.ndarray, shape (len(data) - n + 1,)
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    data = np.asarray(data, dtype=float)
    kernel = np.ones(n) / n
    # 'valid' mode == Mathematica's default ListCorrelate overlap.
    return np.convolve(data, kernel, mode="valid")


def gaussian_kernel(length):
    """Return the normalised Gaussian kernel used by ``ConvolutionFilter``.

    ``Table[Exp[-k^2/100], {k, -length, length}]`` normalised to unit sum.
    Has ``2*length + 1`` points.
    """
    if length <= 0:
        raise ValueError("length must be a positive integer")
    k = np.arange(-length, length + 1)
    kern = np.exp(-(k ** 2) / 100.0)
    return kern / kern.sum()


def convolution_filter(data, length):
    """Gaussian smoothing filter.

    Port of ``ConvolutionFilter[data, len]``.  The original applies the kernel
    with ``ListConvolve[kern, data, {-1-len, 1+len}]``; the overlap spec
    ``{-1-len, 1+len}`` makes the convolution *cyclic* (wrap-around) and returns
    an output the same length as the input, centred on the kernel.  We match
    that here with ``mode='wrap'`` semantics via :func:`numpy.convolve` on a
    periodically extended signal.

    Parameters
    ----------
    data : array_like, 1-D
    length : int
        Half-width of the kernel (kernel has ``2*length + 1`` points).

    Returns
    -------
    numpy.ndarray, same length as ``data``.
    """
    data = np.asarray(data, dtype=float)
    kern = gaussian_kernel(length)
    # Cyclic convolution, same length out, centred kernel.
    extended = np.concatenate([data[-length:], data, data[:length]])
    full = np.convolve(extended, kern, mode="valid")
    return full
