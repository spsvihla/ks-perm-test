"""
kspt.py

A wrapper for the kspt._kspt package.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import kspt._kspt


def max_split_ks_perm_test(
    data: ArrayLike,
    eps: float = 0.005,
    delta: float = 0.01,
    num_bins: Optional[int] = None,
    min_split_size: Optional[int] = None,
    coarse_scan_width: Optional[int] = None,
    seed: Optional[int] = None
):
    """
    Estimates the CDF of the sequence 'data' under the null hypothesis that
    the sequence is exchangeble.

    Parameters
    ----------
    data: ArrayLike
        The target sequence.
    eps: float (optional, default 0.005)
        Desired precision of the estimate.
    delta: float (optional, default 0.001)
        Probability that the estimate lies within the desired precision.
    num_bins: int (optional, default None)
        Number of bins to use when constructing empirical CDF for KS distance
        compitaton. By default set to sqrt(data.size).
    min_split_size: int (optional, default None)
        Minimum size of a split when searching for maxmimum. By default, set
        to 0.1 * data.size.
    coarse_scan_width: int (optional, default None)
        Width of interval when performing coarse scan. By default, set to 
        0.05 * data.size.
    seed: int (optional, default None)
        Random seed.

    Returns
    -------
    float
        Estimate of the CDF at the given point.

    Raises
    -------
    ValueError
        If num_bins, min_split_size, or coarse_scan_width are invalid.
    """
    seed = int(seed) if seed is not None else None
    num_samples = np.ceil(1.0 / (2 * eps**2) * np.log(2 / delta))
    num_bins = np.sqrt(data.size) if num_bins is None else num_bins
    min_split_size = 0.1 * data.size if min_split_size is None else min_split_size
    coarse_scan_width = 0.05 * data.size if coarse_scan_width is None else coarse_scan_width
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    if min_split_size < 1:
        raise ValueError(f"min_split_size must be at least 1, got {min_split_size}")
    if coarse_scan_width < 1:
        raise ValueError(f"coarse_scan_width must be at least 1, got {coarse_scan_width}")
    if min_split_size >= data.size // 2:
        raise ValueError(f"min_split_size ({min_split_size}) too large for data size {data.size}")
    if coarse_scan_width >= data.size - 2 * min_split_size:
        raise ValueError(f"coarse_scan_width ({coarse_scan_width}) too large for data size {data.size} and min_split_size {min_split_size}")
    obs, samples = kspt._kspt.rand_max_split_ks(data, int(num_samples), int(num_bins),
                                                int(min_split_size), int(coarse_scan_width),
                                                seed)
    samples.sort()
    result = np.searchsorted(samples, obs, side='left') / num_samples
    return result.item()