"""Utilities for specifying search spaces

Authors
 * Artem Ploujnikov 2026
"""

import torch


def linear(
    min: int | float,
    max: int | float,
    step: int | float = 1
) -> list[int | float]:
    """Represents a linear range of values to try. This
    function is provided mainly for readability

    Arguments
    ---------
    min : Number
        The minimum value
    max : Number
        The maximum value
    step : Number
        The step size

    Returns
    -------
    values : list[Number]
        The materialized range
    """
    return torch.arange(min, max + step, step).tolist()
