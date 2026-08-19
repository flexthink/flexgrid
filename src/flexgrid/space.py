"""Utilities for specifying search spaces.

Authors
 * Artem Ploujnikov 2026
"""

import math
from typing import Literal

import torch


def linear(
    min: int | float,
    max: int | float,
    step: int | float = 1,
    decimals: int | None = None,
    scale: Literal["increment", "gauge"] = "increment",
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
    decimals : int | None
        The number of decimal places
    scale : {"increment", "gauge"}
        ``"increment"`` (the default) starts at ``min`` and repeatedly adds
        ``step``. ``"gauge"`` includes multiples of ``step`` that fall within
        the range, plus ``min`` and ``max`` themselves.

    Returns
    -------
    values : list[Number]
        The materialized range
    """
    dtype = None
    if (
        isinstance(min, float)
        or isinstance(max, float)
        or isinstance(step, float)
    ):
        dtype = torch.float64
    if scale == "increment":
        items = torch.arange(
            min,
            max + step,
            step,
            dtype=dtype,
        )
    elif scale == "gauge":
        if step <= 0:
            raise ValueError("step must be positive when scale is 'gauge'")
        if min > max:
            raise ValueError("min must not exceed max when scale is 'gauge'")

        first_multiple = math.ceil(min / step)
        last_multiple = math.floor(max / step)
        multiples = torch.arange(
            first_multiple,
            last_multiple + 1,
            dtype=dtype,
        ) * step
        items = torch.unique(
            torch.cat((
                torch.tensor([min, max], dtype=dtype),
                multiples,
            )),
            sorted=True,
        )
    else:
        raise ValueError("scale must be either 'increment' or 'gauge'")
    if decimals is not None:
        items = items.round(decimals=decimals)
    return items.tolist()
