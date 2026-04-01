"""Utilities for specifying search spaces

Authors
 * Artem Ploujnikov 2026
"""


from numbers import Number


def linear(min: Number, max: Number, step: Number = 1) -> list[Number]:
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
    return list(range(min, max+1, step))
