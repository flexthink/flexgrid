"""
Utilities

Authors
 * Artem Ploujnikov 2026
"""
import sys


def split_args(args: list | None = None, sep: str = "--") -> tuple[list, list]:
    """Splits a single list of arguments into two lists - one
    for the hyperparameter search and one for the downstream
    script

    Arguments
    ---------
    args : list | None
        A list of arguments (usually a list of strings)
        If none is passed, sys.args will be used and the first argument
        will be removed (because it is the current script)
    sep : str
        The separator

    Returns
    -------
    own: list
        The list of arguments to the current script
    downstream : list
        The list of arguments to be passed down to the downstream
        script
    """
    if args is None:
        args = sys.argv[1:]
    try:
        sep_idx = args.index(sep)
        own = args[:sep_idx]
        downstream = args[sep_idx + 1:]
    except ValueError:
        own = args
        downstream = []
    return own, downstream
