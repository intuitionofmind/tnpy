"""
Auxiliary functions
"""

import os
import re
import numpy as np
from itertools import accumulate
from ast import literal_eval
from numpy.lib.stride_tricks import as_strided
from collections.abc import Iterable


def get_precision(a: float, max_prec=8, min_prec: None | int = 2):
    """
    Determine the number (smaller than `max_prec`)
    of nonzero digits in `a` after the decimal place
    """
    # convert decimal part to string
    a_str = str(round(a, ndigits=max_prec)).split(".", 1)[-1]
    prec = 0
    for i, digit in enumerate(a_str):
        if digit != "0":
            prec = i + 1
    if min_prec is not None and prec < min_prec:
        prec = min_prec
    return prec


def flatten_list(xs):
    """
    Flatten arbitrarily nested list
    (https://stackoverflow.com/q/2158395/10444934)
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_list(x)
        else:
            yield x


def merge_shape(shape: tuple[int, ...], axmerge: tuple[int, ...]):
    """Find shape of NumPy array after merging axes"""
    return tuple(
        np.prod(shape[j - i : j]) for i, j in zip(axmerge, accumulate(axmerge))
    )


def merge_axes(array: np.ndarray, axmerge: tuple[int, ...]):
    """Merge axes of a NumPy array"""
    return np.reshape(array, merge_shape(array.shape, axmerge))


def get_invperm(perm: list[int]):
    """Find inverse of a permutation `perm`"""
    invperm = [0] * len(perm)
    for i, j in enumerate(perm):
        invperm[j] = i
    return invperm


def is_diagonal(a: np.ndarray):
    """
    Check if `a` is a diagonal matrix

    Adapted from https://stackoverflow.com/a/64127402/10444934
    """
    # create a view of `a` with diagonal elements removed
    m = a.shape[0]
    p, q = a.strides
    nodiag_view = as_strided(a[:, 1:], (m - 1, m), (p + q, q), writable=False)
    # check that `nodiag_view` contains 0 only
    return (nodiag_view == 0).all()


def dir2param(folder: str) -> dict[str]:
    """Convert folder name to dictionary of parameters"""
    # remove parent directory
    assert folder.endswith(os.sep)
    tmp = folder[:-1].rpartition(os.sep)[-1]
    tmp = re.split("-|_", tmp)
    # deal with minus sign
    idx = [i for i, x in enumerate(tmp) if x == ""]
    for i in reversed(idx):
        tmp[i + 1] = "-" + tmp[i + 1]
        tmp.pop(i)
    n = len(tmp)
    param = dict()
    for i in range(0, n, 2):
        if tmp[i+1] == "*": continue
        param[tmp[i]] = float(tmp[i + 1])
    return param


def param2dir(param: dict[str]) -> str:
    """Convert dictionary of parameters to folder name"""
    if param is None:
        # return empty string
        return ""
    plist = []
    for key, value in param.items():
        if isinstance(value, int):
            plist.append(f"{key}-{value}")
        else:
            prec = get_precision(value)
            plist.append(f"{key}-{value:.{prec}f}")
    return "_".join(plist)


def split_measkey(key: str) -> list[str]:
    """
    Split measurement key into
    bond + [name of operators]

    Examples
    ----
    ```
    (two operators)
    "NhId"    -> ["Nh", "Id"]
    (bond name + plaquette)
    "xy2"     -> ["xy", "2"]
    (two sites + two operators)
    "xySpSm"  -> ["xy", "Sp", "Sm"]
    (two sites + two operators + plaquette)
    "xwSpSm2" -> ["xy", "Sp", "Sm", "2"]
    (1st neighbor bond + two operators)
    "x1SpSm"  -> ["x1", "Sp", "Sm"]
    "y2_SpSm" -> ["y2_", "Sp", "Sm"]
    ```
    """
    return re.findall("[w-z][w-z]|[x-y][1-2][_]?|[A-Z][a-z]+|[1-4]", key)


def meas_process(
    coeffs: list[complex|float], 
    keys: list[str], measures: dict[str, complex|float]
): 
    result = 0.0
    assert len(coeffs) == len(keys)
    for key, coeff in zip(keys, coeffs):
        if coeff == 0: continue
        add = coeff * measures[key]
        result += add
    return result


def str_perm(s1: str, s2: str):
    """
    Find the permutation that changes string
    from `s1` to `s2`
    (assuming no repeated characters)
    """
    # assert no duplicate characters in s1
    assert len(s1) == len(set(s1))
    # Create a dictionary to store the positions of characters in s1
    char_positions = {char: i for i, char in enumerate(s1)}
    # Build the permutation list using list comprehension
    perm = [char_positions[char] for char in s2]
    # assert that s2 contains the same characters as s1
    assert len(perm) == len(s1)
    return perm


def dict_loadtxt(filename: str, manual_fix=None) -> dict[str]:
    """
    Load parameters from info file specified by `filename`
    """
    param = {}
    try:
        with open(filename, "r") as f:
            # `strip()` removes leading and trailing whitespace
            line = f.readline()
            while line.strip() != "":
                # split line into (key, value) pairs
                key, value = line.strip().split(maxsplit=1)
                try:
                    param.update({key: literal_eval(value)})
                except ValueError:
                    param.update({key: value})
                line = f.readline()
    except FileNotFoundError:
        if manual_fix is not None:
            pass
        else:
            raise
    if manual_fix is not None:
        assert isinstance(manual_fix, dict)
        param.update(manual_fix)
    # convert read lines into dictionary
    return param
