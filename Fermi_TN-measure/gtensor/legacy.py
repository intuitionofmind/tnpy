"""
**Use with caution**:

Legacy version of GTensor tensordot
(Grassmann metrics are manually input
instead of automatically determined from dual)
"""

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from itertools import product
from typing import Sequence
from .core import GTensor, zeros, _DEBUG
from .tools import gen_gidx

def _process_contract_axes(axes, a_ndim: int, b_ndim: int):
    """
    (Auxiliary)
    Determine the axes to be contracted in
    the two GTensors and set the Grassmann metric

    Parameters
    ----------
    axes: two or three sequence of ints
        axes[0]: axes of tensor A to be contracted
        axes[1]: axes of tensor B to be contracted
        axes[2]: (optional) Grassmann metric (a series of +1 and -1)
            default: [+1] * number of pairs of indices to be contracted

    ndimA, ndimB: int
        number of axes in tensor A, B respectively

    Returns
    -----------------
    a_axis, b_axis: list[int]
        axes of A, B to be contracted
    gMetric: np.ndarray
        Grassmann metric
    npairs: int
        number of pairs of contracted axes
    """
    if not 2 <= len(axes) <= 3:
        raise ValueError("`axes` must be a sequence of length 2 or 3")
    try:
        # contract only one pair of indices
        if not axes[0] % 1 == axes[1] % 1 == 0:
            raise ValueError("`axes` not integer")
        axisA, axisB = [axes[0]], [axes[1]]
        if len(axes) == 3:
            # use the metric given in `axes`
            gMetrics = np.asarray([axes[2]], dtype=int)
        else:
            gMetrics = np.full(shape=(1,), dtype=int, fill_value=1)
    except TypeError:
        # contract over multiple pairs of indices
        if not (isinstance(axes[0], Sequence) and isinstance(axes[1], Sequence)):
            raise TypeError(
                "`axes` must be sequence of int or sequence of sequences of int"
            )
        axisA, axisB = list(axes[0]), list(axes[1])
        if len(axes) == 3:
            # use the metric given in `axes`
            gMetrics = np.asarray(axes[2], dtype=int)
        else:
            # default: all +1
            gMetrics = np.full(shape=(len(axisA),), dtype=int, fill_value=1)
    # count the number of pairs of indices to be contracted
    if len(axisA) == len(axisB) == len(gMetrics):
        npairs = len(axisA)
    else:
        raise ValueError("elements of `axes` must have the same length")
    # verify the elements in the Grassmann metric
    for i in gMetrics:
        if not abs(i) == 1:
            raise ValueError(
                "3rd element of `axes` must be 1 or -1 or a sequence of 1 and -1"
            )
    # process negative number of axes
    for axis, ndim in [(axisA, a_ndim), (axisB, b_ndim)]:
        for i, idx in enumerate(axis):
            if not idx % 1 == 0:
                raise ValueError(
                    "`axes` must be a sequence of int or a sequence of sequences of int"
                )
            if idx < 0:
                axis[i] = ndim + idx
            if not 0 <= axis[i] < ndim:
                raise ValueError("axis out of range")
    return axisA, axisB, gMetrics, npairs


def tensordot(a: GTensor, b: GTensor, axes, verify_dual=False):
    """
    (Legacy version)

    Contract two GTensors `a` and `b`.
    The free indices of `a` are put before `b`.

    Parameters
    --------------
    axes[0:1]: (sequence of) int
        indices in a and b to be contracted

    axes[2] : optional. default [1] * number of contracted pair of axes
        Grassmann metric. Arrow direction:
        +1 : a -> b (default)
        -1 : a <- b

    verify_dual: bool
        if True, verify whether the given gMetrics
        are compatible with the dual of contracted axes

    Returns
    --------------
    c: GTensor
        contraction result
    """
    axisA, axisB, gMetric, npairs = _process_contract_axes(axes, a.ndim, b.ndim)
    if verify_dual:
        for axA, axB, g in zip(axisA, axisB, gMetric):
            assert g == (
                1 if (a.dual[axA], b.dual[axB]) == (1,0)
                else -1
            )

    # put the indices to be contracted together (match indices from inner to outer)
    perma = [i for i in range(a.ndim) if i not in axisA] + axisA[::-1]
    permb = axisB + [i for i in range(b.ndim) if i not in axisB]
    a = a.transpose(*perma)
    b = b.transpose(*permb)
    # initialize result GTensor
    cshape = tuple(a.shape[i][:-npairs] + b.shape[i][npairs:] for i in range(2))
    c = zeros(
        shape=cshape,
        dual=(a.dual[:-npairs] + b.dual[npairs:]),
        parity=(a.parity + b.parity) % 2,
    )
    for (gIdxA, blockA), (gIdxB, blockB) in product(a.blocks.items(), b.blocks.items()):
        # sum over terms with the same Grassmann index in the contracted axes
        if gIdxA[-npairs:][::-1] == gIdxB[:npairs]:
            # new Grassmann index = free A index + free B index
            gIdxC = gIdxA[:-npairs] + gIdxB[npairs:]
            # new block
            c.blocks[gIdxC] += (
                np.prod(gMetric**gIdxB[:npairs])
                * torch.tensordot(blockA, blockB, [
                    list(range(a.ndim-1, a.ndim-npairs-1, -1)), 
                    list(range(npairs))
                ])
            )
    if _DEBUG:
        c.verify()
    return c


def trace(
    a: GTensor,
    axis1: int | list[int] = 0,
    axis2: int | list[int] = 1,
    gMetric: int | list[int] = 1
):
    """
    Find the trace of a GTensor `a`

    Parameters
    ----------
    axis1, axis2: int or list[int]
        Pair(s) of axes to be traced
    """
    # check axis1, axis2
    try:
        axis1 = list(axis1)
    except TypeError:
        axis1 = [axis1]
    try:
        axis2 = list(axis2)
    except TypeError:
        axis2 = [axis2]
    # check gMetric
    if gMetric == 1 or gMetric == -1:
        gMetric = [gMetric] * len(axis1)
        absorb = gMetric == -1
    else:
        gMetric = list(gMetric)
        # determine whether absorption of gMetric
        # into `a` is needed for trace
        absorb = False
        for i in gMetric:
            if i == 1:
                pass
            elif i == -1:
                absorb = True
                break
            else:
                raise ValueError("`gMetric` can only be 1 or -1")
    # check args consistency
    assert (
        len(axis1) == len(axis2) == len(gMetric)
    ), "`axis1`, `axis2`, `gMetric` must be of the same length"
    npairs = len(axis1)
    # proper transposition
    perm = (
        axis1 + axis2
        + [i for i in list(range(a.ndim)) if i not in axis1 and i not in axis2]
    )
    a = a.transpose(*perm)
    # absorb Grassmann metric
    if absorb is True:
        for gIdx in a.blocks:
            a.blocks[gIdx] *= np.prod(np.asarray(gMetric) ** gIdx[0:npairs])
    # initialize trace (shape, dual and parity)
    trshape = tuple(a.shape[i][2 * npairs :] for i in range(2))
    trdual = a.dual[2 * npairs :]
    tr = zeros(trshape, trdual, parity=a.parity)
    axmerge = (npairs,) + (npairs,) + (1,) * (a.ndim - 2 * npairs)
    order = (1, -1) + (1,) * (a.ndim - 2 * npairs)
    dualMerge = (1, 0) + tr.dual
    a = a.merge_axes(axmerge, dualMerge, order, False)
    for gIdx in gen_gidx(tr.ndim, tr.parity):
        for n in range(2):
            tr.blocks[gIdx] += torch.trace(a.blocks[(n, n) + gIdx])
    return tr
