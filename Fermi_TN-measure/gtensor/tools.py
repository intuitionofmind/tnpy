from itertools import accumulate, product
from operator import itemgetter
from typing import Sequence
import numpy as np
from torch import Tensor


def perm_parity(perm):
    """
    Determine the parity of a permutation `perm`.
    """
    cnt = 0  # number of inversions
    perm = list(perm)
    for i in reversed(range(1, len(perm))):
        maxIdx = max(enumerate(perm[: i + 1]), key=itemgetter(1))[0]
        if maxIdx != i:
            perm[i], perm[maxIdx] = perm[maxIdx], perm[i]
            cnt += 1
    return cnt % 2


def subperm_parity(perm, gIdxNew):
    """
    Determine the parity of a sub-permutation on
    axes with fermion (Grassmann) index 1
    """
    # generate the sub-permutation of index = 1 axes
    # the permutation will in general not consisting all numbers from 0 to len(perm)
    subperm = [j for i, j in zip(gIdxNew, perm) if i == 1]
    return perm_parity(subperm)


def get_permsign(
    dual: tuple[int, ...], gIdx: tuple[int, ...], pairs: list[tuple[int, int]]
) -> int:
    """
    find the fermionic sign produced by bringing
    pairs of contracted axes together (moving 1st axis
    to the front or next to the 2nd axis of the pair)

    ALWAYS permute the kets (dual = 0) behind the bras (dual = 1):
    ```
    <bra| --> |ket>
    ```

    Parameters
    ----------
    dual: tuple[int, ...]
        the dual of axes from all contracted axes
    gIdx: tuple[int, ...]
        the Grassmann index of axes from all contracted axes
    pairs: list[tuple[int, int]]
        pairs of contracted axes

    Returns
    -------
    sign: int
        fermionic sign arised from this permutation
    """
    # used to track the sign
    tmp_gIdx = list(gIdx)
    sign = 1
    for i, j in pairs:
        assert i < j
        assert dual[i] ^ dual[j], "contracted axes should have different dual"
        # <i| --> |j>: permute 'qs[j]' behind 'qs[i]'
        if 1 == dual[i]:
            sign *= (-1) ** (tmp_gIdx[j] * sum(tmp_gIdx[(i + 1) : j]))
            # set to zero as this very pair has been already contracted
            # !do not pop out since we need to keep other pairs' position unchanged
            tmp_gIdx[i], tmp_gIdx[j] = 0, 0
        # |i> <-- <j| permute 'qs[i]' behind 'qs[j]'
        else:
            sign *= (-1) ** (tmp_gIdx[i] * sum(tmp_gIdx[(i + 1) : (j + 1)]))
            tmp_gIdx[i], tmp_gIdx[j] = 0, 0
    return sign


# ---- Index generators ----


def gen_gidx(ndim, parity=0):
    """Grassmann index generator (of given parity)"""
    assert parity in (0, 1)
    if ndim == 0:
        if parity != 0:
            raise ValueError("scalar must be parity even")
        yield tuple()
        return
    for i in range(2**ndim):
        binary = bin(i)[2:].rjust(ndim, "0")
        if binary.count("1") % 2 == parity:
            yield tuple(int(j) for j in binary)


def gidx_shape(gIdx: tuple[int, ...], shape: tuple[tuple[int, ...], tuple[int, ...]]):
    """Get block shape corresponding to the Grassmann index `gIdx`"""
    assert len(gIdx) == len(shape[0])
    return tuple(shape[gIdx[i]][i] for i in range(len(gIdx)))


def gen_nidx(shape):
    """Normal index generator"""
    for nIdx in product(*map(range, shape)):
        yield nIdx


def matshape_from_block(
    blocks: dict[tuple[int, ...], Tensor]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    From the blocks (assumed self-consistent and complete)
    of a matrix, get its shape
    """
    key0 = next(iter(blocks.keys()))
    assert len(key0) == 2
    parity = sum(key0) % 2
    shape = (
        (blocks[(0, 0)].shape, blocks[(1, 1)].shape)
        if parity == 0
        else (
            (blocks[(0, 1)].shape[0], blocks[(1, 0)].shape[1]),
            (blocks[(1, 0)].shape[0], blocks[(0, 1)].shape[1]),
        )
    )
    return shape


def shape_from_block(
    blocks: dict[tuple[int, ...], Tensor]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    From the blocks (always assumed to be valid and complete) of a GTensor, get its shape
    """
    key0 = next(iter(blocks.keys()))
    # number of axes
    ndim = len(key0)
    # parity
    parity = sum(key0) % 2
    if parity == 0:
        raise NotImplementedError
    else:
        raise NotImplementedError


# ---- Auxilliary function for Axes Merging ----


def merge_shape(shape, axmerge):
    """Return the shape of tensor after merging axes"""
    # convert to standard shape format
    if isinstance(shape[0], Sequence):
        # check structure of `shape`
        assert len(shape) == 2 and len(shape[0]) == len(shape[1])
        shape = tuple(tuple(shapepar) for shapepar in shape)  # convert to tuple
    elif shape[0] % 1 == 0:
        shape = (tuple(shape),) * 2
    shapeNew = [[0] * len(axmerge), [0] * len(axmerge)]
    for ax, (i, j) in enumerate(zip(accumulate((0,) + axmerge), axmerge)):
        shapePartial = (shape[0][i : i + j], shape[1][i : i + j])
        for par in range(2):
            for gIdx in gen_gidx(j, par):
                blkshape = gidx_shape(gIdx, shapePartial)
                shapeNew[par][ax] += np.prod(blkshape)
    # convert list to tuple
    return tuple(tuple(shapepar) for shapepar in shapeNew)


def find_remain_axis(free_axes: list[int], ndim: int):
    full_set = set(range(ndim))
    list_set = set(free_axes)
    remain_list = sorted(list(full_set - list_set))
    return remain_list


def regularize_axes(axes: list[int], ndim: int):
    """Deal with negative axes id"""
    for ax in axes:
        assert -ndim <= ax < ndim
    return [
        (ax if ax >= 0 else ax+ndim) for ax in axes
    ]
