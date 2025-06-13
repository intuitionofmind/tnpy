import torch
torch.set_default_dtype(torch.float64)
from torch import Tensor
import numpy as np
from . import tools
import utils
from copy import deepcopy
from itertools import accumulate, product, chain
from typing import Sequence, Literal
from math import sqrt, prod
from opt_einsum import contract, contract_path
from string import ascii_letters

# import atexit
# import line_profiler as lp
# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

_DEBUG = False
# SVD trucation criterion
EPSS_DEFAULT = 1.0e-15
# COND_SHIFT = 1.0e-08
# np.seterr(all='raise')


class GTensorInconsistent(Exception): ...


# ---- Class GTensor ----

class GTensor:
    r"""
    Fermion (Grassmann) tensors

    Parameters
    ----
    shape: two kinds of input are accepted (compulsory)
        1. tuple of two tuples of the same length
            shape[0]: normal index dimensions when all gIdx = 0
            shape[1]: normal index dimensions when all gIdx = 1
            Tensor shape corresponding to gIdx can be found using
                T.blocks[gIdx].shape == tools.gidx_shape(gIdx, T.shape)
        2. tuple of ints
            in this case it is assumed that normal index dimensions
            are independent of gIdx; converted to (shape, shape)

        Example: if a tensor T_{i1 i2 i3}^{n1 n2 n3} has axes of dimension

            | gIdx    | 0   | 1   |
            | ------- | --- | --- |
            | dim{i1} | 3   | 4   |
            | dim{i2} | 4   | 2   |
            | dim{i3} | 2   | 5   |

        Then
            T.shape = ((3,4,2), (4,2,5))

    dual: list of 0 or 1 (compulsory)
        type of each axis (|k> or <b|)

    parity: int 0 or 1 (default None)
        parity of the GTensor

    fill_value: complex number (default None)
        default value to be filled into unspecified entries of GTensor

    block: dictionary with the structure (default None)
        specify GTensor elements
        key  : Grassmann index
        value: the tensor corresponding to the Grassmann index
    """

    # class objects and their types
    _shape: tuple[tuple[int, ...], tuple[int, ...]]
    _dual: tuple[int, ...]
    _parity: int
    _ndim: int
    _size: int
    blocks: dict[tuple[int, ...], Tensor]

    def __init__(
        self,
        shape: tuple[tuple[int, ...], tuple[int, ...]],
        dual: tuple[int, ...],
        parity: None | Literal[0, 1] = None,
        fill_value=None,
        blocks: None | dict[tuple[int, ...], Tensor] = None,
    ):
        # get `shape` and `ndim` (number of axes)
        if isinstance(shape[0], Sequence) or isinstance(shape[0], Tensor):
            # check structure of `shape`
            assert len(shape) == 2 and len(shape[0]) == len(shape[1])
            self._shape = tuple(
                tuple(blkshape) for blkshape in shape
            )  # convert to tuple
        elif shape[0] % 1 == 0:
            self._shape = (tuple(shape),) * 2
        self._ndim = len(self._shape[0])
        self._dual = tuple(dual)
        assert len(self._shape[0]) == len(self._shape[1]) == len(self._dual), \
            "inconsistenct ndim of shape and dual"

        # initialize by given values (`blocks`)
        if blocks is not None:
            assert isinstance(blocks, dict)
            # get `parity`
            for gIdx in blocks.keys():
                self._parity = sum(gIdx) % 2
                break
            # if `parity` is given, check if it agrees with `blocks`
            if parity is not None and parity != self.parity:
                raise ValueError("`parity` and `blocks` are inconsistent")
            # check shape and parity consistency and data type
            for gIdx, blk in blocks.items():
                assert sum(gIdx) % 2 == self.parity
                assert isinstance(blk, Tensor), "block values should be PyTorch tensors"
                if blocks[gIdx].shape != tools.gidx_shape(gIdx, self.shape):
                    raise ValueError("`shape` and `blocks` are inconsistent")
            # unspecified elements are filled with 0
            if fill_value is None:
                fill_value = 0.0
        # initialize GTensor without given elements
        else:
            blocks = dict()
            assert (
                parity is not None
            ), "`parity` must be specified when `blocks` is not given"
            self._parity = parity

        # assign values to blocks
        self.blocks = dict()
        for gIdx in tools.gen_gidx(self.ndim, self.parity):
            # shape of the block
            blkshape = tools.gidx_shape(gIdx, self.shape)
            # value specified by `blocks`
            if gIdx in blocks:
                if blocks[gIdx].shape != blkshape:
                    raise ValueError("`blocks` has inconsistent shape")
                self.blocks[gIdx] = blocks.pop(gIdx)
            # value not specified by `blocks`
            else:
                # value not specified: fill with `empty`
                if fill_value is None:
                    self.blocks[gIdx] = torch.empty(blkshape, dtype=torch.cdouble)
                # value specified by `fill_value`
                else:
                    self.blocks[gIdx] = torch.full(
                        blkshape, fill_value, dtype=torch.cdouble
                    )
        # total number of elements
        self._size = sum(block.numel() for block in self.blocks.values())
        # if there are still blocks left, these blocks have wrong parity or ndim
        if blocks:
            raise ValueError("`blocks` inconsistent")
        if _DEBUG:
            self.verify()

    @property
    def shape(self):
        """
        Even and odd dimension of each axis

        - `shape[0]`: Even dimensions
        - `shape[1]`: Odd dimensions
        """
        return self._shape

    @property
    def DS(self):
        """Total dimension of each exis"""
        return tuple(e + o for e, o in zip(self._shape[0], self._shape[1]))

    @property
    def DE(self):
        """Even dimension of each exis"""
        return self._shape[0]

    @property
    def DO(self):
        """Odd dimension of each exis"""
        return self._shape[1]

    @property
    def dual(self):
        """Dual (bra or ket) of each axis"""
        return self._dual

    @property
    def ndim(self):
        """Number of axes"""
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def parity(self):
        return self._parity

    def __mul__(self, value):
        a = self.copy()
        for gIdx in a.blocks.keys():
            a.blocks[gIdx] *= value
        return a

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value):
        a = self.copy()
        for gIdx in a.blocks.keys():
            a.blocks[gIdx] /= value
        return a

    def __neg__(self):
        a = self.copy()
        for gIdx in a.blocks.keys():
            a.blocks[gIdx] *= -1
        return a

    def __add__(self, b):
        if not isinstance(b, GTensor):
            raise TypeError("A GTensor can only be added with another GTensor")
        if not (
            b.shape == self.shape and b.parity == self.parity and b.dual == self.dual
        ):
            raise ValueError("incompatible Grassmann tensors")
        c = self.copy()
        for gIdx, b_block in b.blocks.items():
            if b_block.numel() == 0:
                continue
            c.blocks[gIdx] += b_block
        return c

    def __sub__(self, b):
        if not isinstance(b, GTensor):
            raise TypeError("A GTensor can only be added with another GTensor")
        if not consistency_check(self, b):
            raise ValueError("incompatible Grassmann tensors")
        c = self.copy()
        for gIdx, b_block in b.blocks.items():
            if b_block.numel() == 0:
                continue
            c.blocks[gIdx] -= b_block
        return c

    def update(self, blocks: dict[tuple[int, ...], Tensor]):
        """Update blocks of GTensor from input"""
        if not isinstance(blocks, dict):
            raise TypeError("`blocks` must be a dict")
        for gIdx in self.blocks.keys():
            if gIdx in blocks:
                if not blocks[gIdx].shape == tools.gidx_shape(gIdx, self.shape):
                    raise ValueError("shape error")
                if self.blocks[gIdx].numel() == 0:
                    del blocks[gIdx]
                    continue
                else:
                    self.blocks[gIdx] = blocks[gIdx]
                    del blocks[gIdx]
        if blocks:
            raise ValueError("invalid element in `blocks`")
        if _DEBUG:
            self.verify()

    def verify(self):
        """Verify structure of GTensor"""
        # dual is assigned to each axis
        for d in self.dual:
            if not (d == 0 or d == 1):
                raise GTensorInconsistent()
        if len(self.dual) != self.ndim:
            raise GTensorInconsistent()
        # all blocks are consistent with the tensor shape
        for gIdx in tools.gen_gidx(self.ndim, self.parity):
            if not (self.blocks[gIdx].shape == tools.gidx_shape(gIdx, self.shape)):
                raise GTensorInconsistent()
        return True

    def copy(self, detach=False):
        """
        Make a copy of the GTensor.
        """
        a = GTensor(self.shape, self.dual, self.parity)
        for gIdx, block in self.blocks.items():
            if block.numel() == 0:
                continue
            a.blocks[gIdx] = block.detach().clone() if detach else block.clone()
        if _DEBUG:
            a.verify()
        return a

    def item(self) -> complex:
        """
        Return the only element (as number, not size-1 array)
        of a 0-dimensional (even parity) Grassmann tensor
        """
        if self.shape == ((), ()) and self.parity == 0:
            return self.blocks[()][()].item()
            # return (next(iter(self.blocks.values()))).item()
        else:
            raise ValueError("only dim-0 even-parity tensors have `value`")

    def diagonal(self, real=False) -> dict[int, Tensor]:
        """
        Extract diagonal elements of
        2D even-parity matrix, returned as a dict
        `{0: even-diagonal, 1: odd-diagonal}`

        Parameters
        ----
        real: bool
            when True, discard the imaginary part of the diagonal elements
        """
        assert (self.ndim == 2 and self.parity == 0), \
            "only 2D even tensors have well-defined diagonal elements"
        diag = dict((
            p, self.blocks[(p,p)].diagonal().real if real
            else self.blocks[(p,p)].diagonal()
        ) for p in range(2))
        return diag

    def transpose(self, *perm: int):
        """Transpose (permute) axes"""
        if len(perm) != self.ndim:
            raise ValueError("`perm` must have length `ndim`")
        # trivial permutation
        if perm == tuple(range(self.ndim)):
            return self.copy()
        blocksNew = dict()
        for gIdx, block in self.blocks.items():
            # transpose the Grassmann indices
            gIdxNew = tuple(gIdx[i] for i in perm)
            # transpose the block component
            blocksNew[gIdxNew] = block.permute(perm) * (-1)**tools.subperm_parity(perm, gIdxNew)
        # get shape of transposed tensor
        shapeNew = tuple(tuple(
            self.shape[par][i] for i in perm
        ) for par in range(2))
        # get dual of transposed tensor
        dualNew = tuple(self.dual[i] for i in perm)
        # create transposed GTensor
        gtNew = GTensor(shapeNew, dualNew, blocks=blocksNew)
        if _DEBUG:
            gtNew.verify()
        return gtNew

    @property
    def T(self):
        """Reverse axis order"""
        perm = tuple(reversed(range(self.ndim)))
        return self.transpose(*perm)
    
    def pconj(self, map_axes: None | list[int] = None):
        """
        Multiply fermion sign on axes with dual = 1

        Parameters
        ----
        map_axes: None or list[int]
            axes of pconj that can be acted on by a linear map
            (by default, all axes of the tensor are assumed 
            able to be acted on by a linear map)
        """
        a = self.copy()
        flipper = torch.ones(a.ndim, dtype=int)
        if map_axes is None: 
            map_axes = list(range(self.ndim))
        no_need = True
        for ax in map_axes:
            if a.dual[ax] == 1:
                flipper[ax] = -1
                no_need = False
        if not no_need:
            for gIdx in a.blocks.keys():
                a.blocks[gIdx] *= torch.prod(flipper**torch.tensor(gIdx))
        return a

    def gconj(self):
        """
        Hermitian conjugate:
        - axis order is reversed
        - tensor elements are complex conjugated
        - bras/kets are changed to kets/bras
        """
        shapeNew = tuple(self.shape[par][::-1] for par in range(2))
        dualNew = tuple(1 - d for d in self.dual[::-1])
        a = GTensor(shapeNew, dualNew, self.parity)
        perm = torch.arange(a.ndim - 1, -1, -1)
        for gIdx, block in self.blocks.items():
            a.blocks[gIdx[::-1]] = block.conj().permute(*perm)
        if _DEBUG:
            a.verify()
        return a

    @property
    def gT(self):
        """
        Grassmann conjugate followed by
        full transpose to preserve axis order
        """
        return self.gconj().transpose(*tuple(reversed(range(self.ndim))))

    @property
    def real(self):
        """Get the real part of the GTensor"""
        a = self.copy()
        for gIdx, block in a.blocks.items():
            a.blocks[gIdx] = block.real
        return a

    @property
    def imag(self):
        """Get the imaginary part of the GTensor"""
        a = self.copy()
        for gIdx, block in a.blocks.items():
            a.blocks[gIdx] = block.imag
        return a

    def flip_dual(self, axes: int | list[int], minus=True, change_dual=True):
        """
        Change dual of `axes` for tensor

        This function only makes sense in the
        context of contraction with another tensor

        Parameters
        ----
        axes: int or list[int]
            axes of `a` to change dual
        minus: bool or list[bool]
            whether to multiply -1 to odd sector of `axes`
        """
        assert minus is True or minus is False
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]
        if len(axes) == 0:
            return self.copy()
        # deal with negative numbers in axes
        axes = tools.regularize_axes(axes, self.ndim)
        # change dual of `axes`
        dualNew = (
            tuple(1 - d if ax in axes else d for ax, d in enumerate(self.dual))
            if change_dual
            else self.dual
        )
        if minus is True:
            # multiply -1 to odd sector of `axes`
            flipper = np.array(
                [-1 if i in axes else 1 for i in range(self.ndim)], dtype=int
            )
            blocksNew = dict(
                (gIdx, val * np.prod(flipper**gIdx))
                for gIdx, val in self.blocks.items()
            )
            # construct the new GTensor with flipped sign
            aNew = GTensor(self.shape, dualNew, blocks=blocksNew)
        else:
            aNew = self.copy()
            aNew._dual = dualNew
        if _DEBUG:
            aNew.verify()
        return aNew

    def dot_diag(self, s: "GTensor", axes: list[int]):
        """
        Tensordot of `self` with a diagonal matrix `s` (even parity).
        The contracted axis of `self` is replaced by
        the free axis of `diagmat`

        Parameters
        ----
        s: GTensor
            diagonal Grassmann matrix
        axes: list[int]
            axes[0]: contracted axis of `self`
            axes[1]: contracted axis of `s` (0 or 1)
        """
        assert len(axes) == 2
        ax0 = axes[0] if axes[0] >= 0 else axes[0] + self.ndim
        ax1 = axes[1]
        assert 0 <= ax0 <= self.ndim and 0 <= ax1 <= 1
        assert self.dual[ax0] != s.dual[ax1]
        dualNew = tuple(
            self.dual[ax] if ax != ax0 else s.dual[1 - ax1] for ax in range(self.ndim)
        )
        a = GTensor(self.shape, dual=dualNew, parity=self.parity)
        minus = (
            True
            if ((ax1 == 0 and s.dual[ax1] == 1) or (ax1 == 1 and s.dual[ax1] == 0))
            else False
        )
        sdiag = s.diagonal()
        shapeDummy = tuple(-1 if i == ax0 else 1 for i in range(self.ndim))
        for gIdx, block in self.blocks.items():
            n = gIdx[ax0]
            a.blocks[gIdx] = (block * sdiag[n].reshape(*shapeDummy)) * (
                (-1) ** n if minus else 1
            )
        return a

    def flatten(self):
        """
        Flatten the Grassmann tensor to a 1D PyTorch array.
        """
        return torch.cat(
            [self.blocks[gIdx].reshape(-1) for gIdx in tools.gen_gidx(self.ndim, self.parity)]
        )

    # @profile
    def merge_axes(
        self, axmerge: tuple[int, ...], dualMerge=None, order=None, auto_dual=True
    ):
        """
        Merge consecutive axes of the tensor

        Parameters
        ----------
        axmerge : sequence of int
            number of axes contained in new axis; must satisfy
            `sum(axmerge) == self.ndim`

        dualMerge: tuple[int, ...] or None

        order : sequence of +1 or -1 for each group of axes(default +1 for each)
            order of merging axes
            +1: normal order ("row major")
            -1: reversed order ("column major")
            must satisfy
                len(axmerge) == len(order)

        auto_dual: bool
            when True, if a group of axes to be merged have the same dual,
            the merged axis will also have this dual

        Example
        -------
        If we want to merge axes of a rank-8 tensor `a` as
            a_{(0,1,2), (3,4), (5,6,7)}
        Then set
            axmerge = (3,2,3)
        """
        # check `axmerge`
        ndimMerge = len(axmerge)
        if not (
            ndimMerge <= self.ndim
            and sum(axmerge) == self.ndim
            and all(1 <= ax <= self.ndim for ax in axmerge)
        ):
            raise ValueError("`axmerge` is incompatible with tensor shape")
        # get dual of result tensor
        if auto_dual is True:
            assert dualMerge is None
            dualMerge = get_dual(axmerge, self.dual, "merge")
        else:
            if dualMerge is None:
                dualMerge = (0,) * ndimMerge
            else:
                dualMerge = tuple(dualMerge)
        assert len(dualMerge) == ndimMerge
        # number of axis in axis-merged tensor
        # check `order`
        if order is None:
            order = (1,) * ndimMerge
        else:
            if not isinstance(order, Sequence):
                raise TypeError("`order` must be a sequence of 1 or -1 if specified")
            if not len(order) == ndimMerge:
                raise ValueError("`order` must have the same length as `axmerge`")
            for i in order:
                if not abs(i) == 1:
                    raise ValueError(
                        "`order` must be a sequence of 1 or -1 if specified"
                    )
        # when order = -1, reverse the order of the corresponding group of axes
        perm = []
        for i, j, g in zip(accumulate((0,) + axmerge), axmerge, order):
            perm += list(range(i, i + j))[::g]
        # transpose axis-split tensor according to `order`
        tsSplit = self.transpose(*perm)
        # initialize axis-merged tensor
        shapeMerge = tools.merge_shape(self.shape, axmerge)
        tsMerge = GTensor(shapeMerge, dualMerge, self.parity)
        # if axmerge = (r,s,...), each new block is divided into
        # 2**(r-1) x 2**(s-1) x ... sub-blocks of different sizes
        nBlocks = 2 ** (np.array(axmerge) - 1)
        # move elements to reshaped tensor
        # gIdxMerge: new Grassmann index of the block after merging
        for gIdxMerge in tools.gen_gidx(ndimMerge, self.parity):
            if tsMerge.blocks[gIdxMerge].numel() == 0:
                continue
            # initialize some variables to be used later
            idxStart = np.zeros(ndimMerge, dtype=int)
            idxBlocksLast = (0,) * ndimMerge
            shapeSubMergeLast = (0,) * ndimMerge
            # gIdxSplitGroup: 
            # old Grassmann index in normal order, grouped by axmerge
            # idxBlocks: 
            # label of the old Grassmann index, also the position of the sub-block
            for gIdxSplitGroup, idxBlocks in zip(
                product(*map(tools.gen_gidx, axmerge, gIdxMerge)),
                product(*map(range, nBlocks)),
            ):
                # gIdxSub: old Grassmann index reordered
                # if `order == -1`, restore normal order of corresponding group of axes
                gIdxSplit = tuple()
                for sub, g in zip(gIdxSplitGroup, order):
                    gIdxSplit += sub[::g]
                shapeSubSplit = tsSplit.blocks[gIdxSplit].shape
                shapeSubMerge = utils.merge_shape(shapeSubSplit, axmerge)
                for i in range(ndimMerge):
                    # return to the beginning of this slot
                    if idxBlocks[i] == 0:
                        idxStart[i] = 0
                    elif idxBlocks[i] != idxBlocksLast[i]:
                        idxStart[i] += shapeSubMergeLast[i]
                idxEnd = idxStart + shapeSubMerge
                # restore original normal order and reshape old block
                # and send to the new sub-block
                tsMerge.blocks[gIdxMerge][tuple(map(slice, idxStart, idxEnd))] = (
                    tsSplit.blocks[gIdxSplit].permute(*perm).reshape(shapeSubMerge)
                )
                idxBlocksLast = idxBlocks
                shapeSubMergeLast = shapeSubMerge
        if _DEBUG:
            tsMerge.verify()
        return tsMerge

    def split_axes(
        self,
        axmerge: tuple[int, ...],
        shapeSplit,
        dualSplit=None,
        order=None,
        auto_dual=True,
    ):
        """
        Split axes of the tensor

        Parameters
        ----------
        axmerge: sequence of int
            number of axes splitted from old big axis

        shapeSplit: shape of the normal indices after axes splitting
            two kinds of input are accepted
            1. tuple of two tuples of int
            2. tuple of int (when shape does not depend on gIdx)

        order : sequence of 1 or -1
            order of splitted axes (must satisfy `len(order) = self.ndim`)
            +1: normal order ("row major")
            -1: reversed order ("column major")
        """
        # check `axmerge`
        if not isinstance(axmerge, Sequence):
            raise TypeError("`axmerge` must be a sequence of int")
        axmerge = tuple(axmerge)
        ndimMerge = len(axmerge)
        ndimSplit = sum(axmerge)
        assert ndimMerge == self.ndim
        # generate dual automatically
        if auto_dual:
            assert dualSplit is None
            dualSplit = get_dual(axmerge, self.dual, "split")
        else:
            if dualSplit is None:
                dualSplit = (0,) * ndimSplit
            else:
                dualSplit = tuple(dualSplit)
        assert len(dualSplit) == ndimSplit
        # check structure of `shapeSplit`
        try:
            if shapeSplit[0] % 1 == 0:
                # convert to default form
                shapeSplit = (tuple(shapeSplit),) * 2
        except TypeError:
            pass
        assert (
            isinstance(shapeSplit[0], Sequence)
            and len(shapeSplit[0]) == len(shapeSplit[1])
            and sum(axmerge) == len(shapeSplit[0])
        )
        # check shape consistency
        if not self.shape == tools.merge_shape(shapeSplit, axmerge):
            raise ValueError("Shape mismatch")
        # check `order`
        if order is not None:
            if not isinstance(order, Sequence):
                raise TypeError("`order` must be a sequence of 1 or -1 if specified")
            if not len(order) == len(axmerge):
                raise ValueError("`order` must have the same length as `axmerge`")
            for i in order:
                if not abs(i) == 1:
                    raise ValueError(
                        "`order` must be a sequence of 1 or -1 if specified"
                    )
        else:
            order = (1,) * len(axmerge)
        # when order = -1, reverse the order of the corresponding group of axes
        perm = []
        for i, j, t in zip(accumulate((0,) + axmerge), axmerge, order):
            perm += list(range(i, i + j))[::t]
        # initialize axis-split tensor (transposed according to `order`)
        shapeSplitTmp = [[], []]
        for par in range(2):
            for i, j, t in zip(accumulate((0,) + axmerge), axmerge, order):
                shapeSplitTmp[par] += shapeSplit[par][i : i + j][::t]
        tsSplit = empty(shapeSplitTmp, (0,) * len(shapeSplitTmp[0]), self.parity)
        # if axmerge = (r,s,...), each new block is divided into
        # 2**(r-1) x 2**(s-1) x ... sub-blocks of different sizes
        nBlocks = 2 ** (np.asarray(axmerge) - 1)
        # move elements to reshaped tensor
        # gIdxMerge: new Grassmann index of the block after merging
        for gIdxMerge in tools.gen_gidx(ndimMerge, self.parity):
            # initialize some variables to be used later
            idxStart = np.zeros(ndimMerge, dtype=int)
            idxBlocksLast = (0,) * ndimMerge
            shapeSubMergeLast = (0,) * ndimMerge
            # gIdxSplitGroup: old Grassmann index in normal order, grouped by axmerge
            # idxBlocks: label of the old Grassmann index, also the position of the sub-block
            for gIdxSplitGroup, idxBlocks in zip(
                product(*map(tools.gen_gidx, axmerge, gIdxMerge)),
                product(*map(range, nBlocks)),
            ):
                # gIdxSub: old Grassmann index reordered
                # if `order == -1`, restore normal order of corresponding group of axes
                gIdxSplit = tuple()
                for sub, t in zip(gIdxSplitGroup, order):
                    gIdxSplit += sub[::t]
                shapeSubSplit = tsSplit.blocks[gIdxSplit].shape
                # if the block in the tensor after splitting axis is empty
                # just skip filling values into it
                if tsSplit.blocks[gIdxSplit].numel() == 0:
                    continue
                shapeSubMerge = utils.merge_shape(shapeSubSplit, axmerge)
                for i in range(ndimMerge):
                    # return to the beginning of this slot
                    if idxBlocks[i] == 0:
                        idxStart[i] = 0
                    elif idxBlocks[i] != idxBlocksLast[i]:
                        idxStart[i] += shapeSubMergeLast[i]
                idxEnd = idxStart + shapeSubMerge
                # restore original normal order and reshape old block
                # and send to the new sub-block
                shapeSubSplitTmp = [shapeSubSplit[i] for i in perm]
                blk = self.blocks[gIdxMerge][tuple(map(slice, idxStart, idxEnd))]
                tsSplit.blocks[gIdxSplit] = blk.reshape(shapeSubSplitTmp).permute(*perm)
                idxBlocksLast = idxBlocks
                shapeSubMergeLast = shapeSubMerge
        # reverse order for the group with order = -1 to match convention of ordinary tensor operation
        tsSplit = tsSplit.transpose(*perm)
        # set dual from input (or auto calculated)
        tsSplit._dual = dualSplit
        if _DEBUG:
            tsSplit.verify()
        return tsSplit

# ------ Tensor Creation ------


def consistency_check(*tensors: GTensor):
    """
    Check whether the input tensors have
    the same shape, dual and parity
    """
    assert len(tensors) >= 2
    t0 = tensors[0]
    return all(
        (t.shape == t0.shape and t.parity == t0.parity and t.dual == t0.dual)
        for t in tensors[1::]
    )


def zeros(shape, dual: tuple[int, ...], parity=0):
    """Create zero GTensor"""
    return GTensor(shape, dual, parity, fill_value=0)


def zeros_like(a: GTensor):
    """Create zero GTensor of the same shape & parity as the given `a`"""
    return GTensor(a.shape, a.dual, a.parity, fill_value=0)


def ones(shape, dual: tuple[int, ...], parity=0):
    """Create zero GTensor"""
    return GTensor(shape, dual, parity, fill_value=1)


def ones_like(a: GTensor):
    """Create zero GTensor of the same shape & parity as the given `a`"""
    return GTensor(a.shape, a.dual, a.parity, fill_value=1)


def empty(shape, dual: tuple[int, ...], parity=0):
    """Create empty GTensor"""
    return GTensor(shape, dual, parity, fill_value=None)


def empty_like(a: GTensor):
    """Create empty GTensor of the same shape & parity as the given `a`"""
    return GTensor(a.shape, a.dual, a.parity, fill_value=None)


def full(shape, dual: tuple[int, ...], fill_value, parity=0):
    """Create GTensor filled with a given number"""
    return GTensor(shape, dual, fill_value=fill_value, parity=parity)


def full_like(a: GTensor, fill_value):
    """Create GTensor filled with a given number and of the same shape & parity as the given `a`"""
    return GTensor(a.shape, a.dual, a.parity, fill_value)


def eye(n0: int, n1=None, dual=(0, 1), dtype=torch.cdouble):
    """
    Create identity Grassmann matrix (2D)
    of even parity (default `n1 = n0`)
    """
    if n1 is None:
        n1 = n0
    shape = ((n0,) * 2, (n1,) * 2)
    blocks = {(0, 0): torch.eye(n0, dtype=dtype), (1, 1): torch.eye(n1, dtype=dtype)}
    return GTensor(shape, dual, blocks=blocks)


def diag(diag_dict: dict[int, Tensor], dual: tuple[int, int] = (0, 1)):
    """Create diagonal GMatrix (2D) of even parity"""
    n0, n1 = len(diag_dict[0]), len(diag_dict[1])
    shape = ((n0,)*2, (n1,)*2)
    blocks = {
        (0,0): torch.diag(diag_dict[0]), 
        (1,1): torch.diag(diag_dict[1])
    }
    return GTensor(shape, dual, blocks=blocks)


def diagonal(t: GTensor, real=False):
    """
    Extract diagonal elements of
    2D even-parity matrix, returned as a dict
    `{0: even-diagonal, 1: odd-diagonal}`
    """
    return t.diagonal(real=real)


def rand(
    shape, dual: tuple[int, ...], parity=0, 
    complex_init=True, negrand=False
):
    """
    Create random GTensor of given shape and parity

    Parameters
    ----
    negrand: bool
        When False, real/imag part of each element is in range [0,1). 
        When True, real/imag part of each element is in range [-1,1). 
    """
    a = empty(shape, dual, parity)
    for gIdx in a.blocks.keys():
        blkshape = tools.gidx_shape(gIdx, a.shape)
        randblk = torch.rand(*blkshape, dtype=torch.cdouble)
        if negrand:
            randblk = (randblk - (0.5+0.5j)) * 2
        if not complex_init:
            randblk = randblk.real.type(torch.cdouble)
        a.blocks[gIdx] = randblk
    return a


def rand_like(a: GTensor, complex_init=True):
    """
    Create random GTensor of tha same shape
    and parity as the given tensor `a`
    """
    return rand(
        a.shape, a.dual, a.parity, complex_init
    )


# ------ Axes Operation ------

def _process_flip_axes(axes):
    if len(axes) == 0:
        axReg = [[], []]
    elif len(axes) == 2:
        try:
            axReg = list(list(ax) for ax in axes)
        except TypeError:
            axReg = [[axes[0]], [axes[1]]]
    else:
        raise ValueError("`axes` is in wrong format")
    assert len(axReg[0]) == len(axReg[1])
    return axReg


def flip_dual(a: GTensor, axes, minus=True, change_dual=True):
    """
    Change dual of `axes` for tensor `a`

    This function only makes sense in the context of contraction with another tensor

    Parameters
    ----
    axes: int or list[int]
        axes of `a` to change dual
    minus: bool or list[bool]
        whether to multiply -1 to odd sector of `axes`
    """
    return a.flip_dual(axes, minus, change_dual)


def flip2_dual(a: GTensor, b: GTensor, axes, flip="a"):
    """
    When contracting tensors `a` and `b`,
    this function change the dual of their axes
    to be contracted and modify the elements
    so that the contraction result is unchanged.

    Parameters
    ----
    a, b: GTensor
        the tensors to be contracted
    axes: list[int] or list[list[int]]
        axes of `a` and `b` to be contracted
    flip: str ("a" or "b")
        add minus signs to `a` or `b`
    """
    assert flip == "a" or flip == "b"
    axes = _process_flip_axes(axes)
    if len(axes[0]) == 0:
        return a, b
    aNew = a.flip_dual(axes[0], minus=(flip == "a"))
    bNew = b.flip_dual(axes[1], minus=(flip == "b"))
    return aNew, bNew


def transpose(a: GTensor, perm: list[int]):
    """Transpose tensor `a`"""
    return a.transpose(*perm)


def merge_axes(
    a: GTensor, axmerge: tuple[int, ...], dualMerge=None, order=None, auto_dual=True
):
    """Merge axes of tensor `a`"""
    return a.merge_axes(axmerge, dualMerge, order, auto_dual)


def split_axes(
    a: GTensor,
    axmerge: tuple[int, ...],
    shapeSplit,
    dualSplit=None,
    order=None,
    auto_dual=True,
):
    """Split axes of tensor `a`"""
    return a.split_axes(axmerge, shapeSplit, dualSplit, order, auto_dual)


def get_dual(
    axmerge: tuple[int, ...], dual: tuple[int, ...], mode="merge"
) -> tuple[int, ...]:
    """
    Get dual after merging/splitting axes of a tensor
    """
    dualNew = ()
    if mode == "merge":
        assert sum(axmerge) == len(dual)
        accums = tuple(accumulate((0,) + axmerge))
        # assert that merged axes all have the same dual
        for i, j in zip(accums[0:-1], accums[1::]):
            assert all(
                d == dual[i] for d in dual[i:j]
            ), "Merged axes should all have the same dual"
            # the new axis have the same dual as those merged into it
            dualNew += (dual[i],)
    elif mode == "split":
        assert len(axmerge) == len(dual)
        # axes split from one axis have the same dual as the original axis
        for nax, d in zip(axmerge, dual):
            dualNew += (d,) * nax
    else:
        raise ValueError('`mode` can only be "split" or "merge"')
    return dualNew


# ------ Equality Judge ------


def absolute(a: GTensor):
    """Return the absolute value (the blocks) of a Grassmann tensor"""
    abs_a = a.copy()
    for gIdx, block in abs_a.blocks.items():
        abs_a.blocks[gIdx] = torch.abs(block)
    return abs_a


def maxabs(a: GTensor) -> float:
    """
    Return the max absolute value
    of elements in a Grassmann tensor
    """
    if a.size == 0:
        raise ValueError("input tensor has no elements")
    abs_a = absolute(a)
    # find the max absolute value and where it is
    # Initialize variables to track the maximum element and its index
    max_element = float("-inf")
    # max_gIdx = (0,) * abs_a.ndim
    # max_nIdx = None
    # Iterate over abs(blocks) and find the maximum element
    # and its Grassmann and normal index
    for gIdx, block in abs_a.blocks.items():
        if block.numel() == 0:
            continue
        current_max = block.max()
        if current_max > max_element:
            max_element = current_max
            # max_gIdx = gIdx
            # max_nIdx = np.unravel_index(np.argmax(block), block.shape)
    # return a.blocks[max_gIdx][max_nIdx]
    return max_element.item()


def around(a: GTensor, decimals=10):
    """Round the GTensor"""
    ar = a.copy()
    for gIdx, block in ar.blocks.items():
        real_part = torch.round(block.real, decimals=decimals)
        imag_part = torch.round(block.imag, decimals=decimals)
        ar.blocks[gIdx] = torch.complex(real_part, imag_part)
    return ar


def allclose(a: GTensor, b: GTensor, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Judge whether two tensors are approximately equal"""
    if not consistency_check(a, b):
        return False
    # check each block
    for gIdx in a.blocks.keys():
        if not torch.allclose(a.blocks[gIdx], b.blocks[gIdx], rtol, atol, equal_nan):
            return False
    return True


def allclose2(*gts: GTensor, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Judge whether multiple tensors are all approximately equal"""
    assert len(gts) >= 2
    return all(allclose(g, gts[0], rtol, atol, equal_nan) for g in gts[1::])


def array_equal(a: GTensor, b: GTensor):
    """Judge whether two tensors are identical"""
    if not consistency_check(a, b):
        return False
    # check each block
    for gIdx in a.blocks.keys():
        if not torch.equal(a.blocks[gIdx], b.blocks[gIdx]):
            return False
    return True


def array_equal2(*gts: GTensor):
    """Judge whether multiple tensors are all identical"""
    assert len(gts) >= 2
    return all(array_equal(g, gts[0]) for g in gts[1::])


# ------ Summation (Tensordot and Trace) ------


def _get_gMetric(
    axes1: list[int], axes2: list[int], 
    dual1: list[int], dual2: list[int]
):
    """
    Determine gMetric for contraction of
    `axes1` and `axes2` from the their dual

    +1: <b| --> |k> (dual 1, 0)
    -1: |k> <-- <b| (dual 0, 1)
    """
    assert len(axes1) == len(axes2), "Axes to be contracted must be in pairs"
    gMetric = np.ones(len(axes1), dtype=int)
    for i, (ax1, ax2) in enumerate(zip(axes1, axes2)):
        d1, d2 = dual1[ax1], dual2[ax2]
        assert d1 != d2, "Contracted pair of axes must have different dual"
        if d1 == 0:
            gMetric[i] = -1
    return gMetric


def _process_contract_axes(
    axes, a_ndim: int, b_ndim: int
) -> tuple[list[int], list[int], int]:
    """
    (Auxiliary) Determine the axes to be
    contracted in the two GTensors

    Parameters
    ----------
    axes: two or three sequence of ints
        axes of tensor A (`axes[0]`)
        and B (`axes[1]`) to be contracted

    a_ndim, b_ndim: int
        number of axes in tensor A, B respectively

    Returns
    -----------------
    a_axis, b_axis: list[int]
        axes of A, B to be contracted
    npairs: int
        number of pairs of contracted axes
    """
    assert len(axes) == 2
    try:
        # contract only one pair of indices
        if not axes[0] % 1 == axes[1] % 1 == 0:
            raise ValueError("`axes` not integer")
        a_axes, b_axes = [axes[0]], [axes[1]]
    except TypeError:
        # contract over multiple pairs of indices
        if not (isinstance(axes[0], Sequence) and isinstance(axes[1], Sequence)):
            raise TypeError(
                "`axes` must be sequence of int or sequence of sequences of int"
            )
        a_axes, b_axes = list(axes[0]), list(axes[1])
    # count the number of pairs of indices to be contracted
    assert len(a_axes) == len(b_axes)
    npairs = len(a_axes)
    # process negative axis number
    a_axes = tools.regularize_axes(a_axes, a_ndim)
    b_axes = tools.regularize_axes(b_axes, b_ndim)
    return a_axes, b_axes, npairs


def _outer2(a: GTensor, b: GTensor):
    """
    direct product of two Grassmann tensors
    ```
    """
    abshape = tuple(a.shape[par] + b.shape[par] for par in range(2))
    abdual = a.dual + b.dual
    abpar = (a.parity + b.parity) % 2
    ab = zeros(abshape, abdual, abpar)
    for (gIdxA, blockA), (gIdxB, blockB) in product(a.blocks.items(), b.blocks.items()):
        if blockA.numel() == 0 or blockB.numel() == 0:
            continue
        ab.blocks[gIdxA + gIdxB] = torch.tensordot(blockA, blockB, dims=0)
    if _DEBUG:
        ab.verify()
    return ab


def outer(*ts: GTensor):
    """
    direct (outer) product of two
    or more Grassmann tensors

    Example
    ----
    ```
    0   0       0   2
    ↓   ↓       ↓---↓
    a   b  -->  |   |
    ↓   ↓       ↓---↓
    1   1       1   3
    """
    assert len(ts) >= 2
    prod = _outer2(ts[0], ts[1])
    if len(ts) > 2:
        for t in ts[2::]:
            prod = _outer2(prod, t)
    return prod


def tensordot(a: GTensor, b: GTensor, axes):
    """
    Contract two GTensors `a` and `b`.
    The free indices of `a` are put before `b`.

    Parameters
    --------------
    axes: tuple[int, int] or tuple[list[int], list[int]]
        indices in `a` and `b` to be contracted

    Returns
    --------------
    c: GTensor
        contraction result
    """
    axisA, axisB, npairs = _process_contract_axes(axes, a.ndim, b.ndim)
    # check shape consistency
    assert all(
        all(a.shape[p][axA] == b.shape[p][axB] for p in range(2))
        for axA, axB in zip(axisA, axisB)
    ), "contraction shape mismatch"
    # put the indices to be contracted together (match indices from inner to outer)
    perma = [i for i in range(a.ndim) if i not in axisA] + axisA[::-1]
    permb = axisB + [i for i in range(b.ndim) if i not in axisB]
    # the last npairs axes are to be contracted
    # (in reverse order)
    a = a.transpose(*perma)
    # the first npairs axes are to be contracted
    b = b.transpose(*permb)
    # determine gMetric for contraction
    gMetric = _get_gMetric(
        tuple(range(a.ndim - 1, a.ndim - npairs - 1, -1)),
        tuple(range(0, npairs, 1)),
        a.dual,
        b.dual,
    )
    # initialize result GTensor
    cshape = tuple(a.shape[i][:-npairs] + b.shape[i][npairs:] for i in range(2))
    cdual = a.dual[:-npairs] + b.dual[npairs:]
    c = zeros(cshape, cdual, parity=(a.parity + b.parity) % 2)
    for (gIdxA, blockA), (gIdxB, blockB) in product(a.blocks.items(), b.blocks.items()):
        # sum over terms with the same
        # Grassmann index in the contracted axes
        if gIdxA[-1 : -npairs - 1 : -1] == gIdxB[0:npairs]:
            # skip evaluation for empty blocks
            if blockA.numel() == 0 or blockB.numel() == 0:
                continue
            # new Grassmann index = free A index + free B index
            gIdxC = gIdxA[0:-npairs] + gIdxB[npairs:]
            # new block
            try:
                tmp = torch.tensordot(
                    blockA, blockB,
                    [list(range(a.ndim - 1, a.ndim - npairs - 1, -1)), list(range(npairs))],
                )
            except RuntimeError:
                tmp = torch.tensordot(
                    blockA.to(torch.cdouble), blockB.to(torch.cdouble),
                    [list(range(a.ndim - 1, a.ndim - npairs - 1, -1)), list(range(npairs))],
                )
            c.blocks[gIdxC] += np.prod(gMetric ** gIdxB[0:npairs]) * tmp
    if _DEBUG:
        c.verify()
    return c


def dot_diag(a: GTensor, s: GTensor, axes: list[int]):
    """
    Tensordot of tensor `a` with diagonal matrix `s`
    (with even parity).

    Parameters
    ----
    axes: list[int]
        axes[0]: contracted axis of `self`
        axes[1]: contracted axis of `s` (0 or 1)
    """
    return a.dot_diag(s, axes)


def tensordot_keepform(a: GTensor, b: GTensor, axes, anchor="a"):
    """
    Contract two GTensors `a` and `b`, but the contracted
    axes of the `anchor` tensor are replaced by the remaining
    free indices of the other tensor.

    `axes` for anchor tensor must be a sequence of
    consecutive integers in ascending order.

    Parameters
    --------------
    axes: 2 sequences of ints
        indices in `a` and `b` to be contracted
    anchor: str ('a' or 'b')
        selecting the anchor tensor

    Returns
    --------------
    c: contraction result

    Example
    --------------
    When `a.shape = (3,4,1,2,3,5)`
    and  `b.shape = (3,2,4,3,5)`,
    contract ((0,1), (3,2)):
    - anchor == 'a'
        dimension of free index of `b` : (3,2,5)
        replace `a` index: new shape is
            ([3,4],1,2,3,5) -> ([3,2,5],1,2,3,5)
    - anchor == 'b' is not allowed, since (3,2) is not a consecutive list of ascending ints

    When `a.shape = (3,4,1,2,3,5)`
    and  `b.shape = (3,2,3,4,5)`
    contract ((0,1), (2,3))
    - anchor == 'a'
        dimension of free index of `b` : (3,2,5)
        replace `a` index:  new shape is
            ([3,4],1,2,3,5) -> ([3,2,5],1,2,3,5)
    - anchor == 'b'
        free index of `a` : (1,2,3,5)
        replace `b` index:  new shape is
            (3,2,[3,4],5) -> (3,2,[1,2,3,5],5)
    """
    axisA, axisB, _ = _process_contract_axes(axes, a.ndim, b.ndim)
    if anchor == "a":
        axisAnchor, axisOther, ndimOther = axisA, axisB, b.ndim
    elif anchor == "b":
        axisAnchor, axisOther, ndimOther = axisB, axisA, a.ndim
    else:
        raise ValueError("`anchor` must be 'a' or 'b'")
    if len(axisAnchor) > 1:
        for i, j in zip(axisAnchor, axisAnchor[1:]):
            if not j == i + 1:
                raise ValueError(
                    "`axes` for anchor tensor must be a sequence \
                    of ascending consecutive integers"
                )
    # Put the indices to be contracted together and contract
    c = tensordot(a, b, axes)
    # restore index order of the result
    nFreeAxis = ndimOther - len(axisOther)
    order = list(range(c.ndim))
    if anchor == 'a':
        perm = (
            order[:axisAnchor[0]] + order[-nFreeAxis:] 
            + order[axisAnchor[0]:-nFreeAxis]
        )
    else:
        perm = (
            order[nFreeAxis:nFreeAxis+axisAnchor[0]] 
            + order[:nFreeAxis] 
            + order[nFreeAxis+axisAnchor[0]:]
        )
    return c.transpose(*perm)


# @profile
def fncon(tensors: list[GTensor], indices: list[list[int]]):
    """
    Contract several GTensors using
    ncon summation notation (see arXiv 1402.0939).
    Self-contraction (partial trace) is not supported.

    - Contracted pairs of indices are labelled by 
        consecutive positive integers 1, 2, ...
    - Free indices are labelled by 
        consecutive negative integers -1, -2, ...
    - 0 is not allowed.

    The contraction order is defined by `indices`,
    instead of `tensors`
    """
    indices_ = deepcopy(indices)
    # number of contracted pairs of axes
    conlist = range(1, max(sum(indices_, [])) + 1)
    while len(conlist) > 0:
        Icon = []
        for i in range(len(indices_)):
            if conlist[0] in indices_[i]:
                Icon.append(i)
                if len(Icon) == 2:
                    break
        if len(Icon) == 1:
            raise ValueError("Error: self-trace is not implemented")
        else:
            IndCommon = list(set(indices_[Icon[0]]) & set(indices_[Icon[1]]))
            Pos = [[], []]
            for i in range(2):
                for ind in range(len(IndCommon)):
                    Pos[i].append(indices_[Icon[i]].index(IndCommon[ind]))
            A = tensordot(tensors[Icon[0]], tensors[Icon[1]], (Pos[0], Pos[1]))
            for i in range(2):
                for ind in range(len(IndCommon)):
                    indices_[Icon[i]].remove(IndCommon[ind])
            indices_[Icon[0]] = indices_[Icon[0]] + indices_[Icon[1]]
            indices_.pop(Icon[1])
            tensors[Icon[0]] = A
            tensors.pop(Icon[1])
        conlist = list(set(conlist) ^ set(IndCommon))
    while len(indices_) > 1:
        tensors[0] = outer(tensors[0], tensors[1])
        tensors.pop(1)
        indices_[0] = indices_[0] + indices_[1]
        indices_.pop(1)
    indices_ = indices_[0]
    if len(indices_) > 0:
        Order = sorted(
            range(len(indices_)), key=lambda k: indices_[k]
        )[::-1]
        result = transpose(tensors[0], Order)
    else:
        result = tensors[0]
    return result


# @profile
def einsum(subscripts: str, *tensors: GTensor, optimize=True):
    """
    contract several GTensors using
    Einstein summation notation

    manual dimension check should be run 
    to prevent unwanted broadcasting

    Parameters
    ----------
    subscripts: str
        Contraction string
    *tensors: GTensor
        Tensors to be contracted
    """

    def _get_strperm(s1: str, s2: str):
        """
        Find the permutation that brings string `s1` to `s2`
        """
        # Create a dictionary to store the positions of characters in s1
        char_positions = {char: i for i, char in enumerate(s1)}
        # Build the permutation list using list comprehension
        permutation = [char_positions[char] for char in s2]
        return permutation

    # remove blank spaces in the string
    split_subscripts = subscripts.replace(" ", "").split("->", maxsplit=1)
    if not 1 <= len(split_subscripts) <= 2:
        raise ValueError("Invalid contraction string")
    sum_scripts = split_subscripts[0]
    # number of contracted tensors
    n_ts = sum_scripts.count(",") + 1
    assert n_ts == len(
        tensors
    ), "number of GTensors and the string expression do not match"

    fused_str = sum_scripts.replace(",", "")
    assert len(fused_str) == sum(
        [t.ndim for t in tensors]
    ), "string length and total dimensions do not match"

    # all pairs of contracted axes
    pairs = []
    # the order of free axes after contraction
    scripts = ""
    for c in fused_str:
        if fused_str.count(c) == 2:
            first = fused_str.find(c)
            second = fused_str.find(c, first + 1)
            # if the second is found (but ignore if already found before)
            if second > 0 and (first, second) not in pairs:
                pairs.append((first, second))
        elif fused_str.count(c) == 1:
            scripts += c
        else:
            raise ValueError("Invalid contraction string")

    # flatten pairs
    contracted_axes = tuple(chain(*pairs))
    # new dual and shape by removing the contracted pairs
    fused_dual = tuple(chain(*[t.dual for t in tensors]))
    new_dual = tuple(
        d for ax, d in enumerate(fused_dual)
        if ax not in contracted_axes
    )
    fused_shape = tuple(tuple(chain(
        *[t.shape[p] for t in tensors]
    )) for p in range(2))
    new_shape = tuple(tuple(
        axdim for ax, axdim in enumerate(fused_shape[p])
        if ax not in contracted_axes
    ) for p in range(2))

    # calculate contraction
    result = zeros(
        new_shape, new_dual, 
        parity = sum(t.parity for t in tensors) % 2
    )
    path = None
    for gIdxs in product(*[t.blocks.keys() for t in tensors]):
        fused_gIdx = tuple(chain(*gIdxs))
        # the contracted blocks should have same Grassmann index
        # at the pairs of contracted axes
        try:
            for ax1, ax2 in pairs:
                assert fused_gIdx[ax1] == fused_gIdx[ax2]
        except AssertionError:
            continue
        # fermion sign for contraction
        sign = tools.get_permsign(fused_dual, fused_gIdx, pairs)
        # new Grassmann index by removing contracted ones
        gIdxNew = tuple(
            gIdx for i, gIdx in enumerate(fused_gIdx) 
            if i not in contracted_axes
        )
        # blocks to be contracted
        block_tensors = [
            t.blocks[gIdx]
            for t, gIdx in zip(tensors, gIdxs)
        ]
        if path is None and optimize is True:
            path = contract_path(
                sum_scripts + "->" + scripts, *block_tensors,
                optimize="dp"
            )
        result.blocks[gIdxNew] += sign * contract(
            sum_scripts + "->" + scripts, *block_tensors,
            optimize=(path[0] if path is not None else "auto")
        )

    # final transpose (follow numpy convention)
    # without "->" (implicit mode)
    # free axes are sorted in alphabetic order
    if len(split_subscripts) == 1:
        final_scripts = "".join(sorted(scripts))
    # with "->" (explicit mode)
    elif len(split_subscripts) == 2:
        final_scripts = split_subscripts[1]
    if final_scripts != scripts:
        perm = _get_strperm(scripts, final_scripts)
        result = result.transpose(*perm)
    return result


def is_diagonal(a: GTensor):
    """
    Check if a GTensor is a diagonal matrix
    """
    try:
        assert a.ndim == 2 and a.parity == 0
        assert all(utils.is_diagonal(blk) for blk in a.blocks.values())
        return True
    except AssertionError:
        return False


def is_identity_matrix(mat: GTensor):
    """
    Check if matrix `mat` is the same as
    identity up to a transposition
    """
    assert mat.ndim == 2, "`mat` should be a matrix"
    if not all(mat.shape[p][0] == mat.shape[p][1] for p in range(2)):
        return False
    iden = eye(mat.shape[0][0], mat.shape[1][0], dual=(0, 1))
    if mat.dual == (1, 0):
        return allclose(iden, mat.transpose(1, 0))
    else:
        return allclose(iden, mat)


def trace(a: GTensor, axis1=0, axis2=1):
    """
    Find the trace of a GTensor `a` over `axis1` and `axis2`.

    This function actually calls `einsum`.

    Parameters
    ----------
    axis1, axis2: int or list[int]
        Pair(s) of axes to be traced
        `axis1` is put in front of `axis2`
    """
    # check axis1, axis2
    try:
        axis1 = tuple(axis1)
    except TypeError:
        axis1 = (axis1,)
    try:
        axis2 = tuple(axis2)
    except TypeError:
        axis2 = (axis2,)
    assert len(axis1) == len(axis2)
    npairs = len(axis1)
    # generate einsum string
    assert a.ndim <= 26 * 2
    chars = [char for char in ascii_letters[0 : a.ndim]]
    for ax1, ax2 in zip(axis1, axis2):
        chars[ax2] = chars[ax1]
    subscripts = "".join(chars)
    tr = einsum(subscripts, a)
    return tr


def norm(a: GTensor):
    r"""
    Calculate 2-norm of GTensor `a`, which is the
    
    |a| = sqrt(sum_{...} a_{...})
    """
    nrm = sum(
        torch.sum(torch.abs(arr) ** 2) 
        for arr in a.blocks.values()
    )
    return sqrt(nrm)


def round(a: GTensor, decimals=10):
    a2 = a.copy()
    for gIdx, block in a.blocks.items():
        a2.blocks[gIdx] = (
            torch.round(block.real, decimals=decimals)
            + 1.0j * torch.round(block.imag, decimals=decimals)
        )
    return a2


# ------ Reconstruct from 1D array ------


def flatten(a: GTensor):
    """
    Flatten the Grassmann tensor to a 1D PyTorch array.
    """
    return a.flatten()


def unflatten(
    shape: tuple[tuple[int, ...], tuple[int, ...]],
    dual: tuple[int, ...],
    parity: int,
    v: Tensor,
):
    """
    Reconstruct Grassmann tensor from 1D array
    """
    idxStart = 0
    assert len(shape[0]) == len(shape[1]) == len(dual)
    a = zeros(shape, dual, parity)
    for gIdx in tools.gen_gidx(len(shape[0]), parity):
        blkshape = tools.gidx_shape(gIdx, shape)
        blksize = prod(blkshape)
        a.blocks[gIdx] = v[idxStart : idxStart + blksize].reshape(blkshape)
        idxStart += blksize
    return a
