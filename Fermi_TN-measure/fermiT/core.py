import numpy as np 
from itertools import chain
from .tools import *
from copy import deepcopy
from torch import Tensor

# import atexit
# import line_profiler as lp
# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

class FermiT(object):
    DS: np.ndarray
    DE: np.ndarray
    dual: np.ndarray
    val: np.ndarray

    all_signs = {}
    all_dual_signs = {}
    all_index_parameters = {}

    def clear():
        FermiT.all_signs = {}
        FermiT.all_dual_signs = {}
        FermiT.all_index_parameters = {}
        print("clear")

    def __init__(self, DS: np.ndarray, DE: np.ndarray, dual: np.ndarray, val: np.ndarray):
        self.DS = np.asarray(DS, dtype=int)
        self.DE = np.asarray(DE, dtype=int)
        self.dual = np.asarray(dual, dtype=int)
        self.val = val
        if val.shape and not (np.array(val.shape) == self.DS).sum(): 
            raise ValueError("shape error " + str(self.DS) + " " + str(np.array(val.shape)))
    
    def __add__(self, fermiT2: "FermiT"):
        return FermiT(self.DS, self.DE, self.dual, self.val + fermiT2.val)
    
    def __sub__(self, fermiT2: "FermiT"):
        return FermiT(self.DS, self.DE, self.dual, self.val - fermiT2.val)

    def __mul__(self, constant):
        return FermiT(self.DS, self.DE, self.dual, self.val * constant)

    def __rmul__(self, constant):
        return FermiT(self.DS, self.DE, self.dual, self.val * constant)
    
    def __truediv__(self, constant):
        return FermiT(self.DS, self.DE, self.dual, self.val / constant)
    
    def __neg__(self):
        return FermiT(self.DS, self.DE, self.dual, -self.val)

    def _pads(self) -> list[np.ndarray]:
        """
        Create masks to select odd parity
        elements along a given axis
        """
        DS = self.DS
        DE = self.DE
        ndim = self.ndim
        onedim = np.ones(DS, dtype=np.dtype('i1'))
        pads = [None]*ndim
        # print(dim)
        for i in range(ndim):
            shape_pad = np.ones(ndim, dtype=np.dtype('i8'))
            shape_pad[i] = DS[i]
            pad = np.concatenate([
                np.zeros(DE[i], dtype=np.dtype('i1')), 
                np.ones(DS[i]-DE[i], dtype=np.dtype('i1'))
            ], axis=0)
            # print(pad, shape_pad)
            pads[i] = pad.reshape(shape_pad) * onedim
        return pads
    
    def _sign(self, order) -> np.ndarray:
        """
        Calculate minus sign mask produced by transposition
        """
        order = np.array(order, dtype=int)
        DS = self.DS
        DE = self.DE
        name = str(DS *10000 + DE * 10 + order)
        if name in self.all_signs.keys():
            return self.all_signs[name]
        dim = DS.shape[0]
        pads = self._pads()
        signT = np.zeros(DS, dtype=np.dtype('i1'))
        for i in range(dim):
            m1 = np.zeros(DS, dtype=np.dtype('i1'))
            for j in range(i+1, dim):
                if order[j] < order[i]:
                    m1 += pads[order[j]]
            signT += pads[order[i]] * m1
        signT = (-1) ** signT
        self.all_signs[name] = signT
        return signT
    
    def transpose(self, *order: int):
        """Transposition (reordering) of tensor axes"""
        order = np.asarray(order)
        signT = self._sign(order)
        val = (self.val * signT).transpose(order)
        return FermiT(val.shape, self.DE[order], self.dual[order], val)
    
    def gconj(self):
        """Hermitian conjugate"""
        DS = self.DS[::-1]
        DE = self.DE[::-1]
        dual = 1 - self.dual[::-1]
        order = range(self.ndim)[::-1]
        val = np.conj(self.val).transpose(order)
        return FermiT(DS, DE, dual, val)

    def flip_dual(self, axes, minus=True, change_dual=True):
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
        try:                axes = list(axes)
        except TypeError:   axes = [axes]
        if len(axes) == 0:  return self.copy()
        # deal with negative numbers in axes
        axes = regularize_axes(axes, self.ndim)
        aNew = self.copy()
        if change_dual:
            aNew.dual[:] = np.array([
                1-d if ax in axes else d 
                for ax, d in enumerate(self.dual)
            ])
        if minus is True:
            # multiply -1 to odd sector of `axes`
            flipper = np.array([
                -1 if i in axes else 1 for i in range(self.ndim)
            ], dtype=int)
            for gIdx in gen_gidx(self.ndim, self.parity):
                gSlice = get_slice(self.DS, self.DE, gIdx)
                aNew.val[gSlice] *= np.prod(flipper**gIdx)
        return aNew

    def proj_to_parity(self, parity: int):
        """
        Project out elements that break the definite parity.
        The tensor is modified in place.
        
        Parameters
        ----
        parity: int (0, 1)
            the parity of `self` after the projection
        """
        def _get_fpad(t: FermiT, parity=0):
            """Generate projector into given parity"""
            DS = t.DS
            ndim = t.ndim
            if parity == 0:
                fpad = np.ones(DS, dtype=np.dtype('i1'))
            else:
                fpad = np.zeros(DS, dtype=np.dtype('i1'))
            pads = t._pads()
            for i in range(ndim):
                fpad = fpad + pads[i]
            return fpad % 2
        pad = _get_fpad(self, parity)
        self.val = pad * self.val
        if self.val.any():
            assert self.parity == parity

    def copy(self):
        DS = self.DS.copy()
        DE = self.DE.copy()
        dual = self.dual.copy()
        val = self.val.copy()
        return FermiT(DS, DE, dual, val)
    
    def print_info(self):
        print(self.DS, self.DE, self.dual, self.val.shape)

    def blocks(self, gIdx: tuple[int, ...]):
        """Extract even/odd block corresponding to given fermion index `gIdx`"""
        gSlice = get_slice(self.DS, self.DE, gIdx)
        return self.val[gSlice]
    
    def update(self, blkdict: dict[tuple[int, ...], np.ndarray]):
        """Update blocks of FermiT from input"""
        if not isinstance(blkdict, dict):
            raise TypeError('`blocks` must be a dict')
        for gIdx, blk in blkdict.items():
            gSlice = get_slice(self.DS, self.DE, gIdx)
            self.val[gSlice] = blk

    def update_torch(self, blkdict: dict[tuple[int, ...], Tensor]):
        """Update blocks of FermiT from input"""
        if not isinstance(blkdict, dict):
            raise TypeError('`blocks` must be a dict')
        for gIdx, blk in blkdict.items():
            gSlice = get_slice(self.DS, self.DE, gIdx)
            self.val[gSlice] = blk.numpy(force=True)

    def item(self) -> complex:
        """
        Return the only element (as number, not array) 
        of a 0-dimensional (even parity) fermion tensor
        """
        if (not self.DS.any()):
            return self.val.item()
        else:
            raise ValueError('only dim-0 even-parity tensors have `item`')

    def all_blocks(self) -> dict[tuple[int,...], np.ndarray]:
        """
        Return the dictionary of all fermion-index-labelled blocks
        """
        par = self.parity
        blkdict = dict(
            (gIdx, self.blocks(gIdx))
            for gIdx in (
                gen_gidx(self.ndim, parity=par) if par in (0, 1)
                else chain(gen_gidx(self.ndim, 0), gen_gidx(self.ndim, 1))
            )
        )
        return blkdict

    @property
    def ndim(self):
        """Number of axes"""
        return self.val.ndim
    
    @property
    def size(self):
        """Number of nonzero elements"""
        return self.val.size // 2

    @property 
    def parity(self):
        """
        Parity of the fermion tensor: 
        0 (even), 1 (odd), -1 (indefinite)
        
        Zero tensors are all regarded as even parity tensors
        """
        # check if all odd parity blocks are zero
        odd_allzero = True
        for gIdx in gen_gidx(self.ndim, 1):
            if self.blocks(gIdx).any():
                odd_allzero = False
                break 
        # check if all even parity blocks are zero
        even_allzero = True
        for gIdx in gen_gidx(self.ndim, 0):
            if self.blocks(gIdx).any():
                even_allzero = False
                break 
        if (
            (even_allzero is False and odd_allzero is True) or
            (even_allzero is True and odd_allzero is True) # zero tensor
        ):
            return 0
        elif even_allzero is True and odd_allzero is False:
            return 1
        else:
            return -1

    @property
    def DO(self) -> np.ndarray:
        """Odd dimension of axes"""
        return self.DS - self.DE
    
    @property
    def shape(self) -> np.ndarray:
        """Even and odd dimension of axes"""
        return np.stack([self.DE, self.DO])

    @property
    def real(self):
        """Real part"""
        return FermiT(self.DS, self.DE, self.dual, self.val.real)
    
    @property
    def imag(self):
        """Imaginary part"""
        return FermiT(self.DS, self.DE, self.dual, self.val.imag)

    @property
    def T(self):
        """Reverse axis order"""
        return self.transpose(*tuple(reversed(range(self.ndim))))

    @property
    def gT(self):
        """
        Hermitian conjugate followed by 
        full transpose to preserve axis order
        """
        return self.gconj().T


def rand(DS, DE, dual, parity=0, positive=False):
    """Create random FermiT"""
    assert parity in (-1, 0, 1)
    val = np.random.rand(*DS) - 0.5 + 1j*(np.random.rand(*DS) - 0.5)
    if positive: 
        val = np.random.rand(*DS) + 1j * np.random.rand(*DS)
    randt = FermiT(DS, DE, dual, val)
    # create tensor with definite parity
    if parity != -1:
        randt.proj_to_parity(parity)
    return randt


def zeros(DS, DE, dual):
    """Create zero FermiT"""
    return FermiT(DS, DE, dual, val=np.zeros(DS, dtype=complex))


def eye(DS, DE, dual):
    """Create identity FermiT"""
    return FermiT(DS, DE, dual, np.eye(DS[0], dtype=complex))


def _get_pads(DS: np.ndarray, DE: np.ndarray):
    dim = DS.shape[0]
    pads = [None]*dim
    onedim = np.ones(DS, dtype=np.dtype('i1'))
    for i in range(dim):
        shape_pad = np.ones(dim, dtype=np.dtype('i8'))
        shape_pad[i] = DS[i]
        pad = np.concatenate([
            np.zeros(DE[i], dtype=np.dtype('i1')), 
            np.ones(DS[i]-DE[i], dtype=np.dtype('i1'))
        ], axis=0)
        pads[i] = pad.reshape(shape_pad) * onedim
    return pads


def _get_sign(fermiT: FermiT, order):
    """
    Calculate minus sign produced by transposition
    """
    return fermiT._sign(order)


def transpose(tensor: FermiT, order: list[int]):
    return tensor.transpose(*order)


def flip_dual(a: FermiT, axes, minus=True, change_dual=True):
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


def flip2_dual(a: FermiT, b: FermiT, axes, flip="a"):
    """
    When contracting tensors `a` and `b`, 
    this function change the dual of their axes
    to be contracted and modify the elements 
    so that the contraction result is unchanged.

    Parameters
    ----
    a, b: FermiT
        the tensors to be contracted
    axes: list[int] or list[list[int]]
        axes of `a` and `b` to be contracted
    flip: str ("a" or "b")
        add minus signs to `a` or `b`
    """
    assert flip == "a" or flip == "b"
    axes = process_flip_axes(axes)
    if len(axes[0]) == 0:
        return a, b
    aNew = a.flip_dual(axes[0], minus=(flip=="a"))
    bNew = b.flip_dual(axes[1], minus=(flip=="b"))
    return aNew, bNew


def _get_sign_dual(fermiT2: FermiT, conlist2):
    DS = fermiT2.DS
    DE = fermiT2.DE
    dual = fermiT2.dual
    name = str(DS*10000 + DE*10 + dual) + str(np.array(conlist2))
    if name in fermiT2.all_dual_signs.keys():
        return fermiT2.all_dual_signs[name]
    pads = _get_pads(DS, DE)
    sign_dual = np.zeros(DS, dtype=np.dtype('i1'))
    for i in conlist2:
        if dual[i] == 1:
            sign_dual += pads[i]
    sign_dual = (-1) ** sign_dual
    fermiT2.all_dual_signs[name] = sign_dual
    return sign_dual


def tensordot(a: FermiT, b: FermiT, conlists: list):
    DE = []
    dual = []
    order1 = []
    order2 = []
    if isinstance(conlists[0], int):
        conlist1 = [conlists[0]]    
        conlist2 = [conlists[1]]
    else:
        conlist1 = list(conlists[0])
        conlist2 = list(conlists[1])[::-1]
    for i in range(a.DS.shape[0]):
        if i not in conlist1:
            order1.append(i)
            DE.append(a.DE[i])
            dual.append(a.dual[i])
    for i in range(b.DS.shape[0]):
        if i not in conlist2:
            order2.append(i)
            DE.append(b.DE[i])
            dual.append(b.dual[i])
    for i in range(len(conlist2)):
        # check dual match
        if b.dual[conlist2[i]] == a.dual[conlist1[-i-1]]:
            err = (
                "tensordot dual unmatch: "
                + f"tensor 1 axis {conlist1[-i-1]}, dual {a.dual}; "
                + f"tensor 2 axis {conlist2[i]}, dual {b.dual}; "
            )
            raise ValueError(err)
        # check shape match
        if (b.DS[conlist2[i]], b.DE[conlist2[i]]) != (a.DS[conlist1[-i-1]], a.DE[conlist1[-i-1]]):
            err = (
                "tensordot shape unmatch: "
                + f"tensor 1 axis {conlist1[-i-1]}, DE {a.DE}, DO {a.DO}; "
                + f"tensor 2 axis {conlist2[i]}, DE {b.DE}, DO {b.DO}; "
            )
            raise ValueError(err)
    order1 = order1 + conlist1
    order2 = conlist2 + order2
    signT1 = _get_sign(a, order1)
    signT2 = _get_sign(b, order2)
    signT2_dual = _get_sign_dual(b, conlist2)
    # val = np.tensordot(fermiT1.val*signT1, fermiT2.val*signT2*signT2_dual, conlists)
    a.val *= signT1
    b.val *= signT2 * signT2_dual
    val = np.tensordot(a.val, b.val, conlists)
    # undo the change to input tensors
    a.val *= signT1
    b.val *= signT2 * signT2_dual
    return FermiT(val.shape, DE, dual, val)


def outer(fermiT1: FermiT, fermiT2: FermiT):
    """Outer (direct) product of two FermiT's"""
    DS = np.append(fermiT1.DS, fermiT2.DS)
    DE = np.append(fermiT1.DE, fermiT2.DE)
    dual = np.append(fermiT1.dual, fermiT2.dual)
    val = np.multiply.outer(fermiT1.val, fermiT2.val)
    return FermiT(DS, DE, dual, val)


# @profile
def fncon(tensors: list[FermiT], indices: list[list[int]]) -> FermiT:
    """Contract multiple tensors using NCON syntax"""
    indices_ = deepcopy(indices)
    conlist = range(1,max(sum(indices_,[]))+1)
    while len(conlist) > 0:
        Icon = []
        for i in range(len(indices_)):
            if conlist[0] in indices_[i]:
                Icon.append(i)
                if len(Icon) == 2:
                    break
        if len(Icon) == 1:
            raise ValueError("Error: no self trace")
        else:
            IndCommon = list(set(indices_[Icon[0]]) & set(indices_[Icon[1]]))
            Pos = [[],[]]
            for i in range(2):
                for ind in range(len(IndCommon)):
                    Pos[i].append(indices_[Icon[i]].index(IndCommon[ind]))
            A = tensordot(tensors[Icon[0]],tensors[Icon[1]],(Pos[0],Pos[1]))
            
            for i in range(2):
                for ind in range(len(IndCommon)):
                    indices_[Icon[i]].remove(IndCommon[ind])
            indices_[Icon[0]] = indices_[Icon[0]]+indices_[Icon[1]]
            indices_.pop(Icon[1])
            tensors[Icon[0]] = A
            tensors.pop(Icon[1])
        conlist = list(set(conlist)^set(IndCommon))
    while len(indices_) > 1:
        tensors[0] = outer(tensors[0],tensors[1])
        tensors.pop(1)
        indices_[0] = indices_[0]+indices_[1]
        indices_.pop(1)
    indices_ = indices_[0]
    if len(indices_) > 0:
        Order = sorted(range(len(indices_)),key=lambda k:indices_[k])[::-1]
        result = transpose(tensors[0],Order)
    else:
        result = tensors[0]
    return result


def save(filename: str, tensor: FermiT):
    """Save FermiT"""
    assert filename[-4::] == '.npz'
    np.savez(
        filename, DS=tensor.DS, DE=tensor.DE, 
        Dual=tensor.dual, val=tensor.val
    )


def load(filename: str):
    """Load FermiT"""
    tmp = np.load(filename)
    return FermiT(tmp["DS"], tmp["DE"], tmp["Dual"], tmp["val"])


def allclose(fts1:FermiT, fts2:FermiT):
    return np.allclose(fts1.val, fts2.val) \
        and np.array_equal(fts1.dual, fts2.dual) \
        and np.array_equal(fts1.DE, fts2.DE) \
        and np.array_equal(fts1.DS, fts2.DS)


def array_equal(fts1:FermiT, fts2:FermiT):
    return np.array_equal(fts1.val, fts2.val) \
        and np.array_equal(fts1.dual, fts2.dual) \
        and np.array_equal(fts1.DE, fts2.DE) \
        and np.array_equal(fts1.DS, fts2.DS)

