import torch
torch.set_default_dtype(torch.float64)
from .core import GTensor, zeros
from itertools import product
import numpy as np
from .tools import gen_gidx

# ------ Save or load tensor ------


def save(filename: str, tensor: GTensor):
    """Save GTensor to NumPy arrays format file (npz)"""
    assert filename[-4::] == ".npz"
    # save dual and blocks
    dicttmp = {}
    dicttmp["dual"] = np.array(tensor.dual, dtype=int)
    for gIdx, block in tensor.blocks.items():
        # e.g. block (1,0,0,1) is labelled by 'b1001'
        key = "b" + "".join(str(g) for g in gIdx)
        dicttmp[key] = block.numpy()
    np.savez(filename, **dicttmp)


def load(filename: str, dual=None):
    """
    Load GTensor from saved NumPy arrays.
    All loaded tensors has dtype `torch.cdouble`

    Parameters
    ----
    dual: None or tuple[int, ...]
        Manually input dual when loading legacy files
    """
    assert filename.endswith(".npz")
    tmp = np.load(filename)
    # determine parity
    parity = np.sum(np.fromiter((g for g in tmp.files[-1][1::]), int)) % 2
    # determine ndim
    ndim = len(tmp.files[-1]) - 1
    # determine dual
    if dual is None:
        dual = tuple(tmp["dual"])
    else:
        dual = tuple(dual)
    assert len(dual) == ndim
    for d in dual:
        assert d == 0 or d == 1
    # load blocks into a dict
    blocks = dict(
        (gIdx, torch.from_numpy(
            tmp['b' + ''.join(str(g) for g in gIdx)]
        ).type(torch.cdouble))
        for gIdx in gen_gidx(ndim, parity)
    )
    # get shape from input
    ndim = len(next(iter(blocks.keys())))
    shape = [[0] * ndim, [0] * ndim]
    for ax, par in product(range(ndim), range(2)):
        for gIdx in blocks:
            if gIdx[ax] == par:
                shape[par][ax] = (blocks[gIdx].shape)[ax]
                continue
    return GTensor(shape, dual, parity=parity, blocks=blocks)
