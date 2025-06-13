from .core import *
from .linalg import norm
import gtensor as gt
from gtensor.tools import gen_gidx
from gtensor import GTensor
from math import isclose
import torch

def ft2gt(fts: FermiT, chop=False):
    """
    Convert FermiT to GTensor
    
    Parameters
    ----
    chop: bool
        set close-to-0 elements to 0 in a FermiT
    """
    # parity
    parity = fts.parity
    if parity == -1:
        if chop:
            norm_all  = norm(fts)
            fts_even  = fts.copy()
            fts_even.proj_to_parity(0)
            norm_even = norm(fts_even)
            fts_odd   = fts.copy()
            fts_odd.proj_to_parity(1)
            norm_odd  = norm(fts_odd)
            if isclose(norm_all, norm_even):
                fts = fts_even
                parity = 0
            elif isclose(norm_all, norm_odd):
                fts = fts_odd
                parity = 1
            else:
                pass
        else:
            raise NotImplementedError("Unable to convert FermiT with indefinite parity.")
    # shape
    gtshape = (tuple(fts.DE), tuple(fts.DS - fts.DE))
    # blocks (convert to complex numbers)
    blocks = dict(
        (gIdx, torch.from_numpy(fts.blocks(gIdx).astype(complex)))
        for gIdx in gen_gidx(len(fts.DS), parity=parity)
    )
    # dual
    dual = tuple(int(d) for d in fts.dual)
    gts = gt.GTensor(gtshape, dual, blocks=blocks)
    gts.verify()
    return gts

def gt2ft(gts: GTensor):
    """Convert GTensor to FermiT"""
    DE = np.array(gts.DE)
    DS = np.array(gts.DS)
    Dual = np.array(gts.dual, dtype=int)
    fts = zeros(DS, DE, Dual)
    fts.update_torch(gts.blocks)
    assert fts.parity == gts.parity
    return fts
