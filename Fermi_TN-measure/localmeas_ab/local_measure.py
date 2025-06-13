"""
Approximate local measurement on honeycomb lattice TPS
"""

import re
import gtensor as gt
from gtensor import GTensor
from cmath import isclose as cisclose
from sys import float_info
from update_ftps.sutools import *


def meas_site(
    tname: str, op: GTensor, 
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
):
    """
    Approximate measurement on one site
    """
    assert tname in ("Ta", "Tb")
    t = tensors[tname]
    nbond = len(weights)
    # absorb weight
    ax_wt = 0 if tname == "Ta" else 1
    for ax in range(1, nbond+1):
        t = gt.dot_diag(t, weights[f"w{ax}"], [ax, ax_wt])
    meas = gt.tensordot(op, t, (1,0))
    axes = (tuple(range(t.ndim)),) * 2
    map_axes = list(range(t.ndim))
    meas = gt.tensordot(t.gT, meas.pconj(map_axes), axes).item()
    nrm = gt.tensordot(t.gT, t.pconj(map_axes), axes).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real


def meas_bond(
    wtkey: str, ops: list[None | GTensor], 
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
):
    """
    Approximate measurement on a nearest neighbor A-B bond

    Site order for `ops`: [on Ta, on Tb]
    """
    assert len(ops) == 2
    nbond = len(weights)
    assert bool(re.fullmatch(r'w[1-4]', wtkey))
    skip = int(wtkey[1])
    # absorb environment weights
    # the bond weight is absorbed to Ta
    ts = [tensors["Ta"].copy(), tensors["Tb"].copy()]
    for ax in range(1, nbond+1):
        ts[0] = gt.dot_diag(ts[0], weights[f"w{ax}"], [ax, 0])
        if ax != skip:
            ts[1] = gt.dot_diag(ts[1], weights[f"w{ax}"], [ax, 1])
    # apply operators
    opts = list(
        gt.tensordot(op, t, (1,0)) 
        if op is not None else t
        for t, op in zip(ts, ops)
    )
    # fully contract axes
    t1, t2 = ts
    opt1, opt2 = opts
    # physical and environment axes
    axes = [0] + list(ax for ax in range(1, nbond+1) if ax != skip)
    # <psi|op|psi>
    meas = gt.tensordot(
        gt.tensordot(t1.gT, opt1.pconj(axes), [axes]*2),
        gt.tensordot(t2.gT, opt2.pconj(axes), [axes]*2), ((0,1),(0,1))
    ).item()
    # <psi|psi>
    nrm = gt.tensordot(
        gt.tensordot(t1.gT, t1.pconj(axes), [axes]*2),
        gt.tensordot(t2.gT, t2.pconj(axes), [axes]*2), ((0,1),(0,1))
    ).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real
