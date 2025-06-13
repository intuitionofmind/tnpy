"""
Approximate local measurement on TPS
(accept both symmetric and loop convention)
"""

import gtensor as gt
from gtensor import GTensor
from cmath import isclose as cisclose
from sys import float_info
from update_ftps.sutools import *
from update_ftps.sutools import get_dualconv


def meas_site(
    tname: str, op: GTensor, 
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
):
    """
    Approximate measurement on one site

    Parameters
    ----
    tname: str
        name of the tensor to be measured
    op: GTensor
        the operator to be measured
    """
    t4 = (len(tensors) == 4)
    dualconv = get_dualconv(tensors, weights)
    t = absorb_envwts(
        tensors[tname], weights, get_tswts(tname, t4), 
        dualconv, mode="absorb", skip=None, 
    )
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
    Approximate measurement on a bond
    (only support nearest neighbor terms)

    Site order for `ops`:
    [left, right] or [bottom, top]

    Parameters
    ----
    wtkey: str
        weight name on the bond to be measured
    ops: list[None | GTensor]
        the operators to be measured
        (left-right or bottom-top)
    """
    # conv = get_pepsconv(tensors["Ta"])
    # assert conv == "sym"
    assert len(ops) == 2
    t4 = len(tensors) == 4
    dualconv = get_dualconv(tensors, weights)
    # tensors connected by the bond (left-right or botton-top)
    tnames = get_bondts(wtkey, t4, conv="sym")
    
    # absorb environment weights
    # the bond weight is absorbed to the left/bottom tensor
    skips = [None, 3 if "x" in wtkey else 4]
    ts = list(absorb_envwts(
        tensors[tname], weights, get_tswts(tname, t4), 
        dualconv, mode="absorb", skip=skip
    ) for tname, skip in zip(tnames, skips))
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
    axes1, axes2 = (
        ([0,2,3,4], [0,1,2,4]) if "x" in wtkey
        else ([0,1,3,4], [0,1,2,3])
    )
    # <psi|op|psi>
    meas = gt.tensordot(
        gt.tensordot(t1.gT, opt1.pconj(axes1), [axes1]*2),
        gt.tensordot(t2.gT, opt2.pconj(axes2), [axes2]*2), ((0,1),(0,1))
    ).item()
    # <psi|psi>
    nrm = gt.tensordot(
        gt.tensordot(t1.gT, t1.pconj(axes1), [axes1]*2),
        gt.tensordot(t2.gT, t2.pconj(axes2), [axes2]*2), ((0,1),(0,1))
    ).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real


def meas_loop(
    loopname: int, ops: list[None | GTensor], 
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
):
    """
    Approximate measurement on a 2 x 2 loop
    (support up to 2nd neighbor terms)

    Site order for `ops`
    ----
    ```
        1 --- 3
        |     |
        0 --- 2
    ```

    Parameters
    ----
    loopname: int (blue, green, red, pink)
        the 2 x 2 loop on which the measurement is done
    ops: dict[str, GTensor]
        the operators to be measured on sites x,y,z,w
        (on clockwise or counter-clockwise loops)
    """
    t4 = len(tensors) == 4
    dualconv = get_dualconv(tensors, weights)
    # conv = get_pepsconv(tensors["Ta"])
    # assert conv == "sym"
    assert len(ops) == 4
    # tensors on the loop
    tnames = get_loopts(loopname)
    # convert clockwise loop to counter-clockwise loop
    if loopname in (3, 4):
        tnames = tnames[::-1]
    tnames = [tnames[i] for i in (0,3,1,2)]
    if t4 is False:
        for i, tname in enumerate(tnames):
            if tname == "Tc": tnames[i] = "Ta"
            elif tname == "Td": tnames[i] = "Tb"
    # absorb environment and bond weights
    # bond weights are absorbed to 
    # lower-left and upper-right tensors
    skips = [None, [4,1], [2,3], None]
    ts = list(absorb_envwts(
        tensors[tname], weights, get_tswts(tname, t4), 
        dualconv, mode="absorb", skip=skip
    ) for tname, skip in zip(tnames, skips))
    # apply operators
    opts = list(
        gt.tensordot(op, t, (1,0)) 
        if op is not None else t
        for t, op in zip(ts, ops)
    )
    
    # fully contract axes
    # contraction order: XWYZ 
    # physical and environment axes
    ts_axes = [[0,3,4], [0,2,3], [0,4,1], [0,1,2]] 
    einsum_str = 'adeh,cdgh,bafe,cbgf' 
    meas = gt.einsum(einsum_str, *[
        gt.tensordot(t.gT, opt.pconj(axes), [axes]*2)
        for t, opt, axes in zip(ts, opts, ts_axes)
    ]).item()
    nrm = gt.einsum(einsum_str, *[
        gt.tensordot(t.gT, t.pconj(axes), [axes]*2)
        for t, axes in zip(ts, ts_axes)
    ]).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real


def meas_site4(
    ops: list[None | GTensor], direction: str,
    tensors: dict[str, GTensor], 
    weights: dict[str, GTensor],
):
    """
    Approximate measurement on consecutive 
    4 sites in the same row (h) / column (v)

    Site order for `ops`
    ----
    - direction = 'h': (left -> right) 
    - direction = 'v': (bottom -> top)
    """
    # conv = get_pepsconv(tensors["Ta"])
    # assert conv == "sym"
    assert len(ops) == 4
    assert direction in ('h', 'v')
    t4 = (len(tensors) == 4)
    dualconv = get_dualconv(tensors, weights)
    # absorb environment and bond weights
    # bond weights are absorbed to 
    # lower-left and upper-right tensors
    if direction == 'h':
        tnames = ["Ta", "Tb", "Ta", "Tb"]
    else:
        tnames = (
            ["Ta", "Td", "Ta", "Td"] if t4 else 
            ["Ta", "Tb", "Ta", "Tb"]
        )
    skips = [None] + [[3 if direction == 'h' else 4]]*3
    # bond weights are absorbed to 
    # the tensor to the left/bottom
    ts = list(absorb_envwts(
        tensors[tname], weights, get_tswts(tname, t4), 
        dualconv, mode="absorb", skip=skip
    ) for tname, skip in zip(tnames, skips))
    # apply operators
    opts = list(
        gt.tensordot(op, t, (1,0)) 
        if op is not None else t
        for t, op in zip(ts, ops)
    )

    # fully contract axes
    # <psi|op|psi>
    # physical and env axes
    ts_axes = (
        [[0,2,3,4], [0,2,4], [0,2,4], [0,1,2,4]]
        if direction == 'h' else
        [[0,1,3,4], [0,1,3], [0,1,3], [0,1,2,3]]
    )
    # contraction order: XWYZ 
    einsum_str = 'ad,baed,cbfe,cf'
    meas = gt.einsum(einsum_str, *[
        # first contract physical and env axes
        gt.tensordot(t.gT, opt.pconj(axes), [axes]*2)
        for t, opt, axes in zip(ts, opts, ts_axes)
    ]).item()
    nrm = gt.einsum(einsum_str, *[
        gt.tensordot(t.gT, t.pconj(axes), [axes]*2)
        for t, axes in zip(ts, ts_axes)
    ]).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real

