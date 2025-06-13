"""
Approximate local measurement on sites and bonds
"""

import gtensor as gt
from gtensor import GTensor
from cmath import isclose as cisclose
from sys import float_info
from update_ftps_mn.sutools import absorb_envwts


def meas_site(
    i: int, j: int, op: GTensor, 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    """
    Approximate measurement on one site

    weights surrounding a site
    ```
                    y[i,j]
                    |
        x[i,j-1] -- T[i,j] -- x[i,j]
                    |
                    y[i-1,j]
    ```
    """
    N1, N2 = len(ts), len(ts[0])
    assert 0 <= i < N1 and 0 <= j < N2
    # absorb weight
    t = absorb_envwts(ts[i][j], i, j, wxs, wys, mode="absorb")
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
    direction: str, i: int, j: int, 
    ops: list[None | GTensor], 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    """
    Approximate measurement on a nearest neighbor bond

    wxs[i][j]    
    ```
                    y[i,j]              y[i,j+1]
                    ↑                   ↑
        x[i,j-1] → T[i,j] → x[i,j] → T[i,j+1] → x[i,j+1]
                    ↑                   ↑
                    y[i-1,j]            y[i-1,j+1]
    ```

    wys[i][j]
    ```
                        y[i+1,j]
                        ↑
        x[i+1,j-1] → T[i+1,j] → x[i+1,j]
                        ↑
                        y[i,j]
                        ↑
        x[i,j-1] -→- T[i,j] -→- x[i,j]
                        ↑
                        y[i-1,j]
    ```

    Site order for `ops`: [left, right] or [bottom, top]
    """
    N1, N2 = len(ts), len(ts[0])
    assert 0 <= i < N1 and 0 <= j < N2
    assert len(ops) == 2
    assert direction in ("x", "y")
    # absorb environment weights
    # the bond weight is absorbed to t1
    i2, j2 = (
        (i, (j+1)%N2) if direction == 'x' 
        else ((i+1)%N1, j)
    )
    skips1 = [1] if direction == 'x' else [2]
    skips2 = [3] if direction == 'x' else [4]
    t1 = absorb_envwts(ts[i][j], i, j, wxs, wys, [], "absorb")
    t2 = absorb_envwts(ts[i2][j2], i2, j2, wxs, wys, skips2, "absorb")
    # apply operators
    opts = list(
        gt.tensordot(op, t, (1,0)) 
        if op is not None else t
        for t, op in zip((t1, t2), ops)
    )
    # fully contract axes
    opt1, opt2 = opts
    # physical and environment axes
    nbond = 4
    axes1 = list(ax for ax in range(nbond+1) if ax not in skips1)
    axes2 = list(ax for ax in range(nbond+1) if ax not in skips2)
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
    i: int, j: int, ops: list[None | GTensor], 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    """
    Approximate measurement on a square plaquette
    with lower-left corner at [i,j]
    (usually used to measure 2nd neighbor terms)

    Site order for `ops`
    ----
    ```
        1(W) --- 3(Z)
        |        |
        0(X) --- 2(Y)
    ```
    """
    assert len(ops) == 4
    N1, N2 = len(ts), len(ts[0])
    coords = [
        (i,j), ((i+1)%N1,j), (i,(j+1)%N2), ((i+1)%N1,(j+1)%N2)
    ]
    loopts = [ts[i_][j_].copy() for (i_, j_) in coords]
    # absorb env weights
    # bond weights are absorbed into X and Z
    skipss: list[list[int]] = [[], [1,4], [2,3], []]
    for n, ((i_,j_), t, skips) in enumerate(zip(coords, loopts, skipss)):
        loopts[n] = absorb_envwts(t, i_, j_, wxs, wys, skips, "absorb")
    # apply operators
    opts = list(
        gt.tensordot(op, t, (1,0)) 
        if op is not None else t
        for t, op in zip(loopts, ops)
    )
    # fully contract axes
    # contraction order: XWYZ 
    # physical and environment axes
    ts_axes = [[0,3,4], [0,2,3], [0,4,1], [0,1,2]] 
    einsum_str = 'adeh,cdgh,bafe,cbgf' 
    meas = gt.einsum(einsum_str, *[
        gt.tensordot(t.gT, opt.pconj(axes), [axes]*2)
        for t, opt, axes in zip(loopts, opts, ts_axes)
    ]).item()
    nrm = gt.einsum(einsum_str, *[
        gt.tensordot(t.gT, t.pconj(axes), [axes]*2)
        for t, axes in zip(loopts, ts_axes)
    ]).item()
    assert cisclose(nrm, nrm.real)
    return (
        meas.real if (
            cisclose(meas, meas.real) or 
            abs(meas) < float_info.epsilon
        ) else meas
    ) / nrm.real


def meas_loopxy(
    direction: str, i: int, j: int, 
    ops: list[None | GTensor], 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    assert direction in ("x", "y")
    assert len(ops) == 2
    if direction == "x":
        return meas_loop(i, j, [ops[0],None,ops[1],None], ts, wxs, wys)
    else:
        return meas_loop(i, j, [None,ops[0],None,ops[1]], ts, wxs, wys)


def meas_diag1(
    i: int, j: int, ops: list[None | GTensor], 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    """
    Approximate measurement of diagonal bond (0,3)
    on a square plaquette with lower-left corner at [i,j]
    (usually used to measure 2nd neighbor terms)

    Site order for `ops`
    ----
    ```
        1(W) --- 3(Z)
        |        |
        0(X) --- 2(Y)
    ```
    """
    assert len(ops) == 2
    return meas_loop(i, j, [ops[0],None,None,ops[1]], ts, wxs, wys)


def meas_diag2(
    i: int, j: int, ops: list[None | GTensor], 
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    """
    Approximate measurement of diagonal bond (1,2)
    on a square plaquette with lower-left corner at [i,j]
    (usually used to measure 2nd neighbor terms)

    Site order for `ops`
    ----
    ```
        1(W) --- 3(Z)
        |        |
        0(X) --- 2(Y)
    ```
    """
    assert len(ops) == 2
    return meas_loop(i, j, [None,ops[0],ops[1],None], ts, wxs, wys)
