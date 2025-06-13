import gtensor as gt
from gtensor import GTensor
import numpy as np
from . import unitcell_tools as uc


def doublet(
    t0: GTensor, t1: GTensor, op: GTensor|None
):
    """
    Sandwich 1-site operator `op` between 
    original and conjugated PEPS tensor `t0`, `t1`
    """
    assert t0.dual == (0,1,1,0,0)
    assert t1.dual == (1,1,1,0,0)
    if op is None:
        t = gt.fncon(
            [t1, t0], [[1,-1,-3,-5,-7], [1,-2,-4,-6,-8]]
        )
    else:
        assert op.dual == (0,1)
        t = gt.fncon(
            [t1, op, t0], [[1,-1,-3,-5,-7], [1,2], [2,-2,-4,-6,-8]]
        )
    t = t.merge_axes((2,)*4, order=(1,1,-1,-1))
    return t


def meas_site(
    op: GTensor | None, x: int, y: int, 
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `op` at site (x,y)

    ```
        y+1 C1 →11 → T1 →12 → C2
            ↑        ↑        ↑
            8        9        10
            ↑        ↑        ↑
        y   T4 → 6 → M  → 7 → T2
            ↑        ↑        ↑
            3        4        5
            ↑        ↑        ↑
        y-1 C4 → 1 → T3 → 2 → C3
            x-1  x    x+1
    ```
    """
    if op is None:
        return 1.0
    N1, N2 = len(ts0), len(ts0[0])
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    yp1, ym1 = (y+1) % N1, (y-1) % N1
    xp1, xm1 = (x+1) % N2, (x-1) % N2
    m_val = doublet(ts0[y][x], ts1[y][x], op)
    m_nrm = doublet(ts0[y][x], ts1[y][x], None)
    (val, nrm) = [
        gt.fncon([
            c4s[ym1][xm1], t3s[ym1][x], c3s[ym1][xp1],
            t4s[y][xm1], m, t2s[y][xp1],
            c1s[yp1][xm1], t1s[yp1][x], c2s[yp1][xp1]
        ], [
            [1,3], [2,4,1], [5,2], 
            [3,6,8], [7,9,6,4], [10,7,5], 
            [8,11], [11,9,12], [12,10]
        ]).item() for m in (m_val, m_nrm)
    ]
    return val/nrm


def meas_allsites(
    op: GTensor | None, 
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `op` on all sites in the PEPS unit cell
    """
    N1, N2 = len(ts0), len(ts0[0])
    results = np.array([[
        meas_site(op, x, y, ts0, ts1, ctms) for x in range(N2)
    ] for y in range(N1)])
    return results


def meas_bondx(
    ops: list[GTensor | None], x: int, y: int, 
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `ops[0],...,ops[r-1]` on sites `(x,y)...(x+r-1,y)`
    ```
        y+1 C1 → T1 → ... → T1 → C2
            ↑    ↑          ↑    ↑
        y   T4 → M  → ... → M  → T2
            ↑    ↑          ↑    ↑
        y-1 C4 → T3 → ... → T3 → C3
            x-1  x         x+r-1 x+r
    ```
    """
    r = len(ops)
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    N1, N2 = len(ts0), len(ts0[0])
    yp1 = (y+1) % N1
    """
        C1 → -3     y+1
        ↑
        2
        ↑
        T4 → -2     y
        ↑
        1
        ↑
        C4 → -1     y-1
        x-1   
    """
    val = gt.fncon(
        [c4s[y-1][x-1], t4s[y][x-1], c1s[yp1][x-1]], 
        [[-1,1], [1,-2,2], [2,-3]]
    )
    nrm = val.copy()
    indices = [[1,3,5], [-1,2,1], [-2,4,3,2], [5,4,-3]]
    for n, op in enumerate(ops):
        """
            |-→ 5 → T1 → -3     y+1
            |       ↑
            |       4
            |       ↑
            |-→ 3 → M  → -2     y
            |       ↑
            |       2
            |       ↑
            |-→ 1 → T3 → -1     y-1
            x+n-1   x+n
        """
        col = (x+n) % N2
        m_val = doublet(ts0[y][col], ts1[y][col], op)
        m_nrm = doublet(ts0[y][col], ts1[y][col], None)
        val = gt.fncon([val, t3s[y-1][col], m_val, t1s[yp1][col]], indices)
        nrm = gt.fncon([nrm, t3s[y-1][col], m_nrm, t1s[yp1][col]], indices)
    """
        |-→ 5 → C2  y+1
        |       ↑
        |       4
        |       ↑
        |-→ 3 → T2  y
        |       ↑
        |       2
        |       ↑
        |-→ 1 → C3  y-1
        x+r-1   x+r
    """
    col = (x+r) % N2
    indices = [[1,3,5], [2,1], [4,3,2], [5,4]]
    val = gt.fncon([val, c3s[y-1][col], t2s[y][col], c2s[yp1][col]], indices).item()
    nrm = gt.fncon([nrm, c3s[y-1][col], t2s[y][col], c2s[yp1][col]], indices).item()
    return val / nrm


def meas_bondy(
    ops: list[GTensor | None], x: int, y: int, 
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `ops[0],...,ops[r-1]` on sites `(x,y)...(x,y+r-1)`
    ```
        C1 → T1 → C2    y+r
        ↑    ↑    ↑
        T4 → M  → T2    y+r-1
        ↑    ↑    ↑
        :    :    :
        ↑    ↑    ↑
        T4 → M  → T2    y
        ↑    ↑    ↑
        C4 → T3 → C3    y-1
        x-1  x    x+1
    ```
    We can reuse code of `meas_bondx` by swapping x/y label
    and flip about the 45-degree line.
    `ctm` needs to be reordered as `[c3,c2,c1,c4,t2,t1,t4,t3]`
    """
    d = "left"
    for reverse in False, True:
        uc.transform_ts(ts0, d, reverse)
        uc.transform_ts(ts1, d, reverse)
        uc.transform_ctms(ctms, d, reverse)
        if reverse is False:
            result = meas_bondx(ops, y, x, ts0, ts1, ctms)
    return result


def meas_allbonds1(
    ops: list[GTensor | None], 
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `[ops[0], ops[1]]` on 
    all 1st neighbor bonds in the PEPS unit cell
    """
    assert len(ops) == 2
    N1, N2 = len(ts0), len(ts0[0])
    results_x = [[
        meas_bondx(ops, x, y, ts0, ts1, ctms) for x in range(N2)
    ] for y in range(N1)]
    results_y = [[
        meas_bondy(ops, x, y, ts0, ts1, ctms) for x in range(N2)
    ] for y in range(N1)]
    return np.array([results_x, results_y])
