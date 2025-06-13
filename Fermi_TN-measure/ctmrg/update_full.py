import gtensor as gt
from gtensor import GTensor
import gtensor.linalg as gla
from .update import down_move1
from . import unitcell_tools as uc
from .measure import doublet
from itertools import product
from time import time
from math import sqrt

r"""
Axis order of X, aR, Y, bL
```
        1             0           0           1
        ↑            ↙︎           ↙︎            ↑
    2 → X → 0   1 → aR → 2   1 → bL → 2   2 → Y → 0
        ↑                                     ↑
        3                                     3
```
"""

MIN_ITER = 1

def tensor_env(
    x: int, y: int, X: GTensor, Y: GTensor, 
    ctms: list[list[list[GTensor]]]
):
    r"""
    Construct the tensor
    ```
        C1 → T1 ------→------ T1 → C2   y+1
        ↑    ↑                ↑    ↑
        T4 → XX - 0/1   2/3 - YY → T2   y
        ↑    ↑                ↑    ↑
        C4 → T3 ------→------ T3 → C3   y-1
        x-1  x               x+1  x+2
    ```
    or more simply denoted as
    ```
        |-----------|
        |← 0     2 ←| 
        |→ 1     3 →|
        |-----------|
    ```
    `n1/n0` correspond to conjugated/original tensors.
    Duals of `X†`, `Y†` have been modified by flippers

    ```
        Left half                       Right half
        C1 → 7 → T1 → -3           -3 → T1 → 7 → C2
        ↑        ↑                      ↑        ↑
        5        6                      6        5
        ↑        ↑                      ↑        ↑
        T4 → 4 → XX - -1/-2     -1/-2 - YY → 4 → T2
        ↑        ↑                      ↑        ↑
        2        3                      3        2
        ↑        ↑                      ↑        ↑
        C4 → 1 → T3 → -4           -4 → T3 → 1 → C3
    ```
    """
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    N1, N2 = len(c1s), len(c1s[0])
    yp1 = (y+1)%N1
    xp1, xp2 = (x+1)%N2, (x+2)%N2
    c1 = c1s[yp1][x-1]
    c2 = c2s[yp1][xp2]
    c3 = c3s[y-1][xp2]
    c4 = c4s[y-1][x-1]
    t1, t1_ = t1s[yp1][x], t1s[yp1][xp1]
    t2 = t2s[y][xp2]
    t3, t3_ = t3s[y-1][x], t3s[y-1][xp1]
    t4 = t4s[y][x-1]
    Xh = X.gT.flip_dual([1], minus=True).flip_dual([2,3], minus=False)
    Yh = Y.gT.flip_dual([0,1], minus=True).flip_dual([3], minus=False)
    perm = (0,4,1,5,2,6,3,7)
    XX = gt.merge_axes(
        gt.outer(Xh, X).transpose(*perm), (1,1,2,2,2), order=(1,1,1,-1,-1)
    )
    YY = gt.merge_axes(
        gt.outer(Yh, Y).transpose(*perm), (2,2,1,1,2), order=(1,1,1,1,-1)
    )
    lhalf = gt.fncon(
        [c4, t3, t4, XX, c1, t1], 
        [[1,2], [-4,3,1], [2,4,5], [-1,-2,6,4,3], [5,7], [7,6,-3]]
    )
    rhalf = gt.fncon(
        [c3, t3_, t2, YY, c2, t1_], 
        [[2,1], [1,3,-4], [5,4,2], [4,6,-1,-2,3], [7,5], [-3,6,7]]
    )
    env = gt.tensordot(lhalf, rhalf, [[2,3], [2,3]]).transpose(0,2,3,1)
    return env


def tensor_Ra(env: GTensor, bL: GTensor):
    r"""
    Construct the tensor
    ```
        |----------env----------|
        |← -1     -3 ← bL† ← 1 ←|
        |               ↓       |
        |               3       |
        |               ↓       |
        |→ -2     -4 → bL -→ 2 →|
        |-----------------------|
    ```
    """
    bLh = bL.gT
    Ra = gt.fncon(
        [env, bLh, bL], [[-1,1,2,-2], [3,-3,1], [3,-4,2]]
    )
    return Ra


def tensor_Sa(env: GTensor, aR2: GTensor, bL: GTensor, bL2: GTensor):
    r"""
    Construct the tensor
    ```
        |------------env------------|
        |← -1         -2 ← bL† ← 1 ←|
        |                   ↓       |
        |       -3          3       |
        |        ↓          ↓       |
        |→ 5 -→ aR2 -→ 4 → bL2 → 2 →|
        |---------------------------|
    ```
    """
    bLh = bL.gT
    Sa = gt.fncon(
        [env, bLh, bL2, aR2], 
        [[-1,1,2,5], [3,-2,1], [3,4,2], [-3,5,4]]
    )
    return Sa


def tensor_Rb(env: GTensor, aR: GTensor):
    r"""
    Construct the tensor
    ```
        |----------env----------|
        |← 1 ← aR† ← -1     -3 ←|
        |       ↓               |
        |       3               |
        |       ↓               |
        |→ 2 → aR -→ -2     -4 →|
        |-----------------------|
    ```
    """
    aRh = aR.gT
    Rb = gt.fncon(
        [env, aRh, aR], [[1,-3,-4,2], [3,1,-1], [3,2,-2]]
    )
    return Rb


def tensor_Sb(env: GTensor, aR: GTensor, aR2: GTensor, bL2: GTensor):
    r"""
    Construct the tensor
    ```
        |------------env------------|
        |← 1 ← aR† ← -1         -2 ←|
        |       ↓                   |
        |       3         -3        |
        |       ↓          ↓        |
        |→ 2 → aR2 → 4 -→ bL2 -→ 5 →|
        |---------------------------|
    ```
    """
    aRh = aR.gT
    Sb = gt.fncon(
        [env, aRh, aR2, bL2], 
        [[1,-2,5,2], [3,1,-1], [3,2,4], [-3,4,5]]
    )
    return Sb


def inner_prod(
    env: GTensor, aR1: GTensor, bL1: GTensor,
    aR2: GTensor, bL2: GTensor
):
    r"""
    Calculate the norm <Psi(a1,b1)|Psi(a2,b2)>
    ```
        |------------env------------|
        |← 1 ← aR1† ← 2 ← bL1† ← 3 ←|
        |       ↓          ↓        |
        |       4          5        |
        |       ↓          ↓        |
        |→ 6 → aR2 → 7 -→ bL2 → 8 -→|
        |---------------------------|
    ```
    """
    t = gt.fncon(
        [env, aR1.gT, bL1.gT, aR2, bL2],
        [[1,3,8,6], [4,1,2], [5,2,3], [4,6,7], [5,7,8]]
    )
    return t.item()


def inner_prod_local(
    aR1: GTensor, bL1: GTensor,
    aR2: GTensor, bL2: GTensor
):
    r"""
    Calculate the approximate local inner product
    `<aR1 bL1|aR2 bL2>`
    ```
        |-------- 2 --------|
        |← aR1† ← 1 ← bL1† ←|
            ↓          ↓
            3          4
            ↓          ↓
        |← aR2 ←- 5 ← bL2 ←-|
        |-------- 6 --------|
    ```
    """
    t = gt.fncon(
        [aR1.gT, bL1.gT, aR2, bL2], 
        [[3,2,1], [4,1,2], [3,6,5], [4,5,6]]
    )
    return t.item()


def cost_func(
    env: GTensor, aR: GTensor, bL: GTensor,
    aR2: GTensor, bL2: GTensor
):
    r"""
    Calculate the cost function
    ```
        f(a,b)  = | |Psi(a,b)> - |Psi(a2,b2)> |^2
                = <Psi(a,b)|Psi(a,b)> + <Psi(a2,b2)|Psi(a2,b2)>
                    - 2 Re<Psi(a,b)|Psi(a2,b2)>
    ```
    """
    t1 = inner_prod(env, aR, bL, aR, bL)
    t2 = inner_prod(env, aR2, bL2, aR2, bL2)
    t3 = inner_prod(env, aR, bL, aR2, bL2)
    return t1.real + t2.real - 2 * t3.real


def optimize(
    aR0: GTensor, bL0: GTensor, 
    aR2: GTensor, bL2: GTensor, env: GTensor,
    max_iter=20, max_diff=1e-8, check_int=1
):
    """
    Minimize the cost function
    ```
        fix bL:
        d(aR,aR†) = aR† Ra aR - aR† Sa - Sa† aR + T
        minimized by Ra aR = Sa

        fix aR:
        d(bL,bL†) = bL† Rb bL - bL† Sb - Sb† bL + T
        minimized by Rb bL = Sb
    ```
    `aR0`, `bL0` are initial values of `aR`, `bL`
    """
    print("---- Iterative optimization ----")
    print("{:<6s}{:>12s}{:>12s} {:>10s}".format("Step", "Cost", "Diff", "Time/s"))
    aR, bL = aR0.copy(), bL0.copy()
    cost0 = cost_func(env, aR, bL, aR2, bL2)
    for count in range(max_iter):
        time0 = time()
        Ra = tensor_Ra(env, bL)
        Sa = tensor_Sa(env, aR2, bL, bL2)
        aR = gla.solve(Ra, Sa, [[1,3], [1,2]])
        Rb = tensor_Rb(env, aR)
        Sb = tensor_Sb(env, aR, aR2, bL2)
        bL = gla.solve(Rb, Sb, [[1,3], [1,2]])
        cost = cost_func(env, aR, bL, aR2, bL2)
        diff = abs(cost - cost0)
        time1 = time()
        print("{:<6d}{:>12.3e}{:>12.3e} {:>10.3f}".format(
            count, cost, diff, time1-time0
        ))
        if diff < max_diff and count >= MIN_ITER:
            break
        aR0, bL0 = aR.copy(), bL.copy()
        cost0 = cost
    return aR, bL


def fix_gauge(
    env: GTensor, X: GTensor, Y: GTensor, 
    aR: GTensor, bL: GTensor
):
    """
    Fix local gauge of the env tensor around a bond
    """
    env = (env + env.gconj())/2
    return [env, X, Y, aR, bL]


def update_column(
    x: int, gate: GTensor, 
    ts: list[list[GTensor]], ms: list[list[GTensor]],
    ctms: list[list[list[GTensor]]],
    Dmax: int, De: int|None = None, eps = 5e-8,
    max_iter = 100, max_diff = 1e-8, 
    gauge_fix=False, bipartite=False, **kwargs
):
    """
    Update all horizontal bonds in the x-th column
    (i.e. `(x,y) (x+1,y)` for all y)
    and the CTMs in the x-th, (x+1)-th column
    ```
        Absorb           Absorb
        ----->           <-----
        C1 - T1 -     - T1 - C2     y+1
        |    |          |    |
        T4 - T' -     - T' - T2     y
        |    |          |    |
        C4 - T3 -     - T3 - C3     y-1
        x-1  x          x+1  x+2
    ```
    """
    N1, N2 = len(ts), len(ts[0])
    if bipartite:
        assert N1 == N2 == 2
    # apply gate on column and update tensors
    for y in range(N1):
        A, B = ts[y][x], ts[y][(x+1)%N2]
        # local QR/LQ decomposition
        X, aR = gla.qr(A, [0,1])
        X = X.transpose(3,0,1,2)
        aR = aR.transpose(1,0,2)
        bL, Y = gla.lq(B, [0,3])
        Y = Y.transpose(1,2,0,3)
        # get environment
        env = tensor_env(x, y, X, Y, ctms)
        # local gauge fixing
        if gauge_fix:
            env, X, Y, aR, bL = fix_gauge(env, X, Y, aR, bL)
        # apply gate
        """
                -2          -3
                ↓           ↓
                |----gate---|
                ↓           ↓
                1           2
                ↓           ↓
            -1→ aR -→ 3 -→ bL → -4
        """
        tmp = gt.fncon([gate, aR, bL], [[-2,-3,1,2], [1,-1,3], [2,3,-4]])
        aR2, s, bL2 = gla.svd(tmp, 2)
        aR2, bL2 = gla.absorb_sv(aR2, s, bL2)
        # initialize truncated tensors using simple SVD truncation
        aR0, s_cut, bL0 = gla.svd_cutoff(aR2, s, bL2, Dmax, De, eps)
        aR0, bL0 = gla.absorb_sv(aR0, s_cut, bL0)
        # fix aR, bL axis order
        perm_aR, perm_bL = (1,0,2), (1,0,2)
        aR0, aR2 = aR0.transpose(*perm_aR), aR2.transpose(*perm_aR)
        bL0, bL2 = bL0.transpose(*perm_bL), bL2.transpose(*perm_bL)
        # optimize aR, bL
        aR, bL = optimize(aR0, bL0, aR2, bL2, env, max_iter, max_diff)
        aR /= gt.maxabs(aR)
        bL /= gt.maxabs(bL)
        # update and normalize ts
        """
                -3        -1               -1     -3
                ↑        ↙︎                ↙︎       ↑
            -4→ X → 1 → aR → -2     -4 → bL → 1 → Y → -2
                ↑                                 ↑
                -5                                -5
        """
        ts[y][x] = gt.fncon([X, aR], [[1,-3,-4,-5], [-1,1,-2]])
        ts[y][(x+1)%N2] = gt.fncon([bL, Y], [[-1,-4,1], [-2,-3,1,-5]])
        # normalize
        for (i,j) in [(y,x), (y,(x+1)%N2)]:
            ts[i][j] /= gt.maxabs(ts[i][j])
            # update and normalize ms
            ms[i][j] = doublet(
                ts[i][j], ts[i][j].gT.flip_dual([1,2], minus=True)\
                .flip_dual([3,4], minus=False), None
            )
            ms[i][j] /= gt.maxabs(ms[i][j])
        if bipartite:
            # no need to repeat calculation for y = 1
            for x in range(N2):
                ts[y-1][x-1] = ts[y][x]
            break
    

def leftright_move(
    x: int, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]],
    eps=5e-8, cheap=True, **kwargs
):
    """
    Update ctms in the x-th and (x+1)-th column 
    using left and right move

    To use down_move code:
    - left move:    column x   --> row x
    - right move:   column x+1 --> row N2-x-2
    """
    N1, N2 = len(ms), len(ms[0])
    # using the same chi as input ctm
    chi = ctms[0][0][0].DS[0]
    for direction, i in zip(
        ("left", "right"), (x, N2-x-2)
    ):
        for reverse in (False, True):
            uc.transform_ts(ms, direction, reverse)
            uc.transform_ctms(ctms, direction, reverse)
            if reverse is False:
                down_move1(i, ms, ctms, chi, eps=eps, cheap=cheap)


def update_row(
    y: int, gate: GTensor, 
    ts: list[list[GTensor]], ms: list[list[GTensor]],
    ctms: list[list[list[GTensor]]],
    Dmax: int, De: int|None = None, eps = 5e-8,
    max_iter = 100, max_diff = 1e-8, 
    gauge_fix=False, bipartite=False, **kwargs
):
    """
    Update all vertical bonds in the y-th row
    (i.e. `(x,y) (x,y+1)` for all x)
    """
    d = "left"
    for reverse in (False, True):
        uc.transform_ts(ts, d, reverse)
        uc.transform_ts(ms, d, reverse)
        uc.transform_ctms(ctms, d, reverse)
        if reverse is False:
            update_column(
                y, gate, ts, ms, ctms, Dmax, De, eps, 
                max_iter, max_diff, gauge_fix, bipartite
            )


def updown_move(
    y: int, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]],
    eps=5e-8, cheap=True, **kwargs
):
    """
    Update ctms in the y-th and (y+1)-th row 
    using down and up move
    """
    N1, N2 = len(ms), len(ms[0])
    # using the same chi as input ctm
    chi = ctms[0][0][0].DS[0]
    for direction, i in zip(
        ("down", "up"), (y, N1-y-1)
    ):
        for reverse in (False, True):
            uc.transform_ts(ms, direction, reverse)
            uc.transform_ctms(ctms, direction, reverse)
            if reverse is False:
                down_move1(i, ms, ctms, chi, eps=eps, cheap=cheap)


def local_fidelity(
    aR1: GTensor, bL1: GTensor, 
    aR2: GTensor, bL2: GTensor
):
    r"""
    Calculate the fidelity using aR, bL
    between two evolution steps
    ```
                |<aR1 bL1 | aR2 bL2>|^2
        ---------------------------------------
        <aR1 bL1 | aR1 bL1> <aR2 bL2 | aR2 bL2>
    ```
    """
    b12 = inner_prod_local(aR1, bL1, aR2, bL2)
    b11 = inner_prod_local(aR1, bL1, aR1, bL1)
    b22 = inner_prod_local(aR2, bL2, aR2, bL2)
    return abs(b12) / sqrt(abs(b11*b22))

