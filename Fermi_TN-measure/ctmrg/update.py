"""
Corner transfer matrix renormalization group (CTMRG)
from Phys. Rev. Lett. 113, 046402 (2014)

PEPS uses natural convention
```
        :     :
        ↑     ↑
    ..→ M10 → M11 →..
        ↑     ↑
    ..→ M00 → M01 →..
        ↑     ↑
        :     :
```

Tensor axis order
```
        1
        ↑
    2 → M → 0
        ↑
        3
```

Unit cell size: N1 x N2 (N1 <= N2)
- N1: number of rows
- N2: number of columns
"""

import gtensor as gt
import gtensor.linalg as gla
from gtensor import GTensor
from itertools import product
from . import unitcell_tools as uc
from .ctm_io import ctm_dict


def assert_bipartite(
    ts: list[list[GTensor]], detailed=True
):
    """
    Check if an 2D array of tensors are bipartite
    """
    N1, N2 = len(ts), len(ts[0])
    assert N1 == N2 == 2
    if detailed:
        for y, x in product(range(N1), range(N2)):
            assert gt.array_equal(ts[y][x], ts[y-1][x-1])


def get_Lhalf(
    x: int, y: int, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]], cheap=True
):
    r"""
    Get left-half of the 4x4 network used to find projectors

    Axes order
    ----
    ```
        cheap is False      cheap is True

        C1 → T1 → -1                        y+2
        ↑    ↑
        T4 → M -→ -2        -1   -2         y+1
        ↑    ↑              ↑    ↑
        T4 → M -→ -3        T4 → M -→ -3    y
        ↑    ↑              ↑    ↑
        C4 → T3 → -4        C4 → T3 → -4    y-1
        x-1  x              x-1  x
    ```
    """
    N1, N2 = len(ms), len(ms[0])
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    """
    Contraction of cheap part
        -1      -2         y+1
        ↑        ↑
        T4 → 3 → M -→ -3    y
        ↑        ↑
        2        4
        ↑        ↑
        C4 → 1 → T3 → -4    y-1
        x-1      x
    """
    lhalf = gt.fncon([
        c4s[y-1][x-1], t3s[y-1][x], t4s[y][x-1], ms[y][x]
    ], [[1,2], [-4,4,1], [2,3,-1], [-3,-2,3,4]])
    """
    Contraction of full left part
        C1 → 6 → T1 → -1    y+2
        ↑        ↑
        4        5
        ↑        ↑
        T4 → 3 → M -→ -2    y+1
        ↑        ↑
        1        2
        ↑        ↑
        |--right-| -→ -3    y
        |--part--| -→ -4    y-1
        x-1      x  
    """
    if cheap is False:
        yp1, yp2 = (y+1)%N1, (y+2)%N1
        lhalf = gt.fncon([
            lhalf, t4s[yp1][x-1], ms[yp1][x], c1s[yp2][x-1], t1s[yp2][x]
        ], [[1,2,-3,-4],[1,3,4],[-2,5,3,2],[4,6],[6,5,-1]])
    return lhalf


def get_Rhalf(
    x: int, y: int, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]], cheap=True
):
    r"""
    Get left-half of the 4x4 network used to find projectors

    Axes order
    ----
    ```
        cheap is False      cheap is True

        -1→ T1 → C2                         y+2
            ↑    ↑
        -2→ M -→ T2             -2   -1     y+1
            ↑    ↑              ↑    ↑
        -3→ M -→ T2         -3→ M -→ T2     y
            ↑    ↑              ↑    ↑
        -4→ T3 → C3         -4→ T3 → C3     y-1
            x+1  x+2            x+1  x+2
    ```
    """
    N1, N2 = len(ms), len(ms[0])
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    xp1, xp2 = (x+1)%N2, (x+2)%N2
    yp1, yp2 = (y+1)%N1, (y+2)%N1
    """
    Contraction for cheap part
            -2      -1     y+1
            ↑       ↑
        -3→ M → 3 → T2     y
            ↑       ↑
            4       2
            ↑       ↑
        -4→ T3→ 1 → C3     y-1
            x+1     x+2
    """
    rhalf = gt.fncon([
        c3s[y-1][xp2], t3s[y-1][xp1], t2s[y][xp2], ms[y][xp1]
    ], [[2,1],[1,4,-4],[-1,3,2],[3,-2,-3,4]])
    """
    Contraction of full right half
        -1→ T1→ 6 → C2  y+2
            ↑       ↑
            5       4
            ↑       ↑
        -2→ M → 3 → T2  y+1
            ↑       ↑
            2       1
            ↑       ↑
        -3→ |-cheap-|   y
        -4→ |-part--|   y-1
            x+1     x+2
    """
    if cheap is False:
        rhalf = gt.fncon([
            rhalf, t2s[yp1][xp2], ms[yp1][xp1], c2s[yp2][xp2], t1s[yp2][xp1]
        ], [[1,2,-3,-4],[4,3,1],[3,5,-2,2],[6,4],[-1,5,6]])
    return rhalf


def get_proj(
    lhalf: GTensor, rhalf: GTensor, 
    chi: int, chie: int|None = None, 
    eps=5e-8, _self_test=False
):
    r"""
    Construct projectors `Pa`, `Pb` from `R`,`L` 
    obtained from QR/LQ decomposition.

    Projectors Pa, Pb satisfy the approximate identity
    ```
        Pb Pa = 1/sqrt(s) * uh * R * L * v * 1/sqrt(s) 
            = 1/sqrt(s) * uh * u * s * vh * v * 1/sqrt(s) 
            = 1/sqrt(s) * s * 1/sqrt(s) 
            = 1
    ```

    Axes order
    ----
    ```
              → 1     0 →
        0 → R             L → 2
              → 2     1 →

        0 →                  → 1
            Pa → 2    0 → Pb
        1 →                  → 2
    ```

    Returns
    -------
    - Projectors `Pa` and `Pb`, 
    - singular values `s` from SVD of `R * L`
    - SVD truncation error
    """
    Q1, R = gla.qr(lhalf, [2,3], return_q=True)
    L, Q2 = gla.lq(rhalf, [2,3], return_q=True)
    u, s, vh = gla.svd(gt.tensordot(R, L, [[1,2],[0,1]]), 1)
    for flag in [True, False]:
        if _self_test is False and flag is True:
            continue
        # u * s * vh = R * L
        if flag is True:
            s_cut = s.copy()
        else:
            u, s_cut, vh = gla.svd_cutoff(u, s, vh, chi, chie, eps=eps)
        d_cut = (s_cut.DE[0], s_cut.DO[0])
        # create projectors
        uh, v = u.gconj(), vh.gconj()
        s_inv = gla.matrix_inv(
            gla.matrix_sqrt(s_cut, is_diag=True)[0], is_diag=True
        )
        # Pa = L * v * 1/sqrt(s) 
        Pa = gt.dot_diag(gt.tensordot(L, v, (2,0)), s_inv, (2,0))
        # Pb = 1/sqrt(s) * uh * R
        Pb = gt.dot_diag(gt.tensordot(uh, R, (1,0)), s_inv, (0,1))
        if flag is True:
            iden1 = gt.tensordot(Pb, Pa, [[1,2], [0,1]])
            iden1 = gt.round(iden1, decimals=8)
            iden2 = gt.eye(iden1.DE[0], iden1.DO[0])
            assert gt.allclose(iden1, iden2)
    # calculate and output (relative) truncation error
    relE = gla.gsvd_error(s, d_cut)
    return Pa, Pb, relE


def get_proj2(
    lhalf: GTensor, rhalf: GTensor, 
    chi: int, chie: int|None = None, 
    eps=5e-8, _self_test=False
):
    r"""
    Get projector Pa, Pb directly from 
    SVD of lhalf (CL) * rhalf (CR), 
    without need to perform QR first

    ```
        |--|→ 0   2 →|--|     |-→          -→|
        |  |→ 1   3 →|  |     |-→          -→|
        |CL|         |CR|  =  U ---→ s ---→ Vh
        |  |----→----|  |
        |--|----→----|--|
    ```

    Axis order of U, Uh, Vh, V
    ```
                    0 -→ Uh
        |--→ 0      2 --→|
        |--→ 1      1 --→|
        U -→ 2

        V -→ 2
        |--→ 1      1 --→|
        |--→ 0      2 --→|
                    0 -→ Vh
    ```

    The unitary condition Uh U = 1, Vh V = 1 are
    ```
        0 ---→ Uh     V ----→ 1
                |     |
        |-→-o-→-|  =  |-→-o-→-|  =  0 --→-- 1
        |-→-o-→-|     |-→-o-→-|
        |                     |
        U ----→ 1     0 ---→ Vh
    ```
    `o` is the P tensor used to cancel unwanted fermion signs

    Projectors `Pa`, `Pb` are defined as
    ```
                        V -→- 1/sqrt(s) -→- 2
                        |
                        |-→-o-→-|--|
        0 →             |-→-o-→-|  |
            Pa → 2  =           |CR|
        1 →                 0 → |  |
                            1 → |--|

                        0 -→- 1/sqrt(s) -→- Uh
                                            |
                                |--|-→-o-→--|  
               → 1              |  |-→-o-→--| 
        0 → Pb      =           |CL|     
               → 2              |  | → 1
                                |--| → 2
    ```
    so that they satisfy the approximate identity
    ```
                0 -→- 1/sqrt(s) -→- Uh  V -→- 1/sqrt(s) -→- 1
                                    |   |
                        |--|-→-o-→--|   |-→-o-→-|--|
                        |  |-→-o-→--|   |-→-o-→-|  |
        Pb Pa  =        |CL|                    |CR|
                        |  |----------→---------|  |
                        |--|----------→---------|--|

                0 -→- 1/sqrt(s) -→- Uh  V -→- 1/sqrt(s) -→- 1
                                    |   |
                            |-→-o-→-|   |-→-o-→-|
            =               |-→-o-→-|   |-→-o-→-|
                            |                   |
                            U ---→--- s ---→--- Vh

            =   0 -→- 1/sqrt(s) -→- s -→- 1/sqrt(s) -→- 1
            =   0 ---→--- 1
    ```
    """
    a = gt.tensordot(lhalf, rhalf, [[2,3], [2,3]])
    u, s, vh = gla.svd(a, 2, cutoff=False)
    for flag in [True, False]:
        if _self_test is False and flag is True:
            continue
        if flag is True:
            s_cut = s.copy()
        else:
            u, s_cut, vh = gla.svd_cutoff(u, s, vh, chi, chie, eps=eps)
        d_cut = (s_cut.DE[0], s_cut.DO[0])
        uh, v = u.gconj(), vh.gconj()
        s_inv = gla.matrix_inv(
            gla.matrix_sqrt(s_cut, is_diag=True)[0], is_diag=True
        )
        Pa = gt.tensordot(
            rhalf, gt.dot_diag(v, s_inv, [2,0]).pconj([0,1]), [[1,0], [0,1]]
        )
        Pb = gt.tensordot(
            gt.dot_diag(uh, s_inv, [0,1]), lhalf.pconj([0,1]), [[2,1], [0,1]]
        )
        if flag is True:
            iden1 = gt.tensordot(Pb, Pa, [[1,2], [0,1]])
            iden1 = gt.round(iden1, decimals=10)
            iden2 = gt.eye(iden1.DE[0], iden1.DO[0])
            assert gt.allclose(iden1, iden2)
    # calculate and output (relative) truncation error
    relE = gla.gsvd_error(s, d_cut)
    return Pa, Pb, relE


def down_move1(
    y: int, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]], 
    chi: int, chie: int|None = None, eps=5e-8, 
    cheap=True
):
    r"""
    Perform down-move of CTMRG by 
    absorbing the y-th row into the (y-1)-th row

    ```
            ↑    ↑    ↑         ↑    ↑    ↑
        y   T4 → M  → T2  --->  C4 → T3 → C3
            ↑    ↑    ↑
        y-1 C4 → T3 → C3
            x-1  x    x+1       x-1  x    x+1
    ```
    CTM update
    ```
        Update C4           Update C3
        -2                              -1
        ↑                               ↑
        T4 → 3                      3 → T2
        ↑      \                   /    ↑
        2       Pa → -1     -2 → Pb     1
        ↑      /                   \    ↑
        C4 → 1                      2 → C3
        x-1                             x+1
    
        Update T3
                    -2
                    ↑
                4 → M → 2
               /    ↑    \
        -3 → Pb     3     Pa → -1
               \    ↑    /
                5 → T3→ 1
                    x
    ```
    """
    N1, N2 = len(ms), len(ms[0])
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    # find projectors for each column in this row
    Pas = [None] * N2
    Pbs = [None] * N2
    for x in range(N2):
        Lhalf = get_Lhalf(x, y, ms, ctms, cheap)
        Rhalf = get_Rhalf(x, y, ms, ctms, cheap)
        # Pas[x], Pbs[x], _ = get_proj(Lhalf, Rhalf, chi, chie, eps)
        Pas[x], Pbs[x], _ = get_proj2(Lhalf, Rhalf, chi, chie, eps)
    # update CTM for this row
    for x in range(N2):
        xp1 = (x+1)%N2
        c4s[y][x-1] = gt.fncon([
            c4s[y-1][x-1], t4s[y][x-1], Pas[x-1]
        ], [[1,2], [2,3,-2], [3,1,-1]])
        t3s[y][x] = gt.fncon([
            Pas[x], t3s[y-1][x], ms[y][x], Pbs[x-1]
        ], [[2,1,-1], [1,3,5], [2,-2,4,3], [-3,4,5]])
        c3s[y][xp1] = gt.fncon([
            c3s[y-1][xp1], t2s[y][xp1], Pbs[x]
        ], [[1,2], [-1,3,1], [-2,3,2]])
        # normalize 
        c4s[y][x-1] /= gt.maxabs(c4s[y][x-1])
        t3s[y][x] /= gt.maxabs(t3s[y][x])
        c3s[y][xp1] /= gt.maxabs(c3s[y][xp1])


def down_move(
    ms: list[list[GTensor]], ctms: list[list[list[GTensor]]], 
    chi: int, chie: int|None = None, eps=5e-8, 
    cheap=True, bipartite=False
):
    r"""
    Down-move of CTMRG: absorb a row above into the down-edge. 
    `ctms` is updated in place

    When `bipartite is True`
    ----
    The 2x2 unit cell contains only two independent tensors
    ```
            ↑   ↑
        ..→ B → A →..
            ↑   ↑
        ..→ A → B →..
            ↑   ↑
        
        T00 = T11 = A, T01 = T10 = B
    ```
    Therefore
    ```
                            ↑    ↑    ↑     ↑    ↑    ↑
        y+1                 T4 → A  → T2    T4 → B  → T2
                            ↑    ↑    ↑     ↑    ↑    ↑
        y                   C4 → T3 → C3    C4 → T3 → C3
                            x    x+1  x+2   x+1  x+2  x+3

            ↑    ↑    ↑     ↑    ↑    ↑
        y   T4 → A  → T2    T4 → B  → T2  
            ↑    ↑    ↑     ↑    ↑    ↑
        y-1 C4 → T3 → C3    C4 → T3 → C3
            x-1  x    x+1   x    x+1  x+2
    ```
    after obtaining C4, T3, C3 for the (y-1)-th row, 
    all CTMs can be updated with 
    ```
        ctms[n][y][x] = ctms[n][y-1][x-1]
    ```
    """
    N1, N2 = len(ms), len(ms[0])
    if bipartite: 
        assert N1 == N2 == 2
    # process each row
    for y in range(N1):
        down_move1(y, ms, ctms, chi, chie, eps, cheap)
        if bipartite:
            for n, x in product(range(8), range(N2)):
                ctms[n][y-1][x-1] = ctms[n][y][x]
            # no need to repeat calculation for y = 1
            break


def ctmrg_move(
    direction: str, ms: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]], 
    chi: int, chie: int|None = None, eps=5e-8, 
    cheap=True, bipartite=False
):
    r"""
    CTMRG move on the `direction`-edge.
    `ms`, `ctms` are updated in place

    Down-move
    ----
    ```
            ↑    ↑    ↑         ↑    ↑    ↑
        y   T4 → M  → T2  --->  C4 → T3 → C3
            ↑    ↑    ↑
        y-1 C4 → T3 → C3
            x-1  x    x+1       x-1  x    x+1
    ```

    Axes order (no need to transpose)
    ```
        2             1             0
        ↑             ↑             ↑
        T4 → 1    2 → M → 0     1 → T2
        ↑             ↑             ↑
        0             3             2

        1             1             0
        ↑             ↑             ↑
        C4 → 0    2 → T3 → 0    1 → C3
    ```

    Up-move
    ----
    ```
        y+1 C1 → T1 → C2
            ↑    ↑    ↑
        y   T4 → M  → T2  --->  C1 → T1 → C2
            ↑    ↑    ↑         ↑    ↑    ↑
            x-1  x    x+1       x-1  x    x+1
    ```
    To reuse down-move code, 
    we reverse y-label and vertically flip the network.
    We reorder `ctm` to `[c4,c3,c2,c1,t3,t2,t1,t4]`

    Original axes order (flipped vertically)
    ```
        0             3             2
        ↓             ↓             ↓
        T4 → 1    2 → M → 0     1 → T2
        ↓             ↓             ↓
        2             1             0

        0             1             1
        ↓             ↓             ↓
        C1 → 1    0 → T1 → 2    0 → C2
    ```

    Left-move
    ----
    ```
        y+1 C1 → T1 →           C1 →
            ↑    ↑              ↑
        y   T4 → M  →   --->    T4 →
            ↑    ↑              ↑
        y-1 C4 → T3 →           C4 →
            x-1  x              x
    ```
    To reuse down-move code, 
    we swap x/y label and flip about the 45-degree line.
    We reorder `ctm` to `[c3,c2,c1,c4,t2,t1,t4,t3]`

    Original axes order (flipped about 45-degree line)
    ```
        0             0             2
        ↑             ↑             ↑
        T3 → 1    3 → M → 1     1 → T1
        ↑             ↑             ↑
        2             2             0

        0             1             1
        ↑             ↑             ↑
        C4 → 1    0 → T4 → 2    0 → C1
    ```

    Right-move
    ----
    ```
        y+1   → T1 → C2           → C2
                ↑    ↑              ↑
        y     → M  → T2   --->    → T2
                ↑    ↑              ↑
        y-1   → T3 → C3           → C3
                x    x+1            x
    ```
    To reuse down-move code, we reverse x-label, 
    swap x/y label and rotate clockwise by 90 degree.
    We reorder `ctm` to `[c4,c1,c2,c3,t4,t1,t2,t3]`

    Original axes order (rotated clockwise by 90 degree)
    ```
        2             2             0
        ↑             ↑             ↑
        T3 → 1    3 → M → 1     1 → T1
        ↑             ↑             ↑
        0             0             2

        1             1             0
        ↑             ↑             ↑
        C3 → 0    2 → T2 → 0    1 → C2
    ```
    """
    assert direction in ("up", "down", "left", "right")
    if direction == "down":
        down_move(ms, ctms, chi, chie, eps, cheap, bipartite)
        return
    for reverse in (False, True):
        uc.transform_ts(ms, direction, reverse)
        uc.transform_ctms(ctms, direction, reverse)
        if reverse is False:
            down_move(ms, ctms, chi, chie, eps, cheap, bipartite)


def compare_ctms(
    ctms: list[list[list[GTensor]]], 
    ctms0: list[list[list[GTensor]]]
):
    r"""
    Compare new and old CTMs
    """
    diff = 0.0
    nctm, N1, N2 = len(ctms), len(ctms[0]), len(ctms[0][0])
    for n, y, x in product(range(nctm), range(N1), range(N2)):
        tdiff = ctms[n][y][x] - ctms0[n][y][x]
        diff += gt.norm(tdiff)
    diff /= nctm * N1 * N2
    return diff


def converge_test(
    x: int, y: int, ms: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Check CTM convergence by calculating 
    the ratio (n22 n33) / (n32 n23), where
    ```
        y+1 C1 → T1 → C2
            ↑    ↑    ↑
        y   T4 → M  → T2  =  n33
            ↑    ↑    ↑
        y-1 C4 → T3 → C3
            x-1  x    x+1

        y+1 C1 → C2
            ↑    ↑        = n22
        y   C4 → C3
            x    x+1

        y+1 C1 → C2
            ↑    ↑
        y   T4 → T2       =  n32
            ↑    ↑
        y-1 C4 → C3
            x    x+1

        y+1 C1 → T1 → C2
            ↑    ↑    ↑   = n23
        y   C4 → T3 → C3
            x-1  x    x+1
    ```
    """
    N1, N2 = len(ms), len(ms[0])
    c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s = ctms
    yp1, ym1 = (y+1) % N1, (y-1) % N1
    xp1, xm1 = (x+1) % N2, (x-1) % N2
    n33 = gt.fncon([
        c4s[ym1][xm1], t3s[ym1][x], c3s[ym1][xp1],
        t4s[y][xm1], ms[y][x], t2s[y][xp1],
        c1s[yp1][xm1], t1s[yp1][x], c2s[yp1][xp1]
    ], [
        [1,3], [2,4,1], [5,2], 
        [3,6,8], [7,9,6,4], [10,7,5], 
        [8,11], [11,9,12], [12,10]
    ]).item()
    n22 = gt.fncon(
        [c4s[y][x], c3s[y][xp1], c1s[yp1][x], c2s[yp1][xp1]], 
        [[1,2], [3,1], [2,4], [4,3]]
    ).item()
    n32 = gt.fncon([
        c4s[ym1][x], c3s[ym1][xp1], 
        t4s[y][x], t2s[y][xp1],
        c1s[yp1][x], c2s[yp1][xp1]
    ], [
        [1,2], [3,1], [2,4,5], [6,4,3], [5,7], [7,6]
    ]).item()
    n23 = gt.fncon([
        c4s[y][xm1], c1s[yp1][xm1], 
        t3s[y][x], t1s[yp1][x], 
        c3s[y][xp1], c2s[yp1][xp1]
    ], [
        [2,1], [1,3], [5,4,2], [3,4,6], [7,5], [6,7]
    ]).item()
    ratio = n33*n22/n32/n23
    return [ratio, n33, n22, n32, n23]
