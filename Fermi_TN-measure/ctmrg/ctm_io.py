"""
Initialize, load and save corner transfer matrices

Tensor axes order: counter-clockwise
```
C1 → 1    0 → T1 → 2    0 → C2
↑             ↑             ↑
0             1             1

2             1             0
↑             ↑             ↑
T4 → 1    2 → M → 0     1 → T2
↑             ↑             ↑
0             3             2

1             1             0
↑             ↑             ↑
C4 → 0    2 → T3 → 0    1 → C3
```
"""

import gtensor as gt
from gtensor import GTensor
from itertools import product


ctm_dict = dict(
    (label, n) for n, label in 
    enumerate(["c1","c2","c3","c4","t1","t2","t3","t4"])
)


def rand_ctm(
    ms: list[list[GTensor]], chi: int, chie: int|None = None, 
    complex_init = False
):
    r"""
    Generate random CTMs
    """
    N1, N2 = len(ms), len(ms[0])
    if chie is None:
        assert chi % 2 == 0
        chie = chi // 2
    chio = chi - chie
    assert chio >= 0
    c1s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    c2s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    c3s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    c4s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    t1s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    t2s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    t3s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    t4s: list[list[GTensor]] = [[None] * N2 for _ in range(N1)]
    for y, x in product(range(N1), range(N2)):
        De, Do = ms[y][x].shape
        yp, ym = (y+1)%N1, (y-1)%N1
        xp, xm = (x+1)%N2, (x-1)%N2
        c1s[yp][xm] = gt.rand(((chie,chie),(chio,chio)), (0,1), 0, complex_init)
        c2s[yp][xp] = gt.rand(((chie,chie),(chio,chio)), (0,0), 0, complex_init)
        c3s[ym][xp] = gt.rand(((chie,chie),(chio,chio)), (1,0), 0, complex_init)
        c4s[ym][xm] = gt.rand(((chie,chie),(chio,chio)), (1,1), 0, complex_init)
        t1s[yp][x] = gt.rand(((chie,De[1],chie),(chio,Do[1],chio)), (0,0,1), 0, complex_init)
        t2s[y][xp] = gt.rand(((chie,De[0],chie),(chio,Do[0],chio)), (1,0,0), 0, complex_init)
        t3s[ym][x] = gt.rand(((chie,De[3],chie),(chio,Do[3],chio)), (1,1,0), 0, complex_init)
        t4s[y][xm] = gt.rand(((chie,De[2],chie),(chio,Do[2],chio)), (0,1,1), 0, complex_init)
    return [c1s, c2s, c3s, c4s, t1s, t2s, t3s, t4s]


def save_ctm(folder: str, ctms: list[list[list[GTensor]]]):
    r"""Save CTMs to folder"""
    assert folder[-1] == "/"
    N1, N2 = len(ctms[0]), len(ctms[0][0])
    for label, ts in zip(ctm_dict.keys(), ctms):
        for y, x in product(range(N1), range(N2)):
            gt.save(folder + f"{label}-{y}{x}.npz", ts[y][x])


def load_ctm(folder: str, N1: str, N2: str):
    r"""Load CTMs from folder"""
    ctms = [[[
        gt.load(folder + f"{label}-{y}{x}.npz") for x in range(N2)
    ] for y in range(N1)] for label in ctm_dict.keys()]
    return ctms
