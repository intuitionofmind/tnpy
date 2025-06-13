"""
Load iPEPS
"""
import gtensor as gt
import gtensor.linalg as gla
from gtensor import GTensor
from itertools import product


def absorb_wts(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
):
    r"""
    Absorb weight into iPEPS tensor
    (`ts` is changed in place)

    PEPS axis order (natural/symmetric convention)
    ----
    ```
            2  0                 1
            ↑ /                  ↑
        3 → A → 1   0 → x → 1    y
            ↑                    ↑
            4                    0
    ```
    """
    N1, N2 = len(ts), len(ts[0])
    for y, x in product(range(N1), range(N2)):
        f1, f2, f3, f4 = wxs[y][x], wys[y][x], wxs[y][x-1], wys[y-1][x]
        sf1, _ = gla.matrix_sqrt(f1, True)
        sf2, _ = gla.matrix_sqrt(f2, True)
        _, sf3 = gla.matrix_sqrt(f3, True)
        _, sf4 = gla.matrix_sqrt(f4, True)
        ts[y][x] = gt.fncon(
            [ts[y][x],sf1,sf2,sf3,sf4], 
            [[-1,1,2,3,4],[1,-2],[2,-3],[-4,3],[-5,4]]
        )
    return ts


def load_tps_wt(folder: str, N1: int, N2: int, normalize=False):
    r"""
    Load iPEPS with weights into natural convention

    Tensor axis order
    ----
    ```
            1
            ↑
        2 → M → 0
            ↑
            3
    ```
    """
    from update_ftps_mn.tps_io import load_tps
    ts0, wxs, wys = load_tps(folder, N1, N2)
    # absorb weight
    absorb_wts(ts0, wxs, wys)
    if normalize:
        for y, x in product(range(N1), range(N2)):
            ts0[y][x] /= gt.maxabs(ts0[y][x])
    # Hermitian conjugate
    ts1 = [[
        ts0[y][x].gT.flip_dual([1,2], minus=True)\
        .flip_dual([3,4], minus=False) for x in range(N2)
    ] for y in range(N1)]
    return ts0, ts1


def rand_net(shapes: list[list[tuple]], complex_init=False):
    N1, N2 = len(shapes), len(shapes[0])
    ms = [[gt.rand(
        shapes[y][x], (1,1,0,0), 0, complex_init
    ) for x in range(N2)] for y in range(N1)]
    # normalize tensors
    for y, x in product(range(N1), range(N2)):
        ms[y][x] /= gt.maxabs(ms[y][x])
    return ms

