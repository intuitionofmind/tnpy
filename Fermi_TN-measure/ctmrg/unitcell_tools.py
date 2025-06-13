"""
Rearrange the unit cell to apply down-move code
to up/left/right-moves of CTMRG
"""
from gtensor import GTensor
from utils import get_invperm
from .ctm_io import ctm_dict


def reorder_ucell(
    ts: list[list[GTensor]], 
    direction: str, perm: list[int] | None, reverse=False
):
    r"""
    Reorganize the unit cell for tensors 
    relevant to `direction`-move. 
    `ts` is modified in place.

    Parameters
    ----
    perm: list[int]
        axes permutation when `reverse is False`
    """
    if direction == "down":
        # no need to do anything
        return
    N1, N2 = len(ts), len(ts[0])
    if reverse and (direction in ("left", "right")):
        N1, N2 = N2, N1
    iter_down = [(y,x) for y in range(N1) for x in range(N2)]
    perm0 = (
        None if perm is None else
        (get_invperm(perm) if reverse else perm)
    )
    if direction == "up":
        # reverse y-label
        ts2 = [[None]*N2 for _ in range(N1)]
        for y, x in iter_down:
            ts2[N1-y-1][x] = (
                ts[y][x].transpose(*perm0) 
                if perm0 is not None else ts[y][x]
            )
    elif direction == "left":
        # swap x/y label
        if not reverse:
            ts2 = [[None]*N1 for _ in range(N2)]
            for y, x in iter_down:
                ts2[x][y] = (
                    ts[y][x].transpose(*perm0) 
                    if perm0 is not None else ts[y][x]
                )
        else:
            ts2 = [[None]*N2 for _ in range(N1)]
            for y, x in iter_down:
                ts2[y][x] = (
                    ts[x][y].transpose(*perm0) 
                    if perm0 is not None else ts[x][y]
                )
    elif direction == "right":
        # swap x/y label, then reverse y-label
        if not reverse:
            ts2 = [[None]*N1 for _ in range(N2)]
            for y, x in iter_down:
                ts2[N2-x-1][y] = (
                    ts[y][x].transpose(*perm0) 
                    if perm0 is not None else ts[y][x]
                )
        else:
            ts2 = [[None]*N2 for _ in range(N1)]
            for y, x in iter_down:
                ts2[y][x] = (
                    ts[N2-x-1][y].transpose(*perm0) 
                    if perm0 is not None else ts[N2-x-1][y]
                )
    else:
        raise ValueError("Unrecognized CTM network edge direction")
    ts.clear()
    ts.extend(ts2)


def transform_ts(
    ts: list[list[GTensor]], direction: str, 
    reverse=False, match_dual=False
):
    r"""
    Change axis order of 
    - single-layer iPEPS tensors without contracting physical index; or
    - double-layer iPEPS tensors with physical index contracted and virtual indices merged
    to match down-move orientation
    """
    assert direction in ("up", "down", "left", "right")
    if direction == "down":
        return
    ndim = ts[0][0].ndim
    assert ndim in (4, 5)
    if ndim == 4:
        mperm = (
            [0,3,2,1] if direction == "up" else
            [1,0,3,2] if direction == "left" else
            [1,2,3,0] # direction == "right"
        )
    else:
        if direction == "left":
            mperm = [0,2,1,4,3]
        else:
            raise NotImplementedError
    reorder_ucell(ts, direction, mperm, reverse)


def transform_ctms(
    ctms: list[list[list[GTensor]]], direction: str, 
    reverse=False, match_dual=False
):
    r"""
    Change list order of `ctms` and axis order of CTMs
    to match down-move orientation
    """
    assert direction in ("up", "down", "left", "right")
    if direction == "down":
        return
    # change list order of `ctms`
    perm = [ctm_dict[label] for label in (
        ["c4","c3","c2","c1","t3","t2","t1","t4"] if direction == "up" else
        ["c3","c2","c1","c4","t2","t1","t4","t3"] if direction == "left" else
        ["c4","c1","c2","c3","t4","t1","t2","t3"]
    )]
    if reverse:
        perm = get_invperm(perm)
    ctms[:] = [ctms[n] for n in perm]
    # change axis order of CTMs
    for n, cs in enumerate(ctms):
        axperm = (
            None if direction == "right" else
            list(reversed(range(ctms[n][0][0].ndim)))
        )
        reorder_ucell(ctms[n], direction, axperm, reverse)
