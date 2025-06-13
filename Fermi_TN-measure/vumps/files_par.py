import os
import fermiT as ft
from fermiT import FermiT
from fermiT import save, load
from itertools import product
from fermiT.conversion import gt2ft
import gtensor as gt


def save_tensors(tensors: list[None|FermiT], folder: str):
    """Save list of FermiTs"""
    os.makedirs(folder, exist_ok=True)
    for i, t in enumerate(tensors):
        if t is not None:
            save("{}/{}.npz".format(folder, i), t)
    print("Completely save all data!")


def load_tensors(num: int, folder: str, icheck=0):
    """Load list of FermiTs"""
    ftensors = [load("{}/{}.npz".format(folder, i)) for i in range(num)]
    if icheck:
        print("Completely load all data!")
    return ftensors


def save_bMPScanon(fAss: list[list[list[FermiT]]], folder: str):
    """
    Save one boundary MPS (canonical form)
    """
    fCss, fALss, fARss = fAss
    N1, N2 = len(fCss), len(fCss[0])
    fTs = []
    iterators = [(i, j) for j in range(N2) for i in range(N1)]
    for i, j in iterators:
        fTs.append(fCss[i][j])
        fTs.append(fALss[i][j])
        fTs.append(fARss[i][j])
    save_tensors(fTs, folder)


def load_bMPScanon(N1: int, N2: int, folder: str) -> list[list[list[FermiT]]]:
    """
    Load one boundary MPS (canonical form: C, AL, AR)
    """
    num = N1 * N2 * 3
    fTs = load_tensors(num, folder)
    fCss = [[None] * N2 for i in range(N1)]
    fALss = [[None] * N2 for i in range(N1)]
    fARss = [[None] * N2 for i in range(N1)]
    iterators = [(i, j) for j in range(N2) for i in range(N1)]
    for i, j in iterators:
        fCss[i][j] = fTs.pop(0)
        fALss[i][j] = fTs.pop(0)
        fARss[i][j] = fTs.pop(0)
    assert len(fTs) == 0
    return [fCss, fALss, fARss]


def save_fixedpoint(fGXss: list[list[list[FermiT]]], folder: str):
    """
    Save all (4 directions) fixed point environments
    """
    fGUss, fGDss, fGRss, fGLss = fGXss
    N1, N2 = len(fGUss), len(fGUss[0])
    iterators = [(i, j) for j in range(N2) for i in range(N1)]
    fTs = []
    for i, j in iterators:
        fTs.append(fGUss[i][j])
        fTs.append(fGDss[i][j])
        fTs.append(fGLss[i][j])
        fTs.append(fGRss[i][j])
    save_tensors(fTs, folder=folder)


def load_fixedpoint(N1: int, N2: int, folder: str) -> list[list[list[FermiT]]]:
    """
    Save all (4 directions) fixed point environments
    """
    num = N1 * N2 * 4
    fTs = load_tensors(num, folder)
    fGUss = [[None] * N2 for i in range(N1)]
    fGDss = [[None] * N2 for i in range(N1)]
    fGRss = [[None] * N2 for i in range(N1)]
    fGLss = [[None] * N2 for i in range(N1)]
    iterators = [(i,j) for j in range(N2) for i in range(N1)]
    for i, j in iterators:
        fGUss[i][j] = fTs.pop(0)
        fGDss[i][j] = fTs.pop(0)
        fGRss[i][j] = fTs.pop(0)
        fGLss[i][j] = fTs.pop(0)
    assert len(fTs) == 0
    return [fGUss, fGDss, fGLss, fGRss]


def load_peps(
    N1: int, N2: int, tps_dir: str, tps_type: str, project = False
):
    """
    Axis order of loaded tensors fG0ss
    ```
            3  0
            ↑ /
        2 → G → 4
            ↑
            1
    ```
    Parameters
    ----
    project: bool
        project loaded the peps to no-double-occupancy subspace
    """
    assert tps_type in ("AB", "MN")
    if project:
        from phys_models.onesiteop import get_tJconv, makeops_tJft
    # with weights
    try:
        if tps_type == "AB":
            import vumps.tps_io.ab as tio_ab
            assert N1 == N2 == 2
            fA, fB, ftaus = tio_ab.load_tps(tps_dir)
            if project:
                tJ_conv = get_tJconv((fA.DE[0], fA.DO[0]))
                if tJ_conv == 3:
                    Pg = makeops_tJft("Pg", tJ_conv=3)
                    fA = ft.tensordot(Pg, fA, [[1], [0]])
                    fB = ft.tensordot(Pg, fB, [[1], [0]])
            fG0ss = tio_ab.absorb_wts(fA, fB, ftaus, dual=0)
            fG1ss = tio_ab.absorb_wts(fA, fB, ftaus, dual=1)
        elif tps_type == "MN":
            import vumps.tps_io.mn as tio_mn
            fGss, ftyss, ftxss = tio_mn.load_tps(N1, N2, tps_dir)
            if project:
                tJ_conv = get_tJconv((fGss[0][0].DE[0], fGss[0][0].DO[0]))
                if tJ_conv == 3:
                    Pg = makeops_tJft("Pg", tJ_conv=3)
                    for i, j in product(range(N1), range(N2)):
                        fGss[i][j] = ft.tensordot(Pg, fGss[i][j], [[1], [0]])
            fG0ss = tio_mn.absorb_wts(fGss, ftyss, ftxss, dual=0)
            fG1ss = tio_mn.absorb_wts(fGss, ftyss, ftxss, dual=1)
    # without weights (from GTensor format)
    except FileNotFoundError:
        fG0ss = [[
            gt2ft(gt.load(tps_dir + f"T{i}{j}.npz").transpose(0,4,3,2,1))
            for j in range(N2)
        ] for i in range(N1)]
        fG1ss = [[
            fG0ss[i][j].gT.flip_dual([1,2], minus=True)\
                .flip_dual([3,4], minus=False) for j in range(N2)
        ] for i in range(N1)]
    return [fG0ss, fG1ss]


def save_rhoss(
    rho1vss: list[list[FermiT]],
    rho2vss: list[list[FermiT]],
    rho1hss: list[list[FermiT]],
    rho2hss: list[list[FermiT]],
    folder: str,
):
    N1, N2 = len(rho1vss), len(rho1vss[0])
    fTs = []
    iterators = [(i, j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        fTs.append(rho1vss[i][j])
        fTs.append(rho2vss[i][j])
        fTs.append(rho1hss[i][j])
        fTs.append(rho2hss[i][j])
    save_tensors(fTs, folder)


def load_rhoss(
    N1: int, N2: int, folder: str
) -> list[list[list[FermiT]]]:
    num = N1 * N2 * 4
    fTs = load_tensors(num, folder)
    rho1vss = [[None] * N2 for i in range(N1)]
    rho2vss = [[None] * N2 for i in range(N1)]
    rho1hss = [[None] * N2 for i in range(N1)]
    rho2hss = [[None] * N2 for i in range(N1)]
    iterators = [(i, j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        rho1vss[i][j] = fTs.pop(0)
        rho2vss[i][j] = fTs.pop(0)
        rho1hss[i][j] = fTs.pop(0)
        rho2hss[i][j] = fTs.pop(0)
    return [rho1vss, rho2vss, rho1hss, rho2hss]


# 2x2 cell measurements

def save_fixedpoint2(
    fGL2ss: list[list[FermiT]], fGR2ss: list[list[FermiT]], folder: str
):
    N1, N2 = len(fGL2ss), len(fGL2ss[0])
    fTs = []
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        fTs.append(fGL2ss[i][j])
        fTs.append(fGR2ss[i][j])
    save_tensors(fTs, folder)


def load_fixedpoint2(
    N1: int, N2: int, folder: str
) -> list[list[list[FermiT]]]:
    num = N1 * N2 * 2
    fTs = load_tensors(num, folder)
    fGL2ss = [[None] * N2 for i in range(N1)]
    fGR2ss = [[None] * N2 for i in range(N1)]
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        fGL2ss[i][j] = fTs.pop(0)
        fGR2ss[i][j] = fTs.pop(0)
    return [fGL2ss, fGR2ss]


def save_rhodss(
    rhod1ss: list[list[FermiT]], rhod2ss: list[list[FermiT]], 
    folder: str
):
    N1, N2 = len(rhod1ss), len(rhod1ss[0])
    fTs = []
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        fTs.append(rhod1ss[i][j]) # 1 -> 4
        fTs.append(rhod2ss[i][j]) # 2 -> 3
    save_tensors(fTs, folder)


def load_rhodss(
    N1: int, N2: int, folder: str
) -> list[list[list[FermiT]]]:
    num = N1 * N2 * 2
    fTs = load_tensors(num, folder)
    rhod1ss = [[None] * N2 for i in range(N1)]
    rhod2ss = [[None] * N2 for i in range(N1)]
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        rhod1ss[i][j] = fTs.pop(0)
        rhod2ss[i][j] = fTs.pop(0)
    return [rhod1ss, rhod2ss]


def save_rho4ss(rho4ss: list[list[FermiT]], folder: str):
    N1, N2 = len(rho4ss), len(rho4ss[0])
    fTs = []
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        fTs.append(rho4ss[i][j])
    save_tensors(fTs, folder)


def load_rho4ss(
    N1: int, N2: int, folder: str
) -> list[list[FermiT]]:
    num = N1 * N2
    fTs = load_tensors(num, folder)
    rho4ss = [[None] * N2 for i in range(N1)]
    iterators = [(i,j) for i in range(N1) for j in range(N2)]
    for i, j in iterators:
        rho4ss[i][j] = fTs.pop(0)
    return rho4ss
