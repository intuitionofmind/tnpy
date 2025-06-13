import os
from copy import deepcopy
import fermiT as ft
from fermiT import FermiT
from fermiT.linalg import matrix_sqrt
from update_ftps.convert_dual import dual_loop2sym
from update_ftps.sutools import get_dualconv
from update_ftps.tps_io import load_tps
import vumps.files_par as ffile
import update_ftps_mn.tps_io as tio_gt
from fermiT.conversion import gt2ft
from itertools import product


def save_tps(
    fGss: list[list[FermiT]], 
    ftyss: list[list[FermiT]], 
    ftxss: list[list[FermiT]], 
    folder: str
):
    """
    Save PEPS with weight

    Parameters
    ----
    fGss: list[list[FermiT]]
        PEPS local tensors
    ftyss, ftxss: list[list[FermiT]]
        PEPS vertical and horizontal weights
    """
    assert folder.endswith(os.sep)
    N1 = len(fGss)
    N2 = len(fGss[0])

    fTs = []
    iterators = [(i,j) for j in range(N2) for i in range(N1)]
    for i, j in iterators:
        fTs.append(fGss[i][j])
        fTs.append(ftyss[i][j])
        fTs.append(ftxss[i][j])
    
    ffile.save_tensors(fTs, folder=folder)


def load_tpsMN(folder: str) -> tuple[list[list[FermiT]], list[list[FermiT]], list[list[FermiT]]]:
    """
    2 x 2 FermiT unit cell (symmetric convention)
    ```
                y10         y11
                ↑           ↑
        x11 ← A10 → x10 ← A11 → x11
                ↓           ↓
                y00         y01
                ↑           ↑
        x01 ← A00 → x00 ← A01 → x01
                ↓           ↓
                y10         y11
    ```
    """
    ts, wts = load_tps(folder)
    assert len(ts) == 4 and len(wts) == 8
    # change loop to sym convention
    if get_dualconv(ts, wts) == "loop":
        ts, wts = dual_loop2sym(ts, wts)
    # change local tensor axis order
    perm = [0,4,3,2,1]
    for tname in ts.keys():
        ts[tname] = ts[tname].transpose(*perm)
    # T00 = A, T01 = B, T10 = D, T11 = C
    fGss = [
        [gt2ft(ts["Ta"]), gt2ft(ts["Tb"])],
        [gt2ft(ts["Td"]), gt2ft(ts["Tc"])]
    ]
    # y00 = y2, y01 = y1, y10 = y1_, y11 = y2_
    ftyss = [
        [gt2ft(wts["y2"]), gt2ft(wts["y1"])],
        [gt2ft(wts["y1_"]), gt2ft(wts["y2_"])]
    ]
    # x00 = x1, x01 = x2_, x10 = x2, x11 = x1_,
    ftxss = [
        [gt2ft(wts["x1"]), gt2ft(wts["x2_"])],
        [gt2ft(wts["x2"]), gt2ft(wts["x1_"])]
    ]
    return fGss, ftyss, ftxss


def load_tps(
    N1: int, N2: int, folder: str
) -> tuple[list[list[FermiT]], list[list[FermiT]], list[list[FermiT]]]:
    """
    load PEPS with weight

    Parameters
    ----
    N1, N2: int
        unit cell size (number of rows and columns)
    """
    try:
        try:
            fGss, ftxss, ftyss = tio_gt.load_tps(folder, N1, N2)
            for i, j in product(range(N1), range(N2)):
                # convert to fermiT axis order convention
                fGss[i][j] = gt2ft(fGss[i][j].transpose(0,4,3,2,1))
                ftxss[i][j] = gt2ft(ftxss[i][j])
                ftyss[i][j] = gt2ft(ftyss[i][j])
        except FileNotFoundError:
            assert N1 == N2 == 2
            fGss, ftyss, ftxss = load_tpsMN(folder)
    except:
        num = N1 * N2 * 3
        fTs = ffile.load_tensors(num, folder=folder)
        fGss  = [[None]*N2 for i in range(N1)]
        ftyss = [[None]*N2 for i in range(N1)]
        ftxss = [[None]*N2 for i in range(N1)]
        iterators = [(i,j) for j in range(N2) for i in range(N1)]
        for i, j in iterators:
            fGss[i][j]  = fTs.pop(0)
            ftyss[i][j] = fTs.pop(0)
            ftxss[i][j] = fTs.pop(0)
    return fGss, ftyss, ftxss


def absorb_wts(
    fGss: list[list[FermiT]],
    ftyss: list[list[FermiT]],
    ftxss: list[list[FermiT]], dual=0
):
    """
    Input PEPS (symmetric convention)
    ```
                y10         y11
                ↑           ↑
        x11 ← A10 → x10 ← A11 → x11
                ↓           ↓
                y00         y01
                ↑           ↑
        x01 ← A00 → x00 ← A01 → x01
                ↓           ↓
                y10         y11
    ```

    Axis order
    ```
            3  0                 1
            ↑ /                  ↓
        2 ← A → 4   0 → x ← 1    y
            ↓                    ↑
            1                    0
    ```

    After absorbing weights, 
    the tensor duals are all (dual,0,0,1,1)
    ```
          ↑   ↑
        → B → A →
          ↑   ↑
        → A → B →
          ↑   ↑
    ```
    """
    assert dual in (0, 1)
    N1, N2 = len(fGss), len(fGss[0])
    fGss = deepcopy(fGss)
    iterators = [(i,j) for j in range(N2) for i in range(N1)]
    for i, j in iterators:
        f1, f2, f3, f4 = ftyss[i-1][j], ftxss[i][j-1], ftyss[i][j], ftxss[i][j]
        _, sf1 = matrix_sqrt(f1, 0, True)
        _, sf2 = matrix_sqrt(f2, 0, True)
        sf3, _ = matrix_sqrt(f3, 0, True)
        sf4, _ = matrix_sqrt(f4, 0, True)
        fGss[i][j] = ft.fncon(
            [fGss[i][j],sf1,sf2,sf3,sf4],
            [[-1,1,2,3,4],[-2,1],[-3,2],[3,-4],[4,-5]]
        )
        if dual == 1:
            fGss[i][j] = fGss[i][j].gT.flip_dual([1,2], minus=True)\
                .flip_dual([3,4], minus=False)
    return fGss
