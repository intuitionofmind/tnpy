import fermiT as ft
from fermiT import FermiT
from fermiT.conversion import gt2ft
from fermiT.linalg import matrix_sqrt
from update_ftps.convert_dual import dual_loop2sym
from update_ftps.sutools import get_dualconv
from update_ftps.tps_io import load_tps as load_tps_gt
import vumps.files_par as ffile


def save_tps(
    fA: FermiT, fB: FermiT, 
    ftaus: list[FermiT], folder: str
):
    fTs = [fA, fB] + ftaus
    ffile.save_tensors(fTs, folder=folder)


def load_tps(folder: str):
    """
    FermiT axis order (physical index at 0)
    ```
            3
            |
        2 - T - 4
            |
            1
    ```
    """
    try:
        ts, wts = load_tps_gt(folder)
        # change loop to sym convention
        if get_dualconv(ts, wts) == "loop":
            ts, wts = dual_loop2sym(ts, wts)
        # change local tensor axis order
        perm = [0,4,3,2,1]
        for tname in ts.keys():
            ts[tname] = ts[tname].transpose(*perm)
        assert len(ts) == 2
        assert len(wts) == 4
        fA = gt2ft(ts["Ta"])
        fB = gt2ft(ts["Tb"])
        f1 = gt2ft(wts["y1"])
        f2 = gt2ft(wts["x2"])
        f3 = gt2ft(wts["y2"])
        f4 = gt2ft(wts["x1"])
    except:
        fTs = ffile.load_tensors(6, folder=folder)
        fA, fB, f1, f2, f3, f4 = fTs
    return fA, fB, [f1, f2, f3, f4]


def absorb_wts(
    fA: FermiT, fB: FermiT, ftaus: list[FermiT], dual=0
):
    """
    Input PEPS (symmetric convention)
    ```
                ↑       ↓
                f1      f3
                ↑       ↑
        -→ f2 ← B → f4← A → f2 →-   
                ↓       ↓
                f3      f1
                ↑       ↑
        -← f4 ← A → f2← B → f4 ←-
                ↓       ↓
                f1      f3
                ↑       ↓
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
    f1, f2, f3, f4 = ftaus
    sf1a, sf1b = matrix_sqrt(f1, 0, True)
    sf2a, sf2b = matrix_sqrt(f2, 0, True)
    sf3a, sf3b = matrix_sqrt(f3, 0, True)
    sf4a, sf4b = matrix_sqrt(f4, 0, True)
    fAp = ft.fncon(
        [fA,sf1b,sf2b,sf3a,sf4a],
        [[-1,1,2,3,4],[-2,1],[-3,2],[3,-4],[4,-5]]
    )
    fBp = ft.fncon(
        [fB,sf3b,sf4b,sf1a,sf2a],
        [[-1,1,2,3,4],[-2,1],[-3,2],[3,-4],[4,-5]]
    )
    if dual == 1:
        fAp = fAp.gT.flip_dual([1,2], minus=True)
        fAp = fAp.flip_dual([3,4], minus=False)
        fBp = fBp.gT.flip_dual([1,2], minus=True)
        fBp = fBp.flip_dual([3,4], minus=False)
    fGss = [[fAp,fBp],[fBp,fAp]]
    return fGss
