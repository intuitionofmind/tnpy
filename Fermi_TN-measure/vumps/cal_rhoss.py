import fermiT as ft
from fermiT import FermiT
import vumps.files_par as ffile
from itertools import product

# ---- 1-row (or 1-column) measurements ----

def obtain_rho_ver(
    fLs: list[FermiT], fRs: list[FermiT], 
    fGDs: list[FermiT], fGUs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], pos=(0,0)
):
    """
    Calculate 1-site and 2-site density operators 
    along vertical direction

    Parameters
    ---- 
    pos: tuple[int, int]
        - 1-site: accepts operator at `pos`
        - 2-site: accepts operator at `pos` and `pos+y`
    """
    N1 = len(fLs)
    i, j = pos
    i1, i2 = (i + 1)%N1, (i + 2)%N1
    fD = ft.fncon([fGDs[i-1],fLs[i],fG1s[i],fG0s[i],fRs[i]], [[1,2,3,6],[1,4,5,-1],[-5,2,4,-2,7],[-6,3,5,-3,8],[6,7,8,-4]])
    fU = ft.fncon([fGUs[i2],fLs[i1],fG1s[i1],fG0s[i1],fRs[i1]], [[1,2,3,6],[-1,4,5,1],[-5,-2,4,2,7],[-6,-3,5,3,8],[-4,7,8,6]])
    rho_one = ft.fncon([fD,fGUs[i1]], [[1,2,3,4,-1,-2],[1,2,3,4]])
    rho_two = ft.fncon([fD,fU], [[1,2,3,4,-1,-3],[1,2,3,4,-2,-4]])
    return rho_one, rho_two


def obtain_rho_hor(
    fUs: list[FermiT], fDs: list[FermiT], 
    fGLs: list[FermiT], fGRs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], pos=(0,0)
):
    """
    Calculate 1-site and 2-site density operators 
    along horizontal direction

    Parameters
    ---- 
    pos: tuple[int, int]
        - 1-site: accepts operator at `pos`
        - 2-site: accepts operator at `pos` and `pos+x`
    """
    N2 = len(fUs)
    i, j = pos
    j1, j2 = (j + 1)%N2, (j + 2)%N2
    fL = ft.fncon([fGLs[j-1],fDs[j],fG1s[j],fG0s[j],fUs[j]], [[1,2,3,6],[1,4,5,-1],[-5,4,2,7,-2],[-6,5,3,8,-3],[6,7,8,-4]])
    fR = ft.fncon([fGRs[j2],fDs[j1],fG1s[j1],fG0s[j1],fUs[j1]], [[1,2,3,6],[-1,4,5,1],[-5,4,-2,7,2],[-6,5,-3,8,3],[-4,7,8,6]])
    rho_one = ft.fncon([fL,fGRs[j1]], [[1,2,3,4,-1,-2],[1,2,3,4]])
    rho_two = ft.fncon([fL,fR], [[1,2,3,4,-1,-3],[1,2,3,4,-2,-4]])
    return rho_one, rho_two


def obtain_rhoss(
    fXss: list[list[FermiT]], fGXss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], icheck=0
) -> tuple[
    list[list[FermiT]], list[list[FermiT]], 
    list[list[FermiT]], list[list[FermiT]]
]:
    """
    Calculate 1-site and 2-site density operators 
    along vertical and horizontal direction
    """
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    
    N1, N2 = len(fLss), len(fLss[0])
    rho1vss = [[None] * N2 for i in range(N1)]
    rho2vss = [[None] * N2 for i in range(N1)]
    rho1hss = [[None] * N2 for i in range(N1)]
    rho2hss = [[None] * N2 for i in range(N1)]
    
    for i in range(N1):
        fUs, fDs, fGLs, fGRs, fG1s, fG0s \
            = fUss[i], fDss[i], fGLss[i], fGRss[i], fG1ss[i], fG0ss[i]
        for j in range(N2):
            if icheck > 0: print("hor", i, j)
            rho1, rho2 = obtain_rho_hor(fUs, fDs, fGLs, fGRs, fG1s, fG0s, pos=(i,j))
            rho1hss[i][j] = rho1
            rho2hss[i][j] = rho2
    
    for j in range(N2):
        fLs  = [fLss[i][j]  for i in range(N1)]
        fRs  = [fRss[i][j]  for i in range(N1)]
        fGDs = [fGDss[i][j] for i in range(N1)]
        fGUs = [fGUss[i][j] for i in range(N1)]
        fG1s = [fG1ss[i][j] for i in range(N1)]
        fG0s = [fG0ss[i][j] for i in range(N1)]
        for i in range(N1):
            if icheck > 0: print("ver", i, j)
            rho1, rho2 = obtain_rho_ver(fLs, fRs, fGDs, fGUs, fG1s, fG0s, pos=(i,j))
            rho1vss[i][j] = rho1
            rho2vss[i][j] = rho2
    
    return rho1vss, rho2vss, rho1hss, rho2hss


def cal_rhoss(
    N1: int, N2: int, tps_dir: str,
    vumps_dir: str, rhoss_dir: str, tps_type = "AB"
):
    """
    Load PEPS, boundary MPS and
    calculate 1-site and 2-site density operators 
    along vertical and horizontal direction
    """
    # load PEPS
    fG0ss, fG1ss = ffile.load_peps(
        N1, N2, tps_dir, tps_type, project=False
    )
    # load boundary MPS and fixed points
    fGXss = ffile.load_fixedpoint(N1, N2, vumps_dir + "fixed_point/")
    fXss = [
        # only AL tensors are needed
        ffile.load_bMPScanon(N1, N2, vumps_dir + f"{d}/")[1]
        for d in ("up", "down", "left", "right")
    ]
    rho1vss, rho2vss, rho1hss, rho2hss = obtain_rhoss(fXss, fGXss, fG1ss, fG0ss)
    ffile.save_rhoss(rho1vss, rho2vss, rho1hss, rho2hss, rhoss_dir)

# ---- 2nd neighbor (diagonal) measurements ----

def obtain_rhod(
    fUss: list[list[FermiT]], fDss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    fGL2ss: list[list[FermiT]], fGR2ss: list[list[FermiT]]
) -> list[list[list[FermiT]]]:
    """
    calculate density operator on 
    diagonal (2nd neighbor) bonds
    """
    N1, N2 = len(fUss), len(fUss[0])
    rhod1ss = [[None] * N2 for i in range(N1)]
    rhod2ss = [[None] * N2 for i in range(N1)]
    for i, j in product(range(N1), range(N2)):
        i1, i2 = i, (i+1)%N1
        j0, j1, j2, j3 = j-1, j, (j+1)%N2, (j+2)%N2
        fL1 = ft.fncon([
            fGL2ss[i1][j0],fDss[i1][j1],fG1ss[i1][j1],fG0ss[i1][j1],
            fG1ss[i2][j1],fG0ss[i2][j1],fUss[i2][j1]
        ], [
            [1,2,3,7,8,11],[1,4,5,-1],[6,4,2,9,-2],[6,5,3,10,-3],
            [-7,9,7,12,-4],[-8,10,8,13,-5],[11,12,13,-6]
        ])
        fL2 = ft.fncon([
            fGL2ss[i1][j0],fUss[i2][j1],fG1ss[i2][j1],fG0ss[i2][j1],
            fG1ss[i1][j1],fG0ss[i1][j1],fDss[i1][j1]
        ], [
            [11,7,8,2,3,1],[1,4,5,-6],[6,9,2,4,-4],[6,10,3,5,-5],
            [-7,12,7,9,-2],[-8,13,8,10,-3],[11,12,13,-1]
        ])
        fR1 = ft.fncon([
            fGR2ss[i1][j3],fDss[i1][j2],fG1ss[i1][j2],fG0ss[i1][j2],
            fG1ss[i2][j2],fG0ss[i2][j2],fUss[i2][j2]
        ], [
            [1,2,3,7,8,11],[-1,4,5,1],[6,4,-2,9,2],[6,5,-3,10,3],
            [-7,9,-4,12,7],[-8,10,-5,13,8],[-6,12,13,11]
        ])
        fR2 = ft.fncon([
            fGR2ss[i1][j3],fUss[i2][j2],fG1ss[i2][j2],fG0ss[i2][j2],
            fG1ss[i1][j2],fG0ss[i1][j2],fDss[i1][j2]
        ], [
            [11,7,8,2,3,1],[-6,4,5,1],[6,9,-4,4,2],[6,10,-5,5,3],
            [-7,12,-2,9,7],[-8,13,-3,10,8],[-1,12,13,11]
        ])
        # 1 -> 4
        rhod1ss[i][j] = ft.tensordot(
            fL2, fR1, [[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ).transpose(0,2,1,3) 
        # 2 -> 3
        rhod2ss[i][j] = ft.tensordot(
            fL1, fR2, [[0,1,2,3,4,5],[0,1,2,3,4,5]]
        ).transpose(0,2,1,3) 
    return [rhod1ss, rhod2ss]


def cal_rhodss(
    N1: int, N2: int, tps_dir: str,
    vumps_dir: str, rhoss_dir: str, tps_type = "AB"
):
    """
    Load PEPS, boundary MPS and 
    calculate density operator on 2nd neighbors

    Site label in rho
    ----
    ```
            |     |
        --- 2 --- 4 ---
            |     |
        --- 1 --- 3 ---
            |     |
    ```
    """
    # load PEPS
    fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, tps_type)
    # load boundary MPS
    fXss = [
        # only AL tensors are needed
        ffile.load_bMPScanon(N1, N2, vumps_dir + f"{d}/")[1]
        for d in ("up", "down", "left", "right")
    ]
    fUss, fDss = fXss[0], fXss[1]
    # load 2-row fixed points
    fGL2ss, fGR2ss = ffile.load_fixedpoint2(N1, N2, vumps_dir + "fixed_point2/")
    print("---- Calculating diagonal density matrices ----")
    rhod1ss, rhod2ss = obtain_rhod(fUss, fDss, fG1ss, fG0ss, fGL2ss, fGR2ss)
    ffile.save_rhodss(rhod1ss, rhod2ss, rhoss_dir)

# ---- 2x2 cell measurements ----

def obtain_rho4(
    fUss: list[list[FermiT]], fDss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    fGL2ss: list[list[FermiT]], fGR2ss: list[list[FermiT]]
) -> list[list[FermiT]]:
    """
    calculate density operator on 2 x 2 cell
    """
    N1, N2 = len(fUss), len(fUss[0])
    rho4ss = [[None] * N2 for i in range(N1)]
    for i, j in product(range(N1), range(N2)):
        i1, i2 = i, (i+1)%N1
        j0, j1, j2, j3 = j-1, j, (j+1)%N2, (j+2)%N2
        fL = ft.fncon([
            fGL2ss[i1][j0],fDss[i1][j1],fG1ss[i1][j1],
            fG0ss[i1][j1],fG1ss[i2][j1],fG0ss[i2][j1],fUss[i2][j1]
        ], [
            [1,2,3,6,7,10],[1,4,5,-1],[-7,4,2,8,-2],
            [-8,5,3,9,-3],[-9,8,6,11,-4],[-10,9,7,12,-5],[10,11,12,-6]
        ])
        fR = ft.fncon([
            fGR2ss[i1][j3],fDss[i1][j2],fG1ss[i1][j2],
            fG0ss[i1][j2],fG1ss[i2][j2],fG0ss[i2][j2],fUss[i2][j2]
        ], [
            [1,2,3,6,7,10],[-1,4,5,1],[-7,4,-2,8,2],
            [-8,5,-3,9,3],[-9,8,-4,11,6],[-10,9,-5,12,7],[-6,11,12,10]
        ])
        # A Bup Bdown A
        rho4ss[i][j] = ft.tensordot(fL, fR, [[0,1,2,3,4,5],[0,1,2,3,4,5]]) 
    return rho4ss

def cal_rho4ss(
    N1: int, N2: int, tps_dir: str,
    vumps_dir: str, rhoss_dir: str, tps_type = "AB"
):
    """
    Load PEPS, boundary MPS and 
    calculate density operator on 2 x 2 cell

    Site order in rho
    ----
    ```
            |     |
        --- 2 --- 4 ---
            |     |
        --- 1 --- 3 ---
            |     |
    ```
    """
    # load PEPS
    fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, tps_type)
    # load boundary MPS
    fXss = [
        # only AL tensors are needed
        ffile.load_bMPScanon(N1, N2, vumps_dir + f"{d}/")[1]
        for d in ("up", "down", "left", "right")
    ]
    fUss, fDss = fXss[0], fXss[1]
    # load 2-row fixed points
    fGL2ss, fGR2ss = ffile.load_fixedpoint2(N1, N2, vumps_dir + "fixed_point2/")
    print("---- Calculating density matrices ----")
    rho4ss = obtain_rho4(fUss, fDss, fG1ss, fG0ss, fGL2ss, fGR2ss)
    ffile.save_rho4ss(rho4ss, rhoss_dir)
