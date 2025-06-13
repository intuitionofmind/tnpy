"""
Dual and axis order of boundary MPS
- up/down MPS

    ```
    0 → L → 3  0 → C ← 1  0 ← R ← 3
        ↑                     ↑
        1/2                   1/2

        1/2                   1/2
        ↑                     ↑
    0 ← L ← 3  0 ← C → 1  0 → R → 3
    ```

- left/right MPS

    ```
    3                   3
    ↑                   ↓
    R → 1/2       1/2 → R
    ↑                   ↓
    0                   0

    1                   1      
    ↑                   ↓
    C                   C
    ↓                   ↑
    0                   0

    3                   3
    ↓                   ↑
    L → 1/2       1/2 → L
    ↓                   ↑
    0                   0
    ```

i/j contract with conjugated/original PEPS tensor (G1/G0)
"""

from time import time
from itertools import product
import fermiT as ft
from fermiT import FermiT
from vumps.fixed_point import cal_fGLs, cal_fGRs
from vumps.minacc import min_acc_all
from vumps.unitcell_tools import relabel_unitcell, rotate_unitcell

def onestep_line(
    fCs: list[FermiT],
    fALs: list[FermiT], fARs: list[FermiT],
    fALps: list[FermiT], fARps: list[FermiT],
    fG1s: list[FermiT], fG0s: list[FermiT],
    fGLs: list[FermiT], fGRs: list[FermiT], precision=1e-6
):
    r"""
    One iterative step in solving AL, C, AR 
    for one row of the up-bMPS

    PEPS tensor axis convention
    ```
            3  0
            ↑ /
        2 → A → 4
            ↑
            1
    ```

    Parameters
    ----
    fCs, fALs, fARs: list[FermiT]
        Initial value of boundary MPS in current row
    fALps, fARps: list[FermiT]
        Initial value of boundary MPS in next row
    fG1s, fG0s: list[FermiT]
        PEPS tensors in one row (conjugated and original)
    fGLs, fGRs: list[FermiT]
        initial guess of left/right fixed points 
        of the transfer matrix
    """
    N = len(fCs)
    # get AC[i] = AL[i] C[i] (= C[i-1] AR[i]) 
    fACs = [ft.tensordot(fALs[i], fCs[i], [3,0]) for i in range(N)]
    # find left fixed points of AL column transfer matrix
    eGL, fGLs = cal_fGLs(
        fALs, [fALps[i].gT for i in range(N)], 
        fG1s, fG0s, fGLs, precision=precision
    )
    # find right fixed points of AR column transfer matrix
    eGR, fGRs = cal_fGRs(
        fARs, [fARps[i].gT for i in range(N)], 
        fG1s, fG0s, fGRs, precision=precision
    )
    """
    Update AC in the next row

    |-→- 1 -→ AC ←- 7 -←-|
    |          ↑         |
    |         4/5        |
    |          ↑         |
    GL → 2/3 → G → 8/9 → GR  =  0 → AC ← 3
    |          ↑         |          ↑
    |        -2/-3       |         1/2
    |                    |
    |-←- -1        -4 -→-|

    i/j contract with conjugated/original PEPS tensor (G1/G0)
    """
    fACps = [ft.fncon(
        [fGLs[i-1],fACs[i],fG1s[i],fG0s[i],fGRs[(i+1)%N]],
        [[-1,2,3,1],[1,4,5,7],[6,-2,2,4,8],[6,-3,3,5,9],[-4,8,9,7]]
    ) for i in range(N)]
    """
    Update C in the next row

    |-→ 1 -→- C -←- 4 ←-|
    |                   |
    GL --→-- 2/3 --→-- GR  =  0 → C ← 1
    |                   |
    |-←- -1       -2 -→-|
    """
    fCps  = [ft.fncon(
        [fGLs[i],fCs[i],fGRs[(i+1)%N]],
        [[-1,2,3,1],[1,4],[-2,2,3,4]]
    ) for i in range(N)]
    # normalization
    for i in range(N):
        fACps[i] /= ft.linalg.norm(fACps[i])
        fCps[i] /= ft.linalg.norm(fCps[i])
    # updated AL, AR from updated AC and C
    fALps, fARps, errs = min_acc_all(fACps, fCps)
    return fCps, fALps, fARps, max(errs), fGLs, fGRs


def fixed_boundary_normal(
    Dcut: int, Dce: int,
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]],
    fAss: list[list[list[FermiT]]] | None = None, 
    iternum0=6, iternum1=30, iternum2=1, 
    tolerance=1e-6, tps_type="MN", icheck=0
) -> tuple[
    list[list[FermiT]], list[list[FermiT]], 
    list[list[FermiT]], list[int|float]
]:
    """
    Find the up-bMPS for each row of the PEPS

    PEPS tensor axis convention
    ```
            3  0
            ↑ /
        2 → A → 4
            ↑
            1
    ```

    PEPS tensor label (e.g. 3 x 4 unit cell)
    ```
        |-----|-----|-----| (up-bMPS, 0-th row)
        (0,0) (0,1) (0,2) (0,3)
        (1,0) (1,1) (1,2) (1,3)
        (2,0) (2,1) (2,2) (2,3)
    ```
    Note that it is different from the main convention:
    ```
        (2,0) (2,1) (2,2) (2,3)
        (1,0) (1,1) (1,2) (1,3)
        (0,0) (0,1) (0,2) (0,3)
    ```

    Up-bMPS tensor axis convention
    ```
    0 → L → 3  0 → C ← 1  0 ← R ← 3  ==>  0 → AC ← 3
        ↑                     ↑               ↑
        1/2                   1/2            1/2
    ```
    1/2 etc contract with conjugated(G1)/original(G0) PEPS tensor
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    if tps_type == "AB": assert N1 == 2 and N2 == 2
    if fAss is None:
        mode = "random"
        print(f"Calculation mode: {mode}")
        # randomly initialize boundary MPS
        dual = fG1ss[0][0].dual[1]
        fACss = [[ft.rand(
            (Dcut,fG1ss[i][j].DS[3], fG1ss[i][j].DS[3],Dcut), 
            (Dce,fG1ss[i][j].DE[3], fG1ss[i][j].DE[3],Dce), (dual,)*4
        ) for j in range(N2)] for i in range(N1)]
        fCss  = [[ft.rand(
            (Dcut,Dcut), (Dce,Dce), (dual,)*2
        ) for j in range(N2)] for i in range(N1)]
        fALss, fARss = [None] * N1, [None] * N1
        for i in range(N1):
            fALss[i], fARss[i], errs = min_acc_all(fACss[i], fCss[i])
    else:
        Dcut0, Dce0 = fAss[0][0][0].DS[0], fAss[0][0][0].DE[0]
        if Dcut0 == Dcut:
            # initial boundary MPS chi != Dcut
            mode = "update"
            print(f"Calculation mode: {mode}")
            fCss, fALss, fARss = fAss
        else: 
            # initial boundary MPS chi != Dcut
            mode = "expand"
            print(f"Calculation mode: {mode}")
            fCpss, fALpss, fARpss = fAss
            # randomly initialize boundary MPS
            dual = fG1ss[0][0].dual[1]
            fACss = [[ft.rand(
                (Dcut,fG1ss[i][j].DS[3], fG1ss[i][j].DS[3],Dcut), 
                (Dce,fG1ss[i][j].DE[3], fG1ss[i][j].DE[3],Dce), (dual,)*4
            ) for j in range(N2)] for i in range(N1)]
            fCss  = [[ft.rand(
                (Dcut,Dcut), (Dce,Dce), (dual,dual)
            ) for j in range(N2)] for i in range(N1)]
            fALss, fARss = [None] * N1, [None] * N1
            for i in range(N1):
                fALss[i], fARss[i], errs = min_acc_all(fACss[i], fCss[i])
            # randomly initialize left/right fixed points
            dual = fCss[0][0].dual[0]
            fGLss = [[None] * N2 for i in range(N1)]
            fGRss = [[None] * N2 for i in range(N1)]
            for i, j in product(range(N1), range(N2)):
                fGLss[i][j] = ft.rand(
                    (Dcut,fG1ss[i][j].DS[4],fG0ss[i][j].DS[4],Dcut0), 
                    (Dce,fG1ss[i][j].DE[4],fG0ss[i][j].DE[4],Dce0), 
                    (dual,fG1ss[i][j].dual[4],fG0ss[i][j].dual[4],1-dual)
                )
                fGRss[i][j] = ft.rand(
                    (Dcut,fG1ss[i][j].DS[2],fG0ss[i][j].DS[2],Dcut0), 
                    (Dce,fG1ss[i][j].DE[2],fG0ss[i][j].DE[2],Dce0), 
                    (dual,fG1ss[i][j].dual[2],fG0ss[i][j].dual[2],1-dual)
                )
            t0 = time()
            # preparing initial bMPS with a different bond dimension
            for i in range(N1):
                for r0 in range(iternum0):
                    fCss[i], fALss[i], fARss[i], err, fGLss[i-1], fGRss[i-1] \
                    = onestep_line(
                        fCpss[i-1], fALpss[i-1], fARpss[i-1], 
                        fALss[i], fARss[i], 
                        fG1ss[i-1], fG0ss[i-1], fGLss[i-1], fGRss[i-1]
                    )
                    if icheck > 1: print(i, r0, time() - t0, err)
                if tps_type == "AB":
                    # copy updated boundary MPS to the previous row
                    # and shift by one site horizontally
                    fCss[1]  = [fCss[0][1], fCss[0][0]]
                    fALss[1] = [fALss[0][1], fALss[0][0]]
                    fARss[1] = [fARss[0][1], fARss[0][0]]
                    # no need to process the 2nd row
                    break
    dual = fCss[0][0].dual[0]
    fGLss = [[None] * N2 for i in range(N1)]
    fGRss = [[None] * N2 for i in range(N1)]
    for i, j in product(range(N1), range(N2)):
        fGLss[i][j] = ft.rand(
            (Dcut,fG1ss[i][j].DS[4],fG0ss[i][j].DS[4],Dcut), 
            (Dce,fG1ss[i][j].DE[4],fG0ss[i][j].DE[4],Dce), 
            (dual,fG1ss[i][j].dual[4],fG0ss[i][j].dual[4],1-dual)
        )
        fGRss[i][j] = ft.rand(
            (Dcut,fG1ss[i][j].DS[2],fG0ss[i][j].DS[2],Dcut), 
            (Dce,fG1ss[i][j].DE[2],fG0ss[i][j].DE[2],Dce), 
            (dual,fG1ss[i][j].dual[2],fG0ss[i][j].dual[2],1-dual)
        )
    t0 = time()
    epsilon = 1e-6
    num_qualify = 0
    num_break = 2*N1
    for r1 in range(iternum1):
        t1 = time()
        ibefore, iafter = r1%N1, (r1+1)%N1
        fCs, fALs, fARs = fCss[ibefore], fALss[ibefore], fARss[ibefore]
        fALps, fARps = fALss[iafter],  fARss[iafter]
        fGLs, fGRs = fGLss[ibefore], fGRss[ibefore]
        fG1s, fG0s = fG1ss[ibefore], fG0ss[ibefore]
        for r2 in range(iternum2):
            fCps, fALps, fARps, err, fGLs, fGRs = onestep_line(
                fCs, fALs, fARs, fALps, fARps, fG1s, fG0s, 
                fGLs, fGRs, precision=epsilon
            )
            epsilon = min(epsilon, err*1e-3)
            if icheck > 2: 
                print("  sub-process", r2, err)
        fGLss[ibefore], fGRss[ibefore] = fGLs, fGRs
        fCss[iafter], fALss[iafter], fARss[iafter] = fCps, fALps, fARps
        if tps_type == "AB":
            # copy updated boundary MPS to the previous row
            # and shift by one site horizontally
            fCss[ibefore]  = [fCps[1], fCps[0]]
            fALss[ibefore] = [fALps[1], fALps[0]]
            fARss[ibefore] = [fARps[1], fARps[0]]
        if icheck > 1: 
            subtime = time() - t1
            print(
                f"process of normal order: {r1:>2d} {r2:>2d} {err:.4e} "
                + f"sub-time: {subtime:.4f} s"
            )
        if err < tolerance: num_qualify += 1
        if num_qualify > num_break: break
    if icheck > 0: 
        subtime = time() - t0
        print(f"End! {r1:>2d} {r2:>2d} {err:.4e} | time: {subtime:.4f} s")
    return fCss, fALss, fARss, [r1, err]


def fixed_boundary(
    Dcut: int, Dce: int, direction: str, 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]],
    fAss: None | list[list[list[FermiT]]] = None,
    iternum0=6, iternum1=40, iternum2=1, tolerance=1e-6,
    tps_type="MN", icheck=0
):
    """
    Find uniform boundary MPS in canonical gauge

    Parameters
    ----
    Dcut, Dce: int
        Total and even dimension of the boundary MPS virtual indices
    
    direction: str ("up", "down", "left", "right")
        position of the boundary to be calculated
    
    fG1ss, fG0ss: list[list[FermiT]]
        The TPS tensors (conjugated and original)
        in each unit cell
    
    fAss: None or list[list[list[FermiT]]]
        initial guess of boundary MPS in 
        canonical gauge (C, AL, AR), labeled as
        ```
        (2,0) (2,1) (2,2) (2,3)
        (1,0) (1,1) (1,2) (1,3)
        (0,0) (0,1) (0,2) (0,3)
        ```

    iternum0: int
        optimization for initial boundary MPS
        with bond dimension != Dcut
        (not applicable for bMPS with bond dimension = Dcut)
    
    iternum1: int
        max iternum from row i -> i + 1 -> i + 2 -> ...
        row i -> i + 1 is counted as once
    
    iternum2: int
        max iternum in row i -> i + 1. 
        Each row should converge in principle, 
        but in practice 1 is enough 
        (convergence can always be achieved with large enough iternum1)
    """
    if fAss is not None:
        for i in range(3):
            fAss[i] = relabel_unitcell(fAss[i], direction)
    # standard config: down boundary MPS
    fG1pss = rotate_unitcell(fG1ss, direction)
    fG1pss = relabel_unitcell(fG1pss, direction)
    fG0pss = rotate_unitcell(fG0ss, direction)
    fG0pss = relabel_unitcell(fG0pss, direction)
    boundary_params = dict(
        fAss=fAss, tps_type=tps_type,
        iternum0=iternum0, iternum1=iternum1, iternum2=iternum2, 
        tolerance=tolerance, icheck=icheck
    )
    fCss, fALss, fARss, info = fixed_boundary_normal(
        Dcut, Dce, fG1pss, fG0pss, **boundary_params
    )
    fCss = relabel_unitcell(fCss, direction, reverse=True)
    fALss = relabel_unitcell(fALss, direction, reverse=True)
    fARss = relabel_unitcell(fARss, direction, reverse=True)
    return fCss, fALss, fARss, info
