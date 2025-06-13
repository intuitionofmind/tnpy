import numpy as np
import scipy.sparse.linalg as scilas
import fermiT as ft
from fermiT import FermiT
import fermiT as ft

def cal_fGLs(
    fUs: list[FermiT], fDs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], 
    fGLs: None|list[FermiT]=None, precision=1e-6
) -> tuple[np.ndarray, list[FermiT]]:
    r"""
    Calculate the left fixed points
    (i = 0, 1, ..., N-1)
    ```
        GL[i] (M[i+1] ... M[i+N]) = GL[i]
        ==>          GL[i] M[i+1] = GL[i+1]
    ```
    Each column `M[i]` is given by
    (reading from bottom to top)
    ```
        M[i] = D[i] (G1[i] x G0[i]) U[i]
    ```
    The fixed points are normalized by
    ```
        |GL[i]|^2 = 1
    ```
    
    Parameters
    ----
    fG1s, fG0s: list[FermiT]
        PEPS tensors (conjugated and original)
    fGLs: None or list[FermiT]
        initial guess of the fixed point tensor

    PEPS tensor axis convention
    ```
            3  0
            ↑ /
        2 → A → 4
            ↑
            1
    ```

    Contraction order
    ```
        |--- 7 ---- U -- -4
        |           |
        |          8/9
        |           |
        GL - 2/3 -- G --- -2/-3
        |           |
        |          4/5
        |           |
        |--- 1 ---- D -- -1
    ```
    - G is the double tensor G1 x G0
    - Index 6 is for contraction of physical axis of G1, G0. 
    - i/j contract with conjugated/original PEPS tensor (G1/G0)
    """
    N = len(fUs)
    # shape/dual of the fixed points
    DS = (fDs[-1].DS[3], fG1s[-1].DS[4], fG0s[-1].DS[4], fUs[-1].DS[3])
    DE = (fDs[-1].DE[3], fG1s[-1].DE[4], fG0s[-1].DE[4], fUs[-1].DE[3])
    dual = (fDs[-1].dual[3], fG1s[-1].dual[4], fG0s[-1].dual[4], fUs[-1].dual[3])
    axes = [[1,2,3,7],[1,4,5,-1],[6,4,2,8,-2],[6,5,3,9,-3],[7,8,9,-4]]
    # solve GL[N-1] = GL[N-1] (M[0] ... M[N-1])
    def updateV(V: np.ndarray):
        Vf = ft.FermiT(DS, DE, dual, V.reshape(DS))
        for i in range(N):
            Vf = ft.fncon([Vf, fDs[i], fG1s[i], fG0s[i], fUs[i]], axes)
        return Vf.val.reshape(-1)
    Dl = np.prod(DS)
    M = scilas.LinearOperator((Dl,Dl), matvec=updateV)
    e0, v0 = scilas.eigs(
        M, k=1, which="LM", tol=precision, 
        v0 = (None if fGLs is None else fGLs[-1].val.reshape(-1))
    ) # returned v0 is already normalized
    fGLs = [None] * N
    fGLs[-1] = ft.FermiT(DS, DE, dual, v0.reshape(DS))
    # obtain other GLs using GL[i-1] M[i] = GL[i]
    for i in range(N-1):
        fGLs[i] = ft.fncon(
            [fGLs[i-1],fDs[i],fG1s[i],fG0s[i],fUs[i]], axes
        )
        # normalization
        fGLs[i] /= ft.linalg.norm(fGLs[i])
    return e0, fGLs

def cal_fGRs(
    fUs: list[FermiT], fDs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], 
    fGRs: None|list[FermiT]=None, precision=1e-6
) -> tuple[np.ndarray, list[FermiT]]:
    r"""
    Calculate the right fixed points 
    (i = 0, ..., N-1)
    ```
        M[i+1] ... M[i+N] GR[i+1] = GR[i+1]
        ==>                 GR[i] = M[i] GR[i+1]
    ```
    Each column `M[i]` is given by 
    (reading from bottom to top)
    ```
        M[i] = D[i] (G1[i] x G0[i]) U[i]
    ```
    The fixed points are normalized by
    ```
        |GR[i]|^2 = 1
    ```

    Parameters
    ----
    fG1s, fG0s: list[FermiT]
        PEPS tensors (conjugated and original)
    fGRs: list[FermiT]
        initial guess of the fixed point tensor

    PEPS tensor axis convention
    ```
            3  0
            ↑ /
        2 → A → 4
            ↑
            1
    ```

    Contraction order
    ```
        -4 ---- U ---- 7 ---|
                |           |
               8/9          |
                |           |
        -2/-3 - G -- 2/3 -- GR
                |           |
               4/5          |
                |           |
        -1 ---- L ---- 1 ---|
    ```
    - G is the double tensor G1 x G0.
    - In axis order i/j, i belongs to G1 and j belongs to G0. 
    - Index 6 is for the contraction of physical axis of G1 and G0. 
    """
    N = len(fUs)
    # shape/dual of the fixed points
    DS = (fDs[0].DS[0], fG1s[0].DS[2], fG0s[0].DS[2], fUs[0].DS[0])
    DE = (fDs[0].DE[0], fG1s[0].DE[2], fG0s[0].DE[2], fUs[0].DE[0])
    dual = (fDs[0].dual[0], fG1s[0].dual[2], fG0s[0].dual[2], fUs[0].dual[0])
    axes = [[1,2,3,7],[-1,4,5,1],[6,4,-2,8,2],[6,5,-3,9,3],[-4,8,9,7]]
    # solve M[0] ... M[N-1] GR[0] = GR[0]
    def updateV(V: np.ndarray):
        Vf = ft.FermiT(DS, DE, dual, V.reshape(DS))
        for i in reversed(range(N)):
            Vf = ft.fncon(
                [Vf, fDs[i], fG1s[i], fG0s[i], fUs[i]], axes
            )
        return Vf.val.reshape(-1)
    Dl = np.prod(DS)
    M = scilas.LinearOperator((Dl,Dl), matvec=updateV)
    e0, v0 = scilas.eigs(
        M, k=1, which="LM", tol=precision, 
        v0 = (None if fGRs is None else fGRs[0].val.reshape(-1))
    ) # returned v0 is already normalized
    fGRs = [None] * N
    fGRs[0] = ft.FermiT(DS, DE, dual, v0.reshape(DS))
    # obtain other GRs using M[i] GR[i+1] = GR[i]
    for i in reversed(range(1,N)):
        fGRs[i] = ft.fncon(
            [fGRs[(i+1)%N],fDs[i],fG1s[i],fG0s[i],fUs[i]], axes
        )
        fGRs[i] /= ft.linalg.norm(fGRs[i])
    return e0, fGRs

def fixed_point_line(
    fUs: list[FermiT], fDs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], 
    direction: str, precision=1e-6
):
    """
    Details
    ----
    For down, up fixed point, we first transpose 
    the PEPS tensor by (0,2,1,4,3) (mirror by y = x)
    ```
            4  0
            ↑ /
        1 → A → 3
            ↑
            2
    ```
    Then we can use the same contraction indices
    as for left, right fixed points
    with the replacement `fUs <- fRs` and `fDs <- fLs`
    ```
        -1        -2/-3         -4
        |           |           |
        |           |           |
        LL - 4/5 -- G -- 8/9 - RL
        |           |           |
        1          2/3          7
        |           |           |
        |---------- GD ---------|

        |-----------GU----------|
        |           |           |
        1          2/3          7
        |           |           |
        LR - 4/5 -- G -- 8/9 - RR
        |           |           |
        |           |           |
        -1        -2/-3         -4
    ```
    """
    if direction in ("down", "up"):
        fG1s = [fG1.transpose(0,2,1,4,3) for fG1 in fG1s]
        fG0s = [fG0.transpose(0,2,1,4,3) for fG0 in fG0s]
    if   direction in ("left", "down"): 
        eG, fGXs = cal_fGLs(fUs, fDs, fG1s, fG0s, precision=precision)
    elif direction in ("right", "up"):
        eG, fGXs = cal_fGRs(fUs, fDs, fG1s, fG0s, precision=precision)
    else:
        raise ValueError(
            "Unrecognized position of fixed point environment"
        )
    return fGXs

def fixed_point(
    fUss: list[list[FermiT]], fDss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    direction: str, precision=1e-6
) -> list[list[FermiT]]:
    """
    Find the transfer matrix fixed point

    Left/right fixed point axis order
    ----
    ```
        |----- 3 -----|
        |             |
        GL -- 1/2 -- GR
        |             |
        |----- 0 -----|
    ```

    Up/down fixed point
    ----
    ```
        |----- GR ----|
        |      |      |
        0     1/2     3
        |      |      |
        |----- GD ----|
    ```
    """
    N1, N2 = len(fUss), len(fUss[0])
    fGXss = [[None] * N2 for i in range(N1)]
    if   direction in ("left", "right"):
        for i in range(N1):
            fGXss[i] = fixed_point_line(
                fUss[i], fDss[i], fG1ss[i], fG0ss[i], 
                direction=direction, precision=precision
            )
    elif direction in ("down", "up"):
        for j in range(N2):
            fUs  = [fUss[i][j] for i in range(N1)]
            fDs  = [fDss[i][j] for i in range(N1)]
            fG1s = [fG1ss[i][j] for i in range(N1)]
            fG0s = [fG0ss[i][j] for i in range(N1)]
            fGXs = fixed_point_line(
                fUs, fDs, fG1s, fG0s, 
                direction=direction, precision=precision
            )
            for i in range(N1):
                fGXss[i][j] = fGXs[i]
    else:
        raise ValueError("Please input correct direction!")
    return fGXss
