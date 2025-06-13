import numpy as np
import scipy.sparse.linalg as scilas
import fermiT as ft
from fermiT import FermiT

def cal_fGL2s(
    fUss: list[list[FermiT]], fDss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    i0=0, precision=1e-6
) -> list[FermiT]:
    r"""
    Calculate 2-row left fixed points
    (j = 0, ..., N-1)
    ```
        GL[j] M[j+1] = GL[j+1]
    ```
    Each column `M[j]` contains two rows of the PEPS
    (reading from bottom to top)
    ```
        M[j] = D[j] (G1[i0,j] x G0[i0,j]) 
                (G1[i0+1,j] x G0[i0+1,j]) U[j]
    ```
    The fixed points are normalized by
    ```
        |GL[i]|^2 = 1
    ```
    
    Parameters
    ----
    i0: int (default 0)
        the number of the bottom (lower) row
    """
    N1, N2 = len(fUss), len(fUss[0])
    i1, i2 = i0, (i0 + 1)%N1
    # shape/dual of the 2-row fixed points
    DS = (
        fDss[i1][-1].DS[3],  
        fG1ss[i1][-1].DS[4], fG0ss[i1][-1].DS[4], 
        fG1ss[i2][-1].DS[4], fG0ss[i2][-1].DS[4], 
        fUss[i2][-1].DS[3]
    )
    DE = (
        fDss[i1][-1].DE[3],
        fG1ss[i1][-1].DE[4], fG0ss[i1][-1].DE[4],
        fG1ss[i2][-1].DE[4], fG0ss[i2][-1].DE[4],
        fUss[i2][-1].DE[3]
    )
    Dual = (
        fDss[i1][-1].dual[3],
        fG1ss[i1][-1].dual[4], fG0ss[i1][-1].dual[4],
        fG1ss[i2][-1].dual[4], fG0ss[i2][-1].dual[4],
        fUss[i2][-1].dual[3]
    )
    axes = [
        [1,2,3,7,8,12], [1,4,5,-1],
        [6,4,2,9,-2],   [6,5,3,10,-3],
        [11,9,7,13,-4], [11,10,8,14,-5],
        [12,13,14,-6]
    ]

    def updateV(V):
        V = ft.FermiT(DS, DE, Dual, V.reshape(DS))
        for j in range(N2):
            V = ft.fncon([
                V, fDss[i1][j], fG1ss[i1][j], fG0ss[i1][j],
                fG1ss[i2][j], fG0ss[i2][j], fUss[i2][j]
            ], axes)
        return V.val.reshape(-1)
    Dl = np.prod(DS)
    M = scilas.LinearOperator((Dl,Dl), matvec=updateV)
    e0, v0 = scilas.eigs(M, k=1, which="LM", tol=precision)

    fGL2s = [None] * N2 
    fGL2s[-1] = ft.FermiT(DS, DE, Dual, v0.reshape(DS))
    for j in range(N2-1):
        fGL2s[j] = ft.fncon([
            fGL2s[j-1],fDss[i1][j],fG1ss[i1][j],fG0ss[i1][j],
            fG1ss[i2][j],fG0ss[i2][j],fUss[i2][j]
        ], axes)
        # normalization
        fGL2s[j] /= ft.linalg.norm(fGL2s[j])
    return fGL2s

def cal_fGR2s(
    fUss: list[list[FermiT]], fDss: list[list[FermiT]], 
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    i0=0, precision=1e-6
) -> list[FermiT]:
    r"""
    Calculate 2-row right fixed points
    (j = 0, ..., N-1)
    ```
        GR[j] = M[j] GR[j+1]
    ```
    Each column `M[j]` contains two rows of the PEPS
    (reading from bottom to top)
    ```
        M[j] = D[j] (G1[i0,j] x G0[i0,j]) 
                (G1[i0+1,j] x G0[i0+1,j]) U[j]
    ```
    The fixed points are normalized by
    ```
        |GR[i]|^2 = 1
    ```

    Parameters
    ----
    i0: int (default 0)
        the number of the bottom (lower) row
    """
    N1, N2 = len(fUss), len(fUss[0])
    i1, i2 = i0, (i0 + 1)%N1
    # shape/dual of the 2-row fixed points
    DS = (
        fDss[i1][0].DS[0],
        fG1ss[i1][0].DS[2], fG0ss[i1][0].DS[2],
        fG1ss[i2][0].DS[2], fG0ss[i2][0].DS[2],
        fUss[i2][0].DS[0]
    )
    DE = (
        fDss[i1][0].DE[0],
        fG1ss[i1][0].DE[2], fG0ss[i1][0].DE[2],
        fG1ss[i2][0].DE[2], fG0ss[i2][0].DE[2],
        fUss[i2][0].DE[0]
    )
    Dual = (
        fDss[i1][0].dual[0],
        fG1ss[i1][0].dual[2], fG0ss[i1][0].dual[2],
        fG1ss[i2][0].dual[2], fG0ss[i2][0].dual[2],
        fUss[i2][0].dual[0]
    )
    axes = [
        [1,2,3,7,8,12], [-1,4,5,1],
        [6,4,-2,9,2],   [6,5,-3,10,3],
        [11,9,-4,13,7], [11,10,-5,14,8],
        [-6,13,14,12]
    ]

    def updateV(V):
        V = ft.FermiT(DS, DE, Dual, V.reshape(DS))
        for j in range(N2)[::-1]:
            V = ft.fncon([
                V,fDss[i1][j],fG1ss[i1][j],fG0ss[i1][j],
                fG1ss[i2][j],fG0ss[i2][j],fUss[i2][j]
            ], axes)
        return V.val.reshape(-1)
    Dl = np.prod(DS)
    M = scilas.LinearOperator((Dl,Dl), matvec=updateV)
    e0, v0 = scilas.eigs(M, k=1, which="LM", tol=precision)

    fGR2s = [None] * N2
    fGR2s[0] = ft.FermiT(DS, DE, Dual, v0.reshape(DS))
    for j in reversed(range(1,N2)):
        fGR2s[j] = ft.fncon([
            fGR2s[(j+1)%N2],fDss[i1][j],fG1ss[i1][j],fG0ss[i1][j],
            fG1ss[i2][j],fG0ss[i2][j],fUss[i2][j]
        ], axes)
        # normalization
        fGR2s[j] /= ft.linalg.norm(fGR2s[j])
    return fGR2s
