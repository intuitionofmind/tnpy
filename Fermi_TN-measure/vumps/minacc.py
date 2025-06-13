import fermiT as ft
from fermiT import FermiT
from fermiT.linalg import polar, norm

def min_acc_up(
    fAC: FermiT, fC: FermiT, mode="all"
) -> tuple[FermiT, float] | tuple[FermiT, FermiT, float, float]:
    if mode != "right":
        UACL, PACL = polar(fAC, 3, side="right", typef=0)
        UCL, PCL = polar(fC, 1, side="right", typef=0)
        fermiAL = ft.tensordot(UACL, UCL.gconj(), [3,0])
        errL = norm(PACL - PCL)
    if mode != "left":
        PACR, UACR = polar(fAC, 1, side="left", typef=1)
        PCR, UCR = polar(fC, 1, side="left", typef=1)
        fermiAR = ft.tensordot(UCR.gconj(), UACR, [1,0])
        errR = norm(PACR - PCR)
    if   mode == "left":  return fermiAL, errL
    elif mode == "right": return fermiAR, errR
    else: return fermiAL, fermiAR, errL, errR


def min_acc_down(
    fAC: FermiT, fC: FermiT, mode="all"
) -> tuple[FermiT, float] | tuple[FermiT, FermiT, float, float]:
    if mode != "right":
        UACL, PACL = polar(fAC, 3, side="right", typef=1)
        UCL, PCL = polar(fC, 1, side="right", typef=1)
        fermiAL = ft.tensordot(UACL, UCL.gconj(), [3,0])
        errL = norm(PACL - PCL)
    if mode != "left":
        PACR, UACR = polar(fAC, 1, side="left", typef=0)
        PCR, UCR = polar(fC, 1, side="left", typef=0)
        fermiAR = ft.tensordot(UCR.gconj(), UACR, [1,0])
        errR = norm(PACR - PCR)
    if   mode == "left":  return fermiAL, errL
    elif mode == "right": return fermiAR, errR
    else: return fermiAL, fermiAR, errL, errR


def min_acc_all(
    fACs: list[FermiT], fCs: list[FermiT]
) -> tuple[list[FermiT], list[FermiT], list[float]]:
    r"""
    Solve AL (left orthogonal), AR (right orthogonal) 
    from AC, C (same row) that satisfy
    ```
        AL[i] C[i] = AC[i] = C[i-1] AR[i]
    ```
    
    Parameters
    ----
    fACs: list[FermiT]
        the AC tensors of the i-th row of the boundary MPS
    fCs: list[FermiT]
        the C tensors of the i-th row of the boundary MPS

    Details
    ----
    Naively, we have
    ```
        AL = AC C^{-1}
        AR = C^{-1} AC
    ```
    But to make AL, AR orthogonal, we use the polar decomposition
    ```
        AC = UACL PACL,  C = UCL PCL
        AC = PACR UACR,  C = PCR UCR
    ```
    At convergence, PACL ≈ PCL and PACR ≈ PCR, and we obtain
    ```
        AL = UACL PACL PCL^{-1} (UCL)† ≈ UACL (UCL)†
        AR = (UCR)† PCR^{-1} PACR UACR ≈ (UCR)† UACR
    ```
    We can measure the convergence by
    ```
        errL = |PACL - PCL|
        errR = |PACR - PCR|
    ```
    """
    N = len(fACs)
    fALs = [None] * N
    fARs = [None] * N
    errs = []
    for i in range(N):
        if fACs[i].dual[1]:
            fALs[i], errL = min_acc_down(fACs[i], fCs[i], mode="left")
            fARs[i], errR = min_acc_down(fACs[i], fCs[i-1], mode="right")
        else:
            fALs[i], errL = min_acc_up(fACs[i], fCs[i], mode="left")
            fARs[i], errR = min_acc_up(fACs[i], fCs[i-1], mode="right")
        errs += [errL, errR]
    return fALs, fARs, errs
