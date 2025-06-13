import fermiT as ft
import numpy as np
from fermiT import FermiT
from .correlation import fL_single, fLR_val

def cor_4site_line_hor(
    fOs: list[FermiT], 
    fUs: list[FermiT], fDs: list[FermiT], 
    fGLs: list[FermiT], fGRs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], j0=0
):
    """
    Measure `<prod_{j=j0}^{j0+3} O_{i,j}>` on a fixed row (i)

    Parameters
    ----
    fOs: list[FermiT]
        operators on the 4 sites to be measured
    fUs, fDs: list[FermiT]
        AL (left canonical tensors) of the i-th row 
        of the up/down boundary MPSs
    fGLs, fGRs: list[FermiT]
        left/right fixed points of PEPS column transfer matrices
    fG1s, fG2s: list[FermiT]
        the i-th row of the PEPS
    """
    # number of columns
    N2 = len(fUs)
    assert 0 <= j0 < N2
    assert len(fOs) == 4
    Dps, Dpe = fG0s[0].DS[0], fG0s[0].DE[0]
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    fL0 = fL_single(fGLs[j0-1], fUs[j0], fDs[j0], fG1s[j0], fG0s[j0], Id)
    fL1 = fL_single(fGLs[j0-1], fUs[j0], fDs[j0], fG1s[j0], fG0s[j0], fOs[0])
    for j, fO in zip(range(j0+1, j0+4), fOs[1::]):
        fL0 = fL_single(fL0, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], Id)
        fL1 = fL_single(fL1, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], fO)
    val = fLR_val(fL1, fGRs[(j0+4)%N2]) / fLR_val(fL0, fGRs[(j0+4)%N2])
    return val


def cor_4site_hor(
    fOs: list[FermiT], 
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]],
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], j0=0
):
    """
    Measure `<prod_{j=j0}^{j0+3} O_{i,j}>`
    on all rows (i = 0, ..., N1-1)
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    assert len(fOs) == 4
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    vals = [None] * N1
    for i in range(N1):
        vals[i] = cor_4site_line_hor(
            fOs, fUss[i], fDss[i], 
            fGLss[i], fGRss[i], fG1ss[i], fG0ss[i], j0
        )
    return np.array(vals)


def cor_4site_line_ver(
    fOs: list[FermiT], 
    fLs: list[FermiT], fRs: list[FermiT], 
    fGUs: list[FermiT], fGDs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], i0=0
):
    """
    Measure `<prod_{i=i0}^{i0+3} O_{i,j}>` on a fixed column (j)

    By mirror the network by the diagonal line y = x, 
    we can use the horizontal code with replacements
    ```
        U, D, GL, GR -> R, L, GD, GU
    ```
    R,L,GD,GU already have correct axes order.
    However, G1/G0 should be further transposed
    """
    return cor_4site_line_hor(
        fOs, fRs, fLs, fGDs, fGUs, 
        [fG1.transpose(0,2,1,4,3) for fG1 in fG1s], 
        [fG0.transpose(0,2,1,4,3) for fG0 in fG0s], i0
    )

def cor_4site_ver(
    fOs: list[FermiT], 
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]],
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], i0=0
):
    """
    Measure `<prod_{i=0}^3 O_{i,j}>`
    on all columns (j = 0, ..., N2-1)
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    vals = [None] * N2
    for j in range(N2):
        # select the j-th column
        fRs = [fRss[i][j] for i in range(N1)]
        fLs = [fLss[i][j] for i in range(N1)]
        fGDs = [fGDss[i][j] for i in range(N1)]
        fGUs = [fGUss[i][j] for i in range(N1)]
        fG1s = [fG1ss[i][j] for i in range(N1)]
        fG0s = [fG0ss[i][j] for i in range(N1)]
        vals[j] = cor_4site_line_ver(
            fOs, fLs, fRs, fGUs, fGDs, fG1s, fG0s, i0
        )
    return np.array(vals)
