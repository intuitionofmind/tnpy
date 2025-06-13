"""
PEPS tensor axis order (physical axis at 0)
```
        3  0
        ↑ /
    2 → A → 4
        ↑
        1
```
"""

import fermiT as ft
import numpy as np
from fermiT import FermiT
from itertools import product


def fL_single(
    fL: FermiT, fU: FermiT, fD: FermiT, 
    fG1: FermiT, fG0: FermiT, fOa: FermiT
):
    """
    Contract with left fixed point tensor
    (with 1 x 1 PEPS)

    When G is at the j-th column, one should use
    the (j-1)-th fL (previous column) 

    Contraction order and dual convention
    ----
    ```
        |-→-- 8 --→-- U -→ -4
        |             ↑
        ↑            9/10
        |             ↑
        GL -→ 2/3 -→- G -→- -2/-3
        |             ↑
        ↑            4/5
        |             ↑
        |-←-- 1 --←-- D ←- -1
    ```
    - G is the tensor G1 x Oa x G0
    - In axis label i/j, i belongs to G1 and j belongs to G0. 
    - Index 6/7 is for the contraction of G1/G0 and Oa
    """
    return ft.fncon(
        [fL,fD,fG1,fOa,fG0,fU], [
            [1,2,3,8],[1,4,5,-1],[6,4,2,9,-2],
            [6,7],[7,5,3,10,-3],[8,9,10,-4]
        ]
    )


def fL_double(
    fL: FermiT, fU1: FermiT, fU2: FermiT, fD1: FermiT, fD2: FermiT, 
    fG11: FermiT, fG12: FermiT, fG01: FermiT, fG02: FermiT, fOab: FermiT
):
    """
    Contract with left fixed point tensor
    (with 1 x 2 PEPS)

    Contraction order and dual convention
    ----
    ```
        |-→-- 8 --→-- U1 --→- 18 ---→-- U2 -→ -4
        |             ↑                 ↑
        ↑            9/10             19/20
        |             ↑                 ↑
        GL -→ 2/3 -→- G1 -→- 12/13 --→- G2 -→- -2/-3
        |             ↑                 ↑
        ↑            4/5              14/15
        |             ↑                 ↑
        |-←-- 1 --←-- D1 --←- 11 --←--- D2 ←- -1
    ```
    - Index 6,16/7,17 is for the contraction of G11,G01/G12,G02 and Oab
    """
    return ft.fncon(
        [fL,fD1,fG11,fOab,fG01,fU1,fD2,fG12,fG02,fU2], [
            [1,2,3,8],[1,4,5,11],[6,4,2,9,12],
            [6,16,7,17],[7,5,3,10,13],[8,9,10,18],
            [11,14,15,-1],[16,14,12,19,-2],
            [17,15,13,20,-3],[18,19,20,-4]
        ]
    )


def fL_triple(
    fL: FermiT, 
    fU1: FermiT, fU2: FermiT, fU3: FermiT, 
    fD1: FermiT, fD2: FermiT, fD3: FermiT, 
    fG11: FermiT, fG12: FermiT, fG13: FermiT, 
    fG01: FermiT, fG02: FermiT, fG03: FermiT, 
    fOabc: FermiT
):
    """
    Contract with left fixed point tensor
    (with 1 x 3 PEPS)
    """
    return ft.fncon(
        [
            fL,fD1,fG11,fOabc,fG01,fU1,fD2,
            fG12,fG02,fU2,fD3,fG13,fG03,fU3
        ], [
            [1,2,3,8],[1,4,5,11],[6,4,2,9,12],
            [6,16,26,7,17,27],[7,5,3,10,13],[8,9,10,18],
            [11,14,15,21],[16,14,12,19,22],
            [17,15,13,20,23],[18,19,20,28],
            [21,24,25,-1],[26,24,22,29,-2],
            [27,25,23,30,-3],[28,29,30,-4]
        ]
    )


def fLR_val(fL: FermiT, fR: FermiT):
    """
    Contract with right fixed point tensor

    When fL is contracted to the j-th column,
    one should use the (j+1)-th fR (next column)

    Contraction order and dual convention
    ----
    ```
        |--→-- 3 --→--|
        ↑             ↓
        GL -→ 1/2 -→- GR
        ↑             ↓ 
        |--←-- 0 --←--|
    ```
    """
    return ft.tensordot(fL, fR, [[0,1,2,3],[0,1,2,3]]).val


def cor_2site_line_hor(
    fOa: FermiT, fOb: FermiT, num: int,
    fUs: list[FermiT], fDs: list[FermiT], 
    fGLs: list[FermiT], fGRs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], j0 = 0
):
    """
    Measure `<Oa_{i,j0} Ob_{i,j}>` on a fixed row (i)
    and j = j0, j0+1, ..., j0+num-1

    Parameters
    ----
    fOa, fOb: FermiT
        operators to be measured
    fUs, fDs: list[FermiT]
        AL (left canonical tensors) of the i-th row 
        of the up/down boundary MPSs
    fGLs, fGRs: list[FermiT]
        left/right fixed points of PEPS column transfer matrices
    fG1s, fG2s: list[FermiT]
        the i-th row of the PEPS
    """
    assert num >= 1
    # number of columns
    N2 = len(fUs)
    assert 0 <= j0 < N2
    Dps, Dpe = fG0s[0].DS[0], fG0s[0].DE[0]
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    vals = np.zeros((num,), dtype=complex)
    # same-site correlation
    fL0 = fL_single(fGLs[j0-1], fUs[j0], fDs[j0], fG1s[j0], fG0s[j0], Id)
    fL1ab = fL_single(
        fGLs[j0-1], fUs[j0], fDs[j0], fG1s[j0], fG0s[j0], 
        ft.tensordot(fOa, fOb, [1,0])
    )
    vals[0] = fLR_val(fL1ab, fGRs[(j0+1)%N2])/fLR_val(fL0, fGRs[(j0+1)%N2])
    # two-site correlation
    fL1a = fL_single(fGLs[j0-1], fUs[j0], fDs[j0], fG1s[j0], fG0s[j0], fOa)
    for j in range(j0+1, j0+num):
        fL0 = fL_single(fL0, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], Id)
        fL1ab = fL_single(fL1a, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], fOb)
        vals[j-j0] = fLR_val(fL1ab, fGRs[(j+1)%N2])/fLR_val(fL0, fGRs[(j+1)%N2])
        fL1a = fL_single(fL1a, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], Id)
    return vals


def cor_2site_hor(
    fOa: FermiT, fOb: FermiT, 
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]],
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], num = 30
):
    """
    Measure `<Oa_{i,j0} Ob_{i,j}>`
    where j = j0, j0+1, ..., j0+num-1
    starting from all sites (i,j0) in the unit cell
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    valss = [[None] * N2 for i in range(N1)]
    for i, j0 in product(range(N1), range(N2)):
        # select the i-th row
        valss[i][j0] = cor_2site_line_hor(
            fOa, fOb, num, fUss[i], fDss[i], 
            fGLss[i], fGRss[i], fG1ss[i], fG0ss[i], j0=j0
        )
    return np.array(valss)


def cor_2site_line_ver(
    fOa: FermiT, fOb: FermiT, num: int,
    fLs: list[FermiT], fRs: list[FermiT], 
    fGUs: list[FermiT], fGDs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], i0 = 0
):
    """
    Measure `<Oa_{i0,j} Ob_{i,j}>` on a fixed column (j)
    and i = i0, i0+1, ..., i0+num-1

    By mirror the network by the diagonal line y = x, 
    we can use the horizontal code with replacements
    ```
        U, D, GL, GR -> R, L, GD, GU
    ```
    R,L,GD,GU already have correct axes order.
    However, G1/G0 should be further transposed

    Parameters
    ----
    fOa, fOb: FermiT
        operators to be measured
    fUs, fDs: list[FermiT]
        AL (left canonical tensors) of the j-th column 
        of the up/down boundary MPSs
    fGLs, fGRs: list[FermiT]
        up/down fixed points of PEPS row transfer matrices
    fG1s, fG2s: list[FermiT]
        the j-th column of the PEPS
    """
    return cor_2site_line_hor(
        fOa, fOb, num, fRs, fLs, fGDs, fGUs, 
        [fG1.transpose(0,2,1,4,3) for fG1 in fG1s], 
        [fG0.transpose(0,2,1,4,3) for fG0 in fG0s], 
        j0 = i0
    )


def cor_2site_ver(
    fOa: FermiT, fOb: FermiT, 
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]],
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], num = 30
):
    """
    Measure `<Oa_{i0,j} Ob_{i,j}>`
    where i = i0, i0+1, ..., i0+num-1
    starting from all sites (i0,j) in the unit cell
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    valss = [[None] * N2 for i in range(N1)]
    for i0, j in product(range(N1), range(N2)):
        # select the j-th column
        fRs = [fRss[i][j] for i in range(N1)]
        fLs = [fLss[i][j] for i in range(N1)]
        fGDs = [fGDss[i][j] for i in range(N1)]
        fGUs = [fGUss[i][j] for i in range(N1)]
        fG1s = [fG1ss[i][j] for i in range(N1)]
        fG0s = [fG0ss[i][j] for i in range(N1)]
        valss[i0][j] = cor_2site_line_ver(
            fOa, fOb, num, fLs, fRs, 
            fGUs, fGDs, fG1s, fG0s, i0=i0
        )
    return np.array(valss)


def cor_1x1site_line_hor(
    fOa: FermiT, fOb: FermiT,
    fUs: list[FermiT], fDs: list[FermiT], 
    fGLs: list[FermiT], fGRs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT]
):
    """
    Measure `<Oa_{i,0}> <Ob_{i,j}>` 
    on a fixed row (i) for j = 0, 1, ..., N2-1
    """
    N2 = len(fUs)
    Dps, Dpe = fG0s[0].DS[0], fG0s[0].DE[0]
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    fL0, fL1 = fGLs[-1], fGLs[-1]
    fL0 = fL_single(fL0, fUs[0], fDs[0], fG1s[0], fG0s[0], Id)
    fL1a = fL_single(fL1, fUs[0], fDs[0], fG1s[0], fG0s[0], fOa)
    # <Oa_{i,0}>
    vala = fLR_val(fL1a, fGRs[1%N2])/fLR_val(fL0, fGRs[1%N2])
    # <Ob_{i,j}> for j = 0, ..., N2-1
    valbs = np.zeros((N2,), dtype=complex)
    for j in range(N2):
        fL1b = fL_single(fL1, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], fOb)
        valbs[j] = fLR_val(fL1b, fGRs[(j+1)%N2])/fLR_val(fL0, fGRs[(j+1)%N2])
        fL1 = fL_single(fL1, fUs[j%N2], fDs[j%N2], fG1s[j%N2], fG0s[j%N2], Id)
        fL0 = fL_single(fL0, fUs[(j+1)%N2], fDs[(j+1)%N2], fG1s[(j+1)%N2], fG0s[(j+1)%N2], Id)
    return vala, valbs


def spin_correlation_line_hor(
    fUs: list[FermiT], fDs: list[FermiT], 
    fGLs: list[FermiT], fGRs: list[FermiT], 
    fG1s: list[FermiT], fG0s: list[FermiT], 
    num: int, direction="z", shift = False
):
    """
    Measure spin correlation on a fixed row (i)
    ```
    <Sa_{i,j} Sa_{i,j+num}> - <Sa_{i,j}> <Sa_{i,j+num}>
    (a = x,y,z)
    ```
    """
    from phys_models.onesiteop import get_tJconv, makeops_tJft
    assert direction in ("x", "y", "z")
    N2 = len(fUs)
    Dps, Dpe = fG0s[0].DS[0], fG0s[0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    fOa, fOb = tuple(map(makeops_tJft, [f"S{direction}"]*2, [tJ_conv]*2))
    vals = cor_2site_line_hor(
        fOa, fOb, num, fUs, fDs, fGLs, fGRs, fG1s, fG0s
    )
    if shift:
        vala, valbs = cor_1x1site_line_hor(
            fOa, fOb, fUs, fDs, fGLs, fGRs, fG1s, fG0s
        )
        print(vala, valbs)
        for j in range(num): 
            vals[j] -= vala * valbs[j%N2]
    return vals


def spin_correlation_hor(
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]],
    fG1ss: list[list[FermiT]], fG0ss: list[list[FermiT]], 
    num = 10, direction = "z", shift = False
):
    """
    Spin correlation <S(i,j) S(i+num,j)> in horizontal direction
    """
    N1, N2 = len(fG1ss), len(fG1ss[0])
    fUss, fDss, fLss, fRss = fXss
    fGUss, fGDss, fGLss, fGRss = fGXss
    valss = [None] * N1
    for i in range(N1):
        valss[i] = spin_correlation_line_hor(
            fUss[i], fDss[i], fGLss[i], fGRss[i], 
            fG1ss[i], fG0ss[i], num, direction, shift
        )
    return np.array(valss)
