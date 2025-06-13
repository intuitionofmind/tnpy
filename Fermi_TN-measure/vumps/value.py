"""
Calculate expectation values
with pre-calculated environment tensors
"""

import numpy as np
import fermiT as ft
from fermiT import FermiT
from itertools import product


def cal_rho1ss(
    rhoss: list[list[FermiT]], fOa: FermiT, Id: FermiT
):
    """
    Measure `<fOa>` using 1-site density operator
    """
    N1, N2 = len(rhoss), len(rhoss[0])
    valss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        val0 = ft.tensordot(rhoss[i][j], fOa, [[0,1],[0,1]]).val
        norm = ft.tensordot(rhoss[i][j], Id,  [[0,1],[0,1]]).val
        valss[i,j] = val0/norm
    return valss


def cal_rho2ss_gate(
    rhoss: list[list[FermiT]], gate: FermiT, Id: FermiT
):
    """
    Measure gate `<gate>` (normalized) 
    on nearest neighbor bond
    using 2-site density operator
    """
    N1, N2 = len(rhoss), len(rhoss[0])
    valss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        val0 = ft.tensordot(rhoss[i][j], gate, [[0,1,2,3],[0,1,2,3]]).val
        norm = ft.fncon([rhoss[i][j],Id,Id], [[1,2,3,4],[1,3],[2,4]]).val
        valss[i,j] = val0/norm
    return valss


def cal_rho2ss(
    rhoss: list[list[FermiT]], 
    fOa: FermiT, fOb: FermiT, Id: FermiT
):
    """
    Measure `<fOa fOb>` (normalized)
    on nearest neighbor bond
    using 2-site density operator
    """
    N1, N2 = len(rhoss), len(rhoss[0])
    valss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        val0 = ft.fncon([rhoss[i][j],fOa,fOb], [[1,2,3,4],[1,3],[2,4]]).val
        norm = ft.fncon([rhoss[i][j],Id,Id], [[1,2,3,4],[1,3],[2,4]]).val
        valss[i,j] = val0/norm
    return valss


def ground_energy(rhosss: list[list[list[FermiT]]], nearh: FermiT):
    """Measure gate on nearest neighbor bonds"""
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    # get identity operator
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    Id   = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    # calculate values
    valvss = cal_rho2ss_gate(rho2vss, nearh, Id)
    valhss = cal_rho2ss_gate(rho2hss, nearh, Id)
    return [valvss, valhss]


def cal_rhodss(
    rhod1ss: list[list[FermiT]], 
    rhod2ss: list[list[FermiT]], ops: list[FermiT]
):
    """
    Use diagonal density matrix on each unit cell
    to measure `<psi|O1 O2|psi>` (psi is not normalized)

    Operator order in each rho
    ----
    ```
            |     |
        --- 1 --- 3 ---
            |     |
        --- 0 --- 2 ---
            |     |
    ```
    - `rhod1ss` are 0 -> 3
    - `rhod2ss` are 1 -> 2
    """
    assert len(ops) == 2
    # number of rows and columns in PEPS unit cell
    N1, N2 = len(rhod1ss), len(rhod1ss[0])
    val1ss = np.zeros((N1, N2), dtype=complex)
    val2ss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        val1ss[i,j] = ft.fncon([rhod1ss[i][j]]+ops, [[1,2,3,4],[1,3],[2,4]]).val
        val2ss[i,j] = ft.fncon([rhod2ss[i][j]]+ops, [[1,2,3,4],[1,3],[2,4]]).val
    return np.array([val1ss, val2ss])


def cal_rhodss_gate(
    rhod1ss: list[list[FermiT]], 
    rhod2ss: list[list[FermiT]], gate: FermiT
):
    """
    Use diagonal density matrix on each unit cell
    to measure `<psi|fOab|psi>` (psi is not normalized)
    """
    # number of rows and columns in PEPS unit cell
    N1, N2 = len(rhod1ss), len(rhod1ss[0])
    val1ss = np.zeros((N1, N2), dtype=complex)
    val2ss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        val1ss[i,j] = ft.fncon([rhod1ss[i][j],gate], [[1,2,3,4],[1,2,3,4]]).val
        val2ss[i,j] = ft.fncon([rhod2ss[i][j],gate], [[1,2,3,4],[1,2,3,4]]).val
    return np.array([val1ss, val2ss])


def ground_energy2(
    rhodss: list[list[list[FermiT]]], gate: FermiT
):
    """Measure gate on 2nd neighbor bonds"""
    rhod1ss, rhod2ss = rhodss
    # get identity gate
    Dps, Dpe = rhod1ss[0][0].DS[0], rhod1ss[0][0].DE[0]
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    gate0 = ft.outer(Id, Id).transpose(0,2,1,3)
    val0 = cal_rhodss_gate(rhod1ss, rhod2ss, gate)
    norm = cal_rhodss_gate(rhod1ss, rhod2ss, gate0)
    return val0 / norm


def cal_rho4ss(
    rho4ss: list[list[FermiT]], ops: list[FermiT]
):
    """
    Use 2x2 density matrix on each unit cell
    to measure `<psi|O1 O2 O3 O4|psi>` (psi is not normalized)

    Operator order in each rho
    ----
    ```
            |     |
        --- 1 --- 3 ---
            |     |
        --- 0 --- 2 ---
            |     |
    ```

    Parameters
    ----
    rho4ss: list[list[FermiT]]
        density matrices on 2x2 unit cells
        with lower-left corner at site [i][j] 

    ops: list[FermiT]
        operators on sites 0 - 3 to be measured
    """
    assert len(ops) == 4
    # number of rows and columns in PEPS unit cell
    N1, N2 = len(rho4ss), len(rho4ss[0])
    valss = np.zeros((N1, N2), dtype=complex)
    for i, j in product(range(N1), range(N2)):
        rho = rho4ss[i][j]
        valss[i,j] = ft.fncon([rho] + ops, [[1,2,3,4,5,6,7,8],[1,2],[3,4],[5,6],[7,8]]).val
    return valss
