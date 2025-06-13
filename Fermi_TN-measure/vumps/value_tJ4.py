import fermiT as ft
import vumps.value as fval
from fermiT import FermiT
from phys_models.onesiteop import get_tJconv, makeops_tJft
from math import sqrt


def measure_nb2(rhodss: list[list[FermiT]]):
    """
    Measure 2nd neighbor (diagonal) bonds
    using 2x2 density operators
    """
    Dps, Dpe = rhodss[0][0].DS[0], rhodss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    Cpu, Cpd, Cmu, Cmd, Sp, Sm, Sz \
        = tuple(map(makeops_tJft, [
            "Cpu", "Cpd", "Cmu", "Cmd", "Sp", "Sm", "Sz"
        ], [tJ_conv]*7))
    v0 = fval.cal_rho4ss(rhodss, [Id]*4)
    results = {
        # 45-degree diagonal y = x
        "sinD1": (
            fval.cal_rho4ss(rhodss, [Cmu,Id,Id,Cmd])
            - fval.cal_rho4ss(rhodss, [Cmd,Id,Id,Cmu])
        ) / sqrt(2) / v0,
        "hopD1": (
            fval.cal_rho4ss(rhodss, [Cpu,Id,Id,Cmu])
            + fval.cal_rho4ss(rhodss, [Cpd,Id,Id,Cmd])
        ) / v0,
        "corD1": (
            fval.cal_rho4ss(rhodss, [Sp,Id,Id,Sm]) / 2
            + fval.cal_rho4ss(rhodss, [Sm,Id,Id,Sp]) / 2
            + fval.cal_rho4ss(rhodss, [Sz,Id,Id,Sz])
        ) / v0,

        # -45-degree diagonal y = -x
        "sinD2": (
            fval.cal_rho4ss(rhodss, [Id,Cmu,Cmd,Id])
            - fval.cal_rho4ss(rhodss, [Id,Cmd,Cmu,Id])
        ) / sqrt(2) / v0,
        "hopD2": (
            fval.cal_rho4ss(rhodss, [Id,Cpu,Cmu,Id])
            + fval.cal_rho4ss(rhodss, [Id,Cpd,Cmd,Id])
        ) / v0,
        "corD2": (
            fval.cal_rho4ss(rhodss, [Id,Sp,Sm,Id]) / 2
            + fval.cal_rho4ss(rhodss, [Id,Sm,Sp,Id]) / 2
            + fval.cal_rho4ss(rhodss, [Id,Sz,Sz,Id])
        ) / v0
    }
    return results
