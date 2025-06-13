"""
2nd neighbor measurement on t-J model PEPS
"""

import os
import numpy as np
from glob import glob
from math import sqrt
from natsort import natsorted
import fermiT as ft
from fermiT import FermiT
from utils import dir2param
from fermiT.conversion import gt2ft
from phys_models.init_gate import init_gate
from phys_models.onesiteop import get_tJconv, makeops_tJft
from vumps.files_par import load_rhodss
from vumps.value import cal_rhodss, ground_energy2


def singlet(rhodss: list[list[list[FermiT]]]):
    """
    Use 2nd-neighbor rho to calculate singlet pairing
    """
    rhod1ss, rhod2ss = rhodss
    Dps, Dpe = rhod1ss[0][0].DS[0], rhod1ss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    Cmu, Cmd = tuple(map(makeops_tJft, ["Cmu", "Cmd"], [tJ_conv]*2))
    v0 = cal_rhodss(rhod1ss, rhod2ss, [Id]*2)
    tmp = (
        cal_rhodss(rhod1ss, rhod2ss, [Cmu,Cmd])
        - cal_rhodss(rhod1ss, rhod2ss, [Cmd,Cmu])
    ) / v0 / sqrt(2)
    return tmp


def hopping(rhodss: list[list[list[FermiT]]]):
    """
    Use 2nd-neighbor rho to calculate singlet pairing
    """
    rhod1ss, rhod2ss = rhodss
    Dps, Dpe = rhod1ss[0][0].DS[0], rhod1ss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    Cpu, Cpd, Cmu, Cmd = tuple(map(makeops_tJft, [
        "Cpu", "Cpd", "Cmu", "Cmd"
    ], [tJ_conv]*4))
    v0 = cal_rhodss(rhod1ss, rhod2ss, [Id]*2)
    tmp = (
        cal_rhodss(rhod1ss, rhod2ss, [Cpu,Cmu])
        + cal_rhodss(rhod1ss, rhod2ss, [Cpd,Cmd])
    ) / v0
    return tmp


def spincor(rhodss: list[list[list[FermiT]]]):
    """
    Use 2nd-neighbor rho to calculate singlet pairing
    """
    rhod1ss, rhod2ss = rhodss
    Dps, Dpe = rhod1ss[0][0].DS[0], rhod1ss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    Sp, Sm, Sz = tuple(map(makeops_tJft, ["Sp", "Sm", "Sz"], [tJ_conv]*3))
    v0 = cal_rhodss(rhod1ss, rhod2ss, [Id]*2)
    tmp = (
        cal_rhodss(rhod1ss, rhod2ss, [Sp,Sm]) / 2
        + cal_rhodss(rhod1ss, rhod2ss, [Sm,Sp]) / 2
        + cal_rhodss(rhod1ss, rhod2ss, [Sz,Sz])
    ) / v0
    return tmp


def load_vumps_diag(
    rootdir: str, chi: int, search: str, conv=1, sampleid=0, meas_eng=False
):
    """
    load t-J model 2nd neighbor rhoss
    and calculate quantities by VUMPS

    Contains
    ----
    mu: chemical potential
    singlet: singlet pairing on each 2nd neighbor bond
    hopping: hopping on each 2nd neighbor bond
    spincor: spin correlation on each 2nd neighbor bond
    eng: energy per site on each 2nd neighbor bond
    """
    N1, N2 = 2, 2
    rhoss_dirs: list[str] = natsorted(glob(
        f"{rootdir}{search}/conv{conv}-sample{sampleid}/rhodss-{chi}/"
    ))
    assert len(rhoss_dirs) > 0, "No measurement files are found"
    meas = {
        "mu": [], 
        "sinD1s": [], "sinD2s": [], 
        "hopD1s": [], "hopD2s": [], 
        "corD1s": [], "corD2s": []
    }
    if meas_eng: 
        meas["eng2"] = []
        # determine tJ convention
        rhodss = load_rhodss(N1, N2, rhoss_dirs[0])
        Dps, Dpe = rhodss[0][0][0].DS[0], rhodss[0][0][0].DE[0]
        tJ_conv = get_tJconv((Dpe, Dps-Dpe))
        # create NEAREST NEIGHBOR gates
        gate_params = {
            "model": "tJ", "mu": 0.0, 
            "tJ_convention": tJ_conv, "nbond": 4,
            "t": 0.0, "J": 0.5, 
        }
        tmp_params = dir2param(search + "/")
        for key in ("t", "J"):
            try:
                gate_params[key] = tmp_params[key+"2"]
            except KeyError: pass
        ham = gt2ft(init_gate(gate_params, expo=False))
    for rhoss_dir in rhoss_dirs:
        mu = dir2param(
            rhoss_dir.split(rootdir, 1)[-1].split(os.sep, 1)[0]
            + os.sep
        )["mu"] 
        meas["mu"].append(mu)
        # load rhoss
        rhodss = load_rhodss(N1, N2, rhoss_dir)
        sinD1s, sinD2s = singlet(rhodss)
        hopD1s, hopD2s = hopping(rhodss)
        corD1s, corD2s = spincor(rhodss)
        meas["sinD1s"].append(sinD1s)
        meas["sinD2s"].append(sinD2s)
        meas["hopD1s"].append(hopD1s)
        meas["hopD2s"].append(hopD2s)
        meas["corD1s"].append(corD1s)
        meas["corD2s"].append(corD2s)
        # energy per site
        if meas_eng:
            e1s, e2s = ground_energy2(rhodss, ham)
            eng = (np.mean(e1s) + np.mean(e2s))
            meas["eng2"].append(eng)
    # convert to numpy array
    meas = dict(
        (key, np.array(val)) 
        for (key, val) in meas.items()
    )
    # sort mu in ascending order
    order = np.argsort(meas["mu"])
    for key, val in meas.items():
        meas[key] = val[order]
    return meas
