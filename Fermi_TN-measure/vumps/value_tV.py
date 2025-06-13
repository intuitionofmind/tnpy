"""
1-site and 1st neighbor measurement on spinless t-V model PEPS
"""

import os
import numpy as np
from glob import glob
from natsort import natsorted
from utils import dir2param
from fermiT import FermiT
from fermiT.conversion import gt2ft
from phys_models.init_gate import init_gate
from phys_models.onesiteop import makeops_tVft
from vumps.files_par import load_rhoss
from plottools.imshow import imshow_config
from vumps.value import cal_rho1ss, cal_rho2ss, ground_energy


def dens(rhosss: list[list[list[FermiT]]]):
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Num, Id = list(map(makeops_tVft, ["Num", "Id"]))
    valvss = cal_rho1ss(rho1vss, Num, Id)
    valhss = cal_rho1ss(rho1hss, Num, Id)
    return valvss, valhss


def hopping(rhosss: list[list[list[FermiT]]]):
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Cp, Cm, Id = list(map(makeops_tVft, ["Cp", "Cm", "Id"]))
    valvss = cal_rho2ss(rho2vss, Cp, Cm, Id)
    valhss = cal_rho2ss(rho2hss, Cp, Cm, Id)
    return valvss, valhss


def scorder(rhosss: list[list[list[FermiT]]]):
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Cm, Id = list(map(makeops_tVft, ["Cm", "Id"]))
    valvss = cal_rho2ss(rho2vss, Cm, Cm, Id)
    valhss = cal_rho2ss(rho2hss, Cm, Cm, Id)
    return valvss, valhss


def load_vumps(
    rootdir: str, chi: int, t=1.0, V=-0.5, sampleid=0,
):
    """
    load t-J model one-site or 1st neighbor rhoss
    and calculate quantities by VUMPS

    Contains
    ----
    mu: chemical potential
    
    hole: average doping
    
    eng: average energy per site
    
    mag: magnetization (x,y,z) on each site
    
    mag_mean: average staggered magnetization
    
    singlet: singlet pairing on each bond
        <c_{i,up} c_{j,down} - c_{i,down} c_{j,up}> / sqrt(2)
    sc_mean: average magnitude of singlet pairing

    hopping: hopping on each bond
        <c+_{i,up} c_{j,up} - c+_{i,down} c_{j,down}>
    spincor: spin correlation on each bond
        <Sx_i Sx_j + Sy_i Sy_j + Sy_i Sy_j>
    """
    N1, N2 = 2, 2
    assert t == 1.0
    search = f"V-{V:.2f}_mu-*"
    meas = {
        "mu": [], "hole": [], "eng": [], 
        "sinHs": [], "sinVs": [], 
        "hopHs": [], "hopVs": [], 
    }
    assert rootdir[-1] == os.sep
    rhoss_dirs: list[str] = natsorted(glob(
        f"{rootdir}{search}/conv1-sample{sampleid}/rhoss-{chi}/"
    ))
    assert len(rhoss_dirs) > 0, "No measurement files are found"
    # determine tJ convention
    ham = gt2ft(init_gate({
        "model": "tV", "t": t, "V": V, "mu": 0.0, "nbond": 4
    }, expo=False))
    for rhoss_dir in rhoss_dirs:
        mu = dir2param(
            rhoss_dir.split(rootdir, 1)[-1].split(os.sep, 1)[0]
            + os.sep
        )["mu"] 
        meas["mu"].append(mu)
        # load rhoss
        rhosss = load_rhoss(N1, N2, rhoss_dir)
        # doping
        hVs, hHs = dens(rhosss)
        doping = np.mean(np.concatenate((hVs, hHs)))
        meas["hole"].append(doping)
        # energy per site
        eHs, eVs = ground_energy(rhosss, ham)
        eng = (np.sum(eVs) + np.sum(eHs))/N1/N2
        meas["eng"].append(eng)
        # singlet pairing
        sinHs, sinVs = scorder(rhosss)
        meas["sinHs"].append(sinHs)
        meas["sinVs"].append(sinVs)
        # hopping
        hopHs, hopVs = hopping(rhosss)
        meas["hopHs"].append(hopHs)
        meas["hopVs"].append(hopVs)
    meas = dict(
        (key, np.array(val)) 
        for (key, val) in meas.items()
    )
    return meas


def show_config(
    N1: int, N2: int, param: dict[str], rhoss_dir: str,
):
    nearh = gt2ft(init_gate(param, expo=False))
    rhosss = load_rhoss(N1, N2, rhoss_dir)
    hVs, hHs = dens(rhosss)
    eVs, eHs = ground_energy(rhosss, nearh)
    sinVs, sinHs = scorder(rhosss)
    e0 = (np.sum(eVs) + np.sum(eHs))/N1/N2
    imshow_config(
        N1, N2, e0, hVs, None, sinVs, sinHs,
        rhoss_dir + "config.png"
    )
