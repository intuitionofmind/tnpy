"""
1-site and 1st neighbor measurement on t-J model PEPS
"""

import os
import numpy as np
from glob import glob
from math import sqrt
from natsort import natsorted
import fermiT as ft
from fermiT import FermiT
from utils import dir2param, dict_loadtxt
from fermiT.conversion import gt2ft
from phys_models.init_gate import init_gate
from phys_models.onesiteop import get_tJconv, makeops_tJft
from vumps.files_par import load_rhoss
from plottools.imshow import imshow_config
from vumps.value import cal_rho1ss, cal_rho2ss_gate, ground_energy

# ------ 1-site quantities ------

def hole(rhosss: list[list[list[FermiT]]]):
    """
    Use 1-site density operator to calculate doping
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    # get operators
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Nu, Nd = tuple(map(makeops_tJft, ["Nu", "Nd"], [tJ_conv]*2))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    Oh = Id - Nu - Nd
    # calculate values
    valvss = cal_rho1ss(rho1vss, Oh, Id)
    valhss = cal_rho1ss(rho1hss, Oh, Id)
    return [valvss, valhss]


def magnetization(rhosss: list[list[list[FermiT]]]):
    """
    Use 1-site density operator to calculate magnetization
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    N1, N2 = len(rho1vss), len(rho1vss[0])
    # get operators
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))

    Sx, Sy, Sz = list(map(makeops_tJft, ["Sx", "Sy", "Sz"], [tJ_conv]*3))
    Id = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))

    # calculate values
    magvss = np.zeros((3,N1,N2), dtype=complex)
    magvss[0,:,:] = cal_rho1ss(rho1vss, Sx, Id)
    magvss[1,:,:] = cal_rho1ss(rho1vss, Sy, Id)
    magvss[2,:,:] = cal_rho1ss(rho1vss, Sz, Id)
    maghss = np.zeros((3,N1,N2), dtype=complex)
    maghss[0,:,:] = cal_rho1ss(rho1hss, Sx, Id)
    maghss[1,:,:] = cal_rho1ss(rho1hss, Sy, Id)
    maghss[2,:,:] = cal_rho1ss(rho1hss, Sz, Id)
    return [magvss, maghss]


# ------ 2-site quantities (nearest neighbor) ------


def singlet(rhosss: list[list[list[FermiT]]]):
    """
    Use 2-site density operator to calculate 
    singlet pairing on nearest neighbor bonds
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    # get operators
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    # singlet pair operator
    Cmu, Cmd = tuple(map(makeops_tJft, ["Cmu", "Cmd"], [tJ_conv]*2))
    Osin = sqrt(0.5) * (
        ft.outer(Cmu, Cmd) - ft.outer(Cmd, Cmu)
    ).transpose(0,2,1,3)
    Id   = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    # calculate values
    valvss = cal_rho2ss_gate(rho2vss, Osin, Id)
    valhss = cal_rho2ss_gate(rho2hss, Osin, Id)
    return [valvss, valhss]


def hopping(rhosss: list[list[list[FermiT]]]):
    """
    Use 2-site density operator to calculate 
    hopping on nearest neighbor bonds
    ```
        c+_{i,up} c_{j,up} + c+_{i,dn} c_{j,dn}
    ```
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    # get operators
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    Cpu, Cpd, Cmu, Cmd = tuple(map(
        makeops_tJft, ["Cpu", "Cpd", "Cmu", "Cmd"], [tJ_conv]*4
    ))
    Ohop = (
        ft.outer(Cpu, Cmu) + ft.outer(Cpd, Cmd)
    ).transpose(0,2,1,3)
    Id   = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    # calculate values
    valvss = cal_rho2ss_gate(rho2vss, Ohop, Id)
    valhss = cal_rho2ss_gate(rho2hss, Ohop, Id)
    return [valvss, valhss]


def spincor(rhosss: list[list[list[FermiT]]]):
    """
    Use 2-site density operator to calculate 
    spin correlation S_i S_j on nearest neighbor bonds
    ```
        S_i S_j = (1/2) (Sp_i Sm_j + h.c.) + Sz_i Sz_j
    ```
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    # get operators
    Dps, Dpe = rho1vss[0][0].DS[0], rho1vss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    # spin correlation operator
    Sp, Sm, Sz = tuple(map(
        makeops_tJft, ["Sp", "Sm", "Sz"], [tJ_conv]*3
    ))
    Ospc = (
        (ft.outer(Sp, Sm) + ft.outer(Sm, Sp)) * 0.5
        + ft.outer(Sz, Sz)
    ).transpose(0,2,1,3)
    Id   = ft.eye((Dps,Dps), (Dpe,Dpe), dual=(0,1))
    # calculate values
    valvss = cal_rho2ss_gate(rho2vss, Ospc, Id)
    valhss = cal_rho2ss_gate(rho2hss, Ospc, Id)
    return [valvss, valhss]


# ------ Collection of all results ------

def load_vumps(
    rootdir: str, chi: int, search: str, 
    conv=1, sampleid=0, param_fix: dict[str]={}
):
    """
    load t-J model one-site or 1st neighbor rhoss
    and calculate quantities by VUMPS

    `param_fix` allows manual input of parameters

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
    assert rootdir[-1] == os.sep
    rhoss_dirs: list[str] = natsorted(glob(
        f"{rootdir}{search}/conv{conv}-sample{sampleid}/rhoss-{chi}/"
    ))
    assert len(rhoss_dirs) > 0, "No measurement files are found"
    # determine tJ convention
    rho = ft.load(rhoss_dirs[0] + "0.npz")
    Dps, Dpe = rho.DS[0], rho.DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    # create NEAREST NEIGHBOR gates
    gate_params = {
        "model": "tJ", "mu": 0.0, 
        "tJ_convention": tJ_conv, "nbond": 4,
        "t": 3.0, "J": 1.0, 
    }
    tmp_params = dir2param(search + "/")
    gate_params.update({
        k: tmp_params[k] for k in gate_params
        if k in tmp_params
    })
    gate_params.update(param_fix)
    ham = gt2ft(init_gate(gate_params, expo=False))
    meas = {
        "mu": [], "hole": [], "eng": [], 
        "mHs": [], "mVs": [], "mag_mean": [], 
        "sinHs": [], "sinVs": [], "sc_mean": [],
        "hopHs": [], "hopVs": [], 
        "corHs": [], "corVs": []
    }
    for rhoss_dir in rhoss_dirs:
        # determine unit cell size
        try:
            tps_dir = rhoss_dir.rsplit("/", 2)[0] + "/"
            info = dict_loadtxt(tps_dir + "tpsinfo.txt")
            N1, N2 = info["N1"], info["N2"]
        except KeyError:
            N1, N2 = 2, 2
        except FileNotFoundError:
            N1, N2 = 2, 2
        mu = dir2param(
            rhoss_dir.split(rootdir, 1)[-1].split(os.sep, 1)[0]
            + os.sep
        )["mu"] 
        meas["mu"].append(mu)
        # load rhoss
        rhosss = load_rhoss(N1, N2, rhoss_dir)
        # doping
        hVs, hHs = hole(rhosss)
        doping = np.mean(np.concatenate((hVs, hHs)))
        meas["hole"].append(doping)
        # energy per site
        eHs, eVs = ground_energy(rhosss, ham)
        eng = (np.sum(eVs) + np.sum(eHs))/N1/N2
        meas["eng"].append(eng)
        # magnetization
        mHs, mVs = magnetization(rhosss)
        meas["mHs"].append(mHs)
        meas["mVs"].append(mVs)
        meas["mag_mean"].append(np.mean(np.concatenate((
            np.linalg.norm(mVs, axis=0), 
            np.linalg.norm(mHs, axis=0)
        ))))
        # singlet pairing
        sinVs, sinHs = singlet(rhosss)
        meas["sinHs"].append(sinHs)
        meas["sinVs"].append(sinVs)
        meas["sc_mean"].append(
            np.mean(np.concatenate((np.abs(sinHs), np.abs(sinVs))))
        )
        # hopping
        hopVs, hopHs = hopping(rhosss)
        meas["hopHs"].append(hopHs)
        meas["hopVs"].append(hopVs)
        # spin correlation
        corVs, corHs = spincor(rhosss)
        meas["corHs"].append(corHs)
        meas["corVs"].append(corVs)
    # convert to numpy array
    meas = dict(
        (key, np.array(val)) 
        for (key, val) in meas.items()
    )
    # and sort mu in ascending order
    order = np.argsort(meas["mu"])
    for key, val in meas.items():
        meas[key] = val[order]
    return meas

def show_config(
    N1: int, N2: int, param: dict[str], rhoss_dir: str,
):
    assert "tJ" in param["model"]
    rhosss = load_rhoss(N1, N2, rhoss_dir)
    Dps, Dpe = rhosss[0][0][0].DS[0], rhosss[0][0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    param["tJ_convention"] = tJ_conv
    param["mu"] = 0.0
    param["model"] = "tJ"
    try: param.pop("t2")
    except: pass
    try: param.pop("J2")
    except: pass
    nearh = gt2ft(init_gate(param, expo=False))
    hVs, hHs = hole(rhosss)
    mVs, mHs = magnetization(rhosss)
    eVs, eHs = ground_energy(rhosss, nearh)
    sinVs, sinHs = singlet(rhosss)
    e0 = (np.sum(eVs) + np.sum(eHs))/N1/N2
    # to imshow the result
    imshow_config(
        N1, N2, e0, hVs, mVs, sinVs, sinHs,
        rhoss_dir + "config.png"
    )
