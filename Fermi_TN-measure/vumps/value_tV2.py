"""
2nd neighbor measurement on spinless t-V model PEPS
"""

import os
import numpy as np
from glob import glob
from math import sqrt
from natsort import natsorted
import fermiT as ft
from fermiT import FermiT
from utils import dir2param
from phys_models.onesiteop import makeops_tVft
from vumps.files_par import load_rhodss
from vumps.value import cal_rhodss


def scorder(rhodss: list[list[list[FermiT]]]):
    """
    Use 2nd-neighbor rho to calculate 
    pairing = `<c_i c_j>`
    """
    rhod1ss, rhod2ss = rhodss
    Id, Cm = tuple(map(makeops_tVft, ["Id", "Cm"]))
    v0 = cal_rhodss(rhod1ss, rhod2ss, [Id]*2)
    tmp = cal_rhodss(rhod1ss, rhod2ss, [Cm,Cm]) / v0
    return tmp


def hopping(rhodss: list[list[list[FermiT]]]):
    """
    Use 2nd-neighbor rho to calculate 
    hopping = `<c+_i c_j>`
    """
    rhod1ss, rhod2ss = rhodss
    Id, Cp, Cm = tuple(map(makeops_tVft, ["Id", "Cp", "Cm"]))
    v0 = cal_rhodss(rhod1ss, rhod2ss, [Id]*2)
    tmp = cal_rhodss(rhod1ss, rhod2ss, [Cp,Cm]) / v0
    return tmp


def load_vumps_diag(
    rootdir: str, chi: int, t=1.0, V=-0.5, sampleid=0, 
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
    """
    N1, N2 = 2, 2
    if t != 1.0: raise NotImplementedError
    search = f"V-{V:.2f}_mu-*"
    rhoss_dirs: list[str] = natsorted(glob(
        f"{rootdir}{search}/conv1-sample{sampleid}/rhodss-{chi}/"
    ))
    assert len(rhoss_dirs) > 0, "No measurement files are found"
    meas = {
        "mu": [], 
        "sinD1s": [], "sinD2s": [], 
        "hopD1s": [], "hopD2s": [], 
    }
    for rhoss_dir in rhoss_dirs:
        mu = dir2param(
            rhoss_dir.split(rootdir, 1)[-1].split(os.sep, 1)[0]
            + os.sep
        )["mu"] 
        meas["mu"].append(mu)
        # load rhoss
        rhodss = load_rhodss(N1, N2, rhoss_dir)
        sinD1s, sinD2s = scorder(rhodss)
        hopD1s, hopD2s = hopping(rhodss)
        meas["sinD1s"].append(sinD1s)
        meas["sinD2s"].append(sinD2s)
        meas["hopD1s"].append(hopD1s)
        meas["hopD2s"].append(hopD2s)
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
