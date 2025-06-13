"""
Combine measured terms into 
total energy, order parameter, etc.
"""

import numpy as np
from utils import meas_process
from math import sqrt

def cal_dopings(measures: dict[str, complex|float], N1: int, N2: int):
    """
    Extract doping on each site (A, B)
    """
    dope = np.array([
        [measures[f"Nh_t{i}{j}"] for j in range(N2)]
        for i in range(N1)
    ])
    return dope


def cal_mags(measures: dict[str, complex|float], N1: int, N2: int):
    """
    Extract magnetization 
    `<Sx>, <Sy>, <Sz>` on each site

    Returns
    ----
    mag: np.ndarray
        Row 1/2/3 is the Sx/Sy/Sz on all sites
    """
    mags = []
    for a in "xyz":
        mags.append([
            [measures[f"S{a}_t{i}{j}"] for j in range(N2)]
            for i in range(N1)
        ])
    return np.array(mags)


def cal_spincor(
    measures: dict[str, complex|float], 
    N1: int, N2: int, nb2=False
) -> dict[str, complex|float]:
    """
    Extract spin correlation `<S_i S_j>`
    on all 1st and 2nd neighbor bonds
    """
    coeffs = [0.5, 0.5, 1.0]
    terms = ["SpSm", "SmSp", "SzSz"]
    directions = "xydD" if nb2 else "xy"
    spincor = [[[meas_process(
        coeffs, [f"{term}_{d}{i}{j}" for term in terms], measures
    ) for j in range(N2) ] for i in range(N1)] for d in directions]
    return np.array(spincor)


def cal_hopping(
    measures: dict[str, complex|float], 
    N1: int, N2: int, nb2=False
) -> dict[str, complex|float]:
    """
    Extract hopping term
    `<c+_{i,up} c_{j,up} + c+_{i,down} c_{j,down} + h.c.>`
    on all 1st and 2nd neighbor bonds
    """
    coeffs = [1.0, 1.0, -1.0, -1.0]
    terms = ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]
    directions = "xydD" if nb2 else "xy"
    try:
        hopping = [[[meas_process(
            coeffs, [f"{term}_{d}{i}{j}" for term in terms], measures
        ) for j in range(N2) ] for i in range(N1)] for d in directions]
    except KeyError:
        # Heisenberg model
        hopping = [[[
            0.0 for j in range(N2)
        ] for i in range(N1)] for d in directions]
    return np.array(hopping)


def cal_singlet(
    measures: dict[str, complex|float], 
    N1: int, N2: int, nb2=False
) -> dict[str, complex|float]:
    """
    Extract singlet pairing
    `<c_{i,up} c_{j,down} - c_{i,down} c_{j,up}> / sqrt(2)`
    on all 1st and 2nd neighbor bonds
    """
    coeffs = [1/sqrt(2), -1/sqrt(2)]
    terms = ["CmuCmd", "CmdCmu"]
    directions = "xydD" if nb2 else "xy"
    try:
        singlet = [[[meas_process(
            coeffs, [f"{term}_{d}{i}{j}" for term in terms], measures
        ) for j in range(N2) ] for i in range(N1)] for d in directions]
    except KeyError:
        # Heisenberg model
        singlet = [[[
            0.0 for j in range(N2)
        ] for i in range(N1)] for d in directions]
    return np.array(singlet)


def cal_bondEs(
    measures: dict[str], param: dict[str], nb2=False
):
    r"""
    Extract the energy on each 1st and 2nd neighbor bond
    ```
    (-t) sum_s (c+_{i,s} c_{j,s} + h.c.)
    + J (S_i.S_j - (1/4) n_i n_j)
    ```
    For slave fermion representation, (-t) is changed to (+t).
    """
    tJ_conv = param["tJ_convention"]
    try:
        N1, N2 = param["N1"], param["N2"]
    except KeyError:
        N1, N2 = 2, 2
    try:
        t = param["t"]
    except KeyError:
        assert tJ_conv == 0
        t = 0.0
    J = param["J"]
    if tJ_conv != 0:
        coeffs = [-t, -t, t, t, J/2, J/2, J, -J/4]
        terms = [
            "CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd",
            "SpSm", "SmSp", "SzSz", "NudNud"
        ]
    else:
        # pure Heisenberg model
        coeffs = [J/2, J/2, J, -J/4]
        terms = [
            "SpSm", "SmSp", "SzSz", "NudNud"
        ]
    bondEs = [[[meas_process(
        coeffs, [f"{term}_{d}{i}{j}" for term in terms], measures
    ) for j in range(N2) ] for i in range(N1)] for d in "xy"]
    if nb2:
        try: t2 = param["t2"]
        except KeyError: t2 = 0
        try: J2 = param["J2"]
        except KeyError: J2 = 0
        if tJ_conv != 0:
            coeffs = [-t2, -t2, t2, t2, J2/2, J2/2, J2, -J2/4]
            terms = [
                "CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd",
                "SpSm", "SmSp", "SzSz", "NudNud"
            ]
        else:
            # pure Heisenberg model
            coeffs = [J2/2, J2/2, J2, -J2/4]
            terms = [
                "SpSm", "SmSp", "SzSz", "NudNud"
            ]
        bondE2s = [[[meas_process(
            coeffs, [f"{term}_{d}{i}{j}" for term in terms], measures
        ) for j in range(N2) ] for i in range(N1)] for d in "dD"]
    else:
        bondE2s = [[[0.0 for j in range(N2) ] for i in range(N1)] for d in "dD"]
    bondEs = bondEs + bondE2s
    return np.array(bondEs)


def cal_Esite(
    measures: dict[str, complex|float], 
    param: dict[str]
):
    """
    Extract the total energy per site
    (without chemical potential term)
    """
    try:
        N1, N2 = param["N1"], param["N2"]
    except KeyError:
        N1, N2 = 2, 2
    nbond = 2 * N1 * N2
    nb2 = ("t2" in param or "J2" in param)
    bondEs = cal_bondEs(measures, param, nb2)
    # per bond (1st neighbor)
    e1 = np.sum(bondEs[0:2]) / nbond
    # per bond (2nd neighbor)
    e2 = np.sum(bondEs[2:4]) / nbond
    # per site
    return (e1+e2)*2, bondEs


def cal_Ehole(measures: dict[str, complex|float], param: dict[str]):
    """
    Extract the energy per hole
    (without chemical potential term)
    ```
    Ehole = (Esite - E0) / doping
    ```
    where `E0` is the energy per site at half-filling
    (zero doping)
    """
    e0 = -1.169438
    e_site = cal_Esite(measures, param)
    doping = np.mean(cal_dopings(measures))
    return (e_site - e0) / doping


def process_measure(
    measures: dict[str, complex|float], 
    param: dict[str], nb2: bool
) -> dict[str]:
    """
    Get physical quantities from measured terms
    """
    try:
        N1: int = param["N1"]
        N2: int = param["N2"]
    except KeyError:
        N1, N2 = 2, 2
    results = {}
    results["dope"] = cal_dopings(measures, N1, N2)
    results["dope_mean"] = np.mean(results["dope"])
    results["e_site"], results["energy"] = cal_Esite(measures, param)
    results["mag"] = cal_mags(measures, N1, N2)
    results["mag_norm"] = np.linalg.norm(results["mag"], axis=0)
    results["scorder"] = cal_singlet(measures, N1, N2, nb2)
    results["spincor"] = cal_spincor(measures, N1, N2, nb2)
    results["hopping"] = cal_hopping(measures, N1, N2, nb2)
    return results
