"""
Combine measured terms into 
total energy, order parameter, etc.
"""

import numpy as np
from utils import meas_process


def cal_dopings(measures: dict[str, complex|float]):
    """
    Extract doping on each site (A, B)
    """
    dopeA = measures["w1NhId"]
    dopeB = measures["w1IdNh"]
    dope = np.array([dopeA, dopeB])
    return dope


def cal_mags(measures: dict[str, complex|float]):
    """
    Extract magnetization 
    `<Sx>, <Sy>, <Sz>` on each site

    Returns
    ----
    mag: np.ndarray
        Row 1/2/3 is the Sx/Sy/Sz on all sites
    """
    mag = np.stack([
        np.array([
            measures["w1S{}Id".format(a)], 
            measures["w1IdS{}".format(a)]
        ]) for a in ("x", "y", "z")
    ], axis=0)
    return mag


def cal_spincor(
    measures: dict[str, complex|float], nbond: int
) -> dict[str, complex|float]:
    """
    Extract spin correlation `<S_i S_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [0.5, 0.5, 1.0]
    terms = ["SpSm", "SmSp", "SzSz"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_hopping(
    measures: dict[str, complex|float], nbond: int
) -> dict[str, complex|float]:
    """
    Extract hopping term
    `<c+_{i,up} c_{j,up} + c+_{i,down} c_{j,down} + h.c.>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [1.0, 1.0, -1.0, -1.0]
    terms = ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        try:
            results[bond] = meas_process(coeffs, keys, measures)
        except KeyError:
            # Heisenberg model
            results[bond] = 0.0
    return results


def cal_singlet(
    measures: dict[str, complex|float], nbond: int
) -> dict[str, complex|float]:
    """
    Extract singlet pairing
    `<c_{i,up} c_{j,down} - c_{i,down} c_{j,up}> / sqrt(2)`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [1.0, -1.0]
    terms = ["CmuCmd", "CmdCmu"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        try:
            results[bond] = meas_process(coeffs, keys, measures)
        except KeyError:
            # Heisenberg model
            results[bond] = 0.0
    return results


def cal_bondEs(
    measures: dict[str], param: dict[str]
):
    r"""
    Extract the energy on each 1st and 2nd neighbor bond
    ```
    (-t) sum_s (c+_{i,s} c_{j,s} + h.c.)
    + J (S_i.S_j - (1/4) n_i n_j)
    ```
    """
    results = {}
    tJ_conv = param["tJ_convention"]
    nbond = param["nbond"]
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
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_Esite(measures: dict[str, complex|float], param: dict[str]):
    """
    Extract the total energy per site
    (without chemical potential term)
    """
    nbond = param["nbond"]
    energies = cal_bondEs(measures, param)
    # per bond
    energy = sum(e for e in energies.values()) / nbond
    # per site
    return energy * 2


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
    if param["lattice"] == "honeycomb":
        e0 = -0.91955
    elif param["lattice"] == "square":
        e0 = -1.169438
    else: raise ValueError("unknown lattice")
    
    e_site = cal_Esite(measures, param)
    doping = np.mean(cal_dopings(measures))
    return (e_site - e0) / doping


def process_measure(
    measures: dict[str, complex|float], 
    param: dict[str]
) -> dict[str]:
    """
    Get physical quantities from measured terms
    """
    nbond = param["nbond"]
    results = {}
    results["dope"] = cal_dopings(measures)
    results["dope_mean"] = np.mean(results["dope"])
    results["e_site"] = cal_Esite(measures, param)
    results["energy"] = cal_bondEs(measures, param)
    results["mag"] = cal_mags(measures)
    results["mag_norm"] = np.linalg.norm(results["mag"], axis=0)
    results["scorder"] = cal_singlet(measures, nbond)
    results["spincor"] = cal_spincor(measures, nbond)
    results["hopping"] = cal_hopping(measures, nbond)
    return results
