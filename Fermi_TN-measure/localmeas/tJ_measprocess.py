"""
Combine measured terms into 
total energy, order parameter, etc.

On each different loop, X Y Z W refer to 
the four corners of the loop
```
      ↑   ↓
    → W ← Z →
      ↓   ↑
    ← X → Y ←
      ↑   ↓
```
"""

import numpy as np
from math import sqrt
from utils import split_measkey, meas_process
from .local_measure2 import check_t4
from .local_measure2 import get_bonds1, get_bonds2


def cal_dopings(measures: dict[str, complex|float]):
    """
    Extract doping on each site
    (A, B) or (A, B, C, D)
    """
    t4 = check_t4(measures)
    if t4 is False:
        try:
            dopeA = measures["xyNhId1"]
        except KeyError:
            dopeA = measures["wzIdNh1"]
        try:
            dopeB = measures["xyIdNh1"]
        except KeyError:
            dopeB = measures["wzNhId1"]
        dope = np.array([dopeA, dopeB])
    else:
        dopeA = measures["xyNhId1"]
        dopeB = measures["xyIdNh1"]
        dopeC = measures["wzIdNh1"]
        dopeD = measures["wzNhId1"]
        dope = np.array([dopeA, dopeB, dopeC, dopeD])
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
    t4 = check_t4(measures)
    mag = np.stack([
        np.array([
            measures["xyS{}Id1".format(a)],
            measures["xyIdS{}1".format(a)],
            measures["wzIdS{}1".format(a)],
            measures["wzS{}Id1".format(a)],
        ]) if t4 else np.array([
            measures["xyS{}Id1".format(a)], 
            measures["xyIdS{}1".format(a)]
        ]) for a in ("x", "y", "z")
    ], axis=0)
    return mag


def cal_spincor(
    measures: dict[str, complex|float], nb2 = True
) -> dict[str, complex|float]:
    """
    Extract spin correlation `<S_i S_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [0.5, 0.5, 1.0]
    terms = ["SpSm", "SmSp", "SzSz"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_hopping(
    measures: dict[str, complex|float], nb2 = True
) -> dict[str, complex|float]:
    """
    Extract hopping term
    `<c+_{i,up} c_{j,up} + c+_{i,down} c_{j,down} + h.c.>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [1.0, 1.0, -1.0, -1.0]
    terms = ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        try:
            results[bond] = meas_process(coeffs, keys, measures)
        except KeyError:
            # Heisenberg model
            results[bond] = 0.0
    return results


def cal_singlet(
    measures: dict[str, complex|float], nb2 = True
) -> dict[str, complex|float]:
    """
    Extract singlet pairing
    `<c_{i,up} c_{j,down} - c_{i,down} c_{j,up}> / sqrt(2)`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [1.0, -1.0]
    terms = ["CmuCmd", "CmdCmu"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        try:
            results[bond] = meas_process(coeffs, keys, measures) / sqrt(2)
        except KeyError:
            # Heisenberg model
            results[bond] = 0.0
    return results


def cal_bondEs(
    measures: dict[str], param: dict[str], nb2 = True
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
    t4 = check_t4(measures)
    for nb in (1,2) if nb2 else (1,):
        if nb == 1: 
            try:
                t = param["t"]
            except KeyError:
                assert tJ_conv == 0
                t = 0.0
            J = param["J"]
            bonds = get_bonds1(t4)
        else:
            try: t = param["t2"]
            except KeyError: t = 0.0
            try: J = param["J2"]
            except KeyError: J = 0.0
            bonds = get_bonds2(t4)

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
        for bond in bonds:
            sites, plq = split_measkey(bond)
            keys = [sites + term + plq for term in terms]
            results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_Esite(measures: dict[str, complex|float], param: dict[str]):
    """
    Extract the total energy per site
    (without chemical potential term)
    """
    t4 = check_t4(measures)
    energies = cal_bondEs(measures, param)
    # per bond
    energy = sum(e for e in energies.values()) / (8 if t4 else 4)
    # per site
    return energy * 2, energies


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

# -------- two singlet pairs on 4 sites --------

def cal_singlet4(measures: dict[str, complex|float]):
    """
    Combine the measured cccc terms with 
    total Sz = 0 into two pairs of singlets
    """
    results = {}
    for d in list(range(1,5)) + ['h', 'v']:
        results[f"(0,1)(2,3){d}"] = (
            measures[f"udud-{d}"] - measures[f"uddu-{d}"] 
            - measures[f"duud-{d}"] + measures[f"dudu-{d}"]
        ) / 2
        results[f"(0,2)(1,3){d}"] = (
            measures[f"uudd-{d}"] - measures[f"uddu-{d}"] 
            - measures[f"duud-{d}"] + measures[f"dduu-{d}"]
        ) / 2
        results[f"(0,3)(1,2){d}"] = (
            measures[f"uudd-{d}"] - measures[f"udud-{d}"] 
            - measures[f"dudu-{d}"] + measures[f"dduu-{d}"]
        ) / 2
    return results


def process_measure(
    measures: dict[str, complex|float], 
    param: dict[str], nb2=True, meas4=False
) -> dict[str]:
    """
    Get physical quantities from measured terms
    """
    results = {}
    results["dope"] = cal_dopings(measures)
    results["dope_mean"] = np.mean(results["dope"])
    results["e_site"], results["energy"] = cal_Esite(measures, param)
    results["mag"] = cal_mags(measures)
    results["mag_norm"] = np.linalg.norm(results["mag"], axis=0)
    results["scorder"] = cal_singlet(measures, nb2)
    results["spincor"] = cal_spincor(measures, nb2)
    results["hopping"] = cal_hopping(measures, nb2)
    if meas4:
        results["singlet4"] = cal_singlet4(measures)
    return results
