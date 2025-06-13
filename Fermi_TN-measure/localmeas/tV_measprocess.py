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
from utils import split_measkey, meas_process
from .local_measure2 import check_t4
from .local_measure2 import get_bonds1, get_bonds2, bond_wtkeys


def cal_hopping(
    measures: dict[str, complex|float], nb2 = True
):
    """
    Extract hopping term
    `<c+_i c_j + h.c.> = <c+_i c_j - c_i c+_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [1.0, -1.0]
    terms = ["CpCm", "CmCp"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_intE(
    measures: dict[str, complex|float], nb2 = True
):
    """
    Extract interaction term
    `<n_i n_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [1.0]
    terms = ["NumNum"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_scorders(
    measures: dict[str, complex|float], nb2 = True
):
    """
    Extract pairing `<c_i c_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    if nb2: bonds += get_bonds2(t4)
    coeffs = [1.0]
    terms = ["CmCm"]
    for bond in bonds:
        sites, plq = split_measkey(bond)
        keys = [sites + term + plq for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_bondEs(
    measures: dict[str, complex|float], 
    param: dict[str], nb2 = True
):
    r"""
    Extract the energy on each 1st and 2nd neighbor bond
    (-t) <c+_i c_j + h.c.> + V <n_i n_j>
    """
    results = {}
    t4 = check_t4(measures)
    bonds = get_bonds1(t4) 
    results = {}
    t4 = check_t4(measures)
    for nb in (1,2) if nb2 else (1,):
        if nb == 1: 
            t, V = param["t"], param["V"]
            bonds = get_bonds1(t4)
        else:
            try: t = param["t2"]
            except KeyError: t = 0.0
            try: V = param["V2"]
            except KeyError: V = 0.0
            bonds = get_bonds2(t4)
        # pure Heisenberg model
        coeffs = [-t, t, V]
        terms = ["CpCm", "CmCp", "NumNum"]
        for bond in bonds:
            sites, plq = split_measkey(bond)
            keys = [sites + term + plq for term in terms]
            results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_Esite(measures: dict[str], param: dict[str]):
    """
    Extract the total energy per site
    (without chemical potential term)
    """
    t4 = check_t4(measures)
    energies = cal_bondEs(measures, param, nb2=False)
    # per bond
    energy = sum(e for e in energies.values()) / (8 if t4 else 4)
    # per site
    return energy * 2, energies


def cal_dens(measures: dict[str]):
    """
    Extract average particle density per site
    """
    t4 = check_t4(measures)
    if t4 is False:
        try:
            densA = measures["xyNumId1"]
        except KeyError:
            densA = measures["wzIdNum1"]
        try:
            densB = measures["xyIdNum1"]
        except KeyError:
            densB = measures["wzNumId1"]
        dens = np.array([densA, densB])
    else:
        densA = measures["xyNumId1"]
        densB = measures["xyIdNum1"]
        densC = measures["wzIdNum1"]
        densD = measures["wzNumId1"]
        dens = np.array([densA, densB, densC, densD])
    return dens

def process_measure(measures: dict[str], param: dict[str], nb2 = True):
    """
    Get physical quantities from measured terms

    Returns
    ----
    results: dict[str]
        energy per site, doping, SC order, magnetization
    energies: dict[str]
        energy on each bond
    """
    results = {}
    results["dope"] = cal_dens(measures)
    results["dope_mean"] = np.mean(results["dope"])
    results["scorder"] = cal_scorders(measures, nb2)
    results["hopEs"] = cal_hopping(measures, nb2)
    results["intEs"] = cal_intE(measures, nb2)
    results["e_site"], results["energy"] = cal_Esite(measures, param)
    return results
