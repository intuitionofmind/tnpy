"""
Combine measured terms into 
total energy, order parameter, etc.
"""

import numpy as np
from utils import meas_process


def cal_dens(measures: dict[str]):
    """
    Extract average particle density per site
    """
    densA = measures["w1NumId"]
    densB = measures["w1IdNum"]
    dens = np.array([densA, densB])
    return dens


def cal_hopping(measures: dict[str, complex|float], nbond: int):
    """
    Extract hopping term
    `<c+_i c_j + h.c.> = <c+_i c_j - c_i c+_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [1.0, -1.0]
    terms = ["CpCm", "CmCp"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_intE(measures: dict[str, complex|float], nbond: int):
    """
    Extract interaction term
    `<n_i n_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [1.0]
    terms = ["NumNum"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_scorders(measures: dict[str, complex|float], nbond: int):
    """
    Extract pairing `<c_i c_j>`
    on all 1st and 2nd neighbor bonds
    """
    results = {}
    coeffs = [1.0]
    terms = ["CmCm"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_bondEs(
    measures: dict[str, complex|float], 
    param: dict[str]
):
    r"""
    Extract the energy on each 1st and 2nd neighbor bond
    (-t) <c+_i c_j + h.c.> + V <n_i n_j>
    """
    results = {}
    nbond = param["nbond"]
    t, V = param["t"], param["V"]
    coeffs = [-t, t, V]
    terms = ["CpCm", "CmCp", "NumNum"]
    for bond in [f"w{ax}" for ax in range(1, nbond+1)]:
        keys = [bond + term for term in terms]
        results[bond] = meas_process(coeffs, keys, measures)
    return results


def cal_Esite(measures: dict[str], param: dict[str]):
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


def process_measure(measures: dict[str], param: dict[str]):
    """
    Get physical quantities from measured terms

    Returns
    ----
    results: dict[str]
        energy per site, doping, SC order, magnetization
    energies: dict[str]
        energy on each bond
    """
    nbond = param["nbond"]
    results = {}
    results["dope"] = cal_dens(measures)
    results["dope_mean"] = np.mean(results["dope"])
    results["scorder"] = cal_scorders(measures, nbond)
    results["hopEs"] = cal_hopping(measures, nbond)
    results["intEs"] = cal_intE(measures, nbond)
    results["energy"] = cal_bondEs(measures, param)
    results["e_site"] = cal_Esite(measures, param)
    return results
