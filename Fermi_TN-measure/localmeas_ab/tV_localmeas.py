"""
Approximate local measurement 
on 2D TPS with weights for spinless fermion tV model
"""

from itertools import product
from gtensor import GTensor
from phys_models.onesiteop import makeops_tV
from utils import split_measkey
from .local_measure import meas_bond, meas_site


def meas_dens(
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
) -> dict[str, float | complex]:
    """measure fermion density on each site"""
    meas = {}
    op = makeops_tV("Num")
    dens = [
        meas_site(tname, op, tensors, weights)
        for tname in tensors.keys()
    ]
    nbond = len(weights)
    for ax in range(1, nbond+1):
        meas[f"w{ax}NumId"] = dens[0]
        meas[f"w{ax}IdNum"] = dens[1]
    return meas


def meas_hopE(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    measure hopping (kinetic energy) terms
    on 1st neighbor bonds
    """
    meas = {}
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], 
        ["CpCm", "CmCp"]
    ):
        ops = [makeops_tV(op) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def meas_intE(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    measure interaction terms on 1st neighbor bonds
    """
    meas = {}
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], 
        ["NumNum"]
    ):
        ops = [makeops_tV(op) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def meas_sc(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """measure singlet SC order on each 1st neighbor bond"""
    meas = {}
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], 
        ["CmCm"]
    ):
        ops = [makeops_tV(op) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def tV_localmeas(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    Measure 1st and 2nd neighbor terms of t-J model
    """
    # locally measure all relevant terms
    measures = {}
    measures.update(meas_dens(tensors, weights))
    measures.update(meas_hopE(tensors, weights))
    measures.update(meas_intE(tensors, weights))
    measures.update(meas_sc(tensors, weights))
    return measures
   
