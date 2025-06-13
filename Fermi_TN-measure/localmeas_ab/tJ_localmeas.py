"""
Approximate local measurement 
on 2D TPS with weights for tJ model
"""

from itertools import product
from gtensor import GTensor
from phys_models.onesiteop import makeops_tJ
from utils import split_measkey
from .local_measure import meas_bond, meas_site
from update_ftps.sutools import get_tpstJconv


def meas_dope(
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
) -> dict[str, float | complex]:
    """measure doping on each site"""
    meas = {}
    tJ_conv = get_tpstJconv(tensors)
    op = makeops_tJ("Nh", tJ_conv)
    dope = [
        meas_site(tname, op, tensors, weights)
        for tname in tensors.keys()
    ]
    nbond = len(weights)
    for ax in range(1, nbond+1):
        meas[f"w{ax}NhId"] = dope[0]
        meas[f"w{ax}IdNh"] = dope[1]
    return meas


def meas_mag(
    tensors: dict[str, GTensor], weights: dict[str, GTensor]
) -> dict[str, float | complex]:
    """measure doping on each site"""
    meas = {}
    tJ_conv = get_tpstJconv(tensors)
    nbond = len(weights)
    for opname in ["Sx", "Sy", "Sz"]:
        op = makeops_tJ(opname, tJ_conv)
        mag = [
            meas_site(tname, op, tensors, weights)
            for tname in tensors.keys()
        ]
        for ax in range(1, nbond+1):
            meas[f"w{ax}{opname}Id"] = mag[0]
            meas[f"w{ax}Id{opname}"] = mag[1]
    return meas


def meas_hopping(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    measure hopping (kinetic energy) terms
    on 1st neighbor bonds
    """
    meas = {}
    tJ_conv = get_tpstJconv(tensors)
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], 
        ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]
    ):
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def meas_spincor(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    measure spin correlation on 1st neighbor bonds
    """
    meas = {}
    tJ_conv = get_tpstJconv(tensors)
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], 
        ["SpSm", "SmSp", "SzSz", "NudNud"]
    ):
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def meas_singlet(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """measure singlet SC order on each 1st neighbor bond"""
    meas = {}
    tJ_conv = get_tpstJconv(tensors)
    nbond = len(weights)
    for bond, term in product(
        [f"w{ax}" for ax in range(1, nbond+1)], ["CmuCmd", "CmdCmu"]
    ):
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        key = bond + term
        meas[key] = meas_bond(bond, ops, tensors, weights)
    return meas


def tJ_localmeas(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
) -> dict[str, float | complex]:
    """
    Measure 1st and 2nd neighbor terms of t-J model
    """
    tJ_conv = get_tpstJconv(tensors)
    # locally measure all relevant terms
    measures = {}
    # spin terms
    measures.update(meas_dope(tensors, weights))
    measures.update(meas_mag(tensors, weights))
    measures.update(meas_spincor(tensors, weights))
    if tJ_conv == 0: return measures
    # hopping and pairing terms
    measures.update(meas_hopping(tensors, weights))
    measures.update(meas_singlet(tensors, weights))
    return measures
   
