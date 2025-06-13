"""
Approximate local measurement 
on 2D TPS with weights for t-J model
"""

from itertools import product
from gtensor import GTensor
from . import local_measure2 as lm2
from update_ftps.sutools import get_tpstJconv

# -------- 1-site quantities --------

def meas_dope(
    tensors: dict[str, GTensor], 
    weights: dict[str, GTensor], mode="site"
) -> dict[str, float | complex]:
    """measure doping on each site"""
    dope = {}
    for tname in tensors.keys():
        dope.update(lm2.meas_1site("Nh", tname, tensors, weights, mode=mode))
    return dope


def meas_mag(
    tensors: dict[str, GTensor], 
    weights: dict[str, GTensor], mode="site"
) -> dict[str, float | complex]:
    """measure magnetization on each site"""
    mag = {}
    for tname, opname in product(tensors.keys(), ["Sx", "Sy", "Sz"]):
        mag.update(lm2.meas_1site(opname, tname, tensors, weights, mode=mode))
    return mag

# -------- 2-site quantities --------

def meas_hopping(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """
    measure hopping (kinetic energy) terms
    on 1st and 2nd neighbor bonds
    """
    meas = {}
    t4 = len(tensors) == 4
    terms = ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]
    for bond_plq, term in product(lm2.get_bonds1(t4), terms):
        meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode=mode))
    if nb2:
        for bond_plq, term in product(lm2.get_bonds2(t4), terms):
            meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode="loop"))
    return meas


def meas_spincor(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """
    measure spin correlation terms on each bond
    """
    meas = {}
    t4 = len(tensors) == 4
    terms = ["SpSm", "SmSp", "SzSz", "NudNud"]
    for bond_plq, term in product(lm2.get_bonds1(t4), terms):
        meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode=mode))
    if nb2:
        for bond_plq, term in product(lm2.get_bonds2(t4), terms):
            meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode="loop"))
    return meas


def meas_singlet(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """measure singlet SC order on each 1st neighbor bond"""
    meas = {}
    t4 = len(tensors) == 4
    terms = ["CmuCmd", "CmdCmu"]
    for bond_plq, term in product(lm2.get_bonds1(t4), terms):
        meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode=mode))
    if nb2:
        for bond_plq, term in product(lm2.get_bonds2(t4), terms):
            meas.update(lm2.meas_2site(term, bond_plq, tensors, weights, mode="loop"))
    return meas


def tJ_localmeas(
    tensors: dict[str, GTensor],
    weights: dict[str, GTensor], 
    nb2=True, meas_mode="loop"
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
    measures.update(meas_spincor(tensors, weights, nb2, meas_mode))
    if tJ_conv == 0: return measures
    # hopping and pairing terms
    measures.update(meas_hopping(tensors, weights, nb2, meas_mode))
    measures.update(meas_singlet(tensors, weights, nb2, meas_mode))
    return measures
   
