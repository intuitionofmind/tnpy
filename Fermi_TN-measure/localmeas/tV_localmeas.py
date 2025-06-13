"""
Approximate local measurement 
on 2D TPS with weights for spinless fermion tV model
"""

from itertools import product
from gtensor import GTensor
from . import local_measure2 as lm2


def meas_dens(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], mode="site"
) -> dict[str, float | complex]:
    """measure doping on each site"""
    dope = {}
    for tname in tensors.keys():
        dope.update(lm2.meas_1site(
            "Num", tname, tensors, weights, mode=mode, model='tV'
        ))
    return dope


def meas_hopE(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """
    measure hopping (kinetic energy) terms
    on 1st and 2nd neighbor bonds
    """
    meas = {}
    t4 = len(tensors) == 4
    bond_plqs = lm2.get_bonds1(t4)
    if nb2:
        assert mode == "loop"
        bond_plqs += lm2.get_bonds2(t4)
    for bond_plq, term in product(
        bond_plqs, ["CpCm", "CmCp"]
    ):
        meas.update(lm2.meas_2site(
            term, bond_plq, tensors, weights, mode=mode, model='tV'
        ))
    return meas


def meas_intE(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """
    measure spin correlation terms on each bond
    """
    meas = {}
    t4 = len(tensors) == 4
    bond_plqs = lm2.get_bonds1(t4)
    if nb2:
        assert mode == "loop"
        bond_plqs += lm2.get_bonds2(t4)
    for bond_plq, term in product(bond_plqs, ["NumNum"]):
        meas.update(lm2.meas_2site(
            term, bond_plq, tensors, weights, mode=mode, model='tV'
        ))
    return meas


def meas_sc(
    tensors: dict[str, GTensor], weights: dict[str, GTensor], 
    nb2=True, mode="loop"
) -> dict[str, float | complex]:
    """measure singlet SC order on each 1st neighbor bond"""
    meas = {}
    t4 = len(tensors) == 4
    bond_plqs = lm2.get_bonds1(t4)
    if nb2:
        assert mode == "loop"
        bond_plqs += lm2.get_bonds2(t4)
    for bond_plq, term in product(bond_plqs, ["CmCm"]):
        meas.update(lm2.meas_2site(
            term, bond_plq, tensors, weights, mode=mode, model='tV'
        ))
    return meas


def tV_localmeas(
    tensors: dict[str, GTensor],
    weights: dict[str, GTensor], 
    nb2=True, meas_mode="loop"
) -> dict[str, float | complex]:
    """
    Measure 1st and 2nd neighbor terms of t-J model
    """
    # locally measure all relevant terms
    measures = {}
    measures.update(meas_dens(tensors, weights))
    measures.update(meas_hopE(tensors, weights, nb2, meas_mode))
    measures.update(meas_intE(tensors, weights, nb2, meas_mode))
    measures.update(meas_sc(tensors, weights, nb2, meas_mode))
    return measures
   
