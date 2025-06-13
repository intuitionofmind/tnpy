"""
Approximate local measurement 
on 2D TPS with weights for tJ model
"""

from itertools import product
from gtensor import GTensor
from phys_models.onesiteop import makeops_tJ
from utils import split_measkey
from . import local_measure as lm


def get_tJconv(t: GTensor):
    Dphy = (t.DE[0], t.DO[0])
    if Dphy == (2,0):
        tJ_conv = 0
    elif Dphy == (2,1): 
        tJ_conv = 1
    elif Dphy == (1,2):
        tJ_conv = 2
    else:
        raise ValueError("Unrecognized tJ convention")
    return tJ_conv


def meas_dope(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
) -> dict[str, float | complex]:
    """measure doping on each site"""
    meas = {}
    tJ_conv = get_tJconv(ts[0][0])
    N1, N2 = len(ts), len(ts[0])
    op = makeops_tJ("Nh", tJ_conv)
    for i, j in product(range(N1), range(N2)):
        meas[f"Nh_t{i}{j}"] = lm.meas_site(i, j, op, ts, wxs, wys)
    return meas


def meas_mag(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]]
) -> dict[str, float | complex]:
    """measure doping on each site"""
    meas = {}
    tJ_conv = get_tJconv(ts[0][0])
    N1, N2 = len(ts), len(ts[0])
    op = makeops_tJ("Nh", tJ_conv)
    for opname in ["Sx", "Sy", "Sz"]:
        op = makeops_tJ(opname, tJ_conv)
        for i, j in product(range(N1), range(N2)):
            meas[f"{opname}_t{i}{j}"] = lm.meas_site(i, j, op, ts, wxs, wys)
    return meas


def meas_hopping(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]], 
    nb2=False, meas_mode="bond"
) -> dict[str, float | complex]:
    """
    measure hopping (kinetic energy) terms
    on 1st and 2nd neighbor bonds
    """
    meas = {}
    tJ_conv = get_tJconv(ts[0][0])
    N1, N2 = len(ts), len(ts[0])
    for term in ["CpuCmu", "CpdCmd", "CmuCpu", "CmdCpd"]:
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        for i, j, d in product(range(N1), range(N2), "xy"):
            meas[f"{term}_{d}{i}{j}"] = (
                lm.meas_bond(d, i, j, ops, ts, wxs, wys) 
                if meas_mode == "bond" else
                lm.meas_loopxy(d, i, j, ops, ts, wxs, wys) 
            )
        if nb2:
            for i, j in product(range(N1), range(N2)):
                meas[f"{term}_d{i}{j}"] = lm.meas_diag1(i, j, ops, ts, wxs, wys)
                meas[f"{term}_D{i}{j}"] = lm.meas_diag2(i, j, ops, ts, wxs, wys)
    return meas


def meas_spincor(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]], 
    nb2=False, meas_mode="bond"
) -> dict[str, float | complex]:
    """
    measure spin correlation on 1st and 2nd neighbor bonds
    """
    meas = {}
    tJ_conv = get_tJconv(ts[0][0])
    N1, N2 = len(ts), len(ts[0])
    for term in ["SpSm", "SmSp", "SzSz", "NudNud"]:
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        for i, j, d in product(range(N1), range(N2), "xy"):
            meas[f"{term}_{d}{i}{j}"] = (
                lm.meas_bond(d, i, j, ops, ts, wxs, wys) 
                if meas_mode == "bond" else
                lm.meas_loopxy(d, i, j, ops, ts, wxs, wys) 
            )
        if nb2:
            for i, j in product(range(N1), range(N2)):
                meas[f"{term}_d{i}{j}"] = lm.meas_diag1(i, j, ops, ts, wxs, wys)
                meas[f"{term}_D{i}{j}"] = lm.meas_diag2(i, j, ops, ts, wxs, wys)
    return meas


def meas_singlet(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]], 
    nb2=False, meas_mode="bond"
) -> dict[str, float | complex]:
    """measure singlet SC order on 1st and 2nd neighbor bond"""
    meas = {}
    tJ_conv = get_tJconv(ts[0][0])
    N1, N2 = len(ts), len(ts[0])
    for term in ["CmuCmd", "CmdCmu"]:
        ops = [makeops_tJ(op, tJ_conv) for op in split_measkey(term)]
        for i, j, d in product(range(N1), range(N2), "xy"):
            meas[f"{term}_{d}{i}{j}"] = (
                lm.meas_bond(d, i, j, ops, ts, wxs, wys) 
                if meas_mode == "bond" else
                lm.meas_loopxy(d, i, j, ops, ts, wxs, wys) 
            )
        if nb2:
            for i, j in product(range(N1), range(N2)):
                meas[f"{term}_d{i}{j}"] = lm.meas_diag1(i, j, ops, ts, wxs, wys)
                meas[f"{term}_D{i}{j}"] = lm.meas_diag2(i, j, ops, ts, wxs, wys)
    return meas


def tJ_localmeas(
    ts: list[list[GTensor]], 
    wxs: list[list[GTensor]], wys: list[list[GTensor]], 
    nb2=False, meas_mode="bond"
) -> dict[str, float | complex]:
    """
    Measure 1st and 2nd neighbor terms of t-J model
    """
    tJ_conv = get_tJconv(ts[0][0])
    # locally measure all relevant terms
    measures = {}
    # spin terms
    measures.update(meas_dope(ts, wxs, wys))
    measures.update(meas_mag(ts, wxs, wys))
    measures.update(meas_spincor(ts, wxs, wys, nb2, meas_mode))
    if tJ_conv == 0: return measures
    # hopping and pairing terms
    measures.update(meas_hopping(ts, wxs, wys, nb2, meas_mode))
    measures.update(meas_singlet(ts, wxs, wys, nb2, meas_mode))
    return measures
   
