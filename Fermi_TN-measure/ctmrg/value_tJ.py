import numpy as np
from numpy import ndarray
import gtensor as gt
from gtensor import GTensor
import ctmrg.measure as meas
from phys_models.onesiteop import makeops_tJ


def get_tJconv(Dphy: tuple[int, int]):
    if Dphy == (2,0):   return 0
    elif Dphy == (2,1): return 1
    elif Dphy == (1,2): return 2
    else: raise ValueError("Unrecognized tJ convention")


def get_ops(names: list[str], tJ_conv: int):
    return [makeops_tJ(name, tJ_conv) for name in names]


def cal_dopings(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure doping on each site
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    Nh = makeops_tJ("Nh", tJ_conv)
    doping = meas.meas_allsites(Nh, ts0, ts1, ctms)
    return doping


def cal_mags(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure magnetization `<S^a>` (a = x,y,z) on each site
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    ops = get_ops(["Sx","Sy","Sz"], tJ_conv)
    return np.stack([
        meas.meas_allsites(op, ts0, ts1, ctms) for op in ops
    ])


def cal_hopping(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure hopping on each 1st neighbor bond
    `<c+_{i,up} c_{j,up} + c+_{i,dn} c_{j,dn}>` 
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    Cpu, Cpd, Cmu, Cmd = get_ops(["Cpu","Cpd","Cmu","Cmd"], tJ_conv)
    hopping = (
        meas.meas_allbonds1([Cpu, Cmu], ts0, ts1, ctms)
        + meas.meas_allbonds1([Cpd, Cmd], ts0, ts1, ctms)
    )
    return hopping


def cal_singlet(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure singlet pairing on each 1st neighbor bond
    `<c_{i,up} c_{j,dn} - c_{i,dn} c_{j,up}> / sqrt(2)`
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    Cmu, Cmd = get_ops(["Cmu","Cmd"], tJ_conv)
    singlet = (
        meas.meas_allbonds1([Cmu, Cmd], ts0, ts1, ctms)
        - meas.meas_allbonds1([Cmd, Cmu], ts0, ts1, ctms)
    ) / np.sqrt(2)
    return singlet


def cal_spincor(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure spin correlation on each 1st neighbor bond
    `<S_i S_j>`
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    Sp, Sm, Sz = get_ops(["Sp","Sm","Sz"], tJ_conv)
    spincor = (
        meas.meas_allbonds1([Sp, Sm], ts0, ts1, ctms)
        + meas.meas_allbonds1([Sz, Sz], ts0, ts1, ctms)
    )
    return spincor


def cal_ninj(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]],
    ctms: list[list[list[GTensor]]]
):
    """
    Measure `<n_i n_j>` on each 1st neighbor bond
    """
    Dphy = (ts0[0][0].DE[0], ts0[0][0].DO[0])
    tJ_conv = get_tJconv(Dphy)
    Nud = makeops_tJ("Nud", tJ_conv)
    ninj = meas.meas_allbonds1([Nud, Nud], ts0, ts1, ctms)
    return ninj


def cal_Esite(
    hopping1: ndarray, spincor1: ndarray, 
    ninj1: ndarray, param: dict[str]
):
    """
    Convert site/bond measurements to energy per site of t-J model
    """
    t, J = [param[key] for key in ["t", "J"]]
    N1, N2 = hopping1[0].shape
    ebond = (
        -t * (hopping1 + hopping1.conj())
        + J * (spincor1 - ninj1/4)
    )
    esite = np.sum(ebond) / (N1 * N2)
    return esite, ebond


def load_measure(
    ts0: list[list[GTensor]], ts1: list[list[GTensor]], 
    ctms: list[list[list[GTensor]]], param: dict[str]
) -> dict[str]:
    """
    Get physical quantities from measured terms
    """
    results = {}
    results["dope"] = cal_dopings(ts0, ts1, ctms)
    results["dope_mean"] = np.mean(results["dope"])
    results["mag"] = cal_mags(ts0, ts1, ctms)
    results["mag_norm"] = np.linalg.norm(results["mag"], axis=0)
    results["scorder"] = cal_singlet(ts0, ts1, ctms)
    results["spincor"] = cal_spincor(ts0, ts1, ctms)
    ninj = cal_ninj(ts0, ts1, ctms)
    results["hopping"] = cal_hopping(ts0, ts1, ctms)
    results["e_site"], results["energy"] \
        = cal_Esite(results["hopping"], results["spincor"], ninj, param)
    return results
