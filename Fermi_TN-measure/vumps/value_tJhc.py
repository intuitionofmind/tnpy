import numpy as np
from math import sqrt
import gtensor as gt
from gtensor import GTensor
from fermiT import FermiT
from fermiT.conversion import gt2ft
from phys_models.onesiteop import makeops_tJ
from vumps.value import cal_rho1ss, cal_rho2ss
from vumps.correlation import cor_2site_ver, cor_2site_hor

def transform(opA: GTensor, opB: GTensor):
    """
    Fuse physical operators acting on one honeycomb unit cell
    """
    op = gt.outer(opA, opB).transpose(0,2,1,3)
    op = op.merge_axes((2,2), order=(1,-1))
    return gt2ft(op)


def cal_doping(rhosss: list[list[list[FermiT]]]):
    """
    Calculate doping using nearest neighbor rhos `rhosss`
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Nh, Id = list(map(makeops_tJ, ["Nh", "Id"], [2]*2))
    Id2 = transform(Id, Id)
    dopings = np.array([
        # A (v)
        cal_rho1ss(rho1vss, transform(Nh, Id), Id2)[0][0],
        # A (h)
        cal_rho1ss(rho1hss, transform(Nh, Id), Id2)[0][0],
        # B (v)
        cal_rho1ss(rho1vss, transform(Id, Nh), Id2)[0][0],
        # B (h)
        cal_rho1ss(rho1hss, transform(Id, Nh), Id2)[0][0],
    ])
    return dopings


def cal_mags(rhosss: list[list[list[FermiT]]]):
    """
    Calculate <Sx>, <Sy>, <Sz> using nearest neighbor rhos `rhosss`
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Sx, Sy, Sz, Id = list(map(makeops_tJ, ["Sx", "Sy", "Sz", "Id"], [2]*4))
    Id2 = transform(Id, Id)
    # A (v)
    mags = np.array([
        cal_rho1ss(rho1vss, transform(op, Id), Id2)[0][0]
        for op in [Sx, Sy, Sz]
    ])
    return mags


def cal_sin1(rhosss: list[list[list[FermiT]]]):
    """
    Calculate 1st neighbor singlet pairing
    using nearest neighbor rhos `rhosss`
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Cmu, Cmd, Id = list(map(makeops_tJ, ["Cmu", "Cmd", "Id"], [2]*3))
    Id2 = transform(Id, Id)
    singlets = np.zeros(3, dtype=complex)
    # bond 1
    singlets[0] = (
        cal_rho2ss(rho2vss, transform(Id,Cmu), transform(Cmd,Id), Id2)
        - cal_rho2ss(rho2vss, transform(Id,Cmd), transform(Cmu,Id), Id2)
    )[0][0] / sqrt(2)
    # bond 2
    singlets[1] = (
        cal_rho2ss(rho2hss, transform(Id,Cmu), transform(Cmd,Id), Id2)
        - cal_rho2ss(rho2hss, transform(Id,Cmd), transform(Cmu,Id), Id2)
    )[0][0] / sqrt(2)
    # bond 3
    singlets[2] = cal_rho1ss(
        rho1vss, transform(Cmu, Cmd) - transform(Cmd, Cmu), Id2
    )[0][0] / sqrt(2)
    return singlets


def cal_sin2(rhosss: list[list[list[FermiT]]]):
    """
    Calculate 2nd neighbor singlet pairing
    using nearest neighbor rhos `rhosss`
    """
    rho1vss, rho2vss, rho1hss, rho2hss = rhosss
    Cmu, Cmd, Id = list(map(makeops_tJ, ["Cmu", "Cmd", "Id"], [2]*3))
    Id2 = transform(Id, Id)
    sc2a: complex = (
        cal_rho2ss(rho2vss, transform(Cmu,Id), transform(Cmd,Id), Id2)
        - cal_rho2ss(rho2vss, transform(Cmd,Id), transform(Cmu,Id), Id2)
    )[0][0] / sqrt(2)
    sc2b: complex = (
        cal_rho2ss(rho2vss, transform(Id,Cmu), transform(Id,Cmd), Id2)
        - cal_rho2ss(rho2vss, transform(Id,Cmd), transform(Id,Cmu), Id2)
    )[0][0] / sqrt(2)
    return np.array([sc2a, sc2b])


def cal_sin2s(
    fG0ss: list[list[FermiT]], fG1ss: list[list[FermiT]], 
    fXss: list[list[list[FermiT]]], fGXss: list[list[list[FermiT]]], 
    num=4
):
    """
    Calculate long-range singlet pairing
    in the same direction as 2nd neighbor bonds

    Parameters
    ----
    fG0ss, fG1ss: list[list[FermiT]]
        ket, bra tensors of the PEPS in one unit cell
    fXss: list[list[list[FermiT]]]
        AL of the four boundary MPSs
    fGXss: list[list[list[FermiT]]]
        fixed points of the AL column transfer matrices
        corresponding to the four boundary MPSs
    """
    Cmu, Cmd, Id = list(map(makeops_tJ, ["Cmu", "Cmd", "Id"], [2]*2))
    scXAs = (
        cor_2site_hor(
            transform(Cmu,Id), transform(Cmd,Id), 
            fXss, fGXss, fG1ss, fG0ss, num=num
        ) - cor_2site_hor(
            transform(Cmd,Id), transform(Cmu,Id), 
            fXss, fGXss, fG1ss, fG0ss, num=num
        )
    ) / sqrt(2)
    scXBs = (
        cor_2site_hor(
            transform(Id,Cmu), transform(Id,Cmd), 
            fXss, fGXss, fG1ss, fG0ss, num=num
        ) - cor_2site_hor(
            transform(Id,Cmd), transform(Id,Cmu), 
            fXss, fGXss, fG1ss, fG0ss, num=num
        )
    ) / sqrt(2)
    return [scXAs, scXBs]
