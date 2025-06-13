"""
Terms to be measured measurement
of t-J and Heisenberg model
"""

from itertools import product

def get_allterms(model):
    """Supported terms of 2D TPS measurement"""
    term_list = {}
    if model == "tJ":
        for key in ("hop", "heis", "dens", "sc", "mag", "all"):
            term_list[key] = []
        # hopping terms
        # NOTE: xyCpuCmu = -xyCmuCpu, etc
        bonds = ("xy", "wz", "xw", "yz")
        terms = ("CpuCmu", "CpdCmd")
        for bond, term in product(bonds, terms):
            term_list["hop"].append((bond, term))
        # heisenberg terms 
        # NOTE: SpSm = SmSp when expval is real
        bonds = ("xy", "wz", "xw", "yz")
        terms = ("SpSm", "SzSz", "NudNud")
        for bond, term in product(bonds, terms):
            term_list["heis"].append((bond, term))
        # number density 
        # NOTE: only measured on xy, wz bond
        bonds = ("xy", "wz")
        terms = ("NhId", "IdNh")
        for bond, term in product(bonds, terms):
            term_list["dens"].append((bond, term))
        # SC order parameter
        # formula (e.g. for xy bond): 1/sqrt(2) * (xyCmdCmu - xyCmuCmd)
        bonds = ("xy", "wz", "xw", "yz")
        terms = ("CmdCmu", "CmuCmd")
        for bond, term in product(bonds, terms):
            term_list["sc"].append((bond, term))
        # Magnetization
        # NOTE: only measured on xy, wz bond
        bonds = ("xy", "wz")
        terms = ("SxId", "IdSx", "SyId", "IdSy", "SzId", "IdSz")
        for bond, term in product(bonds, terms):
            term_list["mag"].append((bond, term))

    elif model == "tV":
        for key in ("dens", "sc", "all"):
            term_list[key] = []
        # number density 
        # NOTE: only measured on xy, wz bond
        bonds = ("xy", "wz")
        terms = ("NumId", "IdNum")
        for bond, term in product(bonds, terms):
            term_list["dens"].append((bond, term))
        # SC order parameter
        # formula (e.g. for xy bond): xyCmCm
        bonds = ("xy", "wz", "xw", "yz")
        terms = ("CmCm",)
        for bond, term in product(bonds, terms):
            term_list["sc"].append((bond, term))

    # collect all terms
    for key in term_list:
        if key != "all": 
            term_list["all"] += term_list[key]
    return term_list
