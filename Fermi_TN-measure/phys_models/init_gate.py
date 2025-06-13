"""
    init_gate.py
    =============
    Initialize Trotter gate for imaginary time evolution
"""

import gtensor as gt
import gtensor.linalg as gla
from .onesiteop import *

# ---- operator (gate) on 2 or 3 adjacent sites ----

def init_gate(param: dict[str], expo=True):
    """
    Initialize imag-time evolution gate

    when adding chemical potential, we usually use -μ N
    in line with the grand potential Ω = F - μ N
    we are then calculating the grand canonical partition function

    Parameters
    ----
    expo: bool
        - if True, return exp(-exp * H)
        - if False, return H

    Return value axis order 
    ----
    ```
    2-body gates:   3-body gates:
        i  i+1          i  i+1 i+2
        0   1           0   1   2
        ↓---↓           ↓---↓---↓
        |   |           |       |
        ↓---↓           ↓---↓---↓
        2   3           3   4   5
    ```
    """
    model = param["model"]
    
    # number of nearest neighbors, usually:
    # 1D: nbond = 2
    # 2D: nbond = 4 (square lattice)
    nbond = param["nbond"]

    if model in ("tV", "kitaev"):
        """Kitaev spinless fermion chain"""
        Id, Cp, Cm, Num = tuple(map(makeops_tV, ["Id", "Cp", "Cm", "Num"]))
        # ---- each term in the two-site Hamiltonian ----
        t, V, mu = param["t"], param["V"], param["mu"]
        """
        Hopping: 
            (-t)(c_i^† c_j + c_j^† c_i)
            = (-t)(c_i^† c_j - c_i c_j^†)
        (due to anti-commutation relation)
        """
        hop = (-t) * (gt.outer(Cp, Cm) - gt.outer(Cm, Cp))
        """
        On-site interaction
        - tV:       V n_i n_j
        - kitaev:   V (n_i - 1/2) (n_j - 1/2) 
        """
        inter = (
            V * gt.outer(Num - Id/2, Num - Id/2)
            if model == "kitaev" else
            V * gt.outer(Num, Num)
        )
        """
        Chemical potential (doping)
        - tV:       (-mu/nbond)(n_i + n_j)
        - kitaev:   (-mu/nbond)((n_i-1/2) + (n_j-1/2))
        """
        dope = (
            (-mu / nbond) * (
                gt.outer(Num - Id/2, Id) 
                + gt.outer(Id, Num - Id/2)
            ) if model == "kitaev" else
            (-mu / nbond) * (
                gt.outer(Num, Id) 
                + gt.outer(Id, Num)
            )
        )
        ham = hop + inter + dope

        if model == "kitaev":
            """
            p-wave potential
                (-D)(c_i^† c_j† - c_i c_j)
            """
            D = param["D"]
            pwave = (-D) * (
                gt.outer(Cp, Cp)
                - gt.outer(Cm, Cm)
            )
            ham = ham + pwave
        
        ham = gt.transpose(ham, [0,2,1,3])

    elif model == "tJ":
        """
        t-J/Heisenberg model with only nearest neighbor interaction

        Repeated counting
        - sites:        4 (nbond) times
        - <ij> bonds:   1 time
        """  
        tJ_conv = param["tJ_convention"]
        Sp, Sm, Sz, Nud, Id = tuple(map(
            makeops_tJ, ["Sp", "Sm", "Sz", "Nud", "Id"], [tJ_conv]*5
        ))
        # ---- each term in the two-site Hamiltonian ----
        """
        Heisenberg:
            J (S_i S_j - (1/4) n_i n_j)
        where
            S_i S_j
            = (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)
            = (1/2)(Sp_i Sm_j + Sm_i Sp_j) + Sz_i Sz_j
        and
            n_i = n_{i,up} + n_{i,dn}
        """
        J = param["J"]
        heis = J * (
            (1/2) * (
                gt.outer(Sp, Sm) + gt.outer(Sm, Sp)
            ) + gt.outer(Sz, Sz)
            - (1/4) * gt.outer(Nud, Nud)
        )
        # terms related to un-occupied states
        # valid only at finite doping
        if tJ_conv != 0:
            """
            Hopping:
                (-t) sum_s (c†_{is} c_{js} + c†_{js} c_{is})
                = (-t) sum_s (c†_{is} c_{js} - c_{is} c†_{js})
            (due to anti-commutation relation)
            The c_{is} operator here is in no-double-occupancy subspace

            In Schwinger boson representation
                t sum_s (
                    h†_i h_j b_{is} b†_{js}
                    - h_i h†_j b†_{is} b_{js} 
                )
            """
            Cpu, Cmu, Cpd, Cmd = tuple(map(
                makeops_tJ, ["Cpu", "Cmu", "Cpd", "Cmd"], [tJ_conv]*5
            ))
            t, mu = param["t"], param["mu"]
            tsign = 1 if tJ_conv == 1 else -1
            hop = tsign * t * (
                gt.outer(Cpu, Cmu) + gt.outer(Cpd, Cmd)
                - gt.outer(Cmu, Cpu) - gt.outer(Cmd, Cpd)
            )
            """
            Chemical potential doping:
                (-mu / nbond)(n_i + n_j)
                =  (-mu / nbond)(nb_i + nb_j)
                (nb_i: spinon number at site i)
            """
            dope = (-mu / nbond) * (
                gt.outer(Nud, Id)
                + gt.outer(Id, Nud)
            )
        ham = gt.transpose(
            heis if tJ_conv == 0 
            else (hop + heis + dope), [0,2,1,3]
        )

    elif model == "sigmatJ":
        """
        t-J/Heisenberg model with only nearest neighbor interaction

        Repeated counting
        - sites:        4 (nbond) times
        - <ij> bonds:   1 time
        """  
        tJ_conv = param["tJ_convention"]
        Sp, Sm, Sz, Nud, Id = tuple(map(
            makeops_tJ, ["Sp", "Sm", "Sz", "Nud", "Id"], [tJ_conv]*5
        ))
        # ---- each term in the two-site Hamiltonian ----
        """
        Heisenberg:
            J (S_i S_j - (1/4) n_i n_j)
        where
            S_i S_j
            = (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)
            = (1/2)(Sp_i Sm_j + Sm_i Sp_j) + Sz_i Sz_j
        and
            n_i = n_{i,up} + n_{i,dn}
        """
        J = param["J"]
        heis = J * (
            (1/2) * (
                gt.outer(Sp, Sm) + gt.outer(Sm, Sp)
            ) + gt.outer(Sz, Sz)
            - (1/4) * gt.outer(Nud, Nud)
        )
        # terms related to un-occupied states
        # valid only at finite doping
        if tJ_conv != 0:
            """
            Hopping:
                (-t) sum_s (c†_{is} c_{js} + c†_{js} c_{is})
                = (-t) sum_s (c†_{is} c_{js} - c_{is} c†_{js})
            (due to anti-commutation relation)
            The c_{is} operator here is in no-double-occupancy subspace

            In Schwinger boson representation
                t sum_s (
                    h†_i h_j b_{is} b†_{js}
                    - h_i h†_j b†_{is} b_{js} 
                )
            """
            Cpu, Cmu, Cpd, Cmd = tuple(map(
                makeops_tJ, ["Cpu", "Cmu", "Cpd", "Cmd"], [tJ_conv]*5
            ))
            t, mu = param["t"], param["mu"]
            tsign = 1 if tJ_conv == 1 else -1
            hop = tsign * t * (
                gt.outer(Cpu, Cmu) - gt.outer(Cpd, Cmd)
                - gt.outer(Cmu, Cpu) + gt.outer(Cmd, Cpd)
            )
            """
            Chemical potential doping:
                (-mu / nbond)(n_i + n_j)
                =  (-mu / nbond)(nb_i + nb_j)
                (nb_i: spinon number at site i)
            """
            dope = (-mu / nbond) * (
                gt.outer(Nud, Id)
                + gt.outer(Id, Nud)
            )
        ham = gt.transpose(
            heis if tJ_conv == 0 
            else (hop + heis + dope), [0,2,1,3]
        )
 
    
    elif model == "tJ2":
        """
        t-t'-J1-J2 model

        Repeated counting
        - sites:        12 times
        - <ij> bonds:   4 times
        - <<ij>> bonds: 2 times
        """
        tJ_conv = param["tJ_convention"]
        Sp, Sm, Sz, Nud, Id = tuple(map(
            makeops_tJ, ["Sp", "Sm", "Sz", "Nud", "Id"], [tJ_conv]*5
        ))
        # ---- each term in the 3-site Hamiltonian ----
        J, J2 = param["J"], param["J2"]
        # nearest neighbor interaction
        heis = (J / 4) * (
            (1/2) * (
                gt.outer(Sp, Sm, Id) + gt.outer(Sm, Sp, Id)
                + gt.outer(Id, Sp, Sm) + gt.outer(Id, Sm, Sp)
            ) + gt.outer(Sz, Sz, Id) + gt.outer(Id, Sz, Sz)
            - (
                gt.outer(Nud, Nud, Id) + gt.outer(Id, Nud, Nud)
            ) / 4
        )
        # 2nd neighbor interaction
        heis += (J2 / 2) * (
            (1/2) * (
                gt.outer(Sp, Id, Sm) + gt.outer(Sm, Id, Sp)
            ) + gt.outer(Sz, Id, Sz)
            - gt.outer(Nud, Id, Nud) / 4
        )
        # t and mu terms related to un-occupied states
        # valid only at finite doping
        if tJ_conv != 0:
            Cpu, Cmu, Cpd, Cmd = tuple(map(
                makeops_tJ, ["Cpu", "Cmu", "Cpd", "Cmd"], [tJ_conv]*5
            ))
            t, t2, mu = param["t"], param["t2"], param["mu"]
            tsign = 1 if tJ_conv == 1 else -1
            # nearest neighbor hopping
            hop = tsign * (t / 4) * (
                gt.outer(Cpu, Cmu, Id) + gt.outer(Cpd, Cmd, Id)
                + gt.outer(Id, Cpu, Cmu) + gt.outer(Id, Cpd, Cmd)
                - gt.outer(Cmu, Cpu, Id) - gt.outer(Cmd, Cpd, Id)
                - gt.outer(Id, Cmu, Cpu) - gt.outer(Id, Cmd, Cpd)
            )
            # 2nd neighbor hopping
            hop = tsign * (t2 / 2) * (
                gt.outer(Cpu, Id, Cmu) + gt.outer(Cpd, Id, Cmd)
                - gt.outer(Cmu, Id, Cpu) - gt.outer(Cmd, Id, Cpd)
            )
            # chemical potential terms
            dope = (-mu / 12) * (
                gt.outer(Nud, Id, Id)
                + gt.outer(Id, Nud, Id)
                + gt.outer(Id, Id, Nud)
            )
            raise NotImplementedError
        ham = gt.transpose(
            heis if tJ_conv == 0 
            else (hop + heis + dope), [0,2,4,1,3,5]
        )
    
    else:
        raise ValueError("unrecognized physics model")
    
    # exponentiation exp(-beta * H)
    if expo is False:
        return ham
    else:
        eps = param["eps"]
        # 4- or 6-axis gate
        nsite = ham.ndim // 2
        shapeTmp = ham.shape
        tmp = gt.merge_axes(ham, (nsite,nsite), order=(1,-1))
        gate = gla.expm(-eps * tmp)
        gate = gt.split_axes(gate, (nsite,nsite), shapeTmp, order=(1,-1))
        return gate
