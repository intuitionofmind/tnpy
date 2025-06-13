"""
One-site operators
"""

import gtensor as gt
import torch
torch.set_default_dtype(torch.float64)
from fermiT.conversion import gt2ft

def getIdx(namedict: dict[str], instate: str, outstate: str):
    """
    State index to tensor index conversion
    
    If
        instate -> (g, n)
        outstate -> (g', n')
    
    the corresponding element is at
        `T.blocks[(g',g)][n',n]`

    Parameters
    ----
    namedict: dict
        state name - index dictionary
    
    instate, outstate: str
        state name

    Returns
    ----
    (g',g) , (n',n)
    """
    inIdx = namedict[instate]
    outIdx = namedict[outstate]
    return (outIdx[0], inIdx[0]), (outIdx[1], inIdx[1])

def iden1(namedict: dict[str], shape):
    """One-site identity"""
    if shape is None:
        # will implement "shape from namedict" in the future
        raise NotImplementedError
    iden = gt.zeros(shape, (0,1))
    for state in namedict:
        gs, ns = getIdx(namedict, state, state)
        iden.blocks[gs][ns] = 1
    return iden

# ---- spinless fermion ----

def makeops_tV(name: str):
    """
    Create one site operators for spinless fermion
    """
    namedict = {
        'f': (1,0), # filled
        'e': (0,0)  # empty
    }
    opshape = (1,1)
    
    # identity operator
    if name == "Id":
        op = iden1(namedict, opshape)
    
    # Cp, Cm operators
    elif name == "Cp":
        ## from empty to filled
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'e', 'f')
        op.blocks[gs][ns] = 1
    elif name == "Cm":
        ## from filled to empty
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'f', 'e')
        op.blocks[gs][ns] = 1

    # number operator
    elif name == "Num":
        ## from filled to filled
        op = gt.zeros(opshape, (0,1), 0)
        gs, ns = getIdx(namedict, 'f', 'f')
        op.blocks[gs][ns] = 1

    else:
        raise ValueError("Unrecognized operator name")

    op.verify()
    return op

# ---- spin-1/2 (no double occupancy) ----

def get_tJconv(Dphy: tuple[int, int]):
    """Determine tJ convention from physical index dimension"""
    assert len(Dphy) == 2
    tJ_conv = (
        0 if Dphy == (2,0)
        else 1 if Dphy == (2,1)
        else 2 if Dphy == (1,2)
        else 3 # Dphy = (2,2)
    )
    return tJ_conv

def makeops_tJ(name: str, tJ_conv: int):
    """
    Create one-site operators for t-J or Heisenberg model
    """
    if tJ_conv == 0:
        # Heisenberg model at half filling
        # Schwinger boson formalism, no odd-parity channel
        namedict = {
            'u': (0,0),     # spin up
            'd': (0,1),     # spin down
        }
        opshape = ((2,2), (0,0))
        spin_gIdx = (0,0)
    elif tJ_conv == 1:
        # slave fermion formalism
        # holon as fermion, spinon as boson
        namedict = {
            'u': (0,0),     # spin up
            'd': (0,1),     # spin down
            'h': (1,0)      # hole
        }
        opshape = ((2,2), (1,1))
        spin_gIdx = (0,0)
    elif tJ_conv == 2:
        # slave boson formalism
        # holon as boson, spinon as fermion
        namedict = {
            'u': (1,0),     # spin up
            'd': (1,1),     # spin down
            'h': (0,0),     # hole
        }
        opshape = ((1,1), (2,2))
        spin_gIdx = (1,1)
    elif tJ_conv == 3:
        # slave boson formalism
        # holon as boson, spinon as fermion
        namedict = {
            'u': (1,0),     # spin up
            'd': (1,1),     # spin down
            'h': (0,0),     # hole
            'e': (0,1)      # double-occupancy (unphysical)
        }
        opshape = ((2,2), (2,2))
        spin_gIdx = (1,1)
    else:
        raise ValueError("Unrecognized tJ convention")
    
    # identity operator
    if name == "Id":
        op = iden1(namedict, opshape)
        # in convention 3, 
        # the double occupancy element should be removed
        if tJ_conv == 3:
            gs, ns = getIdx(namedict, 'e', 'e')
            op.blocks[gs][ns] = 0.0

    # projection from convention 3 to 2
    # (remove double occupancy)
    elif name == "Pg":
        # (1+2) <- (2+2)
        op = gt.zeros(((1,2), (2,2)), (0,1), 0)
        # hole <- hole
        op.blocks[(0,0)][(0,0)] = 1.0
        # spin up/down <- spin up/down
        op.blocks[(1,1)][(0,0)] = 1.0
        op.blocks[(1,1)][(1,1)] = 1.0

    # projection for convention 3
    # elements involving double occupancy 
    # is set to 0, but not removed
    elif name == "Pg2":
        # (2+2) <- (2+2)
        op = gt.zeros(((2,2), (2,2)), (0,1), 0)
        # hole <- hole
        op.blocks[(0,0)][(0,0)] = 1.0
        # spin up/down <- spin up/down
        op.blocks[(1,1)][(0,0)] = 1.0
        op.blocks[(1,1)][(1,1)] = 1.0

    # Cp, Cm operators
    elif name == "Cpu":
        assert tJ_conv != 0
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'h', 'u')
        op.blocks[gs][ns] = 1
    elif name == "Cpd":
        assert tJ_conv != 0
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'h', 'd') 
        op.blocks[gs][ns] = 1
    elif name == "Cmu":
        assert tJ_conv != 0
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'u', 'h') 
        op.blocks[gs][ns] = 1
    elif name == "Cmd":
        assert tJ_conv != 0
        op = gt.zeros(opshape, (0,1), 1)
        gs, ns = getIdx(namedict, 'd', 'h') 
        op.blocks[gs][ns] = 1

    # holon number operator
    elif name == "Nh":
        op = gt.zeros(opshape, (0,1), 0)
        if tJ_conv != 0:
            gs, ns = getIdx(namedict, 'h', 'h') 
            op.blocks[gs][ns] = 1

    # spinon number (Nu, Nd) operators
    # restricted to no-double-occupancy subspace
    elif name == "Nu":
        op = gt.zeros(opshape, (0,1), 0)
        gs, ns = getIdx(namedict, 'u', 'u') 
        op.blocks[gs][ns] = 1
    elif name == "Nd":
        op = gt.zeros(opshape, (0,1), 0)
        gs, ns = getIdx(namedict, 'd', 'd') 
        op.blocks[gs][ns] = 1
    elif name == "Nud":
        op = gt.zeros(opshape, (0,1), 0)
        gs, ns = getIdx(namedict, 'u', 'u') 
        op.blocks[gs][ns] = 1
        gs, ns = getIdx(namedict, 'd', 'd') 
        op.blocks[gs][ns] = 1

    # Spin operators
    elif name == "Sz":
        op = gt.zeros(opshape, (0,1), 0)
        op.blocks[spin_gIdx] = torch.tensor([[1.0,0.0], [0.0,-1.0]], dtype=torch.cdouble) / 2
    elif name == "Sx":
        op = gt.zeros(opshape, (0,1), 0)
        op.blocks[spin_gIdx] = torch.tensor([[0.0,1.0], [1.0,0.0]], dtype=torch.cdouble) / 2
    elif name == "Sy":
        op = gt.zeros(opshape, (0,1), 0)
        op.blocks[spin_gIdx] = torch.tensor([[0,-1j], [1j,0]], dtype=torch.cdouble) / 2
    elif name == "Sp":
        op = gt.zeros(opshape, (0,1), 0)
        op.blocks[spin_gIdx] = torch.tensor([[0.0,1.0], [0.0,0.0]], dtype=torch.cdouble)
    elif name == "Sm":
        op = gt.zeros(opshape, (0,1), 0)
        op.blocks[spin_gIdx] = torch.tensor([[0.0,0.0], [1.0,0.0]], dtype=torch.cdouble)

    else:
        raise ValueError("Unrecognized operator name")

    op.verify()
    return op


def makeops(name: str, model: str, **kwargs):
    if 'tV' in model:
        return makeops_tV(name)
    elif 'tJ' in model:
        return makeops_tJ(name, kwargs["tJ_conv"])
    else:
        raise ValueError("Unrecognized model")
    
    
def makeops_tJft(name: str, tJ_conv: int):
    """FermiT wrapper of `makeops_tJ`"""
    return gt2ft(makeops_tJ(name, tJ_conv))


def makeops_tVft(name: str):
    """FermiT wrapper of `makeops_tV`"""
    return gt2ft(makeops_tV(name))


def makeops_ft(name: str, model: str, **kwargs):
    return gt2ft(makeops(name, model, **kwargs))
