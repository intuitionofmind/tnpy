import gtensor as gt
from gtensor import GTensor
import torch
from torch import Tensor
torch.set_default_dtype(torch.float64)


def ising(beta: Tensor):
    """
    Tensor representation of classical 2D Ising model partition function
    """
    t = torch.ones((2,2,2,2), dtype=torch.cdouble)
    t[0,1,0,1] = torch.exp(-4 * beta)
    t[1,0,1,0] = torch.exp(-4 * beta)
    t[0,0,0,0] = torch.exp(4 * beta)
    t[1,1,1,1] = torch.exp(4 * beta)
    t_ = gt.GTensor(
        ((2,)*4, (0,)*4), (1,1,0,0),
        blocks={(0,0,0,0): t}
    )
    return [[t_]]


def mag(beta: Tensor, ctms: list[list[list[GTensor]]]):
    """
    Mea
    """
    pass
