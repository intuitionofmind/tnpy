import opt_einsum as oe
from copy import deepcopy
import itertools
import math

import torch
torch.set_default_dtype(torch.float64)

import tnpy as tp

class SquareXYZ(object):
    r'''
    class of XYZ spin model on a square lattice
    '''

    def __init__(self, Jx: float, Jy: float, Jz: float, cflag=False):
        r'''initialization

        Parmaeters
        ----------
        Jx,y,z: double, coupling constants
        '''

        self._dim_phys = 2
        self._Jx, self._Jy, self._Jz = Jx, Jy, Jz
        self._cflag = cflag

        # identity and spin operators
        self._id = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]])

        self._sx = 0.5*torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0]])

        self._sy = torch.tensor([
            [0.0, -1.j],
            [1.j, 0.0]]).cdouble()

        self._sz = 0.5*torch.tensor([
            [1.0, 0.0],
            [0.0, -1.0]])

        self._sp = torch.tensor([
            [0.0, 1.0],
            [0.0, 0.0]])

        self._sm = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0]])

    @property
    def sx(self):

        return self._sx

    @property
    def sy(self):

        return self._sy

    @property
    def sz(self):

        return self._sz

    def twobody_ham(self):
        r'''
        two-body Hamiltonian operator

        Returns
        -------
        ham, tensor, with the bond order:
        0   1
        |   |
        *---*
        |   |
        2   3
        '''

        ham_shape = [self._dim_phys]*4
        ham = torch.zeros(ham_shape)

        # Kron product is the matrix representation of tensor product
        # 0   2
        # |   |
        # * x *
        # |   |
        # 1   3
        ham += self._Jz*torch.kron(self._sz, self._sz).reshape(ham_shape).permute(0, 2, 1, 3)

        ham += 0.5*(self._Jx-self._Jy)*torch.kron(self._sp, self._sp).reshape(ham_shape).permute(0, 2, 1, 3)
        ham += 0.5*(self._Jx-self._Jy)*torch.kron(self._sm, self._sm).reshape(ham_shape).permute(0, 2, 1, 3)

        ham += 0.5*(self._Jx+self._Jy)*torch.kron(self._sp, self._sm).reshape(ham_shape).permute(0, 2, 1, 3)
        ham += 0.5*(self._Jx+self._Jy)*torch.kron(self._sm, self._sp).reshape(ham_shape).permute(0, 2, 1, 3)

        if self._cflag:
            return ham.cdouble()
        else:
            return ham

    def twobody_img_time_evo(self, ham: torch.tensor, delta: float):
        r'''
        two-body time evolution operator

        Parmaeters
        ----------
        ham: tensor, two-body Hamiltonian gate
        delta: float, time step size
        '''

        ham_mat = ham.reshape(self._dim_phys**2, self._dim_phys**2)

        return torch.linalg.matrix_exp(-delta*ham_mat).reshape(ham.shape)
