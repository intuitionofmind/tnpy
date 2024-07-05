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

    def __init__(self, Jx: float, Jy: float, Jz: float):
        r'''initialization

        Parmaeters
        ----------
        Jx,y,z: double, coupling constants
        '''

        self._dim_phys = 2
        self._Js = Jx, Jy, Jz

        # identity and spin operators
        # view as complex128
        self._ops = [
            torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0]]).cdouble(),

            0.5*torch.tensor([
                [0.0, 1.0],
                [1.0, 0.0]]).cdouble(),

            0.5*torch.tensor([
                [0.0, -1.j],
                [1.j, 0.0]]).cdouble(),

            0.5*torch.tensor([
                [1.0, 0.0],
                [0.0, -1.0]]).cdouble(),
            ]

    @property
    def sx(self):

        return self._ops[1]

    @property
    def sy(self):

        return self._ops[2]

    @property
    def sz(self):

        return self._ops[3]

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

        for i, J in enumerate(self._Js):
            temp = J*torch.kron(self._ops[i+1], self._ops[i+1]).reshape(ham_shape).permute(0, 2, 1, 3)
            ham = ham+temp

        return ham

    def twobody_time_evo(self, ham: torch.tensor, delta: float):
        r'''
        two-body time evolution operator
        '''

        ham_mat = ham.reshape(self._dim_phys**2, self._dim_phys**2)

        return torch.linalg.matrix_exp(-delta*ham_mat).reshape(ham.shape)

class SquareTFIM(object):
    r'''
    class of transverse field Isimg model on a square lattice
    '''

    def __init__(self, J: float, h: float, cflag=False):
        r'''initialization

        Parmaeters
        ----------
        J: float, spin-z coupling
        h: float, transverse field
        '''

        self._J = J
        self._h = h

        # identity and sigma-x, sigma-z operators
        self._ops = [
            torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0]]),

            torch.tensor([
                [0.0, 1.0],
                [1.0, 0.0]]),

            torch.tensor([
                [1.0, 0.0],
                [0.0, -1.0]]),
            ]

        if cflag:
            self._ops = [t.cdouble() for t in self._ops]

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

        ham_shape = [2]*4

        # Kron product is the matrix representation of tensor product
        # 0   2
        # |   |
        # * x *
        # |   |
        # 1   3

        # sigma-z term
        ham = -1.0*self._J*torch.kron(self._ops[2], self._ops[2]).reshape(ham_shape).permute(0, 2, 1, 3)
        # sigma-x term
        ham += -0.25*self._h*torch.kron(self._ops[1], self._ops[1]).reshape(ham_shape).permute(0, 2, 1, 3)

        return ham
