from copy import deepcopy
import itertools
import math
import numpy as np
import scipy
import pickle as pk

import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5)

import tnpy as tp

class ClassicalSquareCTMRG(object):
    r'''class of CTMRG on a square lattice for a classcial model'''
    def __init__(
            self,
            ts: dict,
            rho: int,
            dtype=torch.float64):
        r'''
        Parameters
        ----------
        t:
        '''
        self._dtype = dtype
        self._ts = ts

        self._coords = tuple(self._ts.keys())
        self._size = len(self._coords)
        # sites along two directions
        xs = [c[0] for c in self._coords]
        ys = [c[1] for c in self._coords]
        # remove duplicated items
        xs = list(dict.fromkeys(xs))
        ys = list(dict.fromkeys(ys))
        self._nx, self._ny = len(xs), len(ys)
        # inner bond dimension
        self._chi = self._ts[(0, 0)].shape[0]

        # CTMRG environment tensors
        self._ctm_names = 'C0', 'C1', 'C2', 'C3', 'Ed', 'Eu', 'El', 'Er'
        temp = {}
        for i, n in enumerate(self._ctm_names):
            # generate corner and edge tensors
            if i < 4:
                temp.update({n: torch.rand(rho, rho).to(dtype)})
            else:
                temp.update({n: torch.rand(rho, rho, self._chi, self._chi).to(dtype)})
        # every site is associted with a set of CTM tensors
        for c in self._coords:
            self._ctms.update({c: temp})
        self._rho = rho


    @property
    def coords(self):
        return self._coords


    @property
    def size(self):
        return self._size


    @property
    def nx(self):
        return self._nx


    @property
    def ny(self):
        return self._ny


    @property
    def bond_dim(self):
        return self._chi


    def double_tensors(self):
        return deepcopy(self._dts)


    def update_ctms(
            self,
            ctms: dict):
        r'''update to new CTM tensors'''
        for c, ctm in ctms.items():
            assert c in self._coords, 'Coordinate of new CTM tensors is not correct' 
            for k, v in ctm.items():
                if k not in self._ctm_names:
                    raise ValueError('Name of new CTM tensor is not valid')

        self._ctms = ctms
        self._rho = self._ctms[(0, 0)]['C0'].shape[0]

        return 1
