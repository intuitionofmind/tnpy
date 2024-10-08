from copy import deepcopy
import itertools
import math
import numpy as np
import scipy
import opt_einsum as oe
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




class QuantumSquareCTMRG(object):
    r'''class of CTMRG method on a square lattice for a quantum wavefunction'''
    def __init__(
            self,
            wfs: dict,
            rho: int,
            dtype=torch.float64):
        r'''
        Parameters
        ----------
        wfs: dict, dict of wavefunction tensors
        rho: int, bond dimension of ctm tensors

        # CTM tensors:
        # ctm_names: {C0, C1, C2, C3, Ed, Eu, El, Er}
        #  C2  Eu  C3
        #   *--*--*
        #   |  |  |
        # El*--*--*Er
        #   |  |  |
        #   *--*--*
        #  C0  Ed  C1

        that is, effectively, each site is placed by NINE tensors: 1 double tensor + 8 CTM tensors
        '''
        self._dtype = dtype
        # sorted by the key/coordinate (x, y), firstly by y, then by x
        # self._wfs = dict(sorted(wfs.items(), key=lambda x: (x[0][1], x[0][0])))
        self._wfs = wfs
        self._coords = tuple(self._wfs.keys())
        self._size = len(self._coords)
        # sites along two directions
        xs = [c[0] for c in self._coords]
        ys = [c[1] for c in self._coords]
        # remove duplicated items
        xs = list(dict.fromkeys(xs))
        ys = list(dict.fromkeys(ys))
        self._nx, self._ny = len(xs), len(ys)
        # inner bond dimension
        self._chi = self._dts[(0, 0)].shape[0]

        # double tensors
        #       2 3
        #       | |
        #    0--***--4
        #    1--***--5
        #       | |
        #       6 7
        self._dts = {}
        for c in self._coords:
            self._dts.update(
                    {c: torch.einsum(
                        'ABCDe,abcde->AaBbCcDd',
                        wfs[c].conj(), wfs[c])})

        # CTMRG environment tensors
        self._ctm_names = 'C0', 'C1', 'C2', 'C3', 'Ed', 'Eu', 'El', 'Er'
        temp = {}
        for i, n in enumerate(self._ctm_names):
            # generate corner and edge tensors
            if i < 4:
                temp.update({n: torch.rand(rho, rho).to(dtype)})
            else:
                temp.update({n: torch.rand(rho, rho, self._chi, self._chi).to(dtype)})
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


    def measure_onebody(
            self,
            op: torch.tensor):
        r'''measure onebody operator

        Parameters:
        ----------
        wf: torch.tensor, wavefunction as a unified tensor
        op: torch.tensor, operator to be measured
        '''
        res = []
        for i, c in enumerate(self._coords):
            # left and right environments
            env_l = torch.einsum(
                    'ab,bcde,fc->adef',
                    self._ctms[((c[0]-1) % self._nx, (c[1]-1) % self._ny)]['C0'],
                    self._ctms[((c[0]-1) % self._nx, c[1])]['El'],
                    self._ctms[((c[0]-1) % self._nx, (c[1]+1) % self._ny)]['C2'])
            env_r = torch.einsum(
                    'ab,bcde,fc->adef',
                    self._ctms[((c[0]+1) % self._nx, (c[1]-1) % self._ny)]['C1'],
                    self._ctms[((c[0]+1) % self._nx, c[1])]['Er'],
                    self._ctms[((c[0]+1) % self._nx, (c[1]+1) % self._ny)]['C3'])
            # denominator
            temp = env_l.clone()
            temp = torch.einsum(
                    'eAag,efDd,AaBbCcDd,ghBb->fCch',
                    temp,
                    self._ctms[(c[0], (c[1]-1) % self._ny)]['Ed'],
                    self._dts[c],
                    self._ctms[(c[0], (c[1]+1) % self._ny)]['Eu'])
            den = torch.einsum('abcd,abcd', temp, env_r)
            # numerator
            impure_dt = torch.einsum('ABCDE,Ee,abcde->AaBbCcDd', wf[i].conj(), op, wf[i])
            temp = env_l.clone()
            temp = torch.einsum(
                    'eAag,efDd,AaBbCcDd,ghBb->fCch',
                    temp,
                    self._ctms[(c[0], (c[1]-1) % self._ny)]['Ed'],
                    impure_dt,
                    self._ctms[(c[0], (c[1]+1) % self._ny)]['Eu'])
            num = torch.einsum('abcd,abcd', temp, env_r)
            res.append(num / den)

        return torch.as_tensor(res)

    def measure_twobody(self):
        pass
