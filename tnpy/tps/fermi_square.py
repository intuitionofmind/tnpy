from copy import deepcopy
import itertools
import math
import numpy as np
import opt_einsum as oe
import pickle as pk

import torch
torch.set_default_dtype(torch.float64)
# import torch.nn.functional as tnf

import tnpy
from tnpy import Z2gTensor, GTensor 

class FermiSquareTPS(object):
    r'''
    class of a infinite fermionic tensor product state on a square lattice
    '''

    def __init__(self, nx: int, ny: int, site_tensors: dict, link_tensors: dict, info=None):
        r'''

        Parameters
        ----------
        site_tensors: dict, {(x, y): GTensor}, site tensors, NOT necessarily sorted
        link_tensors: dict, {(x, y): tuple[GTensor]}, link tensors along X- and Y-direction
        '''

        self._nx, self._ny = nx, ny
        self._size = nx*ny

        # sorted by the key/coordinate (x, y), firstly by y, then by x
        self._site_tensors = dict(sorted(site_tensors.items(), key=lambda x: (x[0][1], x[0][0])))
        self._link_tensors = dict(sorted(link_tensors.items(), key=lambda x: (x[0][1], x[0][0])))

        assert len(self._site_tensors) == self._size, 'number of site tensors not consistant with the size'
        self._coords = tuple(self._site_tensors.keys())

        self._info = info

    @classmethod
    def rand(cls, nx: int, ny: int, chi: int, cflag=False):
        r'''
        generate a random UC4FermionicSquareTPS

        Parameters
        ----------
        chi: int, bond dimension of the site tensor
            if even, dimensions of even parity and odd parity sectors are: both chi//2
            if odd, dimensions of even parity and odd parity sectors are: chi//2+1 and chi//2
        '''

        # rank-5 site-tensor
        # the index order convention
        # clockwisely, starting from the WEST:
        #       1 4
        #       |/
        # T: 0--*--2
        #       |
        #       3
        # diagonal link-tensors living on the link appproximate the environment
        # number of link-tensors is 2*self._uc_size
        # arrow, site-tensor, link-tensor conventions as below:
        #   v   v   v   v
        #   |   |   |   |
        #  -A-<-B-<-A-<-B-<
        #   |5  |7  |5  |
        #   v   v   v   v
        #   | 4 | 6 | 4 |
        #  -C-<-D-<-C-<-D-<
        #   |1  |3  |1  |3
        #   v   v   v   v
        #   | 0 | 2 | 0 |
        #  -A-<-B-<-A-<-B-<
        #   |   |   |   |
        # dual for A, B, C, D
        # ALL physical bonds are assumed to be outgoing, 0, vector space
        site_dual = (0, 1, 1, 0, 0)
        link_dual = (0, 1)
        # coordinates of the unit cell
        dim_o = chi // 2
        dim_e = chi-dim_o
        site_shape = [(dim_e, dim_o)]*4
        site_shape.append((1, 1))
        site_tensors, link_tensors = {}, {}
        for y, x in itertools.product(range(ny), range(nx)):
            s = x, y
            temp = GTensor.rand(dual=site_dual, shape=site_shape, cflag=cflag)
            site_tensors[s] = (1.0/temp.max())*temp
            temp_x = GTensor.rand_diag(dual=link_dual, dims=(dim_e, dim_o))
            temp_y = GTensor.rand_diag(dual=link_dual, dims=(dim_e, dim_o))
            link_tensors[s] = [(1.0/temp_x.max())*temp_x, (1.0/temp_y.max())*temp_y]

        return cls(nx, ny, site_tensors, link_tensors)

    @classmethod
    def initialize_from_instance(cls, ins, chi: int):
        r'''
        immport a UC4FermionicSquareTPS

        Parameters
        ----------
        ins: another instance with 
        '''

        assert chi >= ins._chi, 'GTensor must be enlarged'
        dim_diff = chi-ins.chi
        
        def _pad_site_tensor(gt: GTensor, dim: int):

            new_blocks = {}
            for k, v in gt.blocks().items():
                new_blocks[k] = tnf.pad(input=v, pad=(0, 0, 0, dim, 0, dim, 0, dim, 0, dim), mode='constant', value=0.0)

            return GTensor(dual=gt.dual, blocks=new_blocks)

        def _pad_link_tensor(gt: GTensor, dim: int):

            new_blocks = {}
            for k, v in gt.blocks().items():
                new_blocks[k] = tnf.pad(input=v, pad=(0, 1, 0, 1), mode='constant', value=1.0)

            return GTensor(dual=gt.dual, blocks=new_blocks)

        sts, lts = ins.tensors()
        site_tensors, link_tensors = {}, {}
        for s in ins.coords:
            site_tensors[s] = _pad_site_tensor(sts[s], dim_diff)
            link_tensors[s] = [_pad_link_tensor(lts[s][0], dim_diff), _pad_link_tensor(lts[s][1], dim_diff)]

        return cls(site_tensors, link_tensors)

    @property
    def size(self):
        return self._size

    @property
    def coords(self):
        return self._coords

    def site_tensors(self) -> dict:
        return self._site_tensors

    def link_tensors(self) -> dict:
        return self._link_tensors

    def coords(self) -> tuple:
        return self._coords

    def site_envs(self, site: tuple) -> list:
        r'''
        return the environment bond weights around a site
        '''

        envs = []
        envs.append(self._link_tensors[((site[0]-1) % self._nx, site[1])][0])
        envs.append(self._link_tensors[site][1])
        envs.append(self._link_tensors[site][0])
        envs.append(self._link_tensors[(site[0], (site[1]-1) % self._ny)][1])

        return envs

    def sqrt_env(self, env: GTensor) -> GTensor:
        r'''
        compute the square root of a bond weight Gensor
        env: GTensor
        '''

        if (0, 1) == env.dual:
            e = torch.sqrt(env.blocks()[(0, 0)].diag())
            o = torch.sqrt(env.blocks()[(1, 1)].diag())
            new_blocks = {(0, 0): e.diag(), (1, 1): o.diag()}
            cflag = env.cflag
        elif (1, 0) == env.dual:
            e = torch.sqrt(env.blocks()[(0, 0)].diag())
            o = torch.sqrt(env.blocks()[(1, 1)].diag())
            new_blocks = {(0, 0): e.diag(), (1, 1): 1.j*o.diag()}
            cflag = True

        return GTensor(dual=env.dual, shape=env.shape, blocks=new_blocks, cflag=cflag)

    def merged_tensors(self) -> dict:
        r'''
        build the tensors merged with square root of the environments

        Returns
        -------
        mgts: dict, {site: merged tensor}
        '''

        mgts = {}
        for c in self._coords:
            gt = self._site_tensors[c]
            envs = self.site_envs(c)
            half_envs = [self.sqrt_env(t) for t in envs]
            # merge
            mgts[c] = pytn.gcontract('abcde,fa,bg,ch,id->fghie', gt, *half_envs)

        return mgts
