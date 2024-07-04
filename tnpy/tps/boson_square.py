from copy import deepcopy
import itertools
import math
import numpy as np
import scipy
import opt_einsum as oe
import pickle as pk

import torch
torch.set_default_dtype(torch.float64)

import torch.nn.functional as tnf
torch.set_printoptions(precision=5)

import tnpy as tp

class SquareTPS(object):
    r'''
    class of bosonic tensor product states on a square lattice
    '''

    def __init__(self, site_tensors: dict, link_tensors: dict, dim_phys=2):
        r'''initialization

        Parameters
        ----------
        '''

        # for spin-1/2
        self._dim_phys = dim_phys
        # rank-5 site-tensor
        # the index order convention
        # clockwisely, starting from the WEST:
        #       1 4
        #       |/
        # T: 0--*--2
        #       |
        #       3

        # diagonal bond matrix living on the link appproximates the environment
        # the number of bond matrices is 2*self._uc_size
        # order conventions for site-tensors & bond-matrices as below:
        #   |5  |7  |5  |7
        #  -C-4-D-6-C-4-D-6
        #   |1  |3  |1  |3
        #  -A-0-B-2-A-0-B-2
        #   |5  |7  |5  |7
        #  -C-4-D-6-C-4-D-6
        #   |1  |3  |1  |3
        #  -A-0-B-2-A-0-B-2
        #   |   |   |   |

        # sorted by the key/coordinate (x, y), firstly by y, then by x
        self._site_tensors = dict(sorted(site_tensors.items(), key=lambda x: (x[0][1], x[0][0])))
        self._link_tensors = dict(sorted(link_tensors.items(), key=lambda x: (x[0][1], x[0][0])))

        self._coords = tuple(self._site_tensors.keys())
        self._size = len(self._coords)

        # sites along two directions
        xs = [c[0] for c in self._coords]
        ys = [c[1] for c in self._coords]

        # remove duplicated items
        xs = list(dict.fromkeys(xs))
        ys = list(dict.fromkeys(ys))
        self._nx, self._ny = len(xs), len(ys)

        # inner bond dimension
        self._chi = self._site_tensors[(0, 0)].shape[0]

    @classmethod
    def rand(cls, nx: int, ny: int, chi: int, cflag=False):
        r'''
        generate a random UC4FermionicSquareTPS

        Parameters
        ----------
        nx: int, number of sites along x-direction in a unit cell
        ny: int, number of sites along y-direction in a unit cell
        chi: int, bond dimension of the site tensor
        '''

        site_shape = (chi, chi, chi, chi, 2)
        site_tensors, link_tensors = {}, {}

        for x, y in itertools.product(range(nx), range(ny)):
            temp = torch.rand(site_shape)
            lam_x = torch.rand(chi).diag()
            lam_y = torch.rand(chi).diag()

            if cflag:
                temp = temp.cdouble()
                lam_x = lam_x.cdouble()
                lam_y = lam_y.cdouble()

            # normalization
            site_tensors[(x, y)] = temp/torch.linalg.norm(temp)
            link_tensors[(x, y)] = [lam_x/torch.linalg.norm(lam_x), lam_y/torch.linalg.norm(lam_y)]

        return cls(site_tensors, link_tensors)

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

    def site_tensors(self):

        return self._site_tensors

    def link_tensors(self):

        return self._link_tensors

    def site_envs(self, site: tuple, inner_bonds=None) -> list:
        r'''
        return the environment bond weights around a site

        Parameters
        ----------
        site: tuple, coordinate
        inner_bonds: tuple, optional, the inner bonds will be returned by square root of tensors
        '''

        envs = []
        envs.append(self._link_tensors[((site[0]-1) % self._nx, site[1])][0])
        envs.append(self._link_tensors[site][1])
        envs.append(self._link_tensors[site][0])
        envs.append(self._link_tensors[(site[0], (site[1]-1) % self._ny)][1])

        if inner_bonds is not None:
            for j in range(4):
                if j in inner_bonds:
                    envs[j] = torch.sqrt(envs[j])

        return envs

    def merged_tensor(self, site):
        r'''
        return site tensor merged with square root of link tensors around
        '''

        envs = self.site_envs(site, inner_bonds=(0, 1, 2, 3))

        return tp.contract(
                'abcde,Aa,bB,cC,Dd->ABCDe',
                self._site_tensors[site], *envs)
