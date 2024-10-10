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
    r'''class of CTMRG on a square lattice for a classcial model with arbitrary unit cell'''
    def __init__(
            self,
            ts: dict,
            rho: int,
            dtype=torch.float64):
        r'''
        Parameters
        ----------
        ts: dict, dict of rank-4 site tensors, key: coordinate
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
        # convention
        # C:
        # |1
        # *--0, x-direction: 0, y-direction: 1
        # E:
        #    |2
        # 0--*--1, MPS bonds (down to up; left to right): 0, 1; inner bond: 2
        self._ctm_names = 'C0', 'C1', 'C2', 'C3', 'Ed', 'Eu', 'El', 'Er'
        temp = {}
        for i, n in enumerate(self._ctm_names):
            # generate corner and edge tensors
            if i < 4:
                temp.update({n: torch.rand(rho, rho).to(dtype)})
            else:
                temp.update({n: torch.rand(rho, rho, self._chi).to(dtype)})
        # every site is associted with a set of CTM tensors
        self._ctms = {}
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


    def rg_projectors_u(
            self,
            c: tuple[int, int]):
        r'''build projectors for CTMRG up move

        Parameters
        ----------
        c: tuple, coordinate of anchoring point
        '''
        i, j = c
        # *--*--*--* MPS
        # |  |  |  |
        # *--*--*--* MPO
        # |  |  |  |
        mpo = [self._ts[((i+k) % self._nx, j)] for k in range(2)]
        mpo.insert(0, self._ctms[((i-1) % self._nx, j)]['El'])
        mpo.append(self._ctms[((i+2) % self._nx, j)]['Er'])
        jj = (j+1) % self._ny
        mps = [self._ctms[((i+k) % self._nx, jj)]['Eu'] for k in range(2)]
        mps.insert(0, self._ctms[((i-1) % self._nx, jj)]['C2'])
        mps.append(self._ctms[((i+2) % self._nx, jj)]['C3'])
        # MPO-MPS
        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('abc,db->dca', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abc,db->dca', mpo[-1], mps[-1])
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('abcd,efb->eafcd', mpo[k+1], mps[k+1])
        # left and right part
        # *-a-*--d,0
        # |   |
        # *-b-*--e,1
        # |c,2|f,3
        rho_l = torch.einsum('abc,abdef->decf', mpo_mps[0], mpo_mps[1])
        # a,0--*-c-*
        #      |   |
        # b,1--*-d-*
        #      |e,3|f,2
        rho_r = torch.einsum('abcde,cdf->abfe', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((2, 3), (0, 1)), qr_dims=(2, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((2, 3), (0, 1)), qr_dims=(0, 2))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abc,bcd->ad', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

        return pl, pr


    def rg_mu(self):
        r'''
        a CTMRG up step, merge the whole unit cell into up boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny-1, -1, -1), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            # build temporary MPO and MPS
            mpo = [self._ts[((i+k) % self._nx, j)] for k in range(self._nx)]
            mpo.insert(0, self._ctms[((i-1) % self._nx, j)]['El'])
            # (i+self._nx) % self._nx = i
            mpo.append(self._ctms[(i, j)]['Er'])
            jj = (j+1) % self._ny
            mps = [self._ctms[((i+k) % self._nx, jj)]['Eu'] for k in range(self._nx)]
            mps.insert(0, self._ctms[((i-1) % self._nx, jj)]['C2'])
            mps.append(self._ctms[(i, jj)]['C3'])
            # MPO-MPS
            mpo_mps = [None]*(self._nx+2)
            # *--d,0
            # |b
            # *--c,1
            # |a,2
            mpo_mps[0] = torch.einsum('abc,db->dca', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abc,db->dca', mpo[-1], mps[-1])
            # e,0--*--f,2
            #      |b
            # a,1--*--c,3
            #     |d,4
            for k in range(self._nx):
                mpo_mps[k+1] = torch.einsum('abcd,efb->eafcd', mpo[k+1], mps[k+1])
            # use projectors to compress
            mps = [None]*(self._nx+2)
            pl, pr = self.rg_projectors_u(c=((i-1) % self._nx, j))
            mps[0] = torch.einsum('abc,abd->dc', mpo_mps[0], pl)
            for k in range(self._nx):
                pl_prime, pr_prime = self.rg_projectors_u(c=((i+k) % self._nx, j))
                mps[k+1] = torch.einsum('abc,bcdef,deg->agf', pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            mps[-1] = torch.einsum('abc,bcd->ad', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C2'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C3'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._nx):
                self._ctms[((i+k) % self._nx, j)]['Eu'] = mps[k+1] / torch.linalg.norm(mps[k+1])

        return 1


    def rg_projectors_d(
            self,
            c: tuple[int, int]):
        i, j = c
        # |  |  |  |
        # *--*--*--* MPO
        # |  |  |  |
        # *--*--*--* MPS
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
        mpo[-1] = self._ctms[((i+2) % self._nx, j)]['Er']
        for k in range(2):
            mpo[k+1] = self._ts[((i+k) % self._nx, j)]
        jj = (j-1) % self._ny
        mps[0] = self._ctms[((i-1) % self._nx, jj)]['C0']
        mps[-1] = self._ctms[((i+2) % self._nx, jj)]['C2']
        for k in range(2):
            mps[k+1] = self._ctms[((i+k) % self._nx, jj)]['Ed']
        # |b
        # *--c,1
        # |a,2
        # *--d,0
        mpo_mps[0] = torch.einsum('abc,da->dcb', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abc,da->dcb', mpo[-1], mps[-1])
        #      |b,4
        # a,1--*--c,3
        #      |d
        # e,0--*--f,2
        for k in range(2):
            mpo_mps[k+1] = torch.einsum(
                    'abcd,efd->eafcb',
                    mpo[k+1], mps[k+1])
        # left and right part
        # |c,2|f,3
        # *-b-*--e,1
        # |   |
        # *-a-*--d,0
        rho_l = torch.einsum('abc,abdef->decf', mpo_mps[0], mpo_mps[1])
        #      |e,3|f,2
        # b,1--*-d-*
        #      |   |
        # a,0--*-c-*
        rho_r = torch.einsum('abcde,cdf->abfe', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((2, 3), (0, 1)), qr_dims=(2, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((2, 3), (0, 1)), qr_dims=(0, 2))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abc,bcd->ad', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

        return pl, pr


    def rg_md(self):
        r'''
        a CTMRG up step, merge the whole unit cell into up boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            mpo, mps, mpo_mps = [None]*(self._nx+2), [None]*(self._nx+2), [None]*(self._nx+2)
            mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
            # (i+self._nx) % self._nx = i
            mpo[-1] = self._ctms[(i, j)]['Er']
            for k in range(self._nx):
                mpo[k+1] = self._ts[((i+k) % self._nx, j)]
            jj = (j-1) % self._ny
            mps[0] = self._ctms[((i-1) % self._nx, jj)]['C0']
            mps[-1] = self._ctms[(i, jj)]['C2']
            for k in range(self._nx):
                mps[k+1] = self._ctms[((i+k) % self._nx, jj)]['Ed']
            # |b
            # *--c,1
            # |a,2
            # *--d,0
            mpo_mps[0] = torch.einsum('abc,da->dcb', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abc,da->dcb', mpo[-1], mps[-1])
            #      |b,4
            # a,1--*--c,3
            #      |d
            # e,0--*--f,2
            for k in range(self._nx):
                mpo_mps[k+1] = torch.einsum(
                        'abcd,efd->eafcb',
                        mpo[k+1], mps[k+1])
            # use projectors to compress
            mps = [None]*(self._nx+2)
            pl, pr = self.rg_projectors_d(c=((i-1) % self._nx, j))
            # |c,1
            # *--b--*
            # *     *--d,0
            # *--a--*
            mps[0] = torch.einsum('abc,abd->dc', mpo_mps[0], pl)
            #            |f,2
            #      *--c--*--e--*
            # a,0--*     *     *--g,1
            #      *--b--*--d--*
            for k in range(self._nx):
                pl_prime, pr_prime = self.rg_projectors_d(c=((i+k) % self._nx, j))
                mps[k+1] = torch.einsum(
                        'abc,bcdef,deg->agf',
                        pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            #            |d,1
            #      *--c--*
            # a,0--*     *
            #      *--b--*
            mps[-1] = torch.einsum('abc,bcd->ad', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C0'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C1'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._nx):
                self._ctms[((i+k) % self._nx, j)]['Ed'] = mps[k+1] / torch.linalg.norm(mps[k+1])

        return 1

