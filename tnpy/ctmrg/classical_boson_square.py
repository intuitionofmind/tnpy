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
        ts: dict, dict of rank-4 site tensors, {key: coordinate, value: tensor}
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
        # d,0--*-a-*
        #      |   |
        # e,1--*-b-*
        #      |f,3|c,2
        rho_r = torch.einsum('abc,deabf->decf', mpo_mps[3], mpo_mps[2])
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
            #      |d,4
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
        r''''''
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
        mps[-1] = self._ctms[((i+2) % self._nx, jj)]['C1']
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
            mpo_mps[k+1] = torch.einsum('abcd,efd->eafcb', mpo[k+1], mps[k+1])
        # left and right part
        # |c,2|f,3
        # *-b-*--e,1
        # |   |
        # *-a-*--d,0
        rho_l = torch.einsum('abc,abdef->decf', mpo_mps[0], mpo_mps[1])
        #      |f,3|c,2
        # e,1--*-b-*
        #      |   |
        # d,0--*-a-*
        rho_r = torch.einsum('abc,deabf->decf', mpo_mps[3], mpo_mps[2])
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
        a CTMRG down step, merge the whole unit cell into up boundary MPS
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
            mps[-1] = self._ctms[(i, jj)]['C1']
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
                mpo_mps[k+1] = torch.einsum('abcd,efd->eafcb', mpo[k+1], mps[k+1])
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
                mps[k+1] = torch.einsum('abc,bcdef,deg->agf', pr, mpo_mps[k+1], pl_prime)
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


    def rg_projectors_l(
            self,
            c: tuple[int, int]):
        r''''''
        i, j = c
        # *--*--
        # |  |  
        # *--*--
        # |  |
        # *--*--
        # |  |
        # *--*--
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
        mpo[-1] = self._ctms[(i, (j+2) % self._ny)]['Eu']
        for k in range(2):
            mpo[k+1] = self._ts[(i, (j+k) % self._ny)]
        ii = (i-1) % self._nx
        mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C0']
        mps[-1] = self._ctms[(ii, j)]['C2']
        for k in range(2):
            mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['El']
        # |d,1  |c,2
        # *--a--*--b,0
        mpo_mps[0] = torch.einsum('abc,ad->bdc', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abc,ad->bdc', mpo[-1], mps[-1])
        # |f,2  |b,3
        # *--a--*--c,4
        # |e,0  |d,1
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('abcd,efa->edfbc', mpo[k+1], mps[k+1])
        # left and right part:
        # 0  1
        # d  e
        # |  |
        # *--*--f,3
        # |b |c
        # *--*--a,2
        rho_l = torch.einsum('abc,bcdef->deaf', mpo_mps[0], mpo_mps[1])
        # *--*--a,2
        # |b |c
        # *--*--f,3
        # |d |e
        # 0  1
        rho_r = torch.einsum('abc,debcf->deaf', mpo_mps[3], mpo_mps[2])
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


    def rg_ml(self):
        r'''
        a CTMRG left step, merge the whole unit cell into up boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            mpo, mps, mpo_mps = [None]*(self._ny+2), [None]*(self._ny+2), [None]*(self._ny+2)
            mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
            # (j+self._ny) % self._ny = j
            mpo[-1] = self._ctms[(i, j)]['Eu']
            for k in range(self._ny):
                mpo[k+1] = self._ts[(i, (j+k) % self._ny)]
            ii = (i-1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C0']
            mps[-1] = self._ctms[(ii, j)]['C2']
            for k in range(self._ny):
                mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['El']
            # |d,1  |c,2
            # *--a--*--b,0
            mpo_mps[0] = torch.einsum('abc,ad->bdc', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abc,ad->bdc', mpo[-1], mps[-1])
            # |f,2  |b,3
            # *--a--*--c,4
            # |e,0  |d,1
            for k in range(self._ny):
                mpo_mps[k+1] = torch.einsum('abcd,efa->edfbc', mpo[k+1], mps[k+1])
            # use projectors to compress
            mps = [None]*(self._ny+2)
            pl, pr = self.rg_projectors_l(c=(i, (j-1) % self._ny))
            #   |d,1
            # *--*
            # |b |c
            # *--*--a,0
            mps[0] = torch.einsum('abc,bcd->ad', mpo_mps[0], pl)
            #   |g
            # *--*
            # |d |e
            # *--*--f,4
            # |b |c
            # *--*
            #  |a
            for k in range(self._nx):
                pl_prime, pr_prime = self.rg_projectors_l(c=((i+k) % self._nx, j))
                mps[k+1] = torch.einsum('abc,bcdef,deg->agf', pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            # *--*--d,0
            # |b |c
            # *--*
            #   |a,1
            mps[-1] = torch.einsum('abc,dbc->da', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C0'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C2'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._nx):
                self._ctms[((i+k) % self._nx, j)]['El'] = mps[k+1] / torch.linalg.norm(mps[k+1])

        return 1


    def rg_projectors_r(
            self,
            c: tuple[int, int]):
        r''''''
        i, j = c
        # --*--*
        #   |  |  
        # --*--*
        #   |  |
        # --*--*
        #   |  |
        # --*--*
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
        mpo[-1] = self._ctms[(i, (j+2) % self._ny)]['Eu']
        for k in range(2):
            mpo[k+1] = self._ts[(i, (j+k) % self._ny)]
        ii = (i+1) % self._nx
        mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C1']
        mps[-1] = self._ctms[(ii, (j+2) % self._ny)]['C3']
        for k in range(2):
            mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['Er']
        #      |c,2  |d,1
        # a,0--*--b--*
        mpo_mps[0] = torch.einsum('abc,bd->adc', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abc,bd->adc', mpo[-1], mps[-1])
        #    |b,3  |f,2
        # a--*--c--*
        #    |d,1  |e,0
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('abcd,efc->edfba', mpo[k+1], mps[k+1])
        # left and right part:
        #      1  0
        #      e  d
        #      |  |
        # f,3--*--*
        #      |c |b
        # a,2--*--*
        rho_l = torch.einsum('abc,bcdef->deaf', mpo_mps[0], mpo_mps[1])
        # a,2--*--*
        #      |c |b
        # f,3--*--*
        #      |e |d
        #      1  0
        rho_r = torch.einsum('abc,debcf->deaf', mpo_mps[3], mpo_mps[2])
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


    def rg_mr(self):
        r'''
        a CTMRG right move step, merge the whole unit cell into up boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny-1, -1, -1), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            mpo, mps, mpo_mps = [None]*(self._ny+2), [None]*(self._ny+2), [None]*(self._ny+2)
            mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
            # (j+self._ny) % self._ny = j
            mpo[-1] = self._ctms[(i, j)]['Eu']
            for k in range(self._ny):
                mpo[k+1] = self._ts[(i, (j+k) % self._ny)]
            ii = (i+1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C1']
            mps[-1] = self._ctms[(ii, j)]['C3']
            for k in range(self._ny):
                mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['Er']
            #      |c,2  |d,1
            # a,0--*--b--*
            mpo_mps[0] = torch.einsum('abc,bd->adc', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abc,bd->adc', mpo[-1], mps[-1])
            #    |b,3  |f,2
            # a--*--c--*
            #    |d,1  |e,0
            for k in range(self._ny):
                mpo_mps[k+1] = torch.einsum('abcd,efc->edfba', mpo[k+1], mps[k+1])
            # use projectors to compress
            mps = [None]*(self._ny+2)
            pl, pr = self.rg_projectors_r(c=(i, (j-1) % self._ny))
            #        |d,1
            #      *--*
            #      |c |b
            # a,0--*--*
            mps[0] = torch.einsum('abc,bcd->ad', mpo_mps[0], pl)
            #       |g,1
            #      *--*
            #      |e |d
            # f,2--*--*
            #      |c |b
            #      *--*
            #       |a,0
            for k in range(self._nx):
                pl_prime, pr_prime = self.rg_projectors_r(c=((i+k) % self._nx, j))
                mps[k+1] = torch.einsum('abc,bcdef,deg->agf', pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            # d,0--*--*
            #      |c |b
            #      *--*
            #       |a,1
            mps[-1] = torch.einsum('abc,dbc->da', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C1'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C3'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._nx):
                self._ctms[((i+k) % self._nx, j)]['Er'] = mps[k+1] / torch.linalg.norm(mps[k+1])

        return 1

    def test(self):
        print('1')

    def measure_twobody(
            self,
            c: tuple[int, int],
            impure_ts: tuple[torch.tensor, torch.tensor],
            direction='x'):
        r'''measure twobody operator using CTM tensors
        Parameters
        ----------
        c: tuple[int], anchoring coordinate
        '''
        i, j = c
        # forward and backward sites
        bi, fi = (i-1) % self._nx, (i+1) % self._nx
        bj, fj = (j-1) % self._ny, (j+1) % self._ny
        # *--e
        # |c
        # *--d
        # |b
        # *--a
        if 'x' == direction:
            env_l = torch.einsum(
                    'ab,bcd,ec->ade',
                    self._ctms[(bi, bj)]['C0'], 
                    self._ctms[(bi, j)]['El'],
                    self._ctms[(bi, fj)]['C2'])
            env_r = torch.einsum(
                    'ab,bcd,ec->ade', 
                    self._ctms[((i+2) % self._nx, bj)]['C1'],
                    self._ctms[((i+2) % self._nx, j)]['Er'],
                    self._ctms[((i+2) % self._nx, fj)]['C3'])
            # denominator
            temp = env_l.clone()
            temp = torch.einsum(
                    'abc,ade,bfge,chf->dgh',
                    temp,
                    self._ctms[(i, bj)]['Ed'],
                    self._ts[(i, j)],
                    self._ctms[(i, fj)]['Eu'])
            temp = torch.einsum(
                    'abc,ade,bfge,chf->dgh',
                    temp,
                    self._ctms[(fi, bj)]['Ed'],
                    self._ts[(fi, j)],
                    self._ctms[(fi, fj)]['Eu'])
            den = torch.einsum('abc,abc', temp, env_r)
            # neumerator
            temp = env_l.clone()
            temp = torch.einsum(
                    'abc,ade,bfge,chf->dgh',
                    temp,
                    self._ctms[(i, bj)]['Ed'],
                    impure_ts[0],
                    self._ctms[(i, fj)]['Eu'])
            temp = torch.einsum(
                    'abc,ade,bfge,chf->dgh',
                    temp,
                    self._ctms[(fi, bj)]['Ed'],
                    impure_ts[1],
                    self._ctms[(fi, fj)]['Eu'])
            num = torch.einsum('abc,abc', temp, env_r)

        elif 'y' == direction:
            env_d = torch.einsum(
                    'ab,acd,ce->bde',
                    self._ctms[(bi, bj)]['C0'], 
                    self._ctms[(i, bj)]['Ed'],
                    self._ctms[(fi, bj)]['C1'])
            env_u = torch.einsum(
                    'ab,acd,ce->bde',
                    self._ctms[(bi, (j+2) % self._ny)]['C2'],
                    self._ctms[(i, (j+2) % self._ny)]['Eu'],
                    self._ctms[(fi, (j+2) % self._ny)]['C3'])
            # denominator
            temp = env_d.clone()
            temp = torch.einsum(
                    'abc,ade,efgb,chg->dfh',
                    temp,
                    self._ctms[(bi, j)]['El'],
                    self._ts[(i, j)],
                    self._ctms[(fi, j)]['Er'])
            temp = torch.einsum(
                    'abc,ade,efgb,chg->dfh',
                    temp,
                    self._ctms[(bi, fj)]['El'],
                    self._ts[(i, fj)],
                    self._ctms[(fi, fj)]['Er'])
            den = torch.einsum('abc,abc', temp, env_u)
            # neumerator
            temp = env_d.clone()
            temp = torch.einsum(
                    'abc,ade,efgb,chg->dfh',
                    temp,
                    self._ctms[(bi, j)]['El'],
                    impure_ts[0],
                    self._ctms[(fi, j)]['Er'])
            temp = torch.einsum(
                    'abc,ade,efgb,chg->dfh',
                    temp,
                    self._ctms[(bi, fj)]['El'],
                    impure_ts[1],
                    self._ctms[(fi, fj)]['Er'])
            num = torch.einsum('abc,abc', temp, env_u)

        else:
            raise ValueError('direction %s is not not valid' % direction)

        return (num / den)

