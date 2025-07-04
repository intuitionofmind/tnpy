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
        wfs: dict, dict of wavefunction tensors, {key: coordinate, value: rank-5 tensor}
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
        that is, effectively, each site is placed by NINE tensors:
        1 wf tensor + 8 CTM tensors (environments for other sites)
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
        self._chi = self._wfs[(0, 0)].shape[0]
        # double tensors
        #       2 3
        #       | |
        #    0--***--4
        #    1--***--5
        #       | |
        #       6 7
        # the conjugate index is put ahead
        self._dts = {}
        for c in self._coords:
            self._dts.update(
                    {c: torch.einsum('ABCDe,abcde->AaBbCcDd', wfs[c].conj(), wfs[c])})
        # CTMRG environment tensors
        self._ctm_names = 'C0', 'C1', 'C2', 'C3', 'Ed', 'Eu', 'El', 'Er'
        temp = {}
        for i, n in enumerate(self._ctm_names):
            # generate corner and edge tensors
            if i < 4:
                temp.update({n: torch.randn(rho, rho).to(dtype)})
            else:
            # 0--*--1
            #   / \
            #  2   3
                temp.update({n: torch.randn(rho, rho, self._chi, self._chi).to(dtype)})
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
        # x: (i, j), the anchoring point
        # *--*--*--* MPS
        # |  |  |  |
        # *--x--*--* MPO
        # |  |  |  |
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
        mpo[-1] = self._ctms[((i+2) % self._nx, j)]['Er']
        for k in range(2):
            mpo[k+1] = self._dts[((i+k) % self._nx, j)]
        jj = (j+1) % self._ny
        mps[0] = self._ctms[((i-1) % self._nx, jj)]['C2']
        mps[-1] = self._ctms[((i+2) % self._nx, jj)]['C3']
        for k in range(2):
            mps[k+1] = self._ctms[((i+k) % self._nx, jj)]['Eu']
        # MPO-MPS
        # !pay attention to the order of output tensor's bonds
        # *--e,0
        # |b 
        # *--c,1
        # *--d,2
        # |
        # a,3
        mpo_mps[0] = torch.einsum('abcd,eb->ecda', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,eb->ecda', mpo[-1], mps[-1])
        # e,0--*--f,3
        #     |B|b
        # A,1--*--C,4
        # a,2--*--c,5
        #     | |
        #     D d
        #     6 7
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('AaBbCcDd,efBb->eAafCcDd', mpo[k+1], mps[k+1])
        # left and right part
        # *--a--*--e,0
        # *--b--*--f,1
        # *--c--*--g,2
        # |    | |
        # d    h i
        # 3    4 5
        rho_l = torch.einsum('abcd,abcefghi->efgdhi', mpo_mps[0], mpo_mps[1])
        # 0,a--*--d--*
        # 1,b--*--e--*
        # 2,c--*--f--*
        #     | |    |
        #     g h    i
        #     4 5    3
        rho_r = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(3, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)
        '''
        u, s, v = tp.linalg.tsvd(
                torch.einsum('abcdef,abcghi->defghi', rho_l, rho_r),
                group_dims=((0, 1, 2), (3, 4, 5)),
                svd_dims=(3, 0))
        ut, st, vt = u[:, :, :, :self._rho], s[:self._rho], v[:self._rho, :, :, :]
        # ut_dagger = ut.permute(3, 0, 1, 2).conj()
        # vt_dagger = vt.permute(1, 2, 3, 0).conj()
        ut_dagger, vt_dagger = ut.conj(), vt.conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcdef,gdef,gh->abch', rho_r, vt_dagger, sst_inv)
        pr = torch.einsum('ab,cdeb,fghcde->afgh', sst_inv, ut_dagger, rho_l)
        '''

        return pl, pr


    def full_projectors_u(
            self,
            c: tuple):
        r'''build full projectors for CTMRG up move
        '''
        pass


    def rg_projectors_u2(
            self,
            c: tuple[int, int]):
        r'''build projectors for CTMRG up move with six sites
        Parameters
        ----------
        c: tuple, coordinate of anchoring point
        '''
        i, j = c
        # x: (i, j), the anchoring point
        # *--*--*--*--*--* MPS
        # |  |  |  |  |  |
        # *--*--x--*--*--* MPO
        # |  |  |  |  |  |
        mpo, mps, mpo_mps = [None]*6, [None]*6, [None]*6
        mpo[0] = self._ctms[((i-2) % self._nx, j)]['El']
        mpo[-1] = self._ctms[((i+3) % self._nx, j)]['Er']
        for k in range(4):
            mpo[k+1] = self._dts[((i-1+k) % self._nx, j)]
        jj = (j+1) % self._ny
        mps[0] = self._ctms[((i-2) % self._nx, jj)]['C2']
        mps[-1] = self._ctms[((i+3) % self._nx, jj)]['C3']
        for k in range(4):
            mps[k+1] = self._ctms[((i-1+k) % self._nx, jj)]['Eu']
        # MPO-MPS
        # !pay attention to the order of output tensor's bonds
        # *--e,0
        # |b 
        # *--c,1
        # *--d,2
        # |
        # a,3
        mpo_mps[0] = torch.einsum('abcd,eb->ecda', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,eb->ecda', mpo[-1], mps[-1])
        # e,0--*--f,3
        #     |B|b
        # A,1--*--C,4
        # a,2--*--c,5
        #     | |
        #     D d
        #     6 7
        for k in range(4):
            mpo_mps[k+1] = torch.einsum('AaBbCcDd,efBb->eAafCcDd', mpo[k+1], mps[k+1])
        # left and right part
        # *--a--*--e,0
        # *--b--*--f,1
        # *--c--*--g,2
        # |    | |
        # d    h i
        # 3    4 5
        rho_l = torch.einsum(
                'abcd,abcefghi,efgjklmn->dhimnjkl',
                mpo_mps[0], mpo_mps[1], mpo_mps[2])
        # 0,a--*--d--*
        # 1,b--*--e--*
        # 2,c--*--f--*
        #     | |    |
        #     g h    i
        #     4 5    3
        rho_r = torch.einsum(
                'abcd,efgabchi,jklefgmn->dhimnjkl',
                mpo_mps[-1], mpo_mps[-2], mpo_mps[-3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((0, 1, 2, 3, 4), (5, 6, 7)), qr_dims=(5, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((0, 1, 2, 3, 4), (5, 6, 7)), qr_dims=(5, 3))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        return pl, pr


    def rg_mu(self):
        r'''
        a CTMRG up step
        update related up boundary CTM tensors
        effectively merge the boundary MPS down to next row
        '''
        for i, j in itertools.product(range(self._nx), range(self._ny)):
            # x: (i, j)
            # *--*--* MPS, row-k
            # |  |  |
            # *--x--* MPO, row-j
            # |  |  |
            jj = (j+1) % self._ny
            mps, mpo, mpo_mps = [None]*3, [None]*3, [None]*3
            # MPS
            mps[0] = self._ctms[((i-1) % self._nx, jj)]['C2']
            mps[1] = self._ctms[(i, jj)]['Eu']
            mps[2] = self._ctms[((i+1) % self._nx, jj)]['C3']
            # MPO
            mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
            mpo[1] = self._dts[(i, j)]
            mpo[2] = self._ctms[((i+1) % self._nx, j)]['Er']
            # *--e,0
            # |b
            # *--c,1
            # *--d,2
            # |a,3
            mpo_mps[0] = torch.einsum('abcd,eb->ecda', mpo[0], mps[0])
            # e,0--*--f,3
            #     |B|b
            # A,1--*--C,4
            # a,2--*--c,5
            #     |D|d
            #     6 7
            mpo_mps[1] = torch.einsum('AaBbCcDd,efBb->eAafCcDd', mpo[1], mps[1])
            mpo_mps[2] = torch.einsum('abcd,eb->ecda', mpo[2], mps[2])
            # build projectors
            pl, pr = self.rg_projectors_u2(c=((i-1) % self._nx, j))
            pl_prime, pr_prime = self.rg_projectors_u2(c=(i, j))
            # use projectors to compress
            mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], pl)
            mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pr, mpo_mps[1], pl_prime)
            mps[2] = torch.einsum('abcd,bcde->ae', pr_prime, mpo_mps[2])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C2'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['Eu'] = mps[1] / torch.linalg.norm(mps[1])
            self._ctms[((i+1) % self._nx, j)]['C3'] = mps[2] / torch.linalg.norm(mps[2])

        return 1


    def rg_projectors_d(
            self,
            c: tuple[int, int]):
        r'''build projectors for CTMRG down move
        Parameters
        ----------
        c: tuple, coordinate of anchoring point
        '''
        i, j = c
        # x: the anchoring point
        # |  |  |  |
        # *--x--*--* MPO
        # |  |  |  |
        # *--*--*--* MPS
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
        mpo[-1] = self._ctms[((i+2) % self._nx, j)]['Er']
        for k in range(2):
            mpo[k+1] = self._dts[((i+k) % self._nx, j)]
        jj = (j-1) % self._ny
        mps[0] = self._ctms[((i-1) % self._nx, jj)]['C0']
        mps[-1] = self._ctms[((i+2) % self._nx, jj)]['C1']
        for k in range(2):
            mps[k+1] = self._ctms[((i+k) % self._nx, jj)]['Ed']
        # MPO-MPS
        # !pay attention to the order of ouptput tensor's bonds
        # |b,3 
        # *--c,1
        # *--d,2
        # |a
        # *--e,0
        mpo_mps[0] = torch.einsum('abcd,ea->ecdb', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,ea->ecdb', mpo[-1], mps[-1])
        #     |B|b
        # A,1--*--C,4
        # a,2--*--c,5
        #     |D|d
        # e,0--*--f,3
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('AaBbCcDd,efDd->eAafCcBb', mpo[k+1], mps[k+1])
        # left and right part:
        # 3    4 5
        # d    h i
        # |    | |
        # *--b--*--f,1
        # *--c--*--g,2
        # *--a--*--e,0
        rho_l = torch.einsum('abcd,abcefghi->efgdhi', mpo_mps[0], mpo_mps[1])
        #     4 5    3
        #     g h    i
        #     | |    |
        # 1,b--*--e--*
        # 2,c--*--f--*
        # 0,a--*--d--*
        rho_r = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(3, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)
        '''
        u, s, v = tp.linalg.tsvd(
                torch.einsum('abcdef,abcghi->defghi', rho_l, rho_r),
                group_dims=((0, 1, 2), (3, 4, 5)),
                svd_dims=(3, 0))
        ut, st, vt = u[:, :, :, :self._rho], s[:self._rho], v[:self._rho, :, :, :]
        # ut_dagger = ut.permute(3, 0, 1, 2).conj()
        # vt_dagger = vt.permute(1, 2, 3, 0).conj()
        ut_dagger, vt_dagger = ut.conj(), vt.conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcdef,gdef,gh->abch', rho_r, vt_dagger, sst_inv)
        pr = torch.einsum('ab,cdeb,fghcde->afgh', sst_inv, ut_dagger, rho_l)
        '''

        return pl, pr


    def rg_md(self):
        r'''
        a CTMRG down step, merge the whole unit cell into down boundary MPS
        update related CTM tensors
        '''
        for i, j in itertools.product(range(self._nx), range(self._ny)):
            # (i, j) as the anchoring point of MPO
            # |  |  |
            # *--x--* MPO
            # |  |  |
            # *--*--* MPS
            mps, mpo, mpo_mps = [None]*3, [None]*3, [None]*3
            # MPS
            jj = (j-1) % self._ny
            mps[0] = self._ctms[((i-1) % self._nx, jj)]['C0']
            mps[1] = self._ctms[(i, jj)]['Ed']
            mps[2] = self._ctms[((i+1) % self._nx, jj)]['C1']
            # MPO
            mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
            mpo[1] = self._dts[(i, j)]
            mpo[2] = self._ctms[((i+1) % self._nx, j)]['Er']
            # MPO-MPS
            # !pay attention to the order of ouptput tensor's bonds
            # |b,3 
            # *--c,1
            # *--d,2
            # |a
            # *--e,0
            mpo_mps[0] = torch.einsum('abcd,ea->ecdb', mpo[0], mps[0])
            #     |B|b
            # A,1--*--C,4
            # a,2--*--c,5
            #     |D|d
            # e,0--*--f,3
            mpo_mps[1] = torch.einsum('AaBbCcDd,efDd->eAafCcBb', mpo[1], mps[1])
            mpo_mps[2] = torch.einsum('abcd,ea->ecdb', mpo[2], mps[2])
            pl, pr = self.rg_projectors_d(c=((i-1) % self._nx, j))
            pl_prime, pr_prime = self.rg_projectors_d(c=(i, j))
            mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], pl)
            mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pr, mpo_mps[1], pl_prime)
            mps[2] = torch.einsum('abcd,bcde->ae', pr_prime, mpo_mps[2])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C0'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['Ed'] = mps[1] / torch.linalg.norm(mps[1])
            self._ctms[((i+1) % self._nx, j)]['C1'] = mps[2] / torch.linalg.norm(mps[2])

        return 1


    def rg_projectors_l(
            self,
            c: tuple[int, int]):
        r'''build projectors for CTMRG left move
        Parameters
        ----------
        c: tuple, coordinate of anchoring point
        '''
        i, j = c
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        # perspective: from down to up
        # x: (i, j), the anchoring point
        # *--*--
        # |  |
        # *--*--
        # |  |  
        # *--x--
        # |  |
        # *--*--
        mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
        mpo[-1] = self._ctms[(i, (j+2) % self._ny)]['Eu']
        for k in range(2):
            mpo[k+1] = self._dts[(i, (j+k) % self._ny)]
        ii = (i-1) % self._nx
        mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C0']
        mps[-1] = self._ctms[(ii, j)]['C2']
        for k in range(2):
            mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['El']
        # MPO-MPS
        # !pay attention to the order of ouptput tensor's bonds
        # 1     23
        # e     cd
        # |     ||
        # *--a--**--b,0
        mpo_mps[0] = torch.einsum('abcd,ae->becd', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,ae->becd', mpo[-1], mps[-1])
        # 3     45
        # f     Bb
        # |     ||
        # *--A--*--C,6
        # *--a--*--c,7
        # |     ||
        # e     Dd
        # 0     12
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('AaBbCcDd,efAa->eDdfBbCc', mpo[k+1], mps[k+1])
        # left and right part:
        # 345
        # efg
        # |||
        # * *--h,1
        # * *--i,2
        # |||
        # bcd 
        # |||
        # * *--a,0
        rho_l = torch.einsum('abcd,bcdefghi->ahiefg', mpo_mps[0], mpo_mps[1])
        # * *--i,0
        # |||
        # def
        # |||
        # * *--g,1
        # * *--h,2
        # |||
        # abc 
        # 345
        rho_r = torch.einsum('abcdefgh,idef->ighabc', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(3, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(0, 3))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)
        '''
        u, s, v = tp.linalg.tsvd(
                torch.einsum('abcdef,ghidef->abcghi', rho_l, rho_r),
                group_dims=((0, 1, 2), (3, 4, 5)),
                svd_dims=(3, 0))
        ut, st, vt = u[:, :, :, :self._rho], s[:self._rho], v[:self._rho, :, :, :]
        ut_dagger, vt_dagger = ut.conj(), vt.conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcdef,gabc,gh->defh', rho_r, vt_dagger, sst_inv)
        pr = torch.einsum('ab,cdeb,cdefgh->afgh', sst_inv, ut_dagger, rho_l)
        '''

        return pl, pr


    def rg_ml(self):
        r'''
        a CTMRG left step, merge the whole unit cell into down boundary MPS
        update related CTM tensors
        '''
        for i, j in itertools.product(range(self._nx), range(self._ny)):
            # x: (i, j), the anchoring point
            # *--*--
            # |  |
            # *--x--
            # |  |
            # *--*--
            mps, mpo, mpo_mps = [None]*3, [None]*3, [None]*3
            # MPS
            ii = (i-1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C0']
            mps[1] = self._ctms[(ii, j)]['El']
            mps[2] = self._ctms[(ii, (j+1) % self._ny)]['C2']
            # MPO
            mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
            mpo[1] = self._dts[(i, j)]
            mpo[2] = self._ctms[(i, (j+1) % self._ny)]['Eu']
            # MPO-MPS
            # !pay attention to the order of ouptput tensor's bonds
            # 1     23
            # e     cd
            # |     ||
            # *--a--**--b,0
            mpo_mps[0] = torch.einsum('abcd,ae->becd', mpo[0], mps[0])
            # 3     45
            # f     Bb
            # |     ||
            # *--A--*--C,6
            # *--a--*--c,7
            # |     ||
            # e     Dd
            # 0     12
            mpo_mps[1] = torch.einsum('AaBbCcDd,efAa->eDdfBbCc', mpo[1], mps[1])
            mpo_mps[2] = torch.einsum('abcd,ae->becd', mpo[2], mps[2])
            pl, pr = self.rg_projectors_l(c=(i, (j-1) % self._ny))
            pl_prime, pr_prime = self.rg_projectors_l(c=(i, j))
            mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], pl)
            mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi',pr, mpo_mps[1], pl_prime)
            mps[2] = torch.einsum('abcd,ebcd->ea', pr_prime, mpo_mps[2])
            # update related CTM tensors
            self._ctms[(i, (j-1) % self._ny)]['C0'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['El'] = mps[1] / torch.linalg.norm(mps[1])
            self._ctms[(i, (j+1) % self._ny)]['C2'] = mps[2] / torch.linalg.norm(mps[2])

        return 1


    def rg_projectors_r(
            self,
            c: tuple[int, int]):
        r'''build projectors for CTMRG right move
        Parameters
        ----------
        c: tuple, coordinate of anchoring point
        '''
        i, j = c
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        # perspective: from down to up
        # x: the anchoring point
        # --*--*
        #   |  |
        # --*--*
        #   |  |  
        # --x--*
        #   |  |
        # --*--*
        mpo, mps, mpo_mps = [None]*4, [None]*4, [None]*4
        mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
        mpo[-1] = self._ctms[(i, (j+2) % self._ny)]['Eu']
        for k in range(2):
            mpo[k+1] = self._dts[(i, (j+k) % self._ny)]
        ii = (i+1) % self._nx
        mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C1']
        mps[-1] = self._ctms[(ii, (j+2) % self._ny)]['C3']
        for k in range(2):
            mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['Er']
        # MPO-MPS
        # !pay attention to the order of ouptput tensor's bonds
        #      23     1
        #      cd     e
        #      ||     |
        # 0,a--**--b--*
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,be->aecd', mpo[-1], mps[-1])
        #    45     3
        #    Bb     f
        #    ||     |
        # A,6-*--C--*
        # a,7-*--c--*
        #    ||     |
        #    Dd     e
        #    12     0
        for k in range(2):
            mpo_mps[k+1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[k+1], mps[k+1])
        # left and right part:
        #      345
        #      efg
        #      |||
        # h,1--* *
        # i,2--* *
        #      |||
        #      bcd 
        #      |||
        # a,0--* *
        rho_l = torch.einsum('abcd,bcdefghi->ahiefg', mpo_mps[0], mpo_mps[1])
        # i,0--* *
        #      |||
        #      def
        #      |||
        # g,1--* *
        # h,2--* *
        #      |||
        #      abc 
        #      345
        rho_r = torch.einsum('abcdefgh,idef->ighabc', mpo_mps[2], mpo_mps[3])
        # QR and LQ factorizations
        q, r = tp.linalg.tqr(rho_l, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(3, 0))
        q, l = tp.linalg.tqr(rho_r, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(0, 3))
        # build projectors
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)
        '''
        u, s, v = tp.linalg.tsvd(
                torch.einsum('abcdef,ghidef->abcghi', rho_l, rho_r),
                group_dims=((0, 1, 2), (3, 4, 5)),
                svd_dims=(3, 0))
        ut, st, vt = u[:, :, :, :self._rho], s[:self._rho], v[:self._rho, :, :, :]
        ut_dagger, vt_dagger = ut.conj(), vt.conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        pl = torch.einsum('abcdef,gabc,gh->defh', rho_r, vt_dagger, sst_inv)
        pr = torch.einsum('ab,cdeb,cdefgh->afgh', sst_inv, ut_dagger, rho_l)
        '''

        return pl, pr


    def rg_mr(self):
        r'''
        a CTMRG right move step, merge the whole unit cell into up boundary MPS
        update related CTM tensors
        '''
        for i, j in itertools.product(range(self._nx), range(self._ny)):
            # x: (i, j), the anchoring point
            # --*--*
            #   |  |
            # --x--*
            #   |  |
            # --*--*
            mps, mpo, mpo_mps = [None]*3, [None]*3, [None]*3
            # MPS
            ii = (i+1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C1']
            mps[1] = self._ctms[(ii, j)]['Er']
            mps[2] = self._ctms[(ii, (j+1) % self._ny)]['C3']
            # MPO
            mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
            mpo[1] = self._dts[(i, j)]
            mpo[2] = self._ctms[(i, (j+1) % self._ny)]['Eu']
            # MPO-MPS
            # !pay attention to the order of ouptput tensor's bonds
            #      23     1
            #      cd     e
            #      ||     |
            # 0,a--**--b--*
            mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
            #    4 5    3
            #    B b    f
            #    | |    |
            # A,6-*--C--*
            # a,7-*--c--*
            #    | |    |
            #    D d    e
            #    1 2    0
            mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[1], mps[1])
            mpo_mps[2] = torch.einsum('abcd,be->aecd', mpo[2], mps[2])
            pl, pr = self.rg_projectors_r(c=(i, (j-1) % self._ny))
            pl_prime, pr_prime = self.rg_projectors_r(c=(i, j))
            mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], pl)
            mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi',pr, mpo_mps[1], pl_prime)
            mps[2] = torch.einsum('abcd,ebcd->ea', pr_prime, mpo_mps[2])
            # update related CTM tensors
            self._ctms[(i, (j-1) % self._ny)]['C1'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['Er'] = mps[1] / torch.linalg.norm(mps[1])
            self._ctms[(i, (j+1) % self._ny)]['C3'] = mps[2] / torch.linalg.norm(mps[2])

        return 1


    def measure_onebody(
            self,
            c: tuple,
            op: torch.tensor):
        r'''measure onebody operator
        Parameters:
        ----------
        c: coordinate of site
        op: torch.tensor, operator to be measured
        '''
        i, j = c
        # *--f,3
        # |c
        # *--d,1
        # *--e,2
        # |b
        # *--a,0
        env_l = torch.einsum(
                'ab,bcde,fc->adef',
                self._ctms[((i-1) % self._nx, (j-1) % self._ny)]['C0'],
                self._ctms[((i-1) % self._nx, j)]['El'],
                self._ctms[((i-1) % self._nx, (j+1) % self._ny)]['C2'])
        env_r = torch.einsum(
                'ab,bcde,fc->adef',
                self._ctms[((i+1) % self._nx, (j-1) % self._ny)]['C1'],
                self._ctms[((i+1) % self._nx, j)]['Er'],
                self._ctms[((i+1) % self._nx, (j+1) % self._ny)]['C3'])
        # denominator
        temp = env_l.clone()
        # *--g--*--h,3
        # |    |B|b
        # *--A--*--C,1
        # *--a--*--c,2
        # |    |D|d
        # *--e--*--f,0
        temp = torch.einsum(
                'eAag,efDd,AaBbCcDd,ghBb->fCch',
                temp,
                self._ctms[(i, (j-1) % self._ny)]['Ed'],
                self._dts[c],
                self._ctms[(i, (j+1) % self._ny)]['Eu'])
        den = torch.einsum('abcd,abcd', temp, env_r)
        # numerator
        impure_dt = torch.einsum(
                'ABCDE,Ee,abcde->AaBbCcDd',
                self._wfs[c].conj(), op, self._wfs[c])
        temp = env_l.clone()
        temp = torch.einsum(
                'eAag,efDd,AaBbCcDd,ghBb->fCch',
                temp,
                self._ctms[(i, (j-1) % self._ny)]['Ed'],
                impure_dt,
                self._ctms[(i, (j+1) % self._ny)]['Eu'])
        num = torch.einsum('abcd,abcd', temp, env_r)

        print(num, den)
        return (num / den)


    def measure_twobody(
            self,
            c: tuple,
            op: torch.tensor,
            direction='x'):
        r'''measure onebody operator
        Parameters:
        ----------
        c: coordinate of site
        op: twobody operator
        # |0|1
        # * *
        # |2|3
        '''
        # SVD to MPO
        # |0       |0
        # *--2, 2--*
        # |1       |1
        u, s, v = tp.linalg.tsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(1, 0))
        ss = torch.sqrt(s).diag().to(self._dtype)
        us = torch.einsum('abc,bB->aBc', u, ss)
        sv = torch.einsum('Aa,abc->bAc', ss, v)
        op_mpo = us, sv
        i, j = c
        if  'x' == direction:
            # *--f,3
            # |c
            # *--d,1
            # *--e,2
            # |b
            # *--a,0
            env_l = torch.einsum(
                    'ab,bcde,fc->adef',
                    self._ctms[((i-1) % self._nx, (j-1) % self._ny)]['C0'],
                    self._ctms[((i-1) % self._nx, j)]['El'],
                    self._ctms[((i-1) % self._nx, (j+1) % self._ny)]['C2'])
            env_r = torch.einsum(
                    'ab,bcde,fc->adef',
                    self._ctms[((i+2) % self._nx, (j-1) % self._ny)]['C1'],
                    self._ctms[((i+2) % self._nx, j)]['Er'],
                    self._ctms[((i+2) % self._nx, (j+1) % self._ny)]['C3'])
            # denominator
            temp = env_l.clone()
            # *--g--*--h,3
            # |    |B|b
            # *--A--*--C,1
            # *--a--*--c,2
            # |    |D|d
            # *--e--*--f,0
            temp = torch.einsum(
                    'eAag,efDd,AaBbCcDd,ghBb->fCch',
                    temp,
                    self._ctms[(i, (j-1) % self._ny)]['Ed'],
                    self._dts[c],
                    self._ctms[(i, (j+1) % self._ny)]['Eu'])
            temp = torch.einsum(
                    'eAag,efDd,AaBbCcDd,ghBb->fCch',
                    temp,
                    self._ctms[((i+1) % self._nx, (j-1) % self._ny)]['Ed'],
                    self._dts[((i+1) % self._nx, j)],
                    self._ctms[((i+1) % self._nx, (j+1) % self._ny)]['Eu'])
            den = torch.einsum('abcd,abcd', temp, env_r)
            # build impure double tensors
            impure_dts = [None]*2
            impure_dts[0] = torch.einsum(
                    'ABCDE,Efe,abcde->AaBbCfcDd',
                    self._wfs[c].conj(), op_mpo[0], self._wfs[c])
            impure_dts[1] = torch.einsum(
                    'ABCDE,Efe,abcde->AfaBbCcDd',
                    self._wfs[((i+1) % self._nx, j)].conj(),
                    op_mpo[1],
                    self._wfs[((i+1) % self._nx, j)])
            # numerator
            temp = env_l.clone()
            temp = torch.einsum(
                    'eAag,efDd,AaBbCicDd,ghBb->fCich',
                    temp,
                    self._ctms[(i, (j-1) % self._ny)]['Ed'],
                    impure_dts[0],
                    self._ctms[(i, (j+1) % self._ny)]['Eu'])
            temp = torch.einsum(
                    'eAiag,efDd,AiaBbCcDd,ghBb->fCch',
                    temp,
                    self._ctms[((i+1) % self._nx, (j-1) % self._ny)]['Ed'],
                    impure_dts[1],
                    self._ctms[((i+1) % self._nx, (j+1) % self._ny)]['Eu'])
            num = torch.einsum('abcd,abcd', temp, env_r)
        elif 'y' == direction:
            pass
        else:
            raise ValueError('Your direction is not valid')

        return (num / den)
