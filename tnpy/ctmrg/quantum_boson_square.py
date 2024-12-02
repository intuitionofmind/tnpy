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
        that is, effectively, each site is placed by NINE tensors: 1 wf tensor + 8 CTM tensors
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
                temp.update({n: torch.rand(rho, rho).to(dtype)})
            else:
            # 0--*--1
            #   / \
            #  2   3
                temp.update({n: torch.rand(rho, rho, self._chi, self._chi).to(dtype)})
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
            k = (j+1) % self._ny
            mps, mpo, mpo_mps = [None]*3, [None]*3, [None]*3
            # MPS
            mps[0] = self._ctms[((i-1) % self._nx, k)]['C2']
            mps[1] = self._ctms[(i, k)]['Eu']
            mps[2] = self._ctms[((i+1) % self._nx, k)]['C3']
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
            # build projectors
            pl, pr = self.rg_projectors_u(c=((i-1) % self._nx, j))
            pl_prime, pr_prime = self.rg_projectors_u(c=(i, j))
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

        return pl, pr


    def rg_md(self):
        r'''
        a CTMRG down step, merge the whole unit cell into down boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            # |  |  |
            # *--x--* MPO
            # |  |  |
            # *--*--* MPS
            mpo, mps, mpo_mps = [None]*(self._nx+2), [None]*(self._nx+2), [None]*(self._nx+2)
            mpo[0] = self._ctms[((i-1) % self._nx, j)]['El']
            # (i+self._nx) % self._nx = i
            mpo[-1] = self._ctms[(i, j)]['Er']
            for k in range(self._nx):
                mpo[k+1] = self._dts[((i+k) % self._nx, j)]
            jj = (j-1) % self._ny
            mps[0] = self._ctms[((i-1) % self._nx, jj)]['C0']
            mps[-1] = self._ctms[(i, jj)]['C1']
            for k in range(self._nx):
                mps[k+1] = self._ctms[((i+k) % self._nx, jj)]['Ed']
            # MPO-MPS
            mpo_mps[0] = torch.einsum('abcd,ea->ecdb', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abcd,ea->ecdb', mpo[-1], mps[-1])
            for k in range(self._nx):
                mpo_mps[k+1] = torch.einsum('AaBbCcDd,efDd->eAafCcBb', mpo[k+1], mps[k+1])
            # new MPS
            mps = [None]*(self._nx+2)
            pl, pr = self.rg_projectors_d(c=((i-1) % self._nx, j))
            mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], pl)
            for k in range(self._nx):
                pl_prime, pr_prime = self.rg_projectors_d(c=((i+k) % self._nx, j))
                mps[k+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            mps[-1] = torch.einsum('abcd,bcde->ae', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C2'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C3'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._nx):
                self._ctms[((i+k) % self._nx, j)]['Eu'] = mps[k+1] / torch.linalg.norm(mps[k+1])

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
        # x: the anchoring point
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

        return pl, pr


    def rg_ml(self):
        r'''
        a CTMRG left step, merge the whole unit cell into down boundary MPS
        update related CTM tensors
        '''
        for j, i in itertools.product(range(self._ny), range(self._nx)):
            # (i, j) as the anchoring point of MPO
            # perspective: from down to up
            # x: the anchoring point
            # *--*--
            # |  |
            # *--x--
            # |  |
            # *--*--
            mpo, mps, mpo_mps = [None]*(self._ny+2), [None]*(self._ny+2), [None]*(self._ny+2)
            mpo[0] = self._ctms[(i, (j-1) % self._ny)]['Ed']
            # (j+self._ny) % self._ny = j
            mpo[-1] = self._ctms[(i, j)]['Eu']
            for k in range(self._ny):
                mpo[k+1] = self._dts[(i, (j+k) % self._ny)]
            ii = (i-1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C0']
            mps[-1] = self._ctms[(ii, j)]['C2']
            for k in range(self._ny):
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
            for k in range(self._ny):
                mpo_mps[k+1] = torch.einsum(
                        'AaBbCcDd,efAa->eDdfBbCc',
                        mpo[k+1], mps[k+1])
            mps = [None]*(self._ny+2)
            pl, pr = self.rg_projectors_l(c=(i, (j-1) % self._ny))
            #  |e
            # ***
            # ||| 
            # bcd
            # |||
            # ***--a
            mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], pl)
            for k in range(self._ny):
                pl_prime, pr_prime = self.rg_projectors_l(c=(i, (j+k) % self._ny))
                #  |j
                # ***
                # |||
                # efg
                # |||
                # ***--h
                # ***--i
                # |||
                # bcd
                # |||
                # ***
                #  |a
                mps[k+1] = torch.einsum(
                        'abcd,bcdefghi,efgj->ajhi',
                        pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            # ***--e
            # |||
            # bcd
            # ||| 
            # ***
            #  |a
            mps[-1] = torch.einsum('abcd,ebcd->ea', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C0'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C2'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._ny):
                self._ctms[(i, (j+k) % self._ny)]['El'] = mps[k+1] / torch.linalg.norm(mps[k+1])

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
        #  Bb     f
        #  ||     |
        # A-*--C--*
        # a-*--c--*
        #  ||     |
        #  Dd     e
        #  12     0
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
                mpo[k+1] = self._dts[(i, (j+k) % self._ny)]
            ii = (i+1) % self._nx
            mps[0] = self._ctms[(ii, (j-1) % self._ny)]['C1']
            mps[-1] = self._ctms[(ii, j)]['C3']
            for k in range(self._ny):
                mps[k+1] = self._ctms[(ii, (j+k) % self._ny)]['Er']
            # MPO-MPS
            # !pay attention to the order of ouptput tensor's bonds
            #      23     1
            #      cd     e
            #      ||     |
            # 0,a--**--b--*
            mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abcd,be->aecd', mpo[-1], mps[-1])
            #  Bb     f
            #  ||     |
            # A-*--C--*
            # a-*--c--*
            #  ||     |
            #  Dd     e
            #  12     0
            for k in range(self._ny):
                mpo_mps[k+1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[k+1], mps[k+1])
            # use projectors to compress
            mps = [None]*(self._ny+2)
            pl, pr = self.rg_projectors_r(c=(i, (j-1) % self._ny))
            #     |e
            #    ***
            #    ||| 
            #    bcd
            #    |||
            # a--***
            mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], pl)
            for k in range(self._ny):
                pl_prime, pr_prime = self.rg_projectors_r(c=(i, (j+k) % self._ny))
                #     |j
                #    ***
                #    |||
                #    efg
                #    |||
                # h--***
                # i--***
                #    |||
                #    bcd
                #    |||
                #    ***
                #     |a
                mps[k+1] = torch.einsum(
                        'abcd,bcdefghi,efgj->ajhi',
                        pr, mpo_mps[k+1], pl_prime)
                # move to next site
                pl, pr = pl_prime, pr_prime
            # ***--e
            # |||
            # bcd
            # ||| 
            # ***
            #  |a
            mps[-1] = torch.einsum('abcd,ebcd->ea', pr, mpo_mps[-1])
            # update related CTM tensors
            self._ctms[((i-1) % self._nx, j)]['C1'] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[(i, j)]['C3'] = mps[-1] / torch.linalg.norm(mps[-1])
            for k in range(self._ny):
                self._ctms[(i, (j+k) % self._ny)]['Er'] = mps[k+1] / torch.linalg.norm(mps[k+1])

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
