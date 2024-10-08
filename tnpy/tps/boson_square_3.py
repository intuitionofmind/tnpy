from copy import deepcopy import itertools
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

class SquareCTMRG(object):
    r'''
    class of CTMRG method on a square lattice
    '''
    def __init__(
            self,
            dts: dict,
            ctms=None,
            dtype=torch.float64):
        r''' class initialization

        Parameters
        ----------
        dts: dict, double tensors

        # double tensor:
        #       2 3
        #       | |
        #    0--***--4
        #    1--***--5
        #       | |
        #       6 7
        # CTM tensors:
        # ctm_names: {C0, C1, C2, C3, Ed, Eu, El, Er}
        #  C2  Eu  C3
        #   *--*--*
        #   |  |  |
        # El*--*--*Er
        #   |  |  |
        #   *--*--*
        #  C0  Ed  C1

        that is, each site is place on NINE tensors: 1 double tensor + 8 CTM tensors
        '''
        self._dtype = dtype
        # sorted by the key/coordinate (x, y), firstly by y, then by x
        self._dts = dict(sorted(dts.items(), key=lambda x: (x[0][1], x[0][0])))
        self._coords = tuple(self._dts.keys())
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

        # for CTMRG 
        self._ctm_names = 'C0', 'C1', 'C2', 'C4', 'Ed', 'Eu', 'El', 'Er'
        self._ctms = ctms
        if self._ctms is not None:
            for k, v in self._ctms[(0, 0)].items():
                if k not in self._ctm_names:
                    raise ValueError('Input CTM names are not valid')
                else:
                    self._rho = self._ctms[(0, 0)]['C0'].shape[0]
        else:
            self._rho = 0


    @classmethod
    def rand(
            cls,
            nx: int,
            ny: int,
            chi: int,
            rho: int,
            dtype=torch.float64):
        r'''
        generate a random SquareTPS

        Parameters
        ----------
        nx: int, number of sites along x-direction in a unit cell
        ny: int, number of sites along y-direction in a unit cell
        chi: int, bond dimension of double tensor
        rho: int, bond dimension of boundary CTM tensors
        '''
        t_shape = tuple([chi]*8)
        for i, j in itertools.product(range(nx), range(ny)):
            temp = torch.rand(t_shape).to(dtype)
        if rho > 0:
            ctm_names = 'C0', 'C1', 'C2', 'C4', 'Ed', 'Eu', 'El', 'Er'
            ctm = {}
            for i, n in enumerate(ctm_names):
                if i < 4:
                # corners
                    ctm.update({n: torch.rand(rho, rho).to(dtype)})
                else:
                # edges
                    ctm.update({n: torch.rand(rho, rho, chi, chi).to(dtype)})
            ctms = {}
            for i, j in itertools.product(range(nx), range(ny)):
                ctms.update({(i, j): ctm})
        else:
            ctms = None

        return cls(site_tensors=site_tensors, link_tensors=link_tensors, ctms=ctms, dtype=dtype)


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

    def update_ctms(self,
                    new_ctms: dict):
        for 
        self._ctms = new_ctms

        if self._ctms is not None:
            self._rho = self._ctms[(0, 0)][0].shape[0]
        else:
            self._rho = 0

        return 1


    def simple_update_proj(self, time_evo_mpo: tuple):
        r'''
        simple update by projectors

        Parameters
        ----------
        time_evo_mpo: tuple[tensor], time evolution operator MPO
        '''

        for c in self._coords:

            # forward sites
            cx = (c[0]+1) % self._nx, c[1]

            # X-direction
            #   |     |
            # --*--*--*--
            #   |     |
            tens_env = [
                    self.site_envs(c, inner_bonds=(2,)),
                    self.site_envs(cx, inner_bonds=(0,))]
            # merged tensors
            mts = [
                    self.absorb_envs(self._site_tensors[c], tens_env[0]),
                    self.absorb_envs(self._site_tensors[cx], tens_env[1])]

            # apply the time evolution operator
            #      b,1 E,5
            #      |/
            #      *-C,2
            # a,0--*-c,3
            #      |d,4
            te_mts = []
            te_mts.append(torch.einsum('ECe,abcde->abCcdE', time_evo_mpo[0], mts[0]))
            te_mts.append(torch.einsum('AEe,abcde->AabcdE', time_evo_mpo[1], mts[1]))

            # QR and LQ decompositions to bring the rest to canonical forms
            q, r = tp.linalg.tqr(te_mts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            q, l = tp.linalg.tqr(te_mts[1], group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))

            temp = torch.einsum('abc,bcd->ad', r, l)
            # u, s, v = torch.linalg.svd(temp, full_matrices=False)
            u, s, v = tp.linalg.svd(temp)

            # truncate
            ut, st, vt = u[:, :self._chi], s[:self._chi], v[:self._chi, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()

            sst = torch.sqrt(st).to(self._dtype)
            # safe inverse because of the trunction
            sst_inv = (1.0 / sst).diag().to(self._dtype)

            # build projectors
            pr = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

            # update the link tensor
            self._link_tensors[c][0] = (st / torch.linalg.norm(st)).diag().to(self._dtype)

            # apply projectors
            updated_mts = [
                    torch.einsum('abCcde,Ccf->abfde', te_mts[0], pr),
                    torch.einsum('fAa,Aabcde->fbcde', pl, te_mts[1])]

            # remove external environments and update site tensors
            # replace the connected environment by the updated one
            tens_env[0][2], tens_env[1][0] = sst.diag(), sst.diag()

            tens_env_inv = [
                    [torch.linalg.pinv(m).to(self._dtype) for m in tens_env[0]],
                    [torch.linalg.pinv(m).to(self._dtype) for m in tens_env[1]]]
            updated_ts = [
                    self.absorb_envs(updated_mts[0], tens_env_inv[0]),
                    self.absorb_envs(updated_mts[1], tens_env_inv[1])]

            self._site_tensors[c] = updated_ts[0] / torch.linalg.norm(updated_ts[0])
            self._site_tensors[cx] = updated_ts[1] / torch.linalg.norm(updated_ts[1])

            # Y-direction
            if self._ny > 1:
                cy = c[0], (c[1]+1) % self._ny

                tens_env = [
                        self.site_envs(c, inner_bonds=(1,)),
                        self.site_envs(cy, inner_bonds=(3,))]
                # merged tensors
                mts = [
                        self.absorb_envs(self._site_tensors[c], tens_env[0]),
                        self.absorb_envs(self._site_tensors[cy], tens_env[1])]

                # apply the time evolution operator
                #      b,1 E,5
                #      |/
                #      *-C,2
                # a,0--*-c,3
                #      |d,4
                te_mts = []
                te_mts.append(torch.einsum('EBe,abcde->aBbcdE', time_evo_mpo[0], mts[0]))
                te_mts.append(torch.einsum('DEe,abcde->abcDdE', time_evo_mpo[1], mts[1]))

                # QR and LQ decompositions
                q, r = tp.linalg.tqr(te_mts[0], group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                q, l = tp.linalg.tqr(te_mts[1], group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))

                temp = torch.einsum('abc,bcd->ad', r, l)
                # u, s, v = torch.linalg.svd(temp, full_matrices=False)
                u, s, v = tp.linalg.svd(temp)

                # truncate
                ut, st, vt = u[:, :self._chi], s[:self._chi], v[:self._chi, :]
                ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()

                sst = torch.sqrt(st).to(self._dtype)
                sst_inv = (1.0 / sst).diag().to(self._dtype)

                # build projectors
                pr = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
                pl = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

                # update the link tensor
                self._link_tensors[c][1] = (st / torch.linalg.norm(st)).diag().to(self._dtype)

                # apply projectors
                updated_mts = [
                        torch.einsum('aBbcde,Bbf->afcde', te_mts[0], pr),
                        torch.einsum('fDd,abcDde->abcfe', pl, te_mts[1])]

                # two-site wavefunction
                # wf = torch.einsum('abcde,fghbi->acdefghi', *updated_mts)

                # remove external environments and update site tensors
                # replace the connected environment by the updated one
                tens_env[0][1], tens_env[1][3] = sst.diag(), sst.diag()

                tens_env_inv = [
                        [torch.linalg.pinv(m).to(self._dtype) for m in tens_env[0]],
                        [torch.linalg.pinv(m).to(self._dtype) for m in tens_env[1]]]
                updated_ts = [
                        self.absorb_envs(updated_mts[0], tens_env_inv[0]),
                        self.absorb_envs(updated_mts[1], tens_env_inv[1])]

                self._site_tensors[c] = updated_ts[0] / torch.linalg.norm(updated_ts[0])
                self._site_tensors[cy] = updated_ts[1] / torch.linalg.norm(updated_ts[1])

        return 1


    def beta_twobody_measure_ops(self, ops: tuple):
        r'''
        measure bond energy on beta lattice

        Parameters
        ----------
        ops: tuple[tensor], twobody operator
        '''

        mts = {}
        for c in self._coords:
            mts.update({c: self.merged_tensor(c)})

        # measure on all bonds
        res = []
        for c in self._coords:

            # forward sites along two directions
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny

            # X-direction
            mts_conj = {}

            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[2] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[c], envs).conj()
            mts_conj.update({c: temp})

            envs = self.site_envs(cx)
            envs[0] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[cx], envs).conj()
            mts_conj.update({cx: temp})

            nums = [
                    torch.einsum('abCdE,Ee,abcde->Cc', mts_conj[c], ops[0], mts[c]),
                    torch.einsum('AbcdE,Ee,abcde->Aa', mts_conj[cx], ops[1], mts[cx])]
            dens = [
                    torch.einsum('abCde,abcde->Cc', mts_conj[c], mts[c]),
                    torch.einsum('Abcde,abcde->Aa', mts_conj[cx], mts[cx])]

            res.append(torch.einsum('ab,ab', *nums) / torch.einsum('ab,ab', *dens))
            # print('OPs:', torch.einsum('ab,ab', *nums), torch.einsum('ab,ab', *dens))

            # Y-direction
            mts_conj = {}

            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[1] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[c], envs).conj()
            mts_conj.update({c: temp})

            envs = self.site_envs(cy)
            envs[3] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[cy], envs).conj()
            mts_conj.update({cy: temp})

            nums = [
                    torch.einsum('aBcdE,Ee,abcde->Bb', mts_conj[c], ops[0], mts[c]),
                    torch.einsum('abcDE,Ee,abcde->Dd', mts_conj[cy], ops[1], mts[cy])]
            dens = [
                    torch.einsum('aBcde,abcde->Bb', mts_conj[c], mts[c]),
                    torch.einsum('abcDe,abcde->Dd', mts_conj[cy], mts[cy])]

            res.append(torch.einsum('ab,ab', *nums) / torch.einsum('ab,ab', *dens))
            # print('Y', torch.einsum('ab,ab', *dens))

        return torch.mean(torch.as_tensor(res))


    def beta_twobody_measure(self, op: torch.tensor) -> torch.tensor:
        r'''
        measure bond energy on beta lattice

        Parameters
        ----------
        op: tensor, twobody operator
        '''

        # SVD to MPO
        u, s, v = tp.linalg.tsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(0, 0))
        ss = torch.sqrt(s).diag().to(self._dtype)
        us = torch.einsum('Aa,abc->Abc', ss, u)
        sv = torch.einsum('Aa,abc->Abc', ss, v)

        mpo = us, sv

        mts = {}
        for c in self._coords:
            mts.update({c: self.merged_tensor(c)})

        # measure on all bonds
        res = []
        for c in self._coords:
            # X-direction
            cx = (c[0]+1) % self._nx, c[1]

            mts_conj = {}

            # bond Lambda matrices mimic the infinite TPS environments 
            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[2] = torch.eye(self._chi).to(self._dtype)
            temp = self.absorb_envs(mts[c], envs).conj()
            mts_conj.update({c: temp})

            envs = self.site_envs(cx)
            envs[0] = torch.eye(self._chi).to(self._dtype)
            temp = self.absorb_envs(mts[cx], envs).conj()
            mts_conj.update({cx: temp})

            # sandwich
            # numerator
            nums = [
                    torch.einsum('abCdE,fEe,abcde->Cfc', mts_conj[c], mpo[0], mts[c]),
                    torch.einsum('AbcdE,fEe,abcde->Afa', mts_conj[cx], mpo[1], mts[cx])]
            # denominator
            dens = [
                    torch.einsum('abCde,abcde->Cc', mts_conj[c], mts[c]),
                    torch.einsum('Abcde,abcde->Aa', mts_conj[cx], mts[cx])]

            res.append(torch.einsum('abc,abc', *nums) / torch.einsum('ab,ab', *dens))

            # Y-direction
            cy = c[0], (c[1]+1) % self._ny

            mts_conj = {}

            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[1] = torch.eye(self._chi).to(self._dtype)
            temp = self.absorb_envs(mts[c], envs).conj()
            mts_conj.update({c: temp})

            envs = self.site_envs(cy)
            envs[3] = torch.eye(self._chi).to(self._dtype)
            temp = self.absorb_envs(mts[cy], envs).conj()
            mts_conj.update({cy: temp})

            # sandwich
            # numerator
            nums = [
                    torch.einsum('aBcdE,fEe,abcde->Bfb', mts_conj[c], mpo[0], mts[c]),
                    torch.einsum('abcDE,fEe,abcde->Dfd', mts_conj[cy], mpo[1], mts[cy])]
            # denominator
            dens = [
                    torch.einsum('aBcde,abcde->Bb', mts_conj[c], mts[c]),
                    torch.einsum('abcDe,abcde->Dd', mts_conj[cy], mts[cy])]

            res.append(torch.einsum('abc,abc', *nums) / torch.einsum('ab,ab', *dens))

        print(torch.tensor(res))

        return torch.mean(torch.as_tensor(res))

    
    def ctm_singular_vals(self):

        svals = {}
        for c in self._coords:

            mps_d = [self._ctms[c][0], self._ctms[c][4], self._ctms[c][1]]
            mps_u = [self._ctms[c][2], self._ctms[c][5], self._ctms[c][3]]
            mps_l = [self._ctms[c][0], self._ctms[c][6], self._ctms[c][2]]
            mps_r = [self._ctms[c][1], self._ctms[c][7], self._ctms[c][3]]

            svals_u, svals_d, svals_l, svals_r = [], [], [], []

            # up
            rs, ls = [], []
            # QR from left to right
            temp = mps_u[0]
            q, r = tp.linalg.tqr(temp, group_dims=((1,), (0,)), qr_dims=(1, 0))
            rs.append(r)
            # merge R in the next tensor
            temp = torch.einsum('ab,bcde->acde', r, mps_u[1])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            rs.append(r)

            # LQ from right to left
            temp = mps_u[-1]
            q, l = tp.linalg.tqr(temp, group_dims=((1,), (0,)), qr_dims=(0, 1))
            ls.append(l)
            # merge L into the previous tensor
            temp = torch.einsum('abcd,be->aecd', mps_u[1], l)
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            ls.append(l)
        
            ls.reverse()

            for i in range(2):
                u, s, v = tp.linalg.svd(rs[i] @ ls[i])
                svals_u.append(s)

            # down
            rs, ls = [], []
            # QR from left to right
            temp = mps_d[0]
            q, r = tp.linalg.tqr(temp, group_dims=((1,), (0,)), qr_dims=(1, 0))
            rs.append(r)
            # merge R in the next tensor
            temp = torch.einsum('ab,bcde->acde', r, mps_d[1])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            rs.append(r)

            # LQ from right to left
            temp = mps_d[-1]
            q, l = tp.linalg.tqr(temp, group_dims=((1,), (0,)), qr_dims=(0, 1))
            ls.append(l)
            # merge L into the previous tensor
            temp = torch.einsum('abcd,be->aecd', mps_d[1], l)
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            ls.append(l)
        
            ls.reverse()

            for i in range(2):
                u, s, v = tp.linalg.svd(rs[i] @ ls[i])
                svals_d.append(s)

            # left
            rs, ls = [], []
            # QR from down to up
            temp = mps_l[0]
            q, r = tp.linalg.tqr(temp, group_dims=((0,), (1,)), qr_dims=(1, 0))
            rs.append(r)
            # merge R in the next tensor
            temp = torch.einsum('ab,bcde->acde', r, mps_l[1])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            rs.append(r)

            # LQ from up to down
            temp = mps_l[-1]
            q, l = tp.linalg.tqr(temp, group_dims=((0,), (1,)), qr_dims=(0, 1))
            ls.append(l)
            # merge L into the previous tensor
            temp = torch.einsum('abcd,be->aecd', mps_l[1], l)
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            ls.append(l)
        
            ls.reverse()

            for i in range(2):
                u, s, v = tp.linalg.svd(rs[i] @ ls[i])
                svals_l.append(s)

            # right
            rs, ls = [], []
            # QR from down to up
            temp = mps_r[0]
            q, r = tp.linalg.tqr(temp, group_dims=((0,), (1,)), qr_dims=(1, 0))
            rs.append(r)
            # merge R in the next tensor
            temp = torch.einsum('ab,bcde->acde', r, mps_r[1])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            rs.append(r)

            # LQ from up to down
            temp = mps_r[-1]
            q, l = tp.linalg.tqr(temp, group_dims=((0,), (1,)), qr_dims=(0, 1))
            ls.append(l)
            # merge L into the previous tensor
            temp = torch.einsum('abcd,be->aecd', mps_r[1], l)
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            ls.append(l)
        
            ls.reverse()

            for i in range(2):
                u, s, v = tp.linalg.svd(rs[i] @ ls[i])
                svals_r.append(s)

            temp_d = {'u': svals_u, 'd': svals_d, 'l': svals_l, 'r': svals_r}

            svals.update({c: temp_d})

        return svals


    def ctmrg_projectors_u(self, c: tuple, dts: dict):
        r'''
        build ALL up CTMRG projectors

        Parameters:
        -----------
        dts: dict, dict of double tensors

        Returns:
        --------
        each pair of projectors are labelled by the coordinate (x, y)
        '''

        # all coordinates for MPS
        cs = [((c[0]+i) % self._nx, c[1]) for i in range(-1, 3)]
        mps = [self._ctms[cs[0]][2], self._ctms[cs[1]][5], self._ctms[cs[2]][5], self._ctms[cs[3]][3]]
        # all coordinates for MPO
        cs = [((c[0]+i) % self._nx, (c[1]-1) % self._ny) for i in range(-1, 3)]
        mpo = [self._ctms[cs[0]][6], dts[cs[1]], dts[cs[2]], self._ctms[cs[3]][7]]
        # MPO operates on MPS
        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('efBb,AaBbCcDd->eAafCcDd', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('efBb,AaBbCcDd->eAafCcDd', mps[2], mpo[2])
        mpo_mps[3] = torch.einsum('ab,cbde->adec', mps[3], mpo[3])
        # split to two parts
        rho_l = torch.einsum('abcd,abcefghi->efgdhi', mpo_mps[0], mpo_mps[1])
        rho_r = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[2], mpo_mps[3])
        # QR factorizations
        t, r = tp.linalg.tqr(rho_l, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(3, 0))
        t, l = tp.linalg.tqr(rho_r, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = torch.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        return pl, pr


    def ctmrg_mu(self, c: tuple, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site within the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS centered with this site
        cs = [((c[0]+i) % self._nx, c[1]) for i in range(-1, 2)]
        mps = [self._ctms[cs[0]][2], self._ctms[cs[1]][5], self._ctms[cs[2]][3]]
        # build temporary MPO in next row
        cs = [((c[0]+i) % self._nx, (c[1]-1) % self._ny) for i in range(-1, 2)]
        mpo = [self._ctms[cs[0]][6], dts[cs[1]], self._ctms[cs[2]][7]]
        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('efBb,AaBbCcDd->eAafCcDd', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,cbde->adec', mps[2], mpo[2])

        # apply projectors and update CTM tensors
        b = (c[0]-1) % self._nx, c[1]
        ps_b = self.ctmrg_projectors_u(b, dts)
        ps_c = self.ctmrg_projectors_u(c, dts)
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps_b[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps_b[1], mpo_mps[1], ps_c[0])
        mps[2] = torch.einsum('abcd,bcde->ae', ps_c[1], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a up-move means absorbing this row
        self._ctms[cs[0]][2] = mps[0] / norms[0]
        self._ctms[cs[1]][5] = mps[1] / norms[1]
        self._ctms[cs[2]][3] = mps[2] / norms[2]

        return 1


    def ctmrg_mu2(self, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site within the unit cell
        dts: dict, double tensors
        '''

        ps = self.ctmrg_projectors_u(dts)

        for c in self._coords:
            # all coordinates for MPS
            cs = [((c[0]+i) % self._nx, c[1]) for i in range(-1, 3)]
            mps = [self._ctms[cs[0]][2], self._ctms[cs[1]][5], self._ctms[cs[2]][5], self._ctms[cs[3]][3]]
            # all coordinates for MPO
            cs = [((c[0]+i) % self._nx, (c[1]-1) % self._ny) for i in range(-1, 3)]
            mpo = [self._ctms[cs[0]][6], dts[cs[1]], dts[cs[2]], self._ctms[cs[3]][7]]
            # MPO operates on MPS
            mpo_mps = [None]*4
            mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
            mpo_mps[1] = torch.einsum('efBb,AaBbCcDd->eAafCcDd', mps[1], mpo[1])
            mpo_mps[2] = torch.einsum('efBb,AaBbCcDd->eAafCcDd', mps[2], mpo[2])
            mpo_mps[3] = torch.einsum('ab,cbde->adec', mps[3], mpo[3])

            # apply projectors and update CTM tensors
            b = (c[0]-1) % self._nx, c[1]
            d = (c[0]+1) % self._nx, c[1]
            mps = [None]*4
            mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps[b][0])
            mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[b][1], mpo_mps[1], ps[c][0])
            mps[2] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[c][1], mpo_mps[2], ps[d][0])
            mps[2] = torch.einsum('abcd,bcde->ae', ps[d][1], mpo_mps[3])

            norms = [torch.linalg.norm(t) for t in mps]
            # a up-move means absorbing this row
            self._ctms[cs[0]][2] = mps[0] / norms[0]
            self._ctms[cs[1]][5] = mps[1] / norms[1]
            self._ctms[cs[2]][5] = mps[2] / norms[2]
            self._ctms[cs[3]][3] = mps[3] / norms[3]

        return 1


    def ctmrg_projectors_d(self, c: tuple, dts: dict):
        r'''
        build ALL down CTMRG projectors

        Parameters:
        -----------
        dts: dict, dict of double tensors

        Returns:
        --------
        each pair of projectors are labelled by the coordinate (x, y)
        '''

        # all coordinates for MPS
        cs = [((c[0]+i) % self._nx, c[1]) for i in range(-1, 3)]
        mps = [self._ctms[cs[0]][0], self._ctms[cs[1]][4], self._ctms[cs[2]][4], self._ctms[cs[3]][1]]
        # all coordinates for MPO
        cs = [((c[0]+i) % self._nx, (c[1]+1) % self._ny) for i in range(-1, 3)]
        mpo = [self._ctms[cs[0]][6], dts[cs[1]], dts[cs[2]], self._ctms[cs[3]][7]]
        # MPO operates on MPS
        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('efDd,AaBbCcDd->eAafCcBb', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('efDd,AaBbCcDd->eAafCcBb', mps[2], mpo[2])
        mpo_mps[3] = torch.einsum('ab,bcde->adec', mps[3], mpo[3])
        # split to two parts
        rho_l = torch.einsum('abcd,abcefghi->efgdhi', mpo_mps[0], mpo_mps[1])
        rho_r = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[2], mpo_mps[3])
        # QR factorizations
        t, r = tp.linalg.tqr(rho_l, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(3, 0))
        t, l = tp.linalg.tqr(rho_r, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = torch.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        return pl, pr


    def ctmrg_md(self, c: tuple[int], dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site with the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS centered with this site
        cs = [((c[0]+i) % self._nx, c[1]) for i in range(-1, 2)]
        mps = [self._ctms[cs[0]][0], self._ctms[cs[1]][4], self._ctms[cs[2]][1]]
        # build temporary MPO in next row
        cs = [((c[0]+i) % self._nx, (c[1]+1) % self._ny) for i in range(-1, 2)]
        mpo = [self._ctms[cs[0]][6], dts[cs[1]], self._ctms[cs[2]][7]]
        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('efDd,AaBbCcDd->eAafCcBb', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,bcde->adec', mps[2], mpo[2])

        # apply projectors and update CTM tensors
        b = (c[0]-1) % self._nx, c[1]
        ps_b = self.ctmrg_projectors_d(b, dts)
        ps_c = self.ctmrg_projectors_d(c, dts)
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps_b[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps_b[1], mpo_mps[1], ps_c[0])
        mps[2] = torch.einsum('abcd,bcde->ae', ps_c[1], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a down-move means absorbing this row
        self._ctms[cs[0]][0] = mps[0] / norms[0]
        self._ctms[cs[1]][4] = mps[1] / norms[1]
        self._ctms[cs[2]][1] = mps[2] / norms[2]

        return 1


    def ctmrg_projectors_l(self, c: tuple, dts: dict):

        # build temporary MPS centered with this site
        cs = [(c[0], (c[1]+j) % self._ny) for j in range(-1, 3)]
        mps = [self._ctms[cs[0]][0], self._ctms[cs[1]][6], self._ctms[cs[2]][6], self._ctms[cs[3]][2]]
        cs = [((c[0]+1) % self._nx, (c[1]+j) % self._ny) for j in range(-1, 3)]
        mpo = [self._ctms[cs[0]][4], dts[cs[1]], dts[cs[2]], self._ctms[cs[3]][5]]
        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[2], mpo[2])
        mpo_mps[3] = torch.einsum('ab,acde->cbde', mps[3], mpo[3])
        # split into two parts
        rho_l = torch.einsum('abcd,bcdefghi->ahiefg', mpo_mps[0], mpo_mps[1])
        rho_r = torch.einsum('abcdefgh,idef->ighabc', mpo_mps[2], mpo_mps[3])
        # QR factorizations
        t, r = tp.linalg.tqr(rho_l, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(3, 0))
        t, l = tp.linalg.tqr(rho_r, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(0, 3))

        u, s, v = torch.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        return pl, pr


    def ctmrg_ml(self, c: tuple[int], dts: dict):

        cs = [(c[0], (c[1]+j) % self._ny) for j in range(-1, 2)]
        mps = [self._ctms[cs[0]][0], self._ctms[cs[1]][6], self._ctms[cs[2]][2]]
        cs = [((c[0]+1) % self._nx, (c[1]+j) % self._ny) for j in range(-1, 2)]
        mpo = [self._ctms[cs[0]][4], dts[cs[1]], self._ctms[cs[2]][5]]
        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
        # f B b
        # | | |--C 
        # *****
        # | | |--c
        # e D d
        mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,acde->cbde', mps[2], mpo[2])

        # apply projectors and update CTM tensors
        b = c[0], (c[1]-1) % self._ny
        ps_b = self.ctmrg_projectors_l(b, dts)
        ps_c = self.ctmrg_projectors_l(c, dts)
        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps_b[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps_b[1], mpo_mps[1], ps_c[0])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps_c[1], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        self._ctms[cs[0]][0] = mps[0] / norms[0]
        self._ctms[cs[1]][6] = mps[1] / norms[1]
        self._ctms[cs[2]][2] = mps[2] / norms[2]

        return 1


    def ctmrg_projectors_r(self, c: tuple, dts: dict):

        cs = [(c[0], (c[1]+j) % self._ny) for j in range(-1, 3)]
        mps = [self._ctms[cs[0]][1], self._ctms[cs[1]][7], self._ctms[cs[2]][7], self._ctms[cs[3]][3]]
        cs = [((c[0]-1) % self._nx, (c[1]+j) % self._ny) for j in range(-1, 3)]
        mpo = [self._ctms[cs[0]][4], dts[cs[1]], dts[cs[2]], self._ctms[cs[3]][5]]
        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[1], mps[1])
        mpo_mps[2] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[2], mps[2])
        mpo_mps[3] = torch.einsum('abcd,be->aecd', mpo[3], mps[3])
        # split into two parts
        rho_l = torch.einsum('abcd,bcdefghi->ahiefg', mpo_mps[0], mpo_mps[1])
        rho_r = torch.einsum('abcdefgh,idef->ighabc', mpo_mps[2], mpo_mps[3])
        # QR factorizations
        t, r = tp.linalg.tqr(rho_l, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(3, 0))
        t, l = tp.linalg.tqr(rho_r, group_dims=((0, 1, 2), (3, 4, 5)), qr_dims=(0, 3))

        u, s, v = torch.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        pl = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        pr = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        return pl, pr


    def ctmrg_mr(self, c: tuple, dts: dict):

        cs = [(c[0], (c[1]+j) % self._ny) for j in range(-1, 2)]
        mps = [self._ctms[cs[0]][1], self._ctms[cs[1]][7], self._ctms[cs[2]][3]]
        cs = [((c[0]-1) % self._nx, (c[1]+j) % self._ny) for j in range(-1, 2)]
        mpo = [self._ctms[cs[0]][4], dts[cs[1]], self._ctms[cs[2]][5]]
        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[1], mps[1])
        mpo_mps[2] = torch.einsum('abcd,be->aecd', mpo[2], mps[2])

        b = c[0], (c[1]-1) % self._ny
        ps_b = self.ctmrg_projectors_r(b, dts)
        ps_c = self.ctmrg_projectors_r(c, dts)
        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps_b[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps_b[1], mpo_mps[1], ps_c[0])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps_c[1], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        self._ctms[cs[0]][1] = mps[0] / norms[0]
        self._ctms[cs[1]][7] = mps[1] / norms[1]
        self._ctms[cs[2]][3] = mps[2] / norms[2]

        return 1


    def ctm_twobody_measure(self, op: torch.tensor) -> list:
        r'''
        measure a twobody operator by CTMRG 

        Parameters
        ----------
        op: tensor, twobody operator
        '''

        # SVD two-body operator to MPO
        u, s, v = tp.linalg.tsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(0, 0))
        ss = torch.sqrt(s).diag().to(self._dtype)
        us = torch.einsum('Aa,abc->Abc', ss, u)
        sv = torch.einsum('Aa,abc->Abc', ss, v)

        op_mpo = us, sv

        mts, mts_conj = {}, {}
        for c in self._coords:
            t = self.merged_tensor(c)
            mts.update({c: t})
            mts_conj.update({c: t.conj()})

        # double tensors
        dts = self.double_tensors()

        meas = []

        for c in self._coords:
            # X-direction
            cx = (c[0]+1) % self._nx, c[1]

            # impure double tensors
            impure_dts = []
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBbCfcDd', mts_conj[c], op_mpo[0], mts[c]))
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AfaBbCcDd', mts_conj[cx], op_mpo[1], mts[cx]))

            # contraction
            env_l = torch.einsum('ab,bcde,fc->adef', 
                                 self._ctms[((c[0]-1) % self._nx, (c[1]-1) % self._ny)][0],
                                 self._ctms[((c[0]-1) % self._nx, c[1])][6],
                                 self._ctms[((c[0]-1) % self._nx, (c[1]+1) % self._ny)][2])
            env_r = torch.einsum('ab,bcde,fc->adef',
                                 self._ctms[((cx[0]+1) % self._nx, (cx[1]-1) % self._ny)][1],
                                 self._ctms[((cx[0]+1) % self._nx, cx[1])][7],
                                 self._ctms[((cx[0]+1) % self._nx, (cx[1]+1) % self._ny)][3])
            # denominator
            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch',
                                temp,
                                self._ctms[(c[0], (c[1]-1) % self._ny)][4],
                                dts[c],
                                self._ctms[(c[0], (c[1]+1) % self._ny)][5])
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch',
                                temp,
                                self._ctms[(cx[0], (cx[1]-1) % self._ny)][4],
                                dts[cx],
                                self._ctms[(cx[0], (cx[1]+1) % self._ny)][5])
            den = torch.einsum('abcd,abcd', temp, env_r)
            # numerator
            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCicDd,ghBb->fCich',
                                temp,
                                self._ctms[(c[0], (c[1]-1) % self._ny)][4],
                                impure_dts[0],
                                self._ctms[(c[0], (c[1]+1) % self._ny)][5])
            temp = torch.einsum('eAiag,efDd,AiaBbCcDd,ghBb->fCch',
                                temp,
                                self._ctms[(cx[0], (cx[1]-1) % self._ny)][4],
                                impure_dts[1],
                                self._ctms[(cx[0], (cx[1]+1) % self._ny)][5])
            num = torch.einsum('abcd,abcd', temp, env_r)
            meas.append(num / den)

            # Y-direction
            cy = c[0], (c[1]+1) % self._ny

            impure_dts = []
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBfbCcDd', mts_conj[c], op_mpo[0], mts[c]))
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBbCcDfd', mts_conj[cy], op_mpo[1], mts[cy]))

            env_d = torch.einsum('ab,acde,cf->bdef',
                                 self._ctms[((c[0]-1) % self._nx, (c[1]-1) % self._ny)][0],
                                 self._ctms[(c[0], (c[1]-1) % self._ny)][4],
                                 self._ctms[((c[0]+1) % self._nx, (c[1]-1) % self._ny)][1])
            env_u = torch.einsum('ab,acde,cf->bdef',
                                 self._ctms[((cy[0]-1) % self._nx, (cy[1]+1) % self._ny)][2],
                                 self._ctms[(cy[0], (cy[1]+1) % self._ny)][5],
                                 self._ctms[((cy[0]+1) % self._nx, (cy[1]+1) % self._ny)][3])
            # denominator
            temp = env_d.clone()
            temp = torch.einsum('eDdg,efAa,AaBbCcDd,ghCc->fBbh',
                                temp,
                                self._ctms[c][6],
                                dts[c],
                                self._ctms[c][7])
            temp = torch.einsum('eDdg,efAa,AaBbCcDd,ghCc->fBbh', temp, self._ctms[cy][6], dts[cy], self._ctms[cy][7])
            den = torch.einsum('abcd,abcd', temp, env_u)
            # numerator
            temp = env_d.clone()
            temp = torch.einsum('eDdg,efAa,AaBibCcDd,ghCc->fBibh', temp, self._ctms[c][6], impure_dts[0], self._ctms[c][7])
            temp = torch.einsum('eDidg,efAa,AaBbCcDid,ghCc->fBbh', temp, self._ctms[cy][6], impure_dts[1], self._ctms[cy][7])

            num = torch.einsum('abcd,abcd', temp, env_u)
            meas.append(num / den)

        return torch.tensor(meas)


    def ctm_onebody_norm(self, c: tuple, dts: dict):

        env_l = torch.einsum('ab,bcde,fc->adef', self._ctms[c][0], self._ctms[c][6], self._ctms[c][2])
        env_r = torch.einsum('ab,bcde,fc->adef', self._ctms[c][1], self._ctms[c][7], self._ctms[c][3])

        temp = env_l.clone()
        temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], dts[c], self._ctms[c][5])

        return torch.einsum('abcd,abcd', temp, env_r)


    def ctm_onebody_measure(self, op: torch.tensor, dts: dict):

        mts, mts_conj = {}, {}
        for temp_c in self._coords:
            t = self.merged_tensor(temp_c)
            mts.update({temp_c: t})
            mts_conj.update({temp_c: t.conj()})

        res = []
        for c in self._coords:

            env_l = torch.einsum('ab,bcde,fc->adef', self._ctms[c][0], self._ctms[c][6], self._ctms[c][2])
            env_r = torch.einsum('ab,bcde,fc->adef', self._ctms[c][1], self._ctms[c][7], self._ctms[c][3])

            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], dts[c], self._ctms[c][5])
            den = torch.einsum('abcd,abcd', temp, env_r)

            impure_dt = torch.einsum('ABCDE,Ee,abcde->AaBbCcDd', mts_conj[c], op, mts[c])
            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], impure_dt, self._ctms[c][5])
            num = torch.einsum('abcd,abcd', temp, env_r)

            res.append(num / den)

        return torch.tensor(res)


####### below are temporary test functions ########

    def _ctm_22_exact_norm(self, tps, ctms):
        r'''
        test function
        to compute the wavefunction inner product for a 2*2 TPS with CTM tensors
        '''

        assert 2 == self._nx and 2 == self._ny, 'unit cell size is not correct'

        cs, up_es, down_es, left_es, right_es = ctms

        mts, mts_conj = {}, {}
        for i, c in enumerate(self._coords):
            mts.update({c: tps[i].clone()})
            mts_conj.update({c: tps[i].conj().clone()})
        # double tensors
        dts = {}
        for i, c in enumerate(self._coords):
            dts.update({c: torch.einsum('ABCDe,abcde->AaBbCcDd', mts_conj[c], mts[c])})

        meas = []

        # prepare up and down MPS
        mps_u = [t.clone() for t in up_es]
        mps_u.insert(0, cs[2])
        mps_u.append(cs[3])

        # temporary MPO
        mpo = []
        for i in range(self._nx):
            mpo.append(dts[(i, 1)])

        mpo.insert(0, left_es[1])
        mpo.append(right_es[1])

        # MPO-MPS operation
        mps_u[0] = torch.einsum('ab,cbde->adec', mps_u[0], mpo[0])
        mps_u[-1] = torch.einsum('ab,cbde->adec', mps_u[-1], mpo[-1])
        for i in range(self._nx):
            mps_u[i+1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps_u[i+1], mpo[i+1])
        # boundary wavefunction
        bou_u = torch.einsum('abcd,abcefghi,efgjklmn,jklo->dhimno', *mps_u)

        mps_d = [t.clone() for t in down_es]
        mps_d.insert(0, cs[0])
        mps_d.append(cs[1])

        # temporary MPO
        mpo = []
        for i in range(self._nx):
            mpo.append(dts[(i, 0)])

        mpo.insert(0, left_es[0])
        mpo.append(right_es[0])

        mps_d[0] = torch.einsum('ab,bcde->adec', mps_d[0], mpo[0])
        mps_d[-1] = torch.einsum('ab,bcde->adec', mps_d[-1], mpo[-1])
        for i in range(self._nx):
            mps_d[i+1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', mps_d[i+1], mpo[i+1])

        bou_d = torch.einsum('abcd,abcefghi,efgjklmn,jklo->dhimno', *mps_d)

        nor_x = torch.einsum('abcdef,abcdef', bou_u, bou_d)

        # prepare left and right MPS
        mps_l = [t.clone() for t in left_es]
        mps_l.insert(0, cs[0])
        mps_l.append(cs[2])

        mpo = []
        for j in range(self._ny):
            mpo.append(dts[(0, j)])

        mpo.insert(0, down_es[0])
        mpo.append(up_es[0])

        mps_l[0] = torch.einsum('ab,acde->cbde', mps_l[0], mpo[0])
        mps_l[-1] = torch.einsum('ab,acde->cbde', mps_l[-1], mpo[-1])

        for j in range(self._ny):
            mps_l[j+1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps_l[j+1], mpo[j+1])
        
        bou_l = torch.einsum('abcd,bcdefghi,efgjklmn,ojkl->ahimno', *mps_l)

        mps_r = [t.clone() for t in right_es]
        mps_r.insert(0, cs[1])
        mps_r.append(cs[3])

        mpo = []
        for j in range(self._ny):
            mpo.append(dts[(1, j)])

        mpo.insert(0, down_es[1])
        mpo.append(up_es[1])

        mps_r[0] = torch.einsum('abcd,be->aecd', mpo[0], mps_r[0])
        mps_r[-1] = torch.einsum('abcd,be->aecd', mpo[-1], mps_r[-1])
        
        for j in range(self._ny):
            mps_r[j+1] = torch.einsum('fgCc,AaBbCcDd->fDdgBbAa', mps_r[j+1], mpo[j+1])

        bou_r = torch.einsum('abcd,bcdefghi,efgjklmn,ojkl->ahimno', *mps_r)

        nor_y = torch.einsum('abcdef,abcdef', bou_l, bou_r)

        print('Exact norms, X and Y:')
        print(nor_x.item(), nor_y.item(), (nor_x-nor_y).item())

        return 1


    def _ctm_22_RG_norm(self, tps, ctms):
        r'''
        test function
        to compute the wavefunction inner product for a 2*2 TPS with CTM tensors
        '''

        assert 2 == self._nx and 2 == self._ny, 'unit cell size is not correct'

        cs, up_es, down_es, left_es, right_es = ctms

        mts, mts_conj = {}, {}
        for i, c in enumerate(self._coords):
            mts.update({c: tps[i].clone()})
            mts_conj.update({c: tps[i].conj().clone()})
        # double tensors
        dts = {}
        for i, c in enumerate(self._coords):
            dts.update({c: torch.einsum('ABCDe,abcde->AaBbCcDd', mts_conj[c], mts[c])})

        meas = []

        # prepare up and down MPS
        mps_u = [t.clone() for t in up_es]
        mps_u.insert(0, cs[2])
        mps_u.append(cs[3])

        # temporary MPO
        mpo = []
        for i in range(self._nx):
            mpo.append(dts[(i, 1)])

        mpo.insert(0, left_es[1])
        mpo.append(right_es[1])

        # RG
        mps_u = self.ctmrg_move_up(mps_u, mpo)
        # boundary
        bou_u = torch.einsum('ab,acde,cfgh,fi->bdeghi', *mps_u)

        mps_d = [t.clone() for t in down_es]
        mps_d.insert(0, cs[0])
        mps_d.append(cs[1])

        # temporary MPO
        mpo = []
        for i in range(self._nx):
            mpo.append(dts[(i, 0)])

        mpo.insert(0, left_es[0])
        mpo.append(right_es[0])

        mps_d = self.ctmrg_move_down(mps_d, mpo)
        bou_d = torch.einsum('ab,acde,cfgh,fi->bdeghi', *mps_d)

        nor_x = torch.einsum('abcdef,abcdef', bou_u, bou_d)

        # prepare left and right MPS
        mps_l = [t.clone() for t in left_es]
        mps_l.insert(0, cs[0])
        mps_l.append(cs[2])

        mpo = []
        for j in range(self._ny):
            mpo.append(dts[(0, j)])

        mpo.insert(0, down_es[0])
        mpo.append(up_es[0])

        mps_l = self.ctmrg_move_left(mps_l, mpo)
        bou_l = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps_l)

        mps_r = [t.clone() for t in right_es]
        mps_r.insert(0, cs[1])
        mps_r.append(cs[3])

        mpo = []
        for j in range(self._ny):
            mpo.append(dts[(1, j)])

        mpo.insert(0, down_es[1])
        mpo.append(up_es[1])

        mps_r = self.ctmrg_move_right(mps_r, mpo)
        bou_r = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps_r)

        nor_y = torch.einsum('abcdef,abcdef', bou_l, bou_r)

        print('RG norms, X and Y:')
        print(nor_x.item(), nor_y.item(), (nor_x-nor_y).item())

        return 1

    def _ctmrg_22_left_up_rotate(self, dts, init_ctms):

        cs, up_es, down_es, left_es, right_es = init_ctms

        rho = cs[0].shape[0]

        # build a left MPS
        mps_l = [t.clone() for t in left_es]
        mps_l.insert(0, cs[0])
        mps_l.append(cs[2])

        i = 1
        mpo_l = []
        for j in range(self._ny):
            mpo_l.append(dts[(i, j)])

        mpo_l.insert(0, down_es[i])
        mpo_l.append(up_es[i])

        new_mps_l = self.ctmrg_move_left(mps_l, mpo_l)

        # new_mps = self.ctmrg_move_left(mps, mpo)

        # rotate the left MPS to a up one
        # permute head and tail
        mps_u = [t.clone() for t in mps_l]
        mps_u[0] = torch.permute(mps_u[0], (1, 0))
        mps_u[-1] = torch.permute(mps_u[-1], (1, 0))
      
        mpo_u = [t.clone() for t in mpo_l]
        # permute MPO
        mpo_u[0] = torch.permute(mpo_u[0], (1, 0, 2, 3))
        mpo_u[-1] = torch.permute(mpo_u[-1], (1, 0, 2, 3))
        mpo_u[1] = torch.permute(mpo_u[1], (6, 7, 0, 1, 2, 3, 4, 5))
        mpo_u[2] = torch.permute(mpo_u[2], (6, 7, 0, 1, 2, 3, 4, 5))
        new_mps_u = self.ctmrg_move_up(mps_u, mpo_u)

        # rotate back
        new_mps_u[0] = torch.permute(new_mps_u[0], (1, 0))
        new_mps_u[-1] = torch.permute(new_mps_u[-1], (1, 0))

        print('test MPS left-up rotation:')
        for i in range(4):
            print(i, new_mps_l[i].shape)
            print(torch.linalg.norm(new_mps_l[i]-new_mps_u[i]))

        return 1

    def _ctmrg_22_right_down_rotate(self, dts, init_ctms):

        cs, up_es, down_es, left_es, right_es = init_ctms

        rho = cs[0].shape[0]

        # build a right MPS
        mps_r = [t.clone() for t in right_es]
        mps_r.insert(0, cs[1].clone())
        mps_r.append(cs[3].clone())

        i = 1
        mpo_r = []
        for j in range(self._ny):
            mpo_r.append(dts[(i, j)])

        mpo_r.insert(0, down_es[i].clone())
        mpo_r.append(up_es[i].clone())

        new_mps_r = self.ctmrg_move_right(mps_r, mpo_r)

        # rotate the left MPS to a up one
        # permute head and tail
        mps_d = [t.clone() for t in mps_r]
        mps_d[0] = torch.permute(mps_d[0], (1, 0))
        mps_d[-1] = torch.permute(mps_d[-1], (1, 0))
      
        mpo_d = [t.clone() for t in mpo_r]
        # permute MPO
        mpo_d[0] = torch.permute(mpo_d[0], (1, 0, 2, 3))
        mpo_d[-1] = torch.permute(mpo_d[-1], (1, 0, 2, 3))
        mpo_d[1] = torch.permute(mpo_d[1], (6, 7, 0, 1, 2, 3, 4, 5))
        mpo_d[2] = torch.permute(mpo_d[2], (6, 7, 0, 1, 2, 3, 4, 5))

        new_mps_d = self.ctmrg_move_down(mps_d, mpo_d)

        new_mps_d[0] = torch.permute(new_mps_d[0], (1, 0))
        new_mps_d[-1] = torch.permute(new_mps_d[-1], (1, 0))

        print('test MPS right-down rotation:')
        for i in range(4):
            print(i, new_mps_r[i].shape, torch.linalg.norm(new_mps_r[i]-new_mps_d[i]))

        return 1


if __name__ == '__main__':
    print('test')

    rho, D = 8, 4

    mps = []
    mps.append(torch.rand(rho, D, D, rho))
    mps.append(torch.rand(rho, D, D, rho, D, D, D, D))
    mps.append(torch.rand(rho, D, D, rho, D, D, D, D))
    mps.append(torch.rand(rho, D, D, rho))

    n = len(mps)-2

    rs, ls = [], []
    # QR from left to right
    temp = mps[0]
    q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
    rs.append(r)
   
    for i in range(1, n+1):
        # merge R in the next tensor
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, mps[i])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        rs.append(r)

    # LQ from right to left
    temp = mps[-1]
    q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
    ls.append(l)

    for i in range(n, 0, -1):
        # merge L into the previous tensor
        temp = torch.einsum('abcdefgh,defi->abcigh', mps[i], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
        ls.append(l)
    
    ls.reverse()

    # build projectors on each inner bond
    # there are (nx+1) inner bonds
    # left-, right- means on each bond
    prs, pls = [], []
    s_vals = []
    for i in range(n+1):
        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]))
        # truncate
        ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        # inverse of square root
        sst_inv = (1.0 / torch.sqrt(st)).diag()

        s_vals.append(st)
        print(st)

        pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
        pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
        prs.append(pr)
        pls.append(pl)

    # apply projectors to compress the MPO-MPS
    new_mps = [None]*(0+2)
    new_mps[0]= torch.einsum('abcd,abce->ed', mps[0], prs[0])
    new_mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mps[-1])

    for i in range(1, n+1):
        new_mps[i] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i-1], mps[i], prs[i])

    # mps_wf = torch.einsum('abcd,abce->de', *mps)
    # mps_wf = torch.einsum('abcd,abcefghi,efgj->dhij', *mps)
    mps_wf = torch.einsum('abcd,abcefghi,efgjklmn,jklo->dhimno', *mpo_mps)
    print('direct SVD:')
    test_svals = []
    u, s, v = tp.linalg.tsvd(mps_wf, group_dims=((0,), (1, 2, 3)), svd_dims=(1,  0))
    test_svals.append(s)
    print(s)
    u, s, v = tp.linalg.tsvd(mps_wf, group_dims=((0, 1, 2), (3,)), svd_dims=(3,  0))
    print(s)
    test_svals.append(s)

    for x, y in zip(s_vals, test_svals):
        print(torch.linalg.norm(x-y))


