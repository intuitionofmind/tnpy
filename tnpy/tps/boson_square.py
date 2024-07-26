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

    def __init__(self, site_tensors: dict, link_tensors: dict, dim_phys=2, cflag=False):
        r'''initialization

        Parameters
        ----------
        '''

        # for spin-1/2
        self._dim_phys = dim_phys

        self._cflag = cflag

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

        if self._cflag:
            for key in self._site_tensors:
                self._site_tensors[key] = self._site_tensors[key].cdouble()

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

        return deepcopy(self._site_tensors)

    def link_tensors(self):

        return deepcopy(self._link_tensors)

    def site_envs(self, site: tuple, inner_bonds=()) -> list:
        r'''
        return the environment bond weights around a site as a list

        Parameters
        ----------
        site: tuple, coordinate
        inner_bonds: tuple, optional, the inner bonds will be returned by square root of tensors

        Returns
        -------
        envs: tuple[tensor]
        '''

        envs = []
        envs.append(self._link_tensors[((site[0]-1) % self._nx, site[1])][0])
        envs.append(self._link_tensors[site][1])
        envs.append(self._link_tensors[site][0])
        envs.append(self._link_tensors[(site[0], (site[1]-1) % self._ny)][1])

        for j in inner_bonds:
            envs[j] = torch.sqrt(envs[j])

        return envs

    def absorb_envs(self, t, envs):

        r'''
        # find the optimal path
        path_info = oe.contract_path('abcde,Aa,bB,cC,Dd->ABCDe', t, *envs, optimize='optimal')
        print(path_info)
        
        --------------------------------------------------------------------------------
        scaling        BLAS                current                             remaining
        --------------------------------------------------------------------------------
        6           GEMM        Aa,abcde->Abcde                 bB,cC,Dd,Abcde->ABCDe
        6           TDOT        Abcde,bB->AcdeB                    cC,Dd,AcdeB->ABCDe
        6           TDOT        AcdeB,cC->AdeBC                       Dd,AdeBC->ABCDe
        6           TDOT        AdeBC,Dd->ABCDe                          ABCDe->ABCDe)
        '''

        temp = torch.einsum('Aa,abcde->Abcde', envs[0], t)
        temp = torch.einsum('abcde,bB->aBcde', temp, envs[1])
        temp = torch.einsum('abcde,cC->abCde', temp, envs[2])

        return torch.einsum('Dd,abcde->abcDe', envs[3], temp)

    def merged_tensor(self, site):
        r'''
        return site tensor merged with square root of link tensors around

        Parameters
        ----------
        site: tuple[int], coordinate
        '''

        envs = self.site_envs(site, inner_bonds=(0, 1, 2, 3))

        return self.absorb_envs(self._site_tensors[site], envs)

    def simple_update_proj(self, time_evo_mpo: tuple):
        r'''
        simple update by projectors

        Parameters
        ----------
        time_evo_mpo: tuple[tensor], time evolution operator MPO
        '''

        cut_off = self._chi

        for c in self._coords:

            # forward sites along two directions
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny

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
            u, s, v = tp.linalg.svd(temp, full_matrices=False)

            # truncate
            ut, st, vt = u[:, :cut_off], s[:cut_off], v[:cut_off, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()

            sst = torch.sqrt(st)
            # safe inverse because of the trunction
            sst_inv = (1.0 / sst).diag()

            if self._cflag:
                st = st.cdouble()
                sst_inv = sst_inv.cdouble()

            # build projectors
            pr = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

            # update the link tensor
            self._link_tensors[c][0] = (st / torch.linalg.norm(st)).diag()

            # apply projectors
            updated_mts = [
                    torch.einsum('abCcde,Ccf->abfde', te_mts[0], pr),
                    torch.einsum('fAa,Aabcde->fbcde', pl, te_mts[1])]

            # remove external environments and update site tensors
            # replace the connected environment by the updated one
            tens_env[0][2], tens_env[1][0] = sst.diag(), sst.diag()

            tens_env_inv = [
                    [torch.linalg.pinv(m) for m in tens_env[0]],
                    [torch.linalg.pinv(m) for m in tens_env[1]]]
            updated_ts = [
                    self.absorb_envs(updated_mts[0], tens_env_inv[0]),
                    self.absorb_envs(updated_mts[1], tens_env_inv[1])]

            self._site_tensors[c] = updated_ts[0] / torch.linalg.norm(updated_ts[0])
            self._site_tensors[cx] = updated_ts[1] / torch.linalg.norm(updated_ts[1])

            # Y-direction
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
            u, s, v = tp.linalg.svd(temp, full_matrices=False)

            # truncate
            ut, st, vt = u[:, :cut_off], s[:cut_off], v[:cut_off, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()

            sst = torch.sqrt(st)
            sst_inv = (1.0 / sst).diag()

            if self._cflag:
                st = st.cdouble()
                sst_inv = sst_inv.cdouble()

            # build projectors
            pr = torch.einsum('abc,cd->abd', l, vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcd->acd', sst_inv @ ut_dagger, r)

            # update the link tensor
            self._link_tensors[c][1] = (st / torch.linalg.norm(st)).diag()

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
                    [torch.linalg.pinv(m) for m in tens_env[0]],
                    [torch.linalg.pinv(m) for m in tens_env[1]]]
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

    def beta_twobody_measure(self, op):
        r'''
        measure bond energy on beta lattice

        Parameters
        ----------
        op: tensor, twobody operator
        '''

        # SVD to MPO
        u, s, v = tp.linalg.tsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(0, 0))
        ss = torch.sqrt(s).diag()
        us = torch.einsum('Aa,abc->Abc', ss, u)
        sv = torch.einsum('Aa,abc->Abc', ss, v)

        mpo = us, sv

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
            # absorb the operator
            mts_conj.update({c: temp})

            envs = self.site_envs(cx)
            envs[0] = torch.eye(self._chi)
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
            # print('MPO:', torch.einsum('abc,abc', *nums), torch.einsum('ab,ab', *dens), res[-1])

            '''
            # wave-function test
            wf = torch.einsum('abcde,cfghi->abdefghi', mts[c], mts[cx])
            wf_conj = torch.einsum('abcde,cfghi->abdefghi', mts_conj[c], mts_conj[cx])

            num = torch.einsum('abcDefgH,DHdh,abcdefgh', wf_conj, op, wf)
            den = torch.einsum('abcdefgh,abcdefgh', wf_conj, wf)
            print('WF test:', num, den, num / den)
            '''

            # Y-direction
            mts_conj = {}

            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[1] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[c], envs).conj()
            # absorb the operator
            mts_conj.update({c: temp})

            envs = self.site_envs(cy)
            envs[3] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[cy], envs).conj()
            mts_conj.update({cy: temp})

            nums = [
                    torch.einsum('aBcdE,fEe,abcde->Bfb', mts_conj[c], mpo[0], mts[c]),
                    torch.einsum('abcDE,fEe,abcde->Dfd', mts_conj[cy], mpo[1], mts[cy])]
            dens = [
                    torch.einsum('aBcde,abcde->Bb', mts_conj[c], mts[c]),
                    torch.einsum('abcDe,abcde->Dd', mts_conj[cy], mts[cy])]
 
            res.append(torch.einsum('abc,abc', *nums) / torch.einsum('ab,ab', *dens))

        return torch.mean(torch.as_tensor(res))

    def ctmrg(self, rho: int, num_rg=10, init=None):
        r'''
        corner transfer matrix renormalization group method to contract

        Parameters
        ----------
        rho: int, bond dimension of boundary MPS
        num_rg: int, number of RG times

        C2     E      C3
        *-0  0-*-1  0-*
        |     / \     |
        1    2   3    1

        '''

        if init is None:
            # random initialization CTMRG tensors
            # corners
            cs = [torch.rand(rho, rho) for i in range(4)]

            # boundaries
            upper_es = [torch.rand(rho, rho, self._chi, self._chi) for i in range(self._nx)]
            lower_es = deepcopy(upper_es)
            left_es = [torch.rand(rho, rho, self._chi, self._chi) for i in range(self._ny)]
            right_es = deepcopy(left_es)

        # merged tensors as MPO
        mts, mts_conj = {}, {}
        for c in self._coords:
            temp = self.merged_tensor(c)
            mts.update({c: temp})
            mts_conj.update({c: temp.conj()})

        # trace the singular values
        su, sd, sl, sr = [], [], [], []
        for r in range(num_rg):
            # upper MPS
            print('upper')
            mps = deepcopy(upper_es)
            mps.insert(0, cs[2])
            mps.append(cs[3])

            # merge MPO into this boundary MPS
            # inner bonds are "thickened"
            # reversed order from up to down
            for j in range(self._ny):

                # head and tail
                mps[0] = torch.einsum('ab,cbde->adec', mps[0], left_es[-(1+j)])
                mps[-1] = torch.einsum('ab,cbde->adec', mps[-1], right_es[-(1+j)])
                
                for i in range(self._nx):
                    # coordinate for current site
                    c = (i, self._ny-1-j)
                    print(c)
                    # print(c)
                    # f- -g
                    # A-*-C
                    # a- -c
                    #  / \
                    #  D d
                    temp = torch.einsum('ABCDe,fgBb->ACDefgb', mts_conj[c], mps[i+1])
                    mps[i+1] = torch.einsum('ACDefgb,abcde->fAagCcDd', temp, mts[c])

                # compress this MPS
                # QR and LQ factorizations

                # residual tensors:
                # R:
                #     --1
                # 0--*--2
                #     --3
                # L:
                # 0--
                # 1--*--3
                # 2--

                rs, ls = [], []

                # QR from left to right
                temp = mps[0]
                q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
                rs.append(r)
                
                for i in range(self._nx):
                    temp = torch.einsum('abcd,bcdefghi->aefghi', r, mps[i+1])
                    q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
                    rs.append(r)

                # LQ from right to left
                temp = mps[-1]
                q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
                ls.append(l)

                for i in range(self._nx):
                    temp = torch.einsum('abcdefgh,defi->abcigh', mps[-(2+i)], l)
                    q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
                    ls.append(l)
                
                ls.reverse()

                # build projectors on each inner bond
                # there are (nx+1) inner bonds
                # left-, right- means on each bond
                prs, pls = [], []

                sts = []
                for i in range(self._nx+1):
                    u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]), full_matrices=False)
                    # truncate
                    ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                    ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                    # inverse of square root
                    sst_inv = (1.0 / torch.sqrt(st)).diag()

                    sts.append(st)

                    if self._cflag:
                        sst_inv = sst_inv.cdouble()

                    pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
                    pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
                    prs.append(pr)
                    pls.append(pl)

                su.append(sts)

                # apply projectors to compress the MPS
                # head and tail
                mps[0]= torch.einsum('abcd,abce->ed', mps[0], prs[0])
                mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mps[-1])

                for i in range(self._nx):
                    mps[i+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i], mps[i+1], prs[i+1])

                for t in mps:
                    print(t.shape)

                # update CTMRG tensors
                cs[2] = mps[0] / torch.linalg.norm(mps[0])
                cs[3] = mps[-1] / torch.linalg.norm(mps[-1])

                for i in range(self._nx):
                    upper_es[i] = mps[i+1] / torch.linalg.norm(mps[i+1])

            if len(su) > 2:
                # unit cell size
                new_sts, old_sts = su[-1], su[-3]
                for i in range(self._nx+1):
                    print('upper MPS diff:', i, torch.linalg.norm(new_sts[i]-old_sts[i]))

            # lower MPS
            print('lower')
            mps = deepcopy(lower_es)
            mps.insert(0, cs[0])
            mps.append(cs[1])

            for j in range(self._ny):

                # merge MPO into the boundary MPS
                # inner bonds are "thickened"
                # head and tail
                mps[0] = torch.einsum('ab,acde->cbde', mps[0], left_es[j])
                mps[-1] = torch.einsum('ab,bcde->adec', mps[-1], right_es[j])
                
                for i in range(self._nx):
                    # coordinate for current site
                    c = (i, j)
                    print(c)
                    # f- -g
                    # A-*-C
                    # a- -c
                    #  / \
                    #  D d
                    temp = torch.einsum('ABCDe,fgDd->ABCefgd', mts_conj[c], mps[i+1])
                    mps[i+1] = torch.einsum('ABCefgd,abcde->fAagCcBb', temp, mts[c])

                # compress this MPS
                # QR and LQ factorizations

                # residual tensors:
                #        --1
                # R: 0--*--2
                #        --3
                #
                #    0--
                # L: 1--*--3
                #    2--

                rs, ls = [], []

                # QR from left to right
                temp = mps[0]
                q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
                rs.append(r)
                
                for i in range(self._nx):
                    temp = torch.einsum('abcd,bcdefghi->aefghi', r, mps[i+1])
                    q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
                    rs.append(r)

                # LQ from right to left
                temp = mps[-1]
                q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
                ls.append(l)

                for i in range(self._nx):
                    temp = torch.einsum('abcdefgh,defi->abcigh', mps[-(2+i)], l)
                    q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
                    ls.append(l)
                
                ls.reverse()

                # build projectors
                prs, pls = [], []
                for i in range(self._nx+1):
                    u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]), full_matrices=False)
                    # truncate
                    ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                    ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                    # inverse of square root
                    sst_inv = (1.0 / torch.sqrt(st)).diag()

                    if self._cflag:
                        sst_inv = sst_inv.cdouble()

                    pr = torch.einsum('abcd,de,ef->abcf', ls[i], vt_dagger, sst_inv)
                    pl = torch.einsum('ab,bc,cdef->adef', sst_inv, ut_dagger, rs[i])
                    prs.append(pr)
                    pls.append(pl)

                # apply projectors to compress the MPS
                # head and tail
                mps[0]= torch.einsum('abcd,bcde->ae', mps[0], prs[0])
                mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mps[-1])

                for i in range(self._nx):
                    mps[i+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i], mps[i+1], prs[i+1])

                for t in mps:
                    print(t.shape)

                # update CTMRG tensors
                cs[0] = mps[0] / torch.linalg.norm(mps[0])
                cs[1] = mps[-1] / torch.linalg.norm(mps[-1])

                for i in range(self._nx):
                    lower_es[i] = mps[i+1] / torch.linalg.norm(mps[i+1])

            # left MPS
            print('left')
            mps = deepcopy(left_es)
            mps.insert(0, cs[0])
            mps.append(cs[2])

            for i in range(self._nx):

                # merge MPO into the boundary MPS
                # inner bonds are "thickened"
                # head and tail
                mps[0] = torch.einsum('ab,bcde->adec', mps[0], lower_es[i])
                mps[-1] = torch.einsum('ab,acde->cbde', mps[-1], upper_es[i])
                
                for j in range(self._ny):
                    # coordinate for current site
                    c = (i, j)
                    print(c)
                    # f- -g
                    # A-*-C
                    # a- -c
                    #  / \
                    #  D d
                    temp = torch.einsum('ABCDe,fgAa->BCDefga', mts_conj[c], mps[j+1])
                    mps[j+1] = torch.einsum('BCDefga,abcde->fDdgBbCc', temp, mts[c])

                # compress this MPS
                # QR and LQ factorizations

                # residual tensors:
                # R
                # 1 2 3 
                # | | |
                #   *
                #   |
                #   0
                # L
                #   3
                #   |
                #   *
                # | | |
                # 0 1 2 

                rs, ls = [], []

                # QR from botton to top
                temp = mps[0]
                q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 0))
                rs.append(r)
                
                for j in range(self._ny):
                    temp = torch.einsum('abcd,bcdefghi->aefghi', r, mps[j+1])
                    q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
                    rs.append(r)

                # LQ from top to botton
                temp = mps[-1]
                q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
                ls.append(l)

                for j in range(self._ny):
                    temp = torch.einsum('abcdefgh,defi->abcigh', mps[-(2+j)], l)
                    q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
                    ls.append(l)
                
                ls.reverse()

                # build projectors
                prs, pls = [], []
                for j in range(self._ny+1):
                    u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]), full_matrices=False)
                    # truncate
                    ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                    ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                    # inverse of square root
                    sst_inv = (1.0 / torch.sqrt(st)).diag()

                    if self._cflag:
                        sst_inv = sst_inv.cdouble()

                    pr = torch.einsum('abcd,de,ef->abcf', ls[j], vt_dagger, sst_inv)
                    pl = torch.einsum('ab,bc,cdef->adef', sst_inv, ut_dagger, rs[j])
                    prs.append(pr)
                    pls.append(pl)

                # apply projectors to compress the MPS
                # head and tail
                mps[0]= torch.einsum('abcd,abce->ed', mps[0], prs[0])
                mps[-1] = torch.einsum('abcd,ebcd->ae', pls[-1], mps[-1])

                for j in range(self._ny):
                    mps[j+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j], mps[j+1], prs[j+1])

                for t in mps:
                    print(t.shape)

                # update CTMRG tensors
                cs[0] = mps[0] / torch.linalg.norm(mps[0])
                cs[2] = mps[-1] / torch.linalg.norm(mps[-1])

                for j in range(self._ny):
                    left_es[j] = mps[j+1] / torch.linalg.norm(mps[j+1])

            # right MPS
            print('right')
            mps = deepcopy(right_es)
            mps.insert(0, cs[1])
            mps.append(cs[3])

            for i in range(self._nx):

                # merge MPO into the boundary MPS
                # inner bonds are "thickened"
                # head and tail
                mps[0] = torch.einsum('ab,cade->cbde', mps[0], lower_es[-(1+i)])
                mps[-1] = torch.einsum('ab,cade->cbde', mps[-1], upper_es[-(1+i)])
                
                for j in range(self._ny):
                    # coordinate for current site
                    c = (self._nx-1-i, j)
                    print(c)
                    temp = torch.einsum('ABCDe,fgCc->ABDefgc', mts_conj[c], mps[j+1])
                    mps[j+1] = torch.einsum('ABDefgc,abcde->fDdgBbAa', temp, mts[c])

                # compress this MPS
                # QR and LQ factorizations
                # residual tensors:
                # R
                # 1 2 3 
                # | | |
                #   *
                #   |
                #   0
                # L
                #   3
                #   |
                #   *
                # | | |
                # 0 1 2 

                rs, ls = [], []

                # QR from botton to top
                temp = mps[0]
                q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
                rs.append(r)
                
                for j in range(self._ny):
                    temp = torch.einsum('abcd,bcdefghi->aefghi', r, mps[j+1])
                    q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
                    rs.append(r)

                # LQ from top to botton
                temp = mps[-1]
                q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
                ls.append(l)

                for j in range(self._ny):
                    temp = torch.einsum('abcdefgh,defi->abcigh', mps[-(2+j)], l)
                    q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
                    ls.append(l)
                
                ls.reverse()

                # build projectors
                prs, pls = [], []
                for j in range(self._ny+1):
                    u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]), full_matrices=False)
                    # truncate
                    ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                    ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                    # inverse of square root
                    sst_inv = (1.0 / torch.sqrt(st)).diag()

                    if self._cflag:
                        sst_inv = sst_inv.cdouble()

                    pr = torch.einsum('abcd,de,ef->abcf', ls[j], vt_dagger, sst_inv)
                    pl = torch.einsum('ab,bc,cdef->adef', sst_inv, ut_dagger, rs[j])
                    prs.append(pr)
                    pls.append(pl)

                # apply projectors to compress the MPS
                # head and tail
                mps[0]= torch.einsum('abcd,bcde->ae', mps[0], prs[0])
                mps[-1] = torch.einsum('abcd,ebcd->ae', pls[-1], mps[-1])

                for j in range(self._ny):
                    mps[j+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j], mps[j+1], prs[j+1])

                for t in mps:
                    print(t.shape)

                # update CTMRG tensors
                cs[1] = mps[0] / torch.linalg.norm(mps[0])
                cs[3] = mps[-1] / torch.linalg.norm(mps[-1])

                for j in range(self._ny):
                    right_es[j] = mps[j+1] / torch.linalg.norm(mps[j+1])

        return 1
