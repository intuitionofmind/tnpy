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
    the unit cell size at least: nx=2, ny=1
    '''

    def __init__(self, site_tensors: dict, link_tensors: dict, dim_phys=2, cflag=False, ctms=None):
        r'''initialization

        Parameters
        ----------
        site_tensors: dict, {key: coordinate, value: tensor}
        link_tensors: dict, {key: coordinate, value: tensor}
        ctms: dict, {key: coordinate, value: list}, environment tensors for CTMRG
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

        assert self._nx > 1 and self._ny > 0, 'the TPS unit cell should be at least: 2*1'

        # inner bond dimension
        self._chi = self._site_tensors[(0, 0)].shape[0]

        self._ctms = ctms

    @classmethod
    def rand(cls, nx: int, ny: int, chi: int, cflag=False):
        r'''
        generate a random SquareTPS

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
            site_tensors[(x, y)] = temp / torch.linalg.norm(temp)
            link_tensors[(x, y)] = [lam_x / torch.linalg.norm(lam_x), lam_y / torch.linalg.norm(lam_y)]

        return cls(site_tensors, link_tensors)


    @classmethod
    def randn(cls, nx: int, ny: int, chi: int, cflag=False):
        r'''
        generate a random SquareTPS

        Parameters
        ----------
        nx: int, number of sites along x-direction in a unit cell
        ny: int, number of sites along y-direction in a unit cell
        chi: int, bond dimension of the site tensor
        '''

        site_shape = (chi, chi, chi, chi, 2)
        site_tensors, link_tensors = {}, {}

        for x, y in itertools.product(range(nx), range(ny)):
            temp = torch.randn(site_shape)
            lam_x = torch.randn(chi).diag()
            lam_y = torch.randn(chi).diag()

            if cflag:
                temp = temp.cdouble()
                lam_x = lam_x.cdouble()
                lam_y = lam_y.cdouble()

            # normalization
            site_tensors[(x, y)] = temp / torch.linalg.norm(temp)
            link_tensors[(x, y)] = [lam_x / torch.linalg.norm(lam_x), lam_y / torch.linalg.norm(lam_y)]

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


    @property
    def bond_dim(self):

        return self._chi


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


    def unified_tensor(self) -> torch.tensor:
        r'''
        return the TPS tensor in a unit cell as a whole tensor
        new dimension is created as dim=0
        '''

        tens = []
        for c in self._coords:
            tens.append(self.merged_tensor(c))

        return torch.stack(tens, dim=0)


    def double_tensor(self, c: tuple):
        r'''
        return a double tensors

        Parameters:
        ----------
        '''

        temp = self.merged_tensor(c)

        return torch.einsum('ABCDe,abcde->AaBbCcDd', temp.conj(), temp)


    def double_tensors(self, uc_coord: tuple[int]) -> dict:
        r'''
        return double tensors as a dict

        Parameters:
        ----------
        uc_coord: tuple[int], coordinate for current unit cell
        '''

        dts = []
        for j, i in itertools.product(range(self._ny), range(self._nx)):
            c = (uc_coord[0]+i) % self._nx, (uc_coord[1]+j) % self._ny
            temp = self.merged_tensor(c)
            # (i, j) as the inner coordinate within the unit cell
            dts.update({(i, j): torch.einsum('ABCDe,abcde->AaBbCcDd', temp.conj(), temp)})

        return dts


    def init_ctms(self, init_ctms):

        self._ctms = init_ctms

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


    def beta_twobody_measure(self, op: torch.tensor) -> torch.tensor:
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
            # X-direction
            cx = (c[0]+1) % self._nx, c[1]

            mts_conj = {}

            # bond Lambda matrices mimic the infinite TPS environments 
            envs = self.site_envs(c)
            # replace the connected bond by an identity
            envs[2] = torch.eye(self._chi)
            temp = self.absorb_envs(mts[c], envs).conj()
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

            # Y-direction
            cy = c[0], (c[1]+1) % self._ny

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


    def ctmrg_move_up_with_svs(self, mps: list, mpo: list):
        r'''
        one up step of CTMRG

        Parameters:
        ----------
        mps: list[tensor], the boundary MPS
        mpo: list[tensor], the MPO
        '''

        assert len(mps) == self._nx+2, 'this boundary MPS is not valid for length'
        assert len(mpo) == self._nx+2, 'this MPO is not valid for length'

        rho = mps[0].shape[0]

        # MPO-MPS operation
        # need a new variable, otherwise will change the input arguments
        mpo_mps = [None]*(self._nx+2)
        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[-1] = torch.einsum('ab,cbde->adec', mps[-1], mpo[-1])
        for i in range(1, self._nx+1):
            mpo_mps[i] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps[i], mpo[i])

        rs, ls = [], []
        # QR from left to right
        temp = mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        rs.append(r)
        
        for i in range(1, self._nx+1):
            # merge R in the next tensor
            temp = torch.einsum('abcd,bcdefghi->aefghi', r, mpo_mps[i])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
            rs.append(r)

        # LQ from right to left
        temp = mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        ls.append(l)

        for i in range(self._nx, 0, -1):
            # merge L into the previous tensor
            temp = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[i], l)
            q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
            ls.append(l)
        
        ls.reverse()

        # build projectors on each inner bond
        # there are (nx+1) inner bonds
        # left-, right- means on each bond
        prs, pls = [], []

        specs = []
        for i in range(self._nx+1):
            u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]))
            # truncate
            ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
            # inverse of square root
            sst_inv = (1.0 / torch.sqrt(st)).diag()

            specs.append(st)

            if self._cflag:
                sst_inv = sst_inv.cdouble()

            pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
            prs.append(pr)
            pls.append(pl)

        # apply projectors to compress the MPO-MPS
        new_mps = [None]*(self._nx+2)
        new_mps[0]= torch.einsum('abcd,abce->ed', mpo_mps[0], prs[0])
        new_mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mpo_mps[-1])

        for i in range(1, self._nx+1):
            new_mps[i] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i-1], mpo_mps[i], prs[i])

        return new_mps, specs

    def ctmrg_move_up(self, mps: list, mpo: list):
        r'''
        one up step of CTMRG Parameters:
        ----------
        mps: list[tensor], the boundary MPS
        mpo: list[tensor], the MPO
        '''

        assert len(mps) == self._nx+2, 'this boundary MPS is not valid for length'
        assert len(mpo) == self._nx+2, 'this MPO is not valid for length'

        rho = mps[0].shape[0]

        # MPO-MPS operation
        # need a new variable, otherwise will change the input arguments
        mpo_mps = [None]*(self._nx+2)
        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[-1] = torch.einsum('ab,cbde->adec', mps[-1], mpo[-1])
        for i in range(1, self._nx+1):
            mpo_mps[i] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps[i], mpo[i])

        rs, ls = [], []
        # QR from left to right
        temp = mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        rs.append(r)
        
        for i in range(1, self._nx+1):
            # merge R in the next tensor
            temp = torch.einsum('abcd,bcdefghi->aefghi', r, mpo_mps[i])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
            rs.append(r)

        # LQ from right to left
        temp = mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        ls.append(l)

        for i in range(self._nx, 0, -1):
            # merge L into the previous tensor
            temp = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[i], l)
            q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
            ls.append(l)
        
        ls.reverse()

        # build projectors on each inner bond
        # there are (nx+1) inner bonds
        # left-, right- means on each bond
        prs, pls = [], []
        s_vals = []
        for i in range(self._nx+1):
            u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]))
            # truncate
            ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
            # inverse of square root
            sst_inv = (1.0 / torch.sqrt(st)).diag()

            s_vals.append(st)

            if self._cflag:
                sst_inv = sst_inv.cdouble()

            pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
            prs.append(pr)
            pls.append(pl)

        # apply projectors to compress the MPO-MPS
        new_mps = [None]*(self._nx+2)
        new_mps[0]= torch.einsum('abcd,abce->ed', mpo_mps[0], prs[0])
        new_mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mpo_mps[-1])

        for i in range(1, self._nx+1):
            new_mps[i] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i-1], mpo_mps[i], prs[i])

        return new_mps, s_vals


    def ctmrg_move_down(self, mps: list, mpo: list):

        assert len(mps) == self._nx+2, 'this boundary MPS is not valid for length'
        assert len(mpo) == self._nx+2, 'this MPO is not valid for length'

        rho = mps[0].shape[0]

        # MPO-MPS operation
        mpo_mps = [None]*(self._nx+2)
        mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
        mpo_mps[-1] = torch.einsum('ab,bcde->adec', mps[-1], mpo[-1])
        for i in range(1, self._nx+1):
            mpo_mps[i] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', mps[i], mpo[i])

        rs, ls = [], []
        # QR from left to right
        temp = mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        rs.append(r)
        
        for i in range(1, self._nx+1):
            temp = torch.einsum('abcd,bcdefghi->aefghi', r, mpo_mps[i])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
            rs.append(r)

        # LQ from right to left
        temp = mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        ls.append(l)

        for i in range(self._nx, 0, -1):
            temp = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[i], l)
            q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
            ls.append(l)
        
        ls.reverse()

        # build projectors
        prs, pls = [], []
        s_vals = []
        for i in range(self._nx+1):
            u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]))
            # truncate
            ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
            # inverse of square root
            sst_inv = (1.0 / torch.sqrt(st)).diag()

            s_vals.append(st)

            if self._cflag:
                sst_inv = sst_inv.cdouble()

            pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
            prs.append(pr)
            pls.append(pl)

        # apply projectors to compress the MPS
        new_mps = [None]*(self._nx+2)
        # head and tail
        new_mps[0]= torch.einsum('abcd,abce->ed', mpo_mps[0], prs[0])
        new_mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mpo_mps[-1])

        for i in range(self._nx):
            new_mps[i+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i], mpo_mps[i+1], prs[i+1])

        return new_mps, s_vals


    def ctmrg_move_left(self, ucc: tuple):
        r'''
        a left move of CTMRG

        Parameters:
        ----------
        ucc: tuple[int], coordinate of unit cell
        '''

        assert self._ctms is not None, 'CTM tensors not initialized'

        # CTM tensors for current unit cell
        ctms = self._ctms[ucc]
        cs, up_es, down_es, left_es = ctms['c'], ctms['u'], ctms['d'], ctms['l']

        rho = cs[0].shape[0]

        # temporary left boundary MPS: from down to up
        mps = [t for t in left_es]
        mps.insert(0, cs[0])
        mps.append(cs[2])
        # temporary MPO
        mpo = []
        for j in range(self._ny):
            c = ucc[0], (ucc[1]+j) % self._ny
            mpo.append(self.double_tensor(c))

        mpo.insert(0, down_es[0])
        mpo.append(up_es[0])

        mpo_mps = [None]*4
        mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
        mpo_mps[-1] = torch.einsum('ab,acde->cbde', mps[-1], mpo[-1])
        # f B b
        # | | |--C 
        # *****
        # | | |--c
        # e D d
        for j in range(1, self._ny+1):
            mpo_mps[j] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[j], mpo[j])

        # QR and LQ factorizations
        rs, ls = [], []
        # QR from botton to top
        temp = mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(0, 0))
        rs.append(r)
    
        # merge residual matrix into next site
        for j in range(1, self._ny+1):
            temp = torch.einsum('abcd,bcdefghi->aefghi', r, mpo_mps[j])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
            rs.append(r)

        # LQ from top to botton
        temp = mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        ls.append(l)

        for j in range(self._ny, 0, -1):
            temp = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[j], l)
            q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
            ls.append(l)
        
        ls.reverse()

        # build projectors
        # there are (ny+1) inner bonds
        prs, pls = [], []
        s_vals = []
        for j in range(self._ny+1):
            u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]))
            # truncate
            ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
            # inverse of square root
            sst_inv = (1.0 / torch.sqrt(st)).diag()

            s_vals.append(st)

            if self._cflag:
                sst_inv = sst_inv.cdouble()

            pr = torch.einsum('abcd,de->abce', ls[j], vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[j])
            prs.append(pr)
            pls.append(pl)

        # apply projectors to compress the MPO-MPS
        # head and tail
        mps = [None]*4
        mps[0]= torch.einsum('abcd,bcde->ae', mpo_mps[0], prs[0])
        mps[-1] = torch.einsum('abcd,ebcd->ea', pls[-1], mpo_mps[-1])

        for j in range(1, self._ny+1):
            mps[j] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j-1], mpo_mps[j], prs[j])

        # update CTM tensors
        # a left move means uc_coord changes
        new_ucc = (ucc[0]+1) % self._nx, ucc[1]
        ctms = self._ctms[new_ucc]
        cs, left_es = ctms['c'], ctms['l']
        # update left CTM tensors
        cs[0] = mps[0] / torch.linalg.norm(mps[0])
        cs[2] = mps[-1] / torch.linalg.norm(mps[-1])
        for j in range(self._ny):
            left_es[j] = mps[j+1] / torch.linalg.norm(mps[j+1])

        self._ctms[new_ucc].update({'c': cs})
        self._ctms[new_ucc].update({'l': left_es})

        return s_vals


    def ctmrg_move_right(self, mps: list, mpo: list):

        assert len(mps) == self._ny+2, 'this boundary MPS is not valid for length'
        assert len(mpo) == self._ny+2, 'this MPO is not valid for length'

        rho = mps[0].shape[0]

        mpo_mps = [None]*(self._ny+2)
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[-1] = torch.einsum('abcd,be->aecd', mpo[-1], mps[-1])
        
        for j in range(1, self._ny+1):
            mpo_mps[j] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[j], mps[j])

        rs, ls = [], []
        # QR from botton to top
        temp = mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
        rs.append(r)
        
        for j in range(1, self._ny+1):
            temp = torch.einsum('abcd,bcdefghi->aefghi', r, mpo_mps[j])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
            rs.append(r)

        # LQ from top to botton
        temp = mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        ls.append(l)

        for j in range(self._ny, 0, -1):
            temp = torch.einsum('abcdefgh,defi->abcigh', mpo_mps[j], l)
            q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))
            ls.append(l)
        
        ls.reverse()

        # build projectors
        # there are (ny+1) bonds
        prs, pls = [], []
        s_vals = []
        for j in range(self._ny+1):
            u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]))
            # truncate
            ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
            ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
            # inverse of square root
            sst_inv = (1.0 / torch.sqrt(st)).diag()

            s_vals.append(st)

            if self._cflag:
                sst_inv = sst_inv.cdouble()

            pr = torch.einsum('abcd,de->abce', ls[j], vt_dagger @ sst_inv)
            pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[j])
            prs.append(pr)
            pls.append(pl)

        # apply projectors to compress the MPO-MPS
        new_mps = [None]*(self._ny+2)
        # head and tail
        new_mps[0]= torch.einsum('abcd,bcde->ae', mpo_mps[0], prs[0])
        new_mps[-1] = torch.einsum('abcd,ebcd->ea', pls[-1], mpo_mps[-1])

        for j in range(1, self._ny+1):
            new_mps[j] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j-1], mpo_mps[j], prs[j])

        return new_mps, s_vals


    def _ctmrg(self, dts: dict, init_ctms: tuple):
        r'''
        one step of CTMRG
        absorb the unit cell along four directions, respectively

        Parameters:
        dts: dict[torch.tensor], {key: coordinate; value: double tensor
        init_ctms: tuple[torch.tensor], initial CTM tensors
        '''

        cs, up_es, down_es, left_es, right_es = init_ctms

        # temporary corner matrices
        c0, c1, c2, c3 = [], [], [], []

        # up boundary: from left to right
        # build temporary up MPS
        mps = [t.clone() for t in up_es]
        mps.insert(0, cs[2].clone())
        mps.append(cs[3].clone())

        # renormalization:
        # merge a whole unit cell into this MPS
        # j = ...3, 2, 1, 0
        for j in range(self._ny-1, -1, -1):
            # temporary MPO
            mpo = []
            for i in range(self._nx):
                mpo.append(dts[(i, j)])
            # head and tail of MPO
            # from other edge tensors
            mpo.insert(0, left_es[j].clone())
            mpo.append(right_es[j].clone())

            mps = self.ctmrg_move_up(mps, mpo)

            c2.append(mps[0])
            c3.append(mps[-1])

        # update edge tensors
        for i in range(self._nx):
            up_es[i] = mps[i+1].clone() / torch.linalg.norm(mps[i+1])

        # down boundary: from left to right
        mps = [t.clone() for t in down_es]
        mps.insert(0, cs[0].clone())
        mps.append(cs[1].clone())

        # j = 0, 1, 2, 3...
        for j in range(self._ny):
            # temporary MPO
            mpo = []
            for i in range(self._nx):
                mpo.append(dts[(i, j)])

            mpo.insert(0, left_es[j].clone())
            mpo.append(right_es[j].clone())

            mps = self.ctmrg_move_down(mps, mpo)

            c0.append(mps[0])
            c1.append(mps[-1])

        for i in range(self._nx):
            down_es[i] = mps[i+1].clone() / torch.linalg.norm(mps[i+1])

        # left boundary: from down to up
        mps = [t.clone() for t in left_es]
        mps.insert(0, cs[0].clone())
        mps.append(cs[2].clone())
        
        # i = 0, 1, 2, 3...
        for i in range(self._nx):
            mpo = []
            for j in range(self._ny):
                mpo.append(dts[(i, j)])

            mpo.insert(0, down_es[i].clone())
            mpo.append(up_es[i].clone())
            
            mps = self.ctmrg_move_left(mps, mpo)

            c0.append(mps[0])
            c2.append(mps[-1])

        for j in range(self._ny):
            left_es[j] = mps[j+1].clone() / torch.linalg.norm(mps[j+1])

        # right boundary: from down to up
        mps = [t.clone() for t in right_es]
        mps.insert(0, cs[1].clone())
        mps.append(cs[3].clone())

        # i = ...3, 2, 1, 0
        for i in range(self._nx-1, -1, -1):
            mpo = []
            for j in range(self._ny):
                mpo.append(dts[(i, j)])

            mpo.insert(0, down_es[i].clone())
            mpo.append(up_es[i].clone())

            mps = self.ctmrg_move_right(mps, mpo)

            c1.append(mps[0])
            c3.append(mps[-1])

        for j in range(self._ny):
            right_es[j] = mps[j+1].clone() / torch.linalg.norm(mps[j+1])

        # update corners
        temp_cs = c0, c1, c2, c3
        for i in range(4):
            # print(len(temp_cs[i]))
            cs[i] = sum(temp_cs[i])
            cs[i] = cs[i] / torch.linalg.norm(cs[i])

        return cs, up_es, down_es, left_es, right_es

 
    def ctmrg_ud(self, dts: dict, init_ctms: tuple):
        r'''
        one step of CTMRG
        absorb the unit cell along four directions, respectively

        Parameters:
        dts: dict[torch.tensor], {key: coordinate; value: double tensor
        init_ctms: tuple[torch.tensor], initial CTM tensors
        '''

        cs, up_es, down_es, left_es, right_es = init_ctms

        # up boundary: from left to right
        # build temporary up MPS
        mps_u = [t for t in up_es]
        mps_u.insert(0, cs[2])
        mps_u.append(cs[3])

        # renormalization:
        # merge a whole unit cell into this MPS
        # j = ...3, 2, 1, 0
        ss_u = []
        for j in range(self._ny-1, -1, -1):
            # temporary MPO
            mpo_u = []
            for i in range(self._nx):
                mpo_u.append(dts[(i, j)])
            # head and tail of MPO
            # from other edge tensors
            mpo_u.insert(0, left_es[j])
            mpo_u.append(right_es[j])

            mps_u, s = self.ctmrg_move_up_with_svs(mps_u, mpo_u)
            ss_u.append(s)

        # update CTM tensors
        cs[2] = mps_u[0] / torch.linalg.norm(mps_u[0])
        cs[3] = mps_u[-1] / torch.linalg.norm(mps_u[-1])
        # update edge tensors
        for i in range(self._nx):
            up_es[i] = mps_u[i+1] / torch.linalg.norm(mps_u[i+1])

        # down boundary: from left to right
        mps_d = [t for t in down_es]
        mps_d.insert(0, cs[0])
        mps_d.append(cs[1])

        # j = 0, 1, 2, 3...
        ss_d = []
        for j in range(self._ny):
            # temporary MPO
            mpo_d = []
            for i in range(self._nx):
                mpo_d.append(dts[(i, j)])

            mpo_d.insert(0, left_es[j])
            mpo_d.append(right_es[j])

            mps_d, s = self.ctmrg_move_down(mps_d, mpo_d)
            ss_d.append(s)

        cs[0] = mps_d[0] / torch.linalg.norm(mps_d[0])
        cs[1] = mps_d[-1] / torch.linalg.norm(mps_d[-1])

        for i in range(self._nx):
            down_es[i] = mps_d[i+1] / torch.linalg.norm(mps_d[i+1])

        new_ctms = cs, up_es, down_es, left_es, right_es

        return new_ctms, ss_u, ss_d


    def ctmrg_lr(self, dts: dict, init_ctms: tuple):
        r'''
        one step of CTMRG
        absorb the unit cell along four directions, respectively

        Parameters:
        dts: dict[torch.tensor], {key: coordinate; value: double tensor
        init_ctms: tuple[torch.tensor], initial CTM tensors
        '''

        cs, up_es, down_es, left_es, right_es = init_ctms

        svs = []

        # left boundary: from down to up
        mps_l = [t for t in left_es]
        mps_l.insert(0, cs[0])
        mps_l.append(cs[2])

        # i = 0, 1, 2, 3...
        for i in range(self._nx):
            mpo_l = []
            for j in range(self._ny):
                mpo_l.append(dts[(i, j)])

            mpo_l.insert(0, down_es[i])
            mpo_l.append(up_es[i])
            
            mps_l = self.ctmrg_move_left(mps_l, mpo_l)

        cs[0] = mps_l[0] / torch.linalg.norm(mps_l[0])
        cs[2] = mps_l[-1] / torch.linalg.norm(mps_l[-1])
        for j in range(self._ny):
            left_es[j] = mps_l[j+1] / torch.linalg.norm(mps_l[j+1])

        # right boundary: from down to up
        mps_r = [t for t in right_es]
        mps_r.insert(0, cs[1])
        mps_r.append(cs[3])

        # i = ...3, 2, 1, 0
        for i in range(self._nx-1, -1, -1):
            mpo_r = []
            for j in range(self._ny):
                mpo_r.append(dts[(i, j)])

            mpo_r.insert(0, down_es[i])
            mpo_r.append(up_es[i])

            mps_r = self.ctmrg_move_right(mps_r, mpo_r)

        cs[1] = mps_r[0] / torch.linalg.norm(mps_r[0])
        cs[3] = mps_r[-1] / torch.linalg.norm(mps_r[-1])
        for j in range(self._ny):
            right_es[j] = mps_r[j+1] / torch.linalg.norm(mps_r[j+1])

        return cs, up_es, down_es, left_es, right_es, svs


    def ctmrg_cw(self, dts: dict, init_ctms: tuple):
        r'''
        one step of CTMRG
        absorb the unit cell along four directions, respectively

        Parameters:
        dts: dict[torch.tensor], {key: coordinate; value: double tensor
        init_ctms: tuple[torch.tensor], initial CTM tensors
        '''

        cs, up_es, down_es, left_es, right_es = init_ctms

        # up boundary: from left to right
        # build temporary up MPS
        mps_u = [t for t in up_es]
        mps_u.insert(0, cs[2])
        mps_u.append(cs[3])

        # renormalization:
        # merge a whole unit cell into this MPS
        # j = ...3, 2, 1, 0
        svs = []
        for j in range(self._ny-1, -1, -1):
            # temporary MPO
            mpo_u = []
            for i in range(self._nx):
                mpo_u.append(dts[(i, j)])
            # head and tail of MPO
            # from other edge tensors
            mpo_u.insert(0, left_es[j])
            mpo_u.append(right_es[j])

            mps_u, sp = self.ctmrg_move_up_with_svs(mps_u, mpo_u)

            svs.append(sp)

            # update CTM tensors
            cs[2] = mps_u[0] / torch.linalg.norm(mps_u[0])
            cs[3] = mps_u[-1] / torch.linalg.norm(mps_u[-1])
            # update edge tensors
            for i in range(self._nx):
                up_es[i] = mps_u[i+1] / torch.linalg.norm(mps_u[i+1])

        # down boundary: from left to right
        mps_d = [t for t in down_es]
        mps_d.insert(0, cs[0])
        mps_d.append(cs[1])

        # j = 0, 1, 2, 3...
        for j in range(self._ny):
            # temporary MPO
            mpo_d = []
            for i in range(self._nx):
                mpo_d.append(dts[(i, j)])

            mpo_d.insert(0, left_es[j])
            mpo_d.append(right_es[j])

            mps_d = self.ctmrg_move_down(mps_d, mpo_d)

            cs[0] = mps_d[0] / torch.linalg.norm(mps_d[0])
            cs[1] = mps_d[-1] / torch.linalg.norm(mps_d[-1])

            for i in range(self._nx):
                down_es[i] = mps_d[i+1] / torch.linalg.norm(mps_d[i+1])


        '''
        # right boundary: from down to up
        mps_r = [t for t in right_es]
        mps_r.insert(0, cs[1])
        mps_r.append(cs[3])

        # i = ...3, 2, 1, 0
        for i in range(self._nx-1, -1, -1):
            mpo_r = []
            for j in range(self._ny):
                mpo_r.append(dts[(i, j)])

            mpo_r.insert(0, down_es[i])
            mpo_r.append(up_es[i])

            new_mps_r = self.ctmrg_move_right(mps_r, mpo_r)
            mps_r = new_mps_r

        cs[1] = new_mps_r[0] / torch.linalg.norm(new_mps_r[0])
        cs[3] = new_mps_r[-1] / torch.linalg.norm(new_mps_r[-1])
        for j in range(self._ny):
            right_es[j] = new_mps_r[j+1] / torch.linalg.norm(new_mps_r[j+1])

        # down boundary: from left to right
        mps_d = [t for t in down_es]
        mps_d.insert(0, cs[0])
        mps_d.append(cs[1])

        # j = 0, 1, 2, 3...
        for j in range(self._ny):
            # temporary MPO
            mpo_d = []
            for i in range(self._nx):
                mpo_d.append(dts[(i, j)])

            mpo_d.insert(0, left_es[j])
            mpo_d.append(right_es[j])

            new_mps_d = self.ctmrg_move_down(mps_d, mpo_d)
            mps_d = new_mps_d

        cs[0] = new_mps_d[0] / torch.linalg.norm(new_mps_d[0])
        cs[1] = new_mps_d[-1] / torch.linalg.norm(new_mps_d[-1])
        for i in range(self._nx):
            down_es[i] = new_mps_d[i+1] / torch.linalg.norm(new_mps_d[i+1])

        # left boundary: from down to up
        mps_l = [t for t in left_es]
        mps_l.insert(0, cs[0])
        mps_l.append(cs[2])

        # i = 0, 1, 2, 3...
        for i in range(self._nx):
            mpo_l = []
            for j in range(self._ny):
                mpo_l.append(dts[(i, j)])

            mpo_l.insert(0, down_es[i])
            mpo_l.append(up_es[i])
            
            new_mps_l = self.ctmrg_move_left(mps_l, mpo_l)
            mps_l = new_mps_l

        cs[0] = new_mps_l[0] / torch.linalg.norm(new_mps_l[0])
        cs[2] = new_mps_l[-1] / torch.linalg.norm(new_mps_l[-1])
        for j in range(self._ny):
            left_es[j] = new_mps_l[j+1] / torch.linalg.norm(new_mps_l[j+1])
        '''

        return cs, up_es, down_es, left_es, right_es, svs

    '''
    def ctmrg_2(self, dts: dict, ctms: list):
         # up boundary: from left to right
        # renormalization:
        # merge a whole unit cell into this MPS
        # j = ...3, 2, 1, 0
        j = 1
        cs, up_es, left_es, right_es = ctms[0]['c'], ctms[0]['u'], ctms[0]['l'], ctms[0]['r']

        # build temporary up MPS
        mps_u = [t for t in up_es]
        mps_u.insert(0, cs[2])
        mps_u.append(cs[3])

        # temporary MPO
        mpo_u = []
        for i in range(self._nx):
            mpo_u.append(dts[(i, j)])
        # head and tail of MPO
        # from other edge tensors
        mpo_u.insert(0, left_es[j])
        mpo_u.append(right_es[j])

        new_mps_u = self.ctmrg_move_up(mps_u, mpo_u)

        # update CTM tensors
        # 0, 1 are copied
        cs[0] = ctms[2]['c'][0]
        cs[1] = ctms[2]['c'][1]
        cs[2] = new_mps_u[0] / torch.linalg.norm(new_mps_u[0])
        cs[3] = new_mps_u[-1] / torch.linalg.norm(new_mps_u[-1])
        # update edge tensors
        for i in range(self._nx):
            up_es[i] = new_mps_u[i+1] / torch.linalg.norm(new_mps_u[i+1])

        # update
        ctms[2].update({'c': cs})
        ctms[2].update({'u': up_es})

        # right boundary: from down to up
        # i = ...3, 2, 1, 0
        i = 1
        # for i in range(self._nx-1, -1, -1):
            mps_r = [t for t in right_es]
            mps_r.insert(0, cs[1])
            mps_r.append(cs[3])

            mpo_r = []
            for j in range(self._ny):
                mpo_r.append(dts[(i, j)])

            mpo_r.insert(0, down_es[i])
            mpo_r.append(up_es[i])

            new_mps_r = self.ctmrg_move_right(mps_r, mpo_r)

            cs[1] = new_mps_r[0] / torch.linalg.norm(new_mps_r[0])
            cs[3] = new_mps_r[-1] / torch.linalg.norm(new_mps_r[-1])
            for j in range(self._ny):
                right_es[j] = new_mps_r[j+1] / torch.linalg.norm(new_mps_r[j+1])

        # down boundary: from left to right
        # j = 0, 1, 2, 3...
        for j in range(self._ny):
            mps_d = [t for t in down_es]
            mps_d.insert(0, cs[0])
            mps_d.append(cs[1])

            # temporary MPO
            mpo_d = []
            for i in range(self._nx):
                mpo_d.append(dts[(i, j)])

            mpo_d.insert(0, left_es[j])
            mpo_d.append(right_es[j])

            new_mps_d = self.ctmrg_move_down(mps_d, mpo_d)

            cs[0] = new_mps_d[0] / torch.linalg.norm(new_mps_d[0])
            cs[1] = new_mps_d[-1] / torch.linalg.norm(new_mps_d[-1])
            for i in range(self._nx):
                down_es[i] = new_mps_d[i+1] / torch.linalg.norm(new_mps_d[i+1])

        # left boundary: from down to up
        # i = 0, 1, 2, 3...
        for i in range(self._nx):
            mps_l = [t for t in left_es]
            mps_l.insert(0, cs[0])
            mps_l.append(cs[2])
     
            mpo_l = []
            for j in range(self._ny):
                mpo_l.append(dts[(i, j)])

            mpo_l.insert(0, down_es[i])
            mpo_l.append(up_es[i])
            
            new_mps_l = self.ctmrg_move_left(mps_l, mpo_l)

            cs[0] = new_mps_l[0] / torch.linalg.norm(new_mps_l[0])
            cs[2] = new_mps_l[-1] / torch.linalg.norm(new_mps_l[-1])
            for j in range(self._ny):
                left_es[j] = new_mps_l[j+1] / torch.linalg.norm(new_mps_l[j+1])

        return cs, up_es, down_es, left_es, right_es
    '''


    def ctm_onebody_measure(self, tps: torch.tensor, ctms: list, op: torch.tensor) -> list:

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

        for j in range(self._ny):
            # prepare up and down MPS
            mps_u = [t.clone() for t in up_es]
            mps_u.insert(0, cs[2])
            mps_u.append(cs[3])

            mps_d = [t.clone() for t in down_es]
            mps_d.insert(0, cs[0])
            mps_d.append(cs[1])

            # renormalize up MPS until row-j
            # l = ..., j+1
            for l in range(self._ny-1, j, -1):
                # temporary MPO
                mpo = []
                for i in range(self._nx):
                    mpo.append(dts[(i, l)])

                mpo.insert(0, left_es[l])
                mpo.append(right_es[l])

                mps_u = self.ctmrg_move_up(mps_u, mpo)

            # renormalize down MPS until row-j
            # l = 0, ..., j-1
            for l in range(j):
                mpo = []
                for i in range(self._nx):
                    mpo.append(dts[(i, l)])

                mpo.insert(0, left_es[l])
                mpo.append(right_es[l])

                mps_d = self.ctmrg_move_down(mps_d, mpo)

            # build left and right environment tensors
            # *--f,3
            # *--d,1
            # *--e,2
            # *--a,0
            left_env = torch.einsum('ab,bcde,fc->adef', mps_d[0], left_es[j], mps_u[0])
            right_env = torch.einsum('ab,bcde,fc->adef', mps_d[-1], right_es[j], mps_u[-1])

            # pure double tensors
            pure_dts = []
            for i in range(self._nx):
                pure_dts.append(dts[(i, j)])

            # denominator
            temp = left_env.clone()
            for i in range(self._nx):
                temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, mps_d[i+1], pure_dts[i], mps_u[i+1])

            den = torch.einsum('abcd,abcd', temp, right_env)

            # impure double tensor
            for i in range(self._nx):
                impure_dt = torch.einsum('ABCDE,Ee,abcde->AaBbCcDd', mts_conj[(i, j)], op, mts[(i, j)])

                # numerator
                temp = left_env.clone()
                for k in range(self._nx):
                    if i == k:
                        temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, mps_d[k+1], impure_dt, mps_u[k+1])
                    else:
                        temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, mps_d[k+1], pure_dts[k], mps_u[k+1])

                num = torch.einsum('abcd,abcd', temp, right_env)
                meas.append((num / den))

        return torch.tensor(meas)


    def ctm_twobody_measure(self, tps: torch.tensor, ctms: list, op: torch.tensor) -> list:
        r'''
        measure a twobody operator by CTMRG 

        Parameters
        ----------
        op: tensor, twobody operator
        ctm_tensors: tuple, corner transfer matrix tensors from CTMRG

        '''

        cs, up_es, down_es, left_es, right_es = ctms

        # SVD two-body operator to MPO
        u, s, v = tp.linalg.tsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(0, 0))
        ss = torch.sqrt(s).diag()
        us = torch.einsum('Aa,abc->Abc', ss, u)
        sv = torch.einsum('Aa,abc->Abc', ss, v)

        op_mpo = us, sv

        mts, mts_conj = {}, {}
        for i, c in enumerate(self._coords):
            mts.update({c: tps[i].clone()})
            mts_conj.update({c: tps[i].conj().clone()})
        # double tensors
        dts = {}
        for i, c in enumerate(self._coords):
            dts.update({c: torch.einsum('ABCDe,abcde->AaBbCcDd', mts_conj[c], mts[c])})

        meas = []

        # horizontal bonds
        if self._nx > 1:
            for j in range(self._ny):
                # prepare up and down MPS
                mps_u = [t.clone() for t in up_es]
                mps_u.insert(0, cs[2].clone())
                mps_u.append(cs[3].clone())

                mps_d = [t.clone() for t in down_es]
                mps_d.insert(0, cs[0].clone())
                mps_d.append(cs[1].clone())

                # renormalize up MPS until row-j
                # l = ..., j+1
                for l in range(self._ny-1, j, -1):
                    # temporary MPO
                    temp_mpo = []
                    for i in range(self._nx):
                        temp_mpo.append(dts[(i, l)])

                    temp_mpo.insert(0, left_es[l].clone())
                    temp_mpo.append(right_es[l].clone())

                    mps_u = self.ctmrg_move_up(mps_u, temp_mpo)

                # renormalize down MPS until row-j
                # l = 0, ..., j-1
                for l in range(j):
                    temp_mpo = []
                    for i in range(self._nx):
                        temp_mpo.append(dts[(i, l)])

                    temp_mpo.insert(0, left_es[l].clone())
                    temp_mpo.append(right_es[l].clone())

                    mps_d = self.ctmrg_move_down(mps_d, temp_mpo)

                # MPO for very row-j
                mpo = []
                for i in range(self._nx):
                    mpo.append(dts[(i, j)])

                mpo.insert(0, left_es[j].clone())
                mpo.append(right_es[j].clone())

                # build left and right environment tensors
                # *--f,3
                # *--d,1
                # *--e,2
                # *--a,0
                env_l = torch.einsum('ab,bcde,fc->adef', mps_d[0], mpo[0], mps_u[0])
                env_r = torch.einsum('ab,bcde,fc->adef', mps_d[-1], mpo[-1], mps_u[-1])

                # denominator
                # as contraction from left to right
                temp = env_l.clone()
                for i in range(self._nx):
                    temp = torch.einsum(
                            'eAag,efDd,AaBbCcDd,ghBb->fCch',
                            temp, mps_d[i+1], mpo[i+1], mps_u[i+1])

                den = torch.einsum('abcd,abcd', temp, env_r)

                # impure double tensors
                impure_dts = []
                for i in range(self._nx-1):
                    c, cx = (i, j), (i+1, j)
                    impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBbCfcDd', mts_conj[c], op_mpo[0], mts[c]))
                    impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AfaBbCcDd', mts_conj[cx], op_mpo[1], mts[cx]))

                    # numerator
                    temp = env_l.clone()
                    for k in range(self._nx):
                        if i == k:
                            temp = torch.einsum(
                                    'eAag,efDd,AaBbCicDd,ghBb->fCich',
                                    temp, mps_d[k+1], impure_dts[0], mps_u[k+1])
                        elif (i+1) == k:
                            temp = torch.einsum(
                                    'eAiag,efDd,AiaBbCcDd,ghBb->fCch',
                                    temp, mps_d[k+1], impure_dts[1], mps_u[k+1])
                        else:
                            temp = torch.einsum(
                                    'eAag,efDd,AaBbCcDd,ghBb->fCch',
                                    temp, mps_d[k+1], mpo[k+1], mps_u[k+1])

                    num = torch.einsum('abcd,abcd', temp, env_r)
                    meas.append(num / den)

        # vertical bonds
        if self._ny > 1:
            for i in range(self._nx):
                # prepare left and right MPS
                mps_l = [t.clone() for t in left_es]
                mps_l.insert(0, cs[0].clone())
                mps_l.append(cs[2].clone())

                mps_r = [t.clone() for t in right_es]
                mps_r.insert(0, cs[1].clone())
                mps_r.append(cs[3].clone())

                # renormalize left MPS until column-i
                # k = 0, ..., i-1
                for k in range(i):
                    temp_mpo = []
                    for j in range(self._ny):
                        temp_mpo.append(dts[(k, j)])

                    temp_mpo.insert(0, down_es[k].clone())
                    temp_mpo.append(up_es[k].clone())

                    mps_l = self.ctmrg_move_left(mps_l, temp_mpo)

                # renormalize right MPS until column-i
                # k = ..., i+1
                for k in range(self._nx-1, i, -1):
                    temp_mpo = []
                    for j in range(self._ny):
                        temp_mpo.append(dts[(k, j)])

                    temp_mpo.insert(0, down_es[k].clone())
                    temp_mpo.append(up_es[k].clone())

                    mps_r = self.ctmrg_move_right(mps_r, temp_mpo)

                # MPO for very column-i
                # as pure double tensors
                mpo = []
                for j in range(self._ny):
                    mpo.append(dts[(i, j)])

                mpo.insert(0, down_es[i].clone())
                mpo.append(up_es[i].clone())

                '''
                # another way to test norm
                bou_l = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps_l)
                temp = torch.einsum('hiCc,AaBbCcDd,EeFfGgBb,jkFf->hAaEejiDdGgk', *mpo)
                bou_r = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps_r)
                nor_y = torch.einsum('abcdef,abcdefABCDEF,ABCDEF', bou_l, temp, bou_r)
                print(nor_y.item())
                '''

                # build down and up environments
                # 0 1 2 3
                # b d e f
                # | | | |
                # *******
                env_d = torch.einsum('ab,acde,cf->bdef', mps_l[0], mpo[0], mps_r[0])
                env_u = torch.einsum('ab,acde,cf->bdef', mps_l[-1], mpo[-1], mps_r[-1])

                # denominator
                temp = env_d.clone()
                for j in range(self._ny):
                    temp = torch.einsum(
                            'eDdg,efAa,AaBbCcDd,ghCc->fBbh',
                            temp, mps_l[j+1], mpo[j+1], mps_r[j+1])

                den = torch.einsum('abcd,abcd', temp, env_u)

                # impure double tensors
                impure_dts = []
                for j in range(self._ny-1):
                    c, cy = (i, j), (i, j+1)
                    impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBfbCcDd', mts_conj[c], op_mpo[0], mts[c]))
                    impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBbCcDfd', mts_conj[cy], op_mpo[1], mts[cy]))

                    # numerator
                    temp = env_d.clone()
                    for l in range(self._ny):
                        if j == l:
                            temp = torch.einsum(
                                    'eDdg,efAa,AaBibCcDd,ghCc->fBibh',
                                    temp, mps_l[l+1], impure_dts[0], mps_r[l+1])
                        elif (j+1) == l:
                            temp = torch.einsum(
                                    'eDidg,efAa,AaBbCcDid,ghCc->fBbh',
                                    temp, mps_l[l+1], impure_dts[1], mps_r[l+1])
                        else:
                            temp = torch.einsum(
                                    'eDdg,efAa,AaBbCcDd,ghCc->fBbh',
                                    temp, mps_l[l+1], mpo[l+1], mps_r[l+1])

                    num = torch.einsum('abcd,abcd', temp, env_u)
                    meas.append(num / den)

        return torch.tensor(meas)


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
