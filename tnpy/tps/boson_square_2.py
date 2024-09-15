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

    def __init__(self, site_tensors: dict, link_tensors: dict, dim_phys=2, ctms=None, dtype=torch.float64):
        r'''initialization

        Parameters
        ----------
        site_tensors: dict, {key: coordinate, value: tensor}
        link_tensors: dict, {key: coordinate, value: tensor}
        ctms: dict, {key: coordinate, value: list}, environment tensors for CTMRG
            for the value list: [C0, C1, C2, C3, Ed, Eu, El, Er]

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

        # CTM tensors
        # each site is associated with a set of eight CTM tensors
        # {C0, C1, C2, C3, Ed, Eu, El, Er}:
        #
        #  C2  Eu  C3
        #   *--*--*
        #   |  |  |
        # El*--*--*Er
        #   |  |  |
        #   *--*--*
        #  C0  Ed  C1
        '''

        # defalut for spin-1/2
        self._dim_phys = dim_phys

        self._dtype = dtype

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

        assert self._nx > 1 and self._ny > 0, 'the TPS unit cell should be at least: 2*1'

        # inner bond dimension
        self._chi = self._site_tensors[(0, 0)].shape[0]

        self._ctms = ctms

        if self._ctms is not None:
            self._rho = self._ctms[(0, 0)][0].shape[0]
        else:
            self._rho = 0


    @classmethod
    def rand(cls, nx: int, ny: int, chi: int, rho: int, dtype=torch.float64):
        r'''
        generate a random SquareTPS

        Parameters
        ----------
        nx: int, number of sites along x-direction in a unit cell
        ny: int, number of sites along y-direction in a unit cell
        chi: int, bond dimension of the site tensor
        rho: int, bond dimension of boundary CTM tensors
        '''

        site_shape = (chi, chi, chi, chi, 2)
        site_tensors, link_tensors = {}, {}

        for i, j in itertools.product(range(nx), range(ny)):
            temp = torch.rand(site_shape).to(dtype)
            lam_x = torch.rand(chi).diag().to(dtype)
            lam_y = torch.rand(chi).diag().to(dtype)

            # normalization
            site_tensors[(i, j)] = temp / torch.linalg.norm(temp)
            link_tensors[(i, j)] = [lam_x / torch.linalg.norm(lam_x), lam_y / torch.linalg.norm(lam_y)]

        if rho > 0:
            # corners
            cs = [torch.rand(rho, rho).to(dtype) for i in range(4)]
            # edges
            es = [torch.rand(rho, rho, chi, chi).to(dtype) for i in range(4)]
            ctm = cs+es
            ctms = {}
            for i, j in itertools.product(range(nx), range(ny)):
                temp = [t.clone() for t in ctm]
                ctms.update({(i, j): temp})

        else:
            ctms = None

        return cls(site_tensors=site_tensors, link_tensors=link_tensors, ctms=ctms, dtype=dtype)


    @classmethod
    def randn(cls, nx: int, ny: int, chi: int, rho: int, dtype=torch.float64):
        r'''
        generate a random SquareTPS

        Parameters
        ----------
        nx: int, number of sites along x-direction in a unit cell
        ny: int, number of sites along y-direction in a unit cell
        chi: int, bond dimension of the site tensor
        rho: int, bond dimension of boundary CTM tensors
        '''

        site_shape = (chi, chi, chi, chi, 2)
        site_tensors, link_tensors = {}, {}

        for i, j in itertools.product(range(nx), range(ny)):
            temp = torch.randn(site_shape).to(dtype)
            lam_x = torch.randn(chi).diag().to(dtype)
            lam_y = torch.randn(chi).diag().to(dtype)

            # normalization
            site_tensors[(i, j)] = temp / torch.linalg.norm(temp)
            link_tensors[(i, j)] = [lam_x / torch.linalg.norm(lam_x), lam_y / torch.linalg.norm(lam_y)]

        if rho > 0:
            # corners
            cs = [torch.randn(rho, rho).to(dtype) for i in range(4)]
            # edges
            es = [torch.randn(rho, rho, chi, chi).to(dtype) for i in range(4)]
            ctm = cs+es
            ctms = {}
            for i, j in itertools.product(range(nx), range(ny)):
                temp = [t.clone() for t in ctm]
                ctms.update({(i, j): temp})

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

        # print(t.dtype, envs[0].dtype, envs[1].dtype, envs[2].dtype)
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


    def double_tensor(self, c: tuple):
        r'''
        return a double tensors

        Parameters:
        ----------
        '''

        temp = self.merged_tensor(c)

        return torch.einsum('ABCDe,abcde->AaBbCcDd', temp.conj(), temp)


    def double_tensors(self) -> dict:
        r'''
        return double tensors as a dict
        '''

        dts = {}
        for c in self._coords:
            dts.update({c: self.double_tensor(c)})

        return dts


    def unified_tensor(self, requires_grad=False) -> torch.tensor:
        r'''
        return the TPS tensor in a unit cell as a whole tensor
        new dimension is created as dim=0
        '''

        tens = []
        for c in self._coords:
            tens.append(self.merged_tensor(c))

        return torch.stack(tens, dim=0).requires_grad_(requires_grad)


    def update_ctms(self, new_ctms):

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


    def ctmrg_mu(self, c: tuple, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site within the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS and MPO
        mps = [self._ctms[c][2], self._ctms[c][5], self._ctms[c][3]]
        mpo = [self._ctms[c][6], dts[c], self._ctms[c][7]]

        mpo_mps = [None]*3

        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,cbde->adec', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        cb = (c[0]-1) % self._nx, c[1]
        temp_mps = [self._ctms[cb][2], self._ctms[cb][5], self._ctms[c][5], self._ctms[c][3]]
        temp_mpo = [self._ctms[cb][6], dts[cb], dts[c], self._ctms[c][7]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,cbde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,cbde->adec', temp_mps[3], temp_mpo[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        
        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        cf = (c[0]+1) % self._nx, c[1]
        temp_mps = [self._ctms[c][2], self._ctms[c][5], self._ctms[cf][5], self._ctms[cf][3]]
        temp_mpo = [self._ctms[c][6], dts[c], dts[cf], self._ctms[cf][7]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,cbde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,cbde->adec', temp_mps[3], temp_mpo[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # apply projectors and update CTM tensors
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,bcde->ae', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a up-move means absorbing this row
        c_new = c[0], (c[1]-1) % self._ny
        self._ctms[c_new][2] = mps[0] / norms[0]
        self._ctms[c_new][5] = mps[1] / norms[1]
        self._ctms[c_new][3] = mps[2] / norms[2]

        return 1


    def ctmrg_md(self, c: tuple, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site with the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS and MPO
        mps = [self._ctms[c][0], self._ctms[c][4], self._ctms[c][1]]
        mpo = [self._ctms[c][6], dts[c], self._ctms[c][7]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,bcde->adec', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        cb = (c[0]-1) % self._nx, c[1]
        temp_mps = [self._ctms[cb][0], self._ctms[cb][4], self._ctms[c][4], self._ctms[c][1]]
        temp_mpo = [self._ctms[cb][6], dts[cb], dts[c], self._ctms[c][7]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,bcde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,bcde->adec', temp_mps[3], temp_mpo[3])
        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        cf = (c[0]+1) % self._nx, c[1]
        temp_mps = [self._ctms[c][0], self._ctms[c][4], self._ctms[cf][4], self._ctms[cf][1]]
        temp_mpo = [self._ctms[c][6], dts[c], dts[cf], self._ctms[cf][7]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,bcde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,bcde->adec', temp_mps[3], temp_mpo[3])
        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # apply projectors and update CTM tensors
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,bcde->ae', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        c_new = c[0], (c[1]+1) % self._ny
        self._ctms[c_new][0] = mps[0] / norms[0]
        self._ctms[c_new][4] = mps[1] / norms[1]
        self._ctms[c_new][1] = mps[2] / norms[2]

        return 1


    def ctmrg_ml(self, c: tuple, dts: dict):

        # build temporary MPS and MPO
        mps = [self._ctms[c][0], self._ctms[c][6], self._ctms[c][2]]
        mpo = [self._ctms[c][4], dts[c], self._ctms[c][5]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
        # f B b
        # | | |--C 
        # *****
        # | | |--c
        # e D d
        mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,acde->cbde', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        cb = c[0], (c[1]-1) % self._ny 
        temp_mps = [self._ctms[cb][0], self._ctms[cb][6], self._ctms[c][6], self._ctms[c][2]]
        temp_mpo = [self._ctms[cb][4], dts[cb], dts[c], self._ctms[c][5]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,acde->cbde', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,acde->cbde', temp_mps[3], temp_mpo[3])
        
        # QR and LQ from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(0, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        # inverse of square root
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        cf = c[0], (c[1]+1) % self._ny
        temp_mps = [self._ctms[c][0], self._ctms[c][6], self._ctms[cf][6], self._ctms[cf][2]]
        temp_mpo = [self._ctms[c][4], dts[c], dts[cf], self._ctms[cf][5]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,acde->cbde', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,acde->cbde', temp_mps[3], temp_mpo[3])
        
        # QR and LQ from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(0, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a left-move means:
        c_new = (c[0]+1) % self._nx, c[1]
        self._ctms[c_new][0] = mps[0] / norms[0]
        self._ctms[c_new][6] = mps[1] / norms[1]
        self._ctms[c_new][2] = mps[2] / norms[2]

        return 1


    def ctmrg_mr(self, c: tuple, dts: dict):

        # build temporary MPS and MPO
        mps = [self._ctms[c][1], self._ctms[c][7], self._ctms[c][3]]
        mpo = [self._ctms[c][4], dts[c], self._ctms[c][5]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[1], mps[1])
        mpo_mps[2] = torch.einsum('abcd,be->aecd', mpo[2], mps[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        cb = c[0], (c[1]-1) % self._ny 
        temp_mps = [self._ctms[cb][1], self._ctms[cb][7], self._ctms[c][7], self._ctms[c][3]]
        temp_mpo = [self._ctms[cb][4], dts[cb], dts[c], self._ctms[c][5]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('abcd,be->aecd', temp_mpo[0], temp_mps[0])
        temp_mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[1], temp_mps[1])
        temp_mpo_mps[2] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[2], temp_mps[2])
        temp_mpo_mps[3] = torch.einsum('abcd,be->aecd', temp_mpo[3], temp_mps[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        cf = c[0], (c[1]+1) % self._ny
        temp_mps = [self._ctms[c][1], self._ctms[c][7], self._ctms[cf][7], self._ctms[cf][3]]
        temp_mpo = [self._ctms[c][4], dts[c], dts[cf], self._ctms[cf][5]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('abcd,be->aecd', temp_mpo[0], temp_mps[0])
        temp_mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[1], temp_mps[1])
        temp_mpo_mps[2] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[2], temp_mps[2])
        temp_mpo_mps[3] = torch.einsum('abcd,be->aecd', temp_mpo[3], temp_mps[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a right-move means:
        c_new = (c[0]-1) % self._nx, c[1]
        self._ctms[c_new][1] = mps[0] / norms[0]
        self._ctms[c_new][7] = mps[1] / norms[1]
        self._ctms[c_new][3] = mps[2] / norms[2]

        return 1




    def ctmrg_move_up(self, dts: dict):
        r'''
        one up step of CTMRG

        Parameters:
        ----------
        ucc: tuple[int], coordinate as a label of unit cell
        dts: dict, dict of double tensors
        '''

        assert self._ctms is not None, 'CTM tensors not initialized'

        rho = self._rho

        # a up move consist of m*n times of compression
        for c in self._coords:
            # head and tail coordinate for this row
            head, tail = c, ((c[0]+self._nx-1) % self._nx, c[1])

            # build temporary MPS
            mps = [self._ctms[((c[0]+i) % self._nx, c[1])][5] for i in range(self._nx)]
            # head and tail are corner tensors
            mps.insert(0, self._ctms[head][2])
            mps.append(self._ctms[tail][3])

            # build temporary MPO
            mpo = [dts[((c[0]+i) % self._nx, c[1])] for i in range(self._nx)]
            # head and tail are edge tensors
            mpo.insert(0, self._ctms[head][6])
            mpo.append(self._ctms[tail][7])

            # MPO-MPS operation
            mpo_mps = [None]*(self._nx+2)
            mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
            mpo_mps[-1] = torch.einsum('ab,cbde->adec', mps[-1], mpo[-1])
            for i in range(1, self._nx+1):
                mpo_mps[i] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps[i], mpo[i])

            old_wf = torch.einsum('abcd,abcefghi,efgjklmn,jklo->dhimno', *mpo_mps)

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
            prs, pls = [], []
            for i in range(self._nx+1):
                u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[i], ls[i]))
                # truncate
                ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                # inverse of square root
                sst_inv = (1.0 / torch.sqrt(st)).diag()

                if self._cflag:
                    sst_inv = sst_inv.cdouble()

                pr = torch.einsum('abcd,de->abce', ls[i], vt_dagger @ sst_inv)
                pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[i])
                prs.append(pr)
                pls.append(pl)

            # apply projectors to compress the MPO-MPS
            mps = [None]*(self._nx+2)
            mps[0]= torch.einsum('abcd,abce->ed', mpo_mps[0], prs[0])
            mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mpo_mps[-1])

            for i in range(1, self._nx+1):
                mps[i] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i-1], mpo_mps[i], prs[i])

            new_wf = torch.einsum('ab,acde,cfgh,fi->bdeghi', *mps)
            print('u loss:', torch.linalg.norm(new_wf-old_wf))

            # update CTM tensors
            # a up move means a shift: c -> cy
            cy = c[0], (c[1]-1) % self._ny
            head, tail = cy, ((cy[0]+self._nx-1) % self._nx, cy[1])

            temp_ctms = deepcopy(self._ctms)

            for i in range(self._nx):
                temp_ctms[((cy[0]+i) % self._nx, cy[1])][5] = mps[i+1] / torch.linalg.norm(mps[i+1])

            temp_ctms[head][2] = mps[0] / torch.linalg.norm(mps[0])
            temp_ctms[tail][3] = mps[-1] / torch.linalg.norm(mps[-1])

        self._ctms = temp_ctms
        return 1

    def ctmrg_move_down(self, dts: dict):
        r'''
        a down step move of CTMRG
        '''

        assert self._ctms is not None, 'CTM tensors not initialized'

        rho = self._rho

        # a up move consist of m*n times of compression
        coords = (0, 0), (0, 1)
        for c in coords:
        # for c in self._coords:
            # head and tail coordinate for this row
            head, tail = c, ((c[0]+self._nx-1) % self._nx, c[1])

            # build temporary MPS
            mps = [self._ctms[((c[0]+i) % self._nx, c[1])][4] for i in range(self._nx)]
            # head and tail are corner tensors
            mps.insert(0, self._ctms[head][0])
            mps.append(self._ctms[tail][1])

            # build temporary MPO
            mpo = [dts[((c[0]+i) % self._nx, c[1])] for i in range(self._nx)]
            # head and tail are edge tensors
            mpo.insert(0, self._ctms[head][6])
            mpo.append(self._ctms[tail][7])

            # MPO-MPS operation
            mpo_mps = [None]*(self._nx+2)
            mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
            mpo_mps[-1] = torch.einsum('ab,bcde->adec', mps[-1], mpo[-1])
            for i in range(1, self._nx+1):
                mpo_mps[i] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', mps[i], mpo[i])

            old_wf = torch.einsum('abcd,abcefghi,efgjklmn,jklo->dhimno', *mpo_mps)

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
            mps = [None]*(self._nx+2)
            # head and tail
            mps[0]= torch.einsum('abcd,abce->ed', mpo_mps[0], prs[0])
            mps[-1] = torch.einsum('abcd,bcde->ae', pls[-1], mpo_mps[-1])

            for i in range(self._nx):
                mps[i+1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[i], mpo_mps[i+1], prs[i+1])

            new_wf = torch.einsum('ab,acde,cfgh,fi->bdeghi', *mps)
            print('d loss:', torch.linalg.norm(new_wf-old_wf))

            # update CTM tensors
            # a down move means a shift: c -> cy
            cy = c[0], (c[1]+1) % self._ny
            head, tail = cy, ((cy[0]+self._nx-1) % self._nx, cy[1])

            for i in range(self._nx):
                self._ctms[((cy[0]+i) % self._nx, cy[1])][4] = mps[i+1] / torch.linalg.norm(mps[i+1])

            self._ctms[head][0] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[tail][1] = mps[-1] / torch.linalg.norm(mps[-1])

        return 1
 

    def ctmrg_move_left(self, dts: dict):
        r'''
        a left move of directional CTMRG

        Parameters:
        ----------
        dts: dict, double tensors
        '''

        assert self._ctms is not None, 'CTM tensors not initialized'

        rho = self._rho

        # a left move consist of m*n times of compression
        coords = (0, 0), (1, 0)
        # for c in self._coords:
        for c in coords:
            # head and tail coordinate for this column
            head, tail = c, (c[0], (c[1]+self._ny-1) % self._ny)
            # print(c, head, tail)

            # build temporary MPS
            mps = [self._ctms[(c[0], (c[1]+j) % self._ny)][6] for j in range(self._ny)]
            # head and tail are corner tensors
            mps.insert(0, self._ctms[head][0])
            mps.append(self._ctms[tail][2])

            # build temporary MPO
            mpo = [dts[(c[0], (c[1]+j) % self._ny)] for j in range(self._ny)]
            # head and tail are edge tensors
            mpo.insert(0, self._ctms[head][4])
            mpo.append(self._ctms[tail][5])

            mpo_mps = [None]*(self._ny+2)
            mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
            mpo_mps[-1] = torch.einsum('ab,acde->cbde', mps[-1], mpo[-1])
            # f B b
            # | | |--C 
            # *****
            # | | |--c
            # e D d
            for j in range(1, self._ny+1):
                mpo_mps[j] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[j], mpo[j])

            old_wf = torch.einsum('abcd,bcdefghi,efgjklmn,ojkl->ahimno', *mpo_mps)

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
            for j in range(self._ny+1):
                u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]))
                # truncate
                ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                # inverse of square root
                sst_inv = (1.0 / torch.sqrt(st)).diag()

                if self._cflag:
                    sst_inv = sst_inv.cdouble()

                pr = torch.einsum('abcd,de->abce', ls[j], vt_dagger @ sst_inv)
                pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[j])
                prs.append(pr)
                pls.append(pl)

            # apply projectors to compress the MPO-MPS
            mps = [None]*(self._ny+2)
            # head and tail
            mps[0]= torch.einsum('abcd,bcde->ae', mpo_mps[0], prs[0])
            mps[-1] = torch.einsum('abcd,ebcd->ea', pls[-1], mpo_mps[-1])

            for j in range(1, self._ny+1):
                mps[j] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j-1], mpo_mps[j], prs[j])

            new_wf = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps)
            print('l loss:', torch.linalg.norm(new_wf-old_wf))

            # update CTM tensors
            # a left move means a shift: c -> cx
            cx = (c[0]+1) % self._nx, c[1]
            head, tail = cx, (cx[0], (cx[1]+self._ny-1) % self._ny)

            for j in range(self._ny):
                self._ctms[(cx[0], (cx[1]+j) % self._ny)][6] = mps[j+1] / torch.linalg.norm(mps[j+1])

            self._ctms[head][0] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[tail][2] = mps[-1] / torch.linalg.norm(mps[-1])

        return 1
            

    def ctmrg_move_right(self, dts: dict):
        r'''
        a right move of directional CTMRG
        '''

        assert self._ctms is not None, 'CTM tensors not initialized'

        rho = self._rho

        # a right move consist of m*n times of compression
        coords = (1, 0), (0, 0)
        # for c in self._coords:
        for c in coords:
            # head and tail coordinate for this column
            head, tail = c, (c[0], (c[1]+self._ny-1) % self._ny)

            # build temporary MPS
            mps = [self._ctms[(head[0], (c[1]+j) % self._ny)][7] for j in range(self._ny)]
            # head and tail are corner tensors
            mps.insert(0, self._ctms[head][1])
            mps.append(self._ctms[tail][3])

            # build temporary MPO
            mpo = [dts[(head[0], (c[1]+j) % self._ny)] for j in range(self._ny)]
            # head and tail are edge tensors
            mpo.insert(0, self._ctms[head][4])
            mpo.append(self._ctms[tail][5])

            mpo_mps = [None]*(self._ny+2)
            mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
            mpo_mps[-1] = torch.einsum('abcd,be->aecd', mpo[-1], mps[-1])

            for j in range(1, self._ny+1):
                mpo_mps[j] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[j], mps[j])

            old_wf = torch.einsum('abcd,bcdefghi,efgjklmn,ojkl->ahimno', *mpo_mps)

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
            for j in range(self._ny+1):
                u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', rs[j], ls[j]))
                # truncate
                ut, st, vt = u[:, :rho], s[:rho], v[:rho, :]
                ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
                # inverse of square root
                sst_inv = (1.0 / torch.sqrt(st)).diag()

                if self._cflag:
                    sst_inv = sst_inv.cdouble()

                pr = torch.einsum('abcd,de->abce', ls[j], vt_dagger @ sst_inv)
                pl = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, rs[j])
                prs.append(pr)
                pls.append(pl)

            # apply projectors to compress the MPO-MPS
            mps = [None]*(self._ny+2)
            # head and tail
            mps[0]= torch.einsum('abcd,bcde->ae', mpo_mps[0], prs[0])
            mps[-1] = torch.einsum('abcd,ebcd->ea', pls[-1], mpo_mps[-1])

            for j in range(1, self._ny+1):
                mps[j] = torch.einsum('abcd,bcdefghi,efgj->ajhi', pls[j-1], mpo_mps[j], prs[j])

            new_wf = torch.einsum('ab,bcde,cfgh,if->adeghi', *mps)
            print('r loss:', torch.linalg.norm(new_wf-old_wf))

            # update CTM tensors
            # a right move means a shift: c -> cx
            cx = (c[0]-1) % self._nx, c[1]
            head, tail = cx, (cx[0], (cx[1]+self._ny-1) % self._ny)

            for j in range(self._ny):
                self._ctms[(cx[0], (cx[1]+j) % self._ny)][7] = mps[j+1] / torch.linalg.norm(mps[j+1])

            self._ctms[head][1] = mps[0] / torch.linalg.norm(mps[0])
            self._ctms[tail][3] = mps[-1] / torch.linalg.norm(mps[-1])

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
            env_l = torch.einsum('ab,bcde,fc->adef', self._ctms[c][0], self._ctms[c][6], self._ctms[c][2])
            env_r = torch.einsum('ab,bcde,fc->adef', self._ctms[cx][1], self._ctms[cx][7], self._ctms[cx][3])

            # denominator
            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], dts[c], self._ctms[c][5])
            temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[cx][4], dts[cx], self._ctms[cx][5])
            den = torch.einsum('abcd,abcd', temp, env_r)
            # numerator
            temp = env_l.clone()
            temp = torch.einsum('eAag,efDd,AaBbCicDd,ghBb->fCich', temp, self._ctms[c][4], impure_dts[0], self._ctms[c][5])
            temp = torch.einsum('eAiag,efDd,AiaBbCcDd,ghBb->fCch', temp, self._ctms[cx][4], impure_dts[1], self._ctms[cx][5])
            num = torch.einsum('abcd,abcd', temp, env_r)
            meas.append(num / den)

            # Y-direction
            cy = c[0], (c[1]+1) % self._ny

            impure_dts = []
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBfbCcDd', mts_conj[c], op_mpo[0], mts[c]))
            impure_dts.append(torch.einsum('ABCDE,fEe,abcde->AaBbCcDfd', mts_conj[cy], op_mpo[1], mts[cy]))

            env_d = torch.einsum('ab,acde,cf->bdef', self._ctms[c][0], self._ctms[c][4], self._ctms[c][1])
            env_u = torch.einsum('ab,acde,cf->bdef', self._ctms[cy][2], self._ctms[cy][5], self._ctms[cy][3])

            # denominator
            temp = env_d.clone()
            temp = torch.einsum('eDdg,efAa,AaBbCcDd,ghCc->fBbh', temp, self._ctms[c][6], dts[c], self._ctms[c][7])
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


    def ctm_onebody_measure(self, op: torch.tensor):

        mts, mts_conj = {}, {}
        for temp_c in self._coords:
            t = self.merged_tensor(temp_c)
            mts.update({temp_c: t})
            mts_conj.update({temp_c: t.conj()})

        # double tensors
        dts = self.double_tensors()

        # for c in self._coords:

        c = (0, 0)

        # contraction
        env_l = torch.einsum('ab,bcde,fc->adef', self._ctms[c][0], self._ctms[c][6], self._ctms[c][2])
        env_r = torch.einsum('ab,bcde,fc->adef', self._ctms[c][1], self._ctms[c][7], self._ctms[c][3])

        temp = env_l.clone()
        temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], dts[c], self._ctms[c][5])
        den = torch.einsum('abcd,abcd', temp, env_r)

        impure_dt = torch.einsum('ABCDE,Ee,abcde->AaBbCcDd', mts_conj[c], op, mts[c])
        temp = env_l.clone()
        temp = torch.einsum('eAag,efDd,AaBbCcDd,ghBb->fCch', temp, self._ctms[c][4], impure_dt, self._ctms[c][5])
        num = torch.einsum('abcd,abcd', temp, env_r)

        return (num / den)


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


    def ctmrg_mu_test(self, c: tuple, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site within the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS and MPO
        mps = [self._ctms[c][2], self._ctms[c][5], self._ctms[c][3]]
        mpo = [self._ctms[c][6], dts[c], self._ctms[c][7]]

        mpo_mps = [None]*3

        mpo_mps[0] = torch.einsum('ab,cbde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,cbde->adec', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        # cb = (c[0]-1) % self._nx, c[1]
        cb = c
        temp_mps = [self._ctms[cb][2], self._ctms[cb][5], self._ctms[c][5], self._ctms[c][3]]
        temp_mpo = [self._ctms[cb][6], dts[cb], dts[c], self._ctms[c][7]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,cbde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,cbde->adec', temp_mps[3], temp_mpo[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        
        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        # cf = (c[0]+1) % self._nx, c[1]
        cf = c
        temp_mps = [self._ctms[c][2], self._ctms[c][5], self._ctms[cf][5], self._ctms[cf][3]]
        temp_mpo = [self._ctms[c][6], dts[c], dts[cf], self._ctms[cf][7]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,cbde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgBb,AaBbCcDd->fAagCcDd', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,cbde->adec', temp_mps[3], temp_mpo[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # apply projectors and update CTM tensors
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,bcde->ae', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a up-move means absorbing this row
        # c_new = c[0], (c[1]-1) % self._ny
        c_new = c
        self._ctms[c_new][2] = mps[0] / norms[0]
        self._ctms[c_new][5] = mps[1] / norms[1]
        self._ctms[c_new][3] = mps[2] / norms[2]

        return 1


    def ctmrg_md_test(self, c: tuple, dts: dict):
        r'''
        one step of CTMRG for a site

        Parameters:
        ----------
        c: tuple[int], coordinate of a site with the unit cell
        dts: dict, double tensors
        '''

        # build temporary MPS and MPO
        mps = [self._ctms[c][0], self._ctms[c][4], self._ctms[c][1]]
        mpo = [self._ctms[c][6], dts[c], self._ctms[c][7]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,bcde->adec', mps[0], mpo[0])
        mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,bcde->adec', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        # cb = (c[0]-1) % self._nx, c[1]
        cb = c
        temp_mps = [self._ctms[cb][0], self._ctms[cb][4], self._ctms[c][4], self._ctms[c][1]]
        temp_mpo = [self._ctms[cb][6], dts[cb], dts[c], self._ctms[c][7]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,bcde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,bcde->adec', temp_mps[3], temp_mpo[3])
        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        # cf = (c[0]+1) % self._nx, c[1]
        cf = c
        temp_mps = [self._ctms[c][0], self._ctms[c][4], self._ctms[cf][4], self._ctms[cf][1]]
        temp_mpo = [self._ctms[c][6], dts[c], dts[cf], self._ctms[cf][7]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,bcde->adec', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('fgDd,AaBbCcDd->fAagCcBb', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,bcde->adec', temp_mps[3], temp_mpo[3])
        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((3,), (0, 1, 2)), qr_dims=(0, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # apply projectors and update CTM tensors
        mps = [None]*3
        mps[0] = torch.einsum('abcd,abce->ed', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,bcde->ae', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # c_new = c[0], (c[1]+1) % self._ny
        c_new = c
        self._ctms[c_new][0] = mps[0] / norms[0]
        self._ctms[c_new][4] = mps[1] / norms[1]
        self._ctms[c_new][1] = mps[2] / norms[2]

        return 1


    def ctmrg_ml_test(self, c: tuple, dts: dict):

        # build temporary MPS and MPO
        mps = [self._ctms[c][0], self._ctms[c][6], self._ctms[c][2]]
        mpo = [self._ctms[c][4], dts[c], self._ctms[c][5]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('ab,acde->cbde', mps[0], mpo[0])
        # f B b
        # | | |--C 
        # *****
        # | | |--c
        # e D d
        mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', mps[1], mpo[1])
        mpo_mps[2] = torch.einsum('ab,acde->cbde', mps[2], mpo[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        # cb = c[0], (c[1]-1) % self._ny 
        cb = c
        temp_mps = [self._ctms[cb][0], self._ctms[cb][6], self._ctms[c][6], self._ctms[c][2]]
        temp_mpo = [self._ctms[cb][4], dts[cb], dts[c], self._ctms[c][5]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,acde->cbde', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,acde->cbde', temp_mps[3], temp_mpo[3])
        
        # QR and LQ from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(0, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        # inverse of square root
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        # cf = c[0], (c[1]+1) % self._ny
        cf = c
        temp_mps = [self._ctms[c][0], self._ctms[c][6], self._ctms[cf][6], self._ctms[cf][2]]
        temp_mpo = [self._ctms[c][4], dts[c], dts[cf], self._ctms[cf][5]]
        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('ab,acde->cbde', temp_mps[0], temp_mpo[0])
        temp_mpo_mps[1] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[1], temp_mpo[1])
        temp_mpo_mps[2] = torch.einsum('efAa,AaBbCcDd->eDdfBbCc', temp_mps[2], temp_mpo[2])
        temp_mpo_mps[3] = torch.einsum('ab,acde->cbde', temp_mps[3], temp_mpo[3])
        
        # QR and LQ from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(0, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))
        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a left-move means:
        # c_new = (c[0]+1) % self._nx, c[1]
        c_new = c
        self._ctms[c_new][0] = mps[0] / norms[0]
        self._ctms[c_new][6] = mps[1] / norms[1]
        self._ctms[c_new][2] = mps[2] / norms[2]

        return 1


    def ctmrg_mr_test(self, c: tuple, dts: dict):

        # build temporary MPS and MPO
        mps = [self._ctms[c][1], self._ctms[c][7], self._ctms[c][3]]
        mpo = [self._ctms[c][4], dts[c], self._ctms[c][5]]

        mpo_mps = [None]*3
        mpo_mps[0] = torch.einsum('abcd,be->aecd', mpo[0], mps[0])
        mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', mpo[1], mps[1])
        mpo_mps[2] = torch.einsum('abcd,be->aecd', mpo[2], mps[2])

        # build forward and backward projectors
        ps = [None]*4
        # backward one
        # cb = c[0], (c[1]-1) % self._ny 
        cb = c
        temp_mps = [self._ctms[cb][1], self._ctms[cb][7], self._ctms[c][7], self._ctms[c][3]]
        temp_mpo = [self._ctms[cb][4], dts[cb], dts[c], self._ctms[c][5]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('abcd,be->aecd', temp_mpo[0], temp_mps[0])
        temp_mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[1], temp_mps[1])
        temp_mpo_mps[2] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[2], temp_mps[2])
        temp_mpo_mps[3] = torch.einsum('abcd,be->aecd', temp_mpo[3], temp_mps[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        # truncate
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)
        ps[0] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[1] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        # forward one
        # cf = c[0], (c[1]+1) % self._ny
        cf = c
        temp_mps = [self._ctms[c][1], self._ctms[c][7], self._ctms[cf][7], self._ctms[cf][3]]
        temp_mpo = [self._ctms[c][4], dts[c], dts[cf], self._ctms[cf][5]]

        temp_mpo_mps = [None]*4
        temp_mpo_mps[0] = torch.einsum('abcd,be->aecd', temp_mpo[0], temp_mps[0])
        temp_mpo_mps[1] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[1], temp_mps[1])
        temp_mpo_mps[2] = torch.einsum('AaBbCcDd,efCc->eDdfBbAa', temp_mpo[2], temp_mps[2])
        temp_mpo_mps[3] = torch.einsum('abcd,be->aecd', temp_mpo[3], temp_mps[3])

        # QR and LQ factorizations from both ends
        temp = temp_mpo_mps[0]
        q, r = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 0))
        temp = torch.einsum('abcd,bcdefghi->aefghi', r, temp_mpo_mps[1])
        q, r = tp.linalg.tqr(temp, group_dims=((0, 4, 5), (1, 2, 3)), qr_dims=(1, 0))

        temp = temp_mpo_mps[-1]
        q, l = tp.linalg.tqr(temp, group_dims=((0,), (1, 2, 3)), qr_dims=(1, 3))
        temp = torch.einsum('abcdefgh,defi->abcigh', temp_mpo_mps[-2], l)
        q, l = tp.linalg.tqr(temp, group_dims=((3, 4, 5), (0, 1, 2)), qr_dims=(0, 3))

        u, s, v = tp.linalg.svd(torch.einsum('abcd,bcde->ae', r, l))
        ut, st, vt = u[:, :self._rho], s[:self._rho], v[:self._rho, :]
        ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()
        sst_inv = (1.0 / torch.sqrt(st)).diag().to(self._dtype)

        ps[2] = torch.einsum('abcd,de->abce', l, vt_dagger @ sst_inv)
        ps[3] = torch.einsum('ab,bcde->acde', sst_inv @ ut_dagger, r)

        mps = [None]*3
        mps[0] = torch.einsum('abcd,bcde->ae', mpo_mps[0], ps[0])
        mps[1] = torch.einsum('abcd,bcdefghi,efgj->ajhi', ps[1], mpo_mps[1], ps[2])
        mps[2] = torch.einsum('abcd,ebcd->ea', ps[3], mpo_mps[2])

        norms = [torch.linalg.norm(t) for t in mps]
        # a right-move means:
        # c_new = (c[0]-1) % self._nx, c[1]
        c_new = c
        self._ctms[c_new][1] = mps[0] / norms[0]
        self._ctms[c_new][7] = mps[1] / norms[1]
        self._ctms[c_new][3] = mps[2] / norms[2]

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


