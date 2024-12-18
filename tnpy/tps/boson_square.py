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
    def __init__(
            self,
            site_tensors: dict,
            link_tensors: dict,
            dim_phys=2,
            dtype=torch.float64):
        r'''initialization

        Parameters
        ----------
        site_tensors: dict, {key: coordinate, value: tensor}
        link_tensors: dict, {key: coordinate, value: tensor}
        ctms: dict, {key: coordinate, value: list}, environment tensors for CTMRG
        '''
        self._dim_phys = dim_phys
        self._dtype = dtype
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

        assert self._nx > 1 and self._ny > 0, 'the TPS unit cell should be at least: 2*1'

        # inner bond dimension
        self._chi = self._site_tensors[(0, 0)].shape[0]


    @classmethod
    def rand(
            cls,
            nx: int,
            ny: int,
            chi: int,
            dtype=torch.float64):
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

        return cls(site_tensors=site_tensors, link_tensors=link_tensors, dtype=dtype)


    @classmethod
    def randn(
            cls,
            nx: int,
            ny: int,
            chi: int,
            dtype=torch.float64):
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
            lam_x = torch.rand(chi).diag().to(dtype)
            lam_y = torch.rand(chi).diag().to(dtype)
            # normalization
            site_tensors[(i, j)] = temp / torch.linalg.norm(temp)
            link_tensors[(i, j)] = [lam_x / torch.linalg.norm(lam_x), lam_y / torch.linalg.norm(lam_y)]

        return cls(site_tensors=site_tensors, link_tensors=link_tensors, dtype=dtype)


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


    def site_envs(
            self,
            site: tuple[int, int],
            inner_bonds=()) -> list:
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


    def absorb_envs(
            self,
            t: torch.tensor,
            envs):
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

    def merged_tensor(
            self,
            site: tuple[int, int]):
        r'''
        return site tensor merged with square root of link tensors around

        Parameters
        ----------
        site: tuple[int], coordinate
        '''
        envs = self.site_envs(site, inner_bonds=(0, 1, 2, 3))

        return self.absorb_envs(self._site_tensors[site], envs)


    def merged_tensors(self):
        temp = {}
        for c in self._coords:
            temp.update({c: self.merged_tensor(c)})

        return temp


    def pure_double_tensor(
            self,
            c: tuple):
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
            dts.update({c: self.pure_double_tensor(c)})

        return dts


    def unified_tensor(
            self,
            requires_grad=False) -> torch.tensor:
        r'''
        return the TPS tensor in a unit cell as a whole tensor
        new dimension is created as dim=0
        '''
        tens = []
        for c in self._coords:
            tens.append(self.merged_tensor(c))

        return torch.stack(tens, dim=0).requires_grad_(requires_grad)


    def simple_update_proj(
            self,
            time_evo_mpo: tuple[torch.tensor]):
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
            sst_inv = (1.0 / sst).diag().to(self._dtype)
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
                u, s, v = tp.linalg.svd(temp)

                # truncate
                ut, st, vt = u[:, :self._chi], s[:self._chi], v[:self._chi, :]
                ut_dagger, vt_dagger = ut.t().conj(), vt.t().conj()

                sst = torch.sqrt(st)
                sst_inv = (1.0 / sst).diag().to(self._dtype)
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


    def beta_twobody_measure_ops(
            self,
            ops: tuple[torch.tensor]):
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

        return torch.as_tensor(res)


    def beta_twobody_measure(
            self,
            op: torch.tensor) -> torch.tensor:
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

        return torch.as_tensor(res)


class ClassicalSquareCTMRG(object):
    r'''class of CTMRG on a square lattice for a classcial model'''
    pass


class QuantumSquareCTMRG(object):
    r'''class of CTMRG method on a square lattice for a quantum wavefunction'''
    def __init__(
            self,
            wfs: dict,
            rho: int,
            ctms=None,
            dtype=torch.float64):
        r'''initialization

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

        that is, effectively, each site is placed by NINE tensors: 1 double tensor + 8 CTM tensors
        '''
        self._dtype = dtype
        # sorted by the key/coordinate (x, y), firstly by y, then by x
        # self._dts = dict(sorted(dts.items(), key=lambda x: (x[0][1], x[0][0])))
        self._dts = dts
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
        self._chi = self._dts[(0, 0)].shape[0]

        # for CTMRG 
        self._ctm_names = 'C0', 'C1', 'C2', 'C3', 'Ed', 'Eu', 'El', 'Er'
        if ctms is None:
            ctm = {}
            for i, n in enumerate(self._ctm_names):
                # corners and edges
                if i < 4:
                    ctm.update({n: torch.rand(rho, rho).to(dtype)})
                else:
                    ctm.update({n: torch.rand(rho, rho, self._chi, self._chi).to(dtype)})
            ctms = {}
            for i, j in itertools.product(range(self._nx), range(self._ny)):
                ctms.update({(i, j): ctm})
        else:
            for c, ctm in ctms.items():
                assert c in self._coords, 'Coordinate of new CTM tensors is not correct' 
                for k, v in ctm.items():
                    if k not in self._ctm_names:
                        raise ValueError('Name of new CTM tensor is not valid')

        self._ctms = ctms
        self._rho = self._ctms[(0, 0)]['C0'].shape[0]


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
        r'''update CTM tensors'''
        for c, ctm in ctms.items():
            assert c in self._coords, 'Coordinate of new CTM tensors is not correct' 
            for k, v in ctm.items():
                if k not in self._ctm_names:
                    raise ValueError('Name of new CTM tensor is not valid')

        self._ctms = ctms
        self._rho = self._ctms[(0, 0)]['C0'].shape[0]

        return 1


    def measure_onebody(
            self,
            wf: torch.tensor,
            op: torch.tensor):
        r'''measure onebody operator

        Parameters:
        ----------
        wf: torch.tensor, wavefunction as a unified tensor
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
