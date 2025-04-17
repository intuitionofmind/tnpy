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
from tnpy import Z2gTensor, GTensor 

class FermiSquareTPS(object):
    r'''
    class of a infinite fermionic tensor product state on a square lattice
    '''

    def __init__(self, nx: int, ny: int, site_tensors: dict, link_tensors: dict, info=None):
        r'''

        Parameters
        ----------
        nx, ny: int, number of sites in the unit cell along X- and Y-direction
        site_tensors: dict, {(x, y): GTensor}, site tensors, NOT necessarily sorted
        link_tensors: dict, {(x, y): tuple[GTensor]}, link tensors along X- and Y-direction
        '''

        self._nx, self._ny = nx, ny
        self._size = nx*ny

        # sorted by the key/coordinate (x, y), firstly by y, then by x
        self._site_tensors = dict(sorted(site_tensors.items(), key=lambda x: (x[0][1], x[0][0])))
        self._link_tensors = dict(sorted(link_tensors.items(), key=lambda x: (x[0][1], x[0][0])))

        assert len(self._site_tensors) == self._size, 'number of site tensors not consistant with the size'
        self._coords = tuple(self._site_tensors.keys())

        self._info = info

    @classmethod
    def rand(cls, nx: int, ny: int, chi: int, cflag=False):
        r'''
        generate a random UC4FermionicSquareTPS

        Parameters
        ----------
        nx, ny: int, number of sites in the unit cell along X- and Y-direction
        chi: int, bond dimension of the site tensor
            if even, dimensions of even parity and odd parity sectors are: both chi//2
            if odd, dimensions of even parity and odd parity sectors are: chi//2+1 and chi//2
        '''

        # rank-5 site-tensor
        # the index order convention
        # clockwisely, starting from the WEST:
        #       1 4
        #       |/
        # T: 0--*--2
        #       |
        #       3
        # diagonal link-tensors living on the link appproximate the environment
        # number of link-tensors is 2*self._uc_size
        # arrow, site-tensor, link-tensor conventions as below:
        #   v   v   v   v
        #   |   |   |   |
        #  -A-<-B-<-A-<-B-<
        #   |5  |7  |5  |
        #   v   v   v   v
        #   | 4 | 6 | 4 |
        #  -C-<-D-<-C-<-D-<
        #   |1  |3  |1  |3
        #   v   v   v   v
        #   | 0 | 2 | 0 |
        #  -A-<-B-<-A-<-B-<
        #   |   |   |   |
        # dual for A, B, C, D
        # ALL physical bonds are assumed to be outgoing, 0, vector space
        site_dual = (0, 1, 1, 0, 0)
        link_dual = (0, 1)
        # coordinates of the unit cell
        dim_o = chi // 2
        dim_e = chi-dim_o
        site_shape = [(dim_e, dim_o)]*4
        site_shape.append((2, 2))
        site_shape = tuple(site_shape)
        link_shape = (dim_e, dim_o), (dim_e, dim_o)
        site_tensors, link_tensors = {}, {}
        for y, x in itertools.product(range(ny), range(nx)):
            s = x, y
            temp = GTensor.rand(dual=site_dual, shape=site_shape, cflag=cflag)
            site_tensors[s] = (1.0/temp.max())*temp
            temp_x = GTensor.rand_diag(dual=link_dual, shape=link_shape, cflag=cflag)
            temp_y = GTensor.rand_diag(dual=link_dual, shape=link_shape, cflag=cflag)
            link_tensors[s] = [(1.0/temp_x.max())*temp_x, (1.0/temp_y.max())*temp_y]

        return cls(nx, ny, site_tensors, link_tensors)

    @classmethod
    def init_from_blocks(cls, nx: int, ny: int, chi: int, site_blocks: list, link_blocks_x: list, link_blocks_y: list):
        r'''
        initialize from lists of blocks
        '''

        site_dual = (0, 1, 1, 0, 0)
        link_dual = (0, 1)

        dim_o = chi // 2
        dim_e = chi-dim_o
        site_shape = [(dim_e, dim_o)]*4
        site_shape.append((2, 2))
        site_shape = tuple(site_shape)
        link_shape = (dim_e, dim_o), (dim_e, dim_o)

        site_tensors, link_tensors = {}, {}
        i = 0
        for y, x in itertools.product(range(ny), range(nx)):
            s = x, y
            site_tensors[s] = GTensor(dual=site_dual, shape=site_shape, blocks=site_blocks[i])
            lx = GTensor(dual=link_dual, shape=link_shape, blocks=link_blocks_x[i])
            ly = GTensor(dual=link_dual, shape=link_shape, blocks=link_blocks_y[i])
            link_tensors[s] = [lx, ly]
            i += 1

        return cls(nx, ny, site_tensors, link_tensors)

    @classmethod
    def init_from_instance(cls, ins, chi: int):
        r'''
        immport a UC4FermionicSquareTPS

        Parameters
        ----------
        ins: another instance with 
        '''

        assert chi >= ins._chi, 'GTensor must be enlarged'
        dim_diff = chi-ins.chi
        
        def _pad_site_tensor(gt: GTensor, dim: int):

            new_blocks = {}
            for k, v in gt.blocks().items():
                new_blocks[k] = tnf.pad(input=v, pad=(0, 0, 0, dim, 0, dim, 0, dim, 0, dim), mode='constant', value=0.0)

            return GTensor(dual=gt.dual, blocks=new_blocks)

        def _pad_link_tensor(gt: GTensor, dim: int):

            new_blocks = {}
            for k, v in gt.blocks().items():
                new_blocks[k] = tnf.pad(input=v, pad=(0, 1, 0, 1), mode='constant', value=1.0)

            return GTensor(dual=gt.dual, blocks=new_blocks)

        sts, lts = ins.tensors()
        site_tensors, link_tensors = {}, {}
        for s in ins.coords:
            site_tensors[s] = _pad_site_tensor(sts[s], dim_diff)
            link_tensors[s] = [_pad_link_tensor(lts[s][0], dim_diff), _pad_link_tensor(lts[s][1], dim_diff)]

        return cls(site_tensors, link_tensors)

    @property
    def size(self):

        return self._size

    @property
    def coords(self):

        return self._coords

    def tensors(self):

        return self._site_tensors, self._link_tensors

    def site_tensors(self) -> list:

        return list(self._site_tensors.values())

    def link_tensors(self) -> list:
        
        lts = []
        for c in self._coords:
            lts.append(self._link_tensors[c][0])
            lts.append(self._link_tensors[c][1])

        return lts

    def site_envs(self, site: tuple) -> list:
        r'''
        return the environment bond weights around a site
        '''

        envs = []
        envs.append(self._link_tensors[((site[0]-1) % self._nx, site[1])][0])
        envs.append(self._link_tensors[site][1])
        envs.append(self._link_tensors[site][0])
        envs.append(self._link_tensors[(site[0], (site[1]-1) % self._ny)][1])

        return envs

    def sqrt_env(self, env: GTensor) -> GTensor:
        r'''
        compute the square root of a bond weight Gensor
        env: GTensor
        '''

        if (0, 1) == env.dual:
            e = torch.sqrt(env.blocks()[(0, 0)].diag())
            o = torch.sqrt(env.blocks()[(1, 1)].diag())
            new_blocks = {(0, 0): e.diag(), (1, 1): o.diag()}
            cflag = env.cflag
        elif (1, 0) == env.dual:
            e = torch.sqrt(env.blocks()[(0, 0)].diag())
            o = torch.sqrt(env.blocks()[(1, 1)].diag())
            new_blocks = {(0, 0): e.diag(), (1, 1): 1.j*o.diag()}
            cflag = True

        return GTensor(dual=env.dual, shape=env.shape, blocks=new_blocks, cflag=cflag)

    def mixed_site_envs(self, site, ex_bonds: tuple):
        r'''
        return the environment bond weights around a site if it is an external one
        otherwise, return its square root
        '''

        envs = self.site_envs(site)
        for j in range(4):
            if j not in ex_bonds:
                envs[j] = self.sqrt_env(envs[j])

        return envs

    def merged_tensors(self) -> dict:
        r'''
        site tensors merged with square root of the environments

        Returns
        -------
        mgts: dict, {site: merged GTensor}
        '''

        mgts = {}
        for c in self._coords:
            gt = self._site_tensors[c]
            envs = self.site_envs(c)
            half_envs = [self.sqrt_env(t) for t in envs]
            # merge
            mgts[c] = tp.gcontract('abcde,fa,bg,ch,id->fghie', gt, *half_envs)

        return mgts

    def simple_update_proj_sort(self, te_mpo: tuple, average_weights=None, expand=None, fixed_dims=False, ifprint=False):
        r'''
        simple update
        average on 4 loops

        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        average_weights: bool,
        expand: tuple[int], optional
            expand to larger D
            expand[0]: the dominant sector lenght, expand[1]: the secondary sector length
        fixed_dims: bool, if the cut-off dim fixed or not
        '''

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            gts = [self._site_tensors[c], self._site_tensors[cx]]
            # set cut-off
            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[2]
                else:
                    cf = sum(gts[0].shape[2])
            else:
                # expand according to the dominant sector
                temp = self._link_tensors[c][0].blocks()
                se, so = temp[(0, 0)].diag(), temp[(1, 1)].diag()
                if se[0] > so[0]:
                    cf = expand[0], expand[1]
                else:
                    cf = expand[1], expand[0]

            envs = [self.mixed_site_envs(c, ex_bonds=(0, 1, 3)), self.mixed_site_envs(cx, ex_bonds=(1, 2, 3))]
            # inverse of envs, for removing the envs later
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # time evo operation
            gts[0] = tp.gcontract('ECe,abcde->abCcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('AEe,abcde->AabcdE', te_mpo[1], gts[1])
            # QR and LQ factorizations
            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            # overall cutoff
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # only left-conjugation of U and right-conjugation of V are valid under truncation
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            # apply projectors
            gts[0] = tp.gcontract('abCcde,Ccf,fg,gh->abhde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fAa,Aabcde->hbcde', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cx] = (1.0/gts[1].max())*gts[1]

            new_lt = (1.0/s.max())*s

            if ifprint and None == expand:
                diff = 0.0
                for key, val in new_lt.blocks().items():
                    diff += (self._link_tensors[c][0].blocks()[key]-val).norm()
                print('X Lambda changing:', diff.item())

            self._link_tensors[c][0] = new_lt

            # Y-direction
            gts = [self._site_tensors[c], self._site_tensors[cy]]
            # set cut-off
            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[1]
                else:
                    cf = sum(gts[0].shape[1])
            else:
                temp = self._link_tensors[c][1].blocks()
                se, so = temp[(0, 0)].diag(), temp[(1, 1)].diag()
                if se[0] > so[0]:
                    cf = expand[0], expand[1]
                else:
                    cf = expand[1], expand[0]

            envs = [self.mixed_site_envs(c, ex_bonds=(0, 2, 3)), self.mixed_site_envs(cy, ex_bonds=(0, 1, 2))]
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # time evo
            gts[0] = tp.gcontract('EBe,abcde->aBbcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('DEe,abcde->abcDdE', te_mpo[1], gts[1])

            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            # apply projectors            
            gts[0] = tp.gcontract('aBbcde,Bbf,fg,gh->ahcde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fDd,abcDde->abche', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cy] = (1.0/gts[1].max())*gts[1]

            new_lt = (1.0/s.max())*s
            if ifprint and None == expand:
                diff = 0.0
                for key, val in new_lt.blocks().items():
                    diff += (self._link_tensors[c][1].blocks()[key]-val).norm()
                print('Y Lambda changing:', diff.item())

            self._link_tensors[c][1] = new_lt

            # average
            if 'sort' == average_weights:
                # self.sorted_average_weights(c, ifprint=ifprint)
                # self.sorted_average_weights_plaquette(c, ifprint=ifprint)
                self.sorted_average_weights_bond(c, ifprint=ifprint)
            elif 'direct' == average_weights:
                self.direct_average_weights(c, ifprint=ifprint)

        return 1

    def simple_update_proj(self, te_mpo: tuple, sort_weights=False, average_weights=None, average_method=None, expand=None):
        r'''
        simple update
        average on 4 loops

        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        sort_weights: bool, 
            True: combine even and odd singular values together and sort then truncate
            False: truncation even and odd singular values seperately
        average_weights: string,
            'dominance', average by the dominance sector
            'parity', average by parity sectors
        expand: tuple[int], optional
            expand to larger D
        '''

        # factorize to MPO
        # u, s, v = tp.linalg.gtsvd(time_evo, group_dims=((0, 2), (1, 3)), svd_dims=(1, 0))
        # se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
        # ss = GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
        # te_mpo = tp.gcontract('abc,bd->adc', u, ss), tp.gcontract('ab,bcd->acd', ss, v)

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            gts = [self._site_tensors[c], self._site_tensors[cx]]
            envs = [self.mixed_site_envs(c, ex_bonds=(0, 1, 3)), self.mixed_site_envs(cx, ex_bonds=(1, 2, 3))]
            # inverse of envs, for removing the envs later
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[2])
            else:
                cf = gts[0].shape[2][0], gts[0].shape[2][1]
                if isinstance(expand, tuple):
                    cf = expand
            # time evo operation
            gts[0] = tp.gcontract('ECe,abcde->abCcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('AEe,abcde->AabcdE', te_mpo[1], gts[1])
            # QR and LQ factorizations
            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # only left-conjugation of U and right-conjugation of V are valid under truncation
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            '''

            # check identity
            # print('check identity:')
            # print(l.shape, v_dagger.shape, s_inv.shape, u_dagger.shape, r.shape)
            # idt = tp.gcontract('abc,cd,de,ef,fg,gh,hij->abij', l, v_dagger, s_inv, s, s_inv, u_dagger, r)
            # for key, val in idt.blocks().items():
                # print(key)
                # print(val)
 
            # apply projectors
            gts[0] = tp.gcontract('abCcde,Ccf,fg,gh->abhde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fAa,Aabcde->hbcde', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cx].dual == gts[1].dual
            # assert self._link_tensors[c][0].dual == s.dual

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cx] = (1.0/gts[1].max())*gts[1]
            self._link_tensors[c][0] = (1.0/s.max())*s

            # Y-direction
            gts = [self._site_tensors[c], self._site_tensors[cy]]
            envs = [self.mixed_site_envs(c, ex_bonds=(0, 2, 3)), self.mixed_site_envs(cy, ex_bonds=(0, 1, 2))]
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[1])
            else:
                cf = gts[0].shape[1][0], gts[0].shape[1][1]
                if isinstance(expand, tuple):
                    cf = expand
            # time evo
            gts[0] = tp.gcontract('EBe,abcde->aBbcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('DEe,abcde->abcDdE', te_mpo[1], gts[1])

            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            '''
 
            # apply projectors            
            gts[0] = tp.gcontract('aBbcde,Bbf,fg,gh->ahcde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fDd,abcDde->abche', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cy].dual == gts[1].dual
            # assert self._link_tensors[c][1].dual == s.dual

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cy] = (1.0/gts[1].max())*gts[1]
            self._link_tensors[c][1] = (1.0/s.max())*s

            if expand is None and sort_weights is False:
                if 'plaquette' == average_method:
                    self.average_plquette_weights(c, mode=average_weights)
            elif sort_weights is True:
                lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                sds, sms = [], []
                for t in lams:
                    se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                    print(se, so)

        if expand is None and sort_weights is False:
            if 'all' == average_method:
                self.average_all_weights(mode=average_weights)

        return 1

    def twobody_cluster_update(self, te_mpo: tuple, sort_weights=False, average_weights=None):
        r'''
        simple update
        average on 4 loops

        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        sort_weights: bool, 
            True: combine even and odd singular values together and sort then truncate
            False: truncation even and odd singular values seperately
        average_weights: string,
            'dominance', average by the dominance sector
            'parity', average by parity sectors
        expand: tuple[int], optional
            expand to larger D
        '''

        # external bonds for the cluster
        external_bonds = (0, 3), (2, 3), (0, 1), (1, 2)

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            cluster = [c, cx, cy, cxy]
            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=external_bonds[i]) for i, site in enumerate(cluster)]
            # only need two site's env_inv
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(4)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[2])
            else:
                cf = gts[0].shape[2][0], gts[0].shape[2][1]
            # time evo operation
            gts[0] = tp.gcontract('ECe,abcde->abCcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('AEe,abcde->AabcdE', te_mpo[1], gts[1])
            # clockwise QR
            temp = gts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 2, 3, 4, 5), (1,)), qr_dims=(1, 0))
            temp = tp.gcontract('Dd,abcde->abcDe', r, gts[2])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 3, 4), (2,)), qr_dims=(2, 0))
            temp = tp.gcontract('Aa,abcde->Abcde', r, gts[3])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 4), (3,)), qr_dims=(3, 1))
            temp = tp.gcontract('Aabcde,bB->AaBcde', gts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            lx = l
            # counter-clockwise QR
            temp = gts[1]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 3, 4, 5), (2,)), qr_dims=(2, 0))
            temp = tp.gcontract('Dd,abcde->abcDe', r, gts[3])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2, 3, 4), (0,)), qr_dims=(0, 1))
            temp = tp.gcontract('abcde,cC->abCde', gts[2], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 4), (3,)), qr_dims=(3, 1))
            temp = tp.gcontract('abCcde,bB->aBCcde', gts[0], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rx = r
            # build projectors
            rl = tp.gcontract('abc,bcd->ad', rx, lx)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # only left-conjugation of U and right-conjugation of V are valid under truncation
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            # check identity
            # print('check identity:')
            # print(l.shape, v_dagger.shape, s_inv.shape, u_dagger.shape, r.shape)
            # idt = tp.gcontract('abc,cd,de,ef,fg,gh,hij->abij', l, v_dagger, s_inv, s, s_inv, u_dagger, r)
            # for key, val in idt.blocks().items():
                # print(key)
                # print(val)
            '''
 
            # apply projectors
            gts[0] = tp.gcontract('abCcde,Ccf,fg,gh->abhde', gts[0], lx, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fAa,Aabcde->hbcde', s_inv, u_dagger, rx, gts[1])
            # place identity on the connected bonds
            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts[0] = tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[0], *envs_inv[0])
            gts[1] = tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[1], *envs_inv[1])
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cx].dual == gts[1].dual
            # assert self._link_tensors[c][0].dual == s.dual

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cx] = (1.0/gts[1].max())*gts[1]
            self._link_tensors[c][0] = (1.0/s.max())*s

            # Y-direction
            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=external_bonds[i]) for i, site in enumerate(cluster)]
            # only need two site's env_inv
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[2][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(4)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[1])
            else:
                cf = gts[0].shape[1][0], gts[0].shape[1][1]
            # time evo
            gts[0] = tp.gcontract('EBe,abcde->aBbcdE', te_mpo[0], gts[0])
            gts[2] = tp.gcontract('DEe,abcde->abcDdE', te_mpo[1], gts[2])
            # clockwise QR
            temp = gts[2]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 3, 4, 5), (2,)), qr_dims=(2, 0))
            temp = tp.gcontract('Aa,abcde->Abcde', r, gts[3])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 4), (3,)), qr_dims=(3, 1))
            temp = tp.gcontract('abcde,bB->aBcde', gts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2, 3, 4), (0,)), qr_dims=(0, 1))
            temp = tp.gcontract('aBbcde,cC->aBbCde', gts[0], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            ry = r
            # counter-clockwise QR
            temp = gts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 2, 4, 5), (3,)), qr_dims=(3, 0))
            temp = tp.gcontract('Aa,abcde->Abcde', r, gts[1])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 2, 3, 4), (1,)), qr_dims=(1, 0))
            temp = tp.gcontract('Dd,abcde->abcDe', r, gts[3])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2, 3, 4), (0,)), qr_dims=(0, 1))
            temp = tp.gcontract('abcDde,cC->abCDde', gts[2], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ly = l
            # build projectors
            rl = tp.gcontract('abc,bcd->ad', ry, ly)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            '''
 
            # apply projectors            
            gts[0] = tp.gcontract('aBbcde,Bbf,fg,gh->ahcde', gts[0], ly, v_dagger, s_inv)
            gts[2] = tp.gcontract('hg,gf,fDd,abcDde->abche', s_inv, u_dagger, ry, gts[2])
            # place identity on the connected bonds
            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts[0] = tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[0], *envs_inv[0])
            gts[2] = tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[2], *envs_inv[1])
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cy].dual == gts[1].dual
            # assert self._link_tensors[c][1].dual == s.dual
            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cy] = (1.0/gts[1].max())*gts[2]
            self._link_tensors[c][1] = (1.0/s.max())*s

            if sort_weights is False:
                if 'parity' == average_weights:
                    # direct averge two sectors, naively
                    lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                    ses, sos = [], []
                    for t in lams:
                        ses.append(t.blocks()[(0, 0)].diag())
                        sos.append(t.blocks()[(1, 1)].diag())
                    se, so = 0.25*sum(ses), 0.25*sum(sos)
                    print('average parity:', se, so)
                    new_blocks = {(0, 0):torch.tensor(se).diag(), (1, 1):torch.tensor(so).diag()}
                    new_lam = GTensor(dual=(0, 1), shape=lams[0].shape, blocks=new_blocks, cflag=lams[0].cflag)
                    self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1] = tuple([new_lam]*4)

                if 'dominance' == average_weights:
                    # average by dominance part
                    lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                    sds, sms = [], []
                    flags = []
                    for t in lams:
                        se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                        print(se, so)
                        if se[0].item() > (1.0-1E-12):
                            sds.append(se)
                            sms.append(so)
                            flags.append(True)
                        elif so[0].item() > (1.0-1E-12):
                            sds.append(so)
                            sms.append(se)
                            flags.append(False)

                    # flags = [True, False, True, False]
                    # flags = [False, True, False, True]
                    # print(flags)
                    print('average dominance:', 0.25*sum(sds), 0.25*sum(sms))
                    if flags[0]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[c][0] = GTensor(dual=(0, 1), shape=lams[0].shape, blocks=new_blocks, cflag=lams[0].cflag)
                    if flags[1]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[cx][1] = GTensor(dual=(0, 1), shape=lams[1].shape, blocks=new_blocks, cflag=lams[1].cflag)
                    if flags[2]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[cy][0] = GTensor(dual=(0, 1), shape=lams[2].shape, blocks=new_blocks, cflag=lams[2].cflag)
                    if flags[3]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[c][1] = GTensor(dual=(0, 1), shape=lams[3].shape, blocks=new_blocks, cflag=lams[3].cflag)

            elif sort_weights is True:
                lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                sds, sms = [], []
                for t in lams:
                    se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                    print(se, so)

        return 1

    def twobody_mpo_factorize(self, op: GTensor):
                
        u, s, v = tp.linalg.gtsvd(op, group_dims=((0, 2), (1, 3)), svd_dims=(1, 0))
        se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
        ss = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)

        mpos = tp.gcontract('abc,bd->adc', u, ss), tp.gcontract('ab,bcd->acd', ss, v)

        return mpos

    def threebody_mpo_factorize(self, op: GTensor, internal_flags: tuple) -> tuple:
        r'''
        # 0  1  2
        # |  |  |
        # *--*--*
        # |  |  |
        # 3  4  5
        # --> 
        # 0        1        1
        # |        |        |
        # *--1, 0--*--2, 0--*
        # |        |        |
        # 2        3        2

        Parameters
        ----------
        op: GTensor, rank-6 time evo operator
        internal_flags: tuple[int], 1: normal factoriazation; 0: super factoriazation
        '''

        if internal_flags[0]:
            u, s, v = tp.linalg.gtsvd(op, group_dims=((0, 3), (1, 4, 2, 5)), svd_dims=(1, 0))
            se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
            ss = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
            m0 = tp.gcontract('abc,bd->adc', u, ss) 
            res = tp.gcontract('ab,bcdef->acdef', ss, v)
        else:
            u, s, v = tp.linalg.super_gtsvd(op, group_dims=((0, 3), (1, 4, 2, 5)), svd_dims=(1, 0))
            se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
            ss_0 = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):-1.0*so}, cflag=s.cflag)
            ss_1 = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
            m0 = tp.gcontract('abc,bd->adc', u, ss_0)
            #    1  3
            #    |  |
            # 0--*--*
            #    |  |
            #    2  4
            res = tp.gcontract('ab,bcdef->acdef', ss_1, v)

        if internal_flags[1]:
            u, s, v = tp.linalg.gtsvd(res, group_dims=((0, 1, 2), (3, 4)), svd_dims=(2, 0))
            se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
            ss = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
            m1, m2 = tp.gcontract('abcd,cf->abfd', u, ss), tp.gcontract('ab,bcd->acd', ss, v)
        else:
            u, s, v = tp.linalg.super_gtsvd(res, group_dims=((0, 1, 2), (3, 4)), svd_dims=(2, 0))
            se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
            ss_0 = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
            ss_1 = tp.GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):-1.0*so}, cflag=s.cflag)
            m1, m2 = tp.gcontract('abcd,cf->abfd', u, ss_0), tp.gcontract('ab,bcd->acd', ss_1, v)

        mpos = m0, m1, m2

        # test
        # a        d        g
        # |        |        |
        # *--b, b--*--e, e--*
        # |        |        |
        # c        f        h
        # print('three body MPO:', internal_flags)
        # test = tp.gcontract('abc,bdef,egh->adgcfh', *mpos)
        # for key, val in test.blocks().items():
        #     print(key, (val-op.blocks()[key]).norm())

        return mpos


    def sorted_average_weights_bond(self, c: tuple, ifprint=False):
        r'''
        average sorted weights
        '''


        s_all, cuts = [], []
        flags = []
        for d in range(2):
            se = self._link_tensors[c][d].blocks()[(0, 0)].diag()
            so = self._link_tensors[c][d].blocks()[(1, 1)].diag()

            if se[0].item() > so[0].item():
                s = torch.cat((se, so), dim=0)
                flags.append(True)
                cuts.append(se.shape[0])
            else:
                s = torch.cat((so, se), dim=0)
                flags.append(False)
                cuts.append(so.shape[0])

            s_all.append(s)

            if ifprint:
                print(se.shape[0], so.shape[0], se, so)

        s_mean = sum(s_all) / len(s_all)
        s_mean = s_mean / max(s_mean)

        if ifprint:
            print('sorted average:', s_mean)

        cf = self._link_tensors[(0, 0)][0].cflag

        n = 0
        for d in range(2):
            if flags[n]:
                new_se = s_mean[:cuts[n]].clone()
                new_so = s_mean[cuts[n]:].clone()
            else:
                new_so = s_mean[:cuts[n]].clone()
                new_se = s_mean[cuts[n]:].clone()

            new_blocks = {(0, 0): new_se.diag(), (1, 1): new_so.diag()}
            new_gt = GTensor(dual=(0, 1), shape=self._link_tensors[c][d].shape, blocks=new_blocks, cflag=cf)
            self._link_tensors[c][d] = new_gt

            n += 1

        return 1


    def sorted_average_weights_plaquette(self, c: tuple, ifprint=False):
        r'''
        average sorted weights
        '''

        cx = (c[0]+1) % self._nx, c[1]
        cy = c[0], (c[1]+1) % self._ny
        # four bonds for this plaquette
        cds = (c, 0), (c, 1), (cx, 1), (cy, 0)

        s_all, cuts = [], []
        flags = []
        for (cc, d) in cds:
            se = self._link_tensors[cc][d].blocks()[(0, 0)].diag()
            so = self._link_tensors[cc][d].blocks()[(1, 1)].diag()

            if se[0].item() > so[0].item():
                s = torch.cat((se, so), dim=0)
                flags.append(True)
                cuts.append(se.shape[0])
            else:
                s = torch.cat((so, se), dim=0)
                flags.append(False)
                cuts.append(so.shape[0])

            s_all.append(s)

            if ifprint:
                print(se.shape[0], so.shape[0], se, so)

        s_mean = sum(s_all) / len(s_all)
        s_mean = s_mean / max(s_mean)

        if ifprint:
            print('sorted average:', s_mean)

        cf = self._link_tensors[(0, 0)][0].cflag

        n = 0
        for (cc, d) in cds:
            if flags[n]:
                new_se = s_mean[:cuts[n]].clone()
                new_so = s_mean[cuts[n]:].clone()
            else:
                new_so = s_mean[:cuts[n]].clone()
                new_se = s_mean[cuts[n]:].clone()

            new_blocks = {(0, 0): new_se.diag(), (1, 1): new_so.diag()}
            new_gt = GTensor(dual=(0, 1), shape=self._link_tensors[c][d].shape, blocks=new_blocks, cflag=cf)
            self._link_tensors[cc][d] = new_gt

            n += 1

        return 1


    def sorted_average_weights(self, c: tuple, ifprint=False):
        r'''
        average sorted weights
        '''

        s_all, cuts = [], []
        flags = []
        for c in self._coords:
            for d in range(2):
                se = self._link_tensors[c][d].blocks()[(0, 0)].diag()
                so = self._link_tensors[c][d].blocks()[(1, 1)].diag()
                # join two sectors and sort
                if 0 != se.numel() and 0 != so.numel():
                    if se[0].item() > so[0].item():
                        s = torch.cat((se, so), dim=0)
                        flags.append(True)
                        cuts.append(se.shape[0])
                    else:
                        s = torch.cat((so, se), dim=0)
                        flags.append(False)
                        cuts.append(so.shape[0])
                elif 0 == se.numel():
                    s = torch.cat((so, se), dim=0)
                    flags.append(False)
                    cuts.append(so.shape[0])
                elif 0 == so.numel():
                    s = torch.cat((se, so), dim=0)
                    flags.append(True)
                    cuts.append(se.shape[0])
                s_all.append(s)
                if ifprint:
                    print(se.shape[0], so.shape[0], se, so)

        s_mean = sum(s_all)/len(s_all)
        s_mean = s_mean/max(s_mean)

        if ifprint:
            print('sorted average:', s_mean)

        cf = self._link_tensors[(0, 0)][0].cflag

        n = 0
        for c in self._coords:
            for d in range(2):
                if flags[n]:
                    new_se = s_mean[:cuts[n]].clone()
                    new_so = s_mean[cuts[n]:].clone()
                else:
                    new_so = s_mean[:cuts[n]].clone()
                    new_se = s_mean[cuts[n]:].clone()

                new_blocks = {(0, 0): new_se.diag(), (1, 1): new_so.diag()}
                new_gt = GTensor(dual=(0, 1), shape=self._link_tensors[c][d].shape, blocks=new_blocks, cflag=cf)
                self._link_tensors[c][d] = new_gt

                n += 1

        return 1

    def direct_average_weights(self, c: tuple, ifprint=False):
        r'''
        directly average bond weights
        '''

        s_all = []
        shapes = []
        for c in self._coords:
            for d in range(2):
                se = self._link_tensors[c][d].blocks()[(0, 0)].diag()
                so = self._link_tensors[c][d].blocks()[(1, 1)].diag()
                # join two sectors and sort
                if se[0] > so [0]:
                    s = torch.cat((se, so), dim=0)
                else:
                    s = torch.cat((so, se), dim=0)
                s_all.append(s)
                shapes.append((se.shape[0], so.shape[0]))
                if ifprint:
                    print(se.shape[0], so.shape[0], se, so)

        s_mean = sum(s_all)/len(s_all)
        s_mean = s_mean/max(s_mean)

        if ifprint:
            print('direct average:', s_mean)

        cf = self._link_tensors[(0, 0)][0].cflag

        n = 0
        for c in self._coords:
            for d in range(2):
                new_se = s_mean[:shapes[n][0]].clone()
                new_so = s_mean[shapes[n][0]:].clone()
                # print(new_se, new_so)
                new_blocks = {(0, 0): new_se.diag(), (1, 1): new_so.diag()}
                new_gt = GTensor(dual=(0, 1), shape=self._link_tensors[c][d].shape, blocks=new_blocks, cflag=cf)
                self._link_tensors[c][d] = new_gt
                n += 1

        return 1

    def average_plquette_weights(self, c: tuple, mode='dominance', info=None):
        r'''

        Parameters
        ----------
        c: tuple[int], base coordinate of the plaquette
        '''

        def _build_bond_matrix(sd, sm, flag: bool, cflag: bool):

            if flag:
                new_blocks = {(0, 0):sd.diag(), (1, 1):sm.diag()}
                new_shape = (sd.shape[0], sm.shape[0]), (sd.shape[0], sm.shape[0])
            else:
                new_blocks = {(0, 0):sm.diag(), (1, 1):se.diag()}
                new_shape = (sm.shape[0], sd.shape[0]), (sm.shape[0], sd.shape[0])

            return GTensor(dual=(0, 1), shape=new_shape, blocks=new_blocks, cflag=cflag)

        cx = (c[0]+1) % self._nx, c[1]
        cy = c[0], (c[1]+1) % self._ny

        if 'parity' == mode:
            # directly averge two sectors
            lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
            ses, sos = [], []
            for t in lams:
                ses.append(t.blocks()[(0, 0)].diag())
                sos.append(t.blocks()[(1, 1)].diag())
            se, so = 0.25*sum(ses), 0.25*sum(sos)
            print(info, 'Parity average:', se, so)
            new_blocks = {(0, 0):torch.tensor(se).diag(), (1, 1):torch.tensor(so).diag()}
            new_lam = GTensor(dual=(0, 1), shape=lams[0].shape, blocks=new_blocks, cflag=lams[0].cflag)
            self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1] = tuple([new_lam]*4)

        elif 'dominance' == mode:
            # average by dominance parts
            lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
            sds, sms = [], []
            # mark the dominance in which sector
            flags = [True]*4
            for i, t in enumerate(lams):
                se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                print(se, so)
                # find the dominance sector
                # dominance in even
                if se[0].item() > (1.0-1E-12):
                    sds.append(se)
                    sms.append(so)
                    flags[i] = True
                # dominance in odd
                elif so[0].item() > (1.0-1E-12):
                    sds.append(so)
                    sms.append(se)
                    flags[i] = False

            sds_ave, sms_ave = sum(sds)/len(sds), sum(sms)/len(sms)
            print(info, 'Dominance average:', sds_ave, sms_ave)
            self._link_tensors[c][0] = _build_bond_matrix(sds_ave, sms_ave, flags[0], cflag=lams[0].cflag)
            self._link_tensors[cx][1] = _build_bond_matrix(sds_ave, sms_ave, flags[1], cflag=lams[1].cflag)
            self._link_tensors[cy][0] = _build_bond_matrix(sds_ave, sms_ave, flags[2], cflag=lams[2].cflag)
            self._link_tensors[c][1] = _build_bond_matrix(sds_ave, sms_ave, flags[3], cflag=lams[3].cflag)

        else:
            raise ValueError('mode not matched!')

        return 1

    def average_all_weights(self, mode='dominance'):
        r'''

        Parameters
        ----------
        c: tuple[int], base coordinate of the plaquette
        '''

        def _build_bond_matrix(sd, sm, flag: bool, cflag: bool):

            if flag:
                new_blocks = {(0, 0):sd.diag(), (1, 1):sm.diag()}
                new_shape = (sd.shape[0], sm.shape[0]), (sd.shape[0], sm.shape[0])
            else:
                new_blocks = {(0, 0):sm.diag(), (1, 1):se.diag()}
                new_shape = (sm.shape[0], sd.shape[0]), (sm.shape[0], sd.shape[0])

            return GTensor(dual=(0, 1), shape=new_shape, blocks=new_blocks, cflag=cflag)

        if 'parity' == mode:
            # directly averge two sectors
            ses, sos = [], []
            for c in self._coords:
                for i in range(2):
                    ses.append(self._link_tensors[c][i].blocks()[(0, 0)].diag())
                    sos.append(self._link_tensors[c][i].blocks()[(1, 1)].diag())

            se, so = 0.125*sum(ses), 0.125*sum(sos)
            print('Parity average:', se, so)
            new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
            new_shape = (se.shape[0], so.shape[0]), (se.shape[0], so.shape[0])
            cf = self._link_tensors[(0, 0)][0].cflag
            temp = GTensor(dual=(0, 1), shape=new_shape, blocks=new_blocks, cflag=cf)
            for c in self._coords:
                for i in range(2):
                    self._link_tensors[c][i] = temp

        elif 'dominance' == mode:
            # average by dominance parts
            sds, sms = [], []
            # mark the dominance in which sector
            n = 0
            flags = [True]*8
            for c in self._coords:
                for i in range(2):
                    se = self._link_tensors[c][i].blocks()[(0, 0)].diag()
                    so = self._link_tensors[c][i].blocks()[(1, 1)].diag()
                    print(se, so)
                    # dominance in even
                    if se[0].item() > (1.0-1E-10):
                        sds.append(se)
                        sms.append(so)
                        flags[n] = True
                    # dominance in odd
                    elif so[0].item() > (1.0-1E-10):
                        sds.append(so)
                        sms.append(se)
                        flags[n] = False
                    n += 1

            sds_ave, sms_ave = sum(sds)/len(sds), sum(sms)/len(sms)
            print('Dominance average:', sds_ave, sms_ave)
            # flags = [True, True, True, True, True, True, True, True]
            # flags = [True, False, True, False, True, False, True, False]
            print(flags)
            n = 0
            cf = self._link_tensors[(0, 0)][0].cflag
            for c in self._coords:
                for i in range(2):
                    self._link_tensors[c][i] = _build_bond_matrix(sds_ave, sms_ave, flags[n], cflag=cf)
                    n += 1

        else:
            raise ValueError('mode not matched!')

        return 1

    def two_three_cluster_update(self, time_evo2: GTensor, time_evo3: GTensor, sort_weights=False, average_weights=None, expand=None):
        r'''
        simple update
        average on 4 loops

        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        sort_weights: bool, 
            True: combine even and odd singular values together and sort then truncate
            False: truncation even and odd singular values seperately
        average_weights: string,
            'dominance', average by the dominance sector
            'parity', average by parity sectors
        expand: tuple[int], optional
            expand to larger D
        '''

        if time_evo2 is not None:
            te2_mpo = self.twobody_mpo_factorize(time_evo2)

        if time_evo3 is not None:
            te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            if time_evo2 is not None:
                # X-direction
                gts = [self._site_tensors[c], self._site_tensors[cx]]
                envs = [self.mixed_site_envs(c, ex_bonds=(0, 1, 3)), self.mixed_site_envs(cx, ex_bonds=(1, 2, 3))]
                # inverse of envs, for removing the envs later
                envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
                # absorb envs into GTensors
                gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[2])
                else:
                    cf = gts[0].shape[2][0], gts[0].shape[2][1]
                    if isinstance(expand, tuple):
                        cf = expand
                # time evo operation
                gts[0] = tp.gcontract('ECe,abcde->abCcdE', te2_mpo[0], gts[0])
                gts[1] = tp.gcontract('AEe,abcde->AabcdE', te2_mpo[1], gts[1])

                wf = tp.gcontract('abCcde,Ccfghi->abdfghei', gts[0], gts[1])

                # QR and LQ factorizations
                q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                q, l = tp.linalg.super_gtqr(gts[1], group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                rl = tp.gcontract('abc,bcd->ad', r, l)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                # only left-conjugation of U and right-conjugation of V are valid under truncation
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

                s_inv = tp.linalg.ginv(s)

                # check identity
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                # u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                # v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                # s_inv = tp.linalg.ginv(s)

                # build apply projectors
                pr = tp.gcontract('Aab,bc,cd->Aad', l, v_dagger, s_inv)
                pl = tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, r)

                # wf = tp.gcontract('abCcde,Ccfghi->abdfghei', gts[0], gts[1])
                # test_wf = tp.gcontract('abCcde,CcJj,Jjfghi->abdfghei', gts[0], idt, gts[1])
                # for key, val in wf.blocks().items():
                #     print(key)
                #     print(key, (val-test_wf.blocks()[key]).norm())

                gts[0] = tp.gcontract('abCcde,Ccf->abfde', gts[0], pr)
                gts[1] = tp.gcontract('fAa,Aabcde->fbcde', pl, gts[1])
                # place identity on the connected bonds
                envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=s.shape)
                envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=s.shape)

                test_wf = tp.gcontract('abcde,cj,jfghi->abdfghei', gts[0], s, gts[1])
                print('test 2-body WF:')
                res = []
                for key, val in wf.blocks().items():
                    print(key, (val-test_wf.blocks()[key]).norm())
                    res.append((val-test_wf.blocks()[key]).norm())
                print(sum(res))

                # remove envs
                gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]

                # check and update
                # assert self._site_tensors[c].dual == gts[0].dual
                # assert self._site_tensors[cx].dual == gts[1].dual
                # assert self._link_tensors[c][0].dual == s.dual

                self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
                self._site_tensors[cx] = (1.0/gts[1].max())*gts[1]
                self._link_tensors[c][0] = (1.0/s.max())*s

                # Y-direction
                gts = [self._site_tensors[c], self._site_tensors[cy]]
                envs = [self.mixed_site_envs(c, ex_bonds=(0, 2, 3)), self.mixed_site_envs(cy, ex_bonds=(0, 1, 2))]
                envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
                # absorb envs into GTensors
                gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[1])
                else:
                    cf = gts[0].shape[1][0], gts[0].shape[1][1]
                    if isinstance(expand, tuple):
                        cf = expand
                # time evo
                gts[0] = tp.gcontract('EBe,abcde->aBbcdE', te2_mpo[0], gts[0])
                gts[1] = tp.gcontract('DEe,abcde->abcDdE', te2_mpo[1], gts[1])

                q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                q, l = tp.linalg.super_gtqr(gts[1], group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                rl = tp.gcontract('abc,bcd->ad', r, l)
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

                s_inv = tp.linalg.ginv(s)
     
                # apply projectors
                pr = tp.gcontract('Aab,bc,cd->Aad', l, v_dagger, s_inv)
                pl = tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, r)
                gts[0] = tp.gcontract('aBbcde,Bbf->afcde', gts[0], pr)
                gts[1] = tp.gcontract('fDd,abcDde->abcfe', pl, gts[1])
                # place identity on the connected bonds
                envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=s.shape)
                envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=s.shape)
                # remove envs
                gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]
                # check and update
                # assert self._site_tensors[c].dual == gts[0].dual
                # assert self._site_tensors[cy].dual == gts[1].dual
                # assert self._link_tensors[c][1].dual == s.dual

                self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
                self._site_tensors[cy] = (1.0/gts[1].max())*gts[1]
                self._link_tensors[c][1] = (1.0/s.max())*s

                self.average_plquette_weights(c, mode=average_weights, info='2 body')

            # three-body gate
            if time_evo3 is not None:
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))
                # starting from A
                # ABD
                #      *3
                #      |
                #      v
                #      |
                # 0*-<-*1,C,AB,D
                cluster = [c, cx, cxy]
                external_bonds = (0, 1, 3), (2, 3), (0, 1, 2)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                # time evo operation
                mgts[0] = tp.gcontract('ECe,abcde->abCcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('AEBe,abcde->AaBbcdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('DEe,abcde->abcDdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('abCcde,CcFfghi,jklFfm->abdghjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                # QR and LQ
                temp = mgts[0]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[0] = r
                temp = tp.gcontract('fAa,AaBbcde->fBbcde', r, mgts[1])
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[1] = r

                temp = mgts[2]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[1] = l
                temp = tp.gcontract('AaBbcde,Bbf->Aafcde', mgts[1], l)
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[0] = l

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[2])
                else:
                    cf = gts[0].shape[2][0], gts[0].shape[2][1]

                ss = []
                # R: bond left; L: bond right
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('abCcde,Ccf->abfde', mgts[0], prs[0])
                mgts[1] = tp.gcontract('fAa,AaBbcde,Bbg->fgcde', pls[0], mgts[1], prs[1])
                mgts[2] = tp.gcontract('fDd,abcDde->abcfe', pls[1], mgts[2])

                envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # test_wf = tp.gcontract('abcde,cn,nfghi,fo,jklom->abdghjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # # if cut_off is not set
                # print('check ABD WF:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))


                # update 
                self._link_tensors[cluster[0]][0] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[1]][1] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                # ACD
                # 2*-<-*3
                #  |
                #  v
                #  |
                #  *0,B,DC,A
                cluster = [c, cy, cxy]
                external_bonds = (0, 2, 3), (0, 1), (1, 2, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                mgts[0] = tp.gcontract('EBe,abcde->aBbcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('DECe,abcde->abCcDdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('AEe,abcde->AabcdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('aBbcde,fgHhBbi,Hhjklm->acdfgjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[0] = r
                temp = tp.gcontract('fDd,abCcDde->abCcfe', r, mgts[1])
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[1] = r

                temp = mgts[2]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[1] = l
                temp = tp.gcontract('abCcDde,Ccf->abfDde', mgts[1], l)
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[0] = l

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[2])
                else:
                    cf = gts[0].shape[2][0], gts[0].shape[2][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('aBbcde,Bbf->afcde', mgts[0], prs[0])
                mgts[1] = tp.gcontract('fDd,abCcDde,Ccg->abgfe', pls[0], mgts[1], prs[1])
                mgts[2] = tp.gcontract('fAa,Aabcde->fbcde', pls[1], mgts[2])

                # if cut_off is not set
                # test_wf = tp.gcontract('abcde,bB,fghBi,hH,Hjklm->acdfgjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check ACD wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[0]][1] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[1]][0] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                self.average_plquette_weights(c, mode='dominance', info='ABD,ACD')

                # starting from B
                # BAC
                #  2
                #  *
                #  |
                #  v
                #  |
                # 0*--<--*1, A,CB,D
                cluster = [cx, c, cy]
                external_bonds = (1, 2, 3), (0, 3), (0, 1, 2)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                mgts[0] = tp.gcontract('EAe,abcde->AabcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('CEBe,abcde->aBbCcdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('DEe,abcde->abcDdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('Aabcde,fGgAahi,jklGgm->bcdfhjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[0] = l
                temp = tp.gcontract('aBbCcde,Ccf->aBbfde', mgts[1], l)
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[1] = r
                    
                temp = mgts[2]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[1] = l
                temp = tp.gcontract('aBbCcde,Bbf->afCcde', mgts[1], l)
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[0] = r

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[0])
                else:
                    cf = gts[0].shape[0][0], gts[0].shape[0][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('fAa,Aabcde->fbcde', pls[0], mgts[0])
                mgts[1] = tp.gcontract('aBbCcde,Ccf,Bbg->agfde', mgts[1], prs[0], prs[1])
                mgts[2] = tp.gcontract('fDd,abcDde->abcfe', pls[1], mgts[2])

                # test_wf = tp.gcontract('Abcde,aA,fgahi,gG,jklGm->bcdfhjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check BAC wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[1]][0] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[1]][1] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    # print(bare_gts[i].dual, bare_gts[i].shape)
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                # BDC
                # 2*--<--*3
                #        |
                #        v
                #        |
                #        *1,B,DA,C
                cluster = [cx, cxy, cy]
                external_bonds = (0, 2, 3), (1, 2), (0, 1, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                mgts[0] = tp.gcontract('EBe,abcde->aBbcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('DEAe,abcde->AabcDdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('CEe,abcde->abCcdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('aBbcde,FfghBbi,jkFflm->acdghjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[0] = r
                temp = tp.gcontract('fDd,AabcDde->Aabcfe', r, mgts[1])
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[1] = l

                temp = mgts[2]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs [1] = r
                temp = tp.gcontract('fAa,AabcDde->fbcDde', r, mgts[1])
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[0] = l

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[1])
                else:
                    cf = gts[0].shape[1][0], gts[0].shape[1][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('aBbcde,Bbf->afcde', mgts[0], prs[0])
                mgts[1] = tp.gcontract('gAa,fDd,AabcDde->gbcfe', pls[1], pls[0], mgts[1])
                mgts[2] = tp.gcontract('abCcde,Ccf->abfde', mgts[2], prs[1])

                # test_wf = tp.gcontract('abcde,bB,FghBi,fF,jkflm->acdghjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check BDC wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[0]][1] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[2]][0] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    # print(bare_gts[i].dual, bare_gts[i].shape)
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                self.average_plquette_weights(c, mode='dominance', info='BAC,BDC')

                # starting from D
                # DBA
                #      *3,D,BA,C
                #      |
                #      v
                #      |
                # 0*-<-*1
                cluster = [cxy, cx, c]
                external_bonds = (0, 1, 2), (2, 3), (0, 1, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                mgts[0] = tp.gcontract('EDe,abcde->abcDdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('BEAe,abcde->AaBbcdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('CEe,abcde->abCcdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('abcDde,FfDdghi,jkFflm->abcghjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[0] = l
                temp = tp.gcontract('AaBbcde,Bbf->Aafcde', mgts[1], l)
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[1] = l
                temp = mgts[2]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs [1] = r
                temp = tp.gcontract('fAa,AaBbcde->fBbcde', r, mgts[1])
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[0] = r

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[3])
                else:
                    cf = gts[0].shape[3][0], gts[0].shape[3][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('fDd,abcDde->abcfe', pls[0], mgts[0])
                mgts[1] = tp.gcontract('gAa,AaBbcde,Bbf->gfcde', pls[1], mgts[1], prs[0])
                mgts[2] = tp.gcontract('abCcde,Ccf->abfde', mgts[2], prs[1])

                # test_wf = tp.gcontract('abcDe,dD,Fdghi,fF,jkflm->abcghjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check DBA wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))


                envs_inv[0][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[1]][1] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[2]][0] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                # DCA
                # 2*-<-*3,A,CD,B
                #  |
                #  v
                #  |
                #  *0
                cluster = [cxy, cy, c]
                external_bonds = (1, 2, 3), (0, 1), (0, 2, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                mgts[0] = tp.gcontract('EAe,abcde->AabcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('CEDe,abcde->abCcDdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('BEe,abcde->aBbcdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('Aabcde,fgAaHhi,jHhklm->bcdfgjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[0] = l
                temp = tp.gcontract('abCcDde,Ccf->abfDde', mgts[1], l)
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[1] = l
                temp = mgts[2]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[1] = r
                temp = tp.gcontract('fDd,abCcDde->abCcfe', r, mgts[1])
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[0] = r

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[0])
                else:
                    cf = gts[0].shape[0][0], gts[0].shape[0][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                mgts[0] = tp.gcontract('fAa,Aabcde->fbcde', pls[0], mgts[0])
                mgts[1] = tp.gcontract('gDd,abCcDde,Ccf->abfge',  pls[1], mgts[1], prs[0])
                mgts[2] = tp.gcontract('aBbcde,Bbf->afcde', mgts[2], prs[1])

                # test_wf = tp.gcontract('Abcde,aA,fgaHi,hH,jhklm->bcdfgjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check DCA wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[1]][0] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[2]][1] = (1.0/ss[1].max())*ss[1]

                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                self.average_plquette_weights(c, mode='dominance', info='DBA,DCA')

                # starting from C
                # CAB
                #  2,D,BC,A
                #  *
                #  |
                #  v
                #  |
                # 0*--<--*1
                cluster = [cy, c, cx]
                external_bonds = (0, 1, 2), (0, 3), (1, 2, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                # time evo operation
                mgts[0] = tp.gcontract('EDe,abcde->abcDdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('BECe,abcde->aBbCcdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('AEe,abcde->AabcdE', te3_mpo[2], mgts[2])

                # wf = tp.gcontract('abcDde,fDdGghi,Ggjklm->abcfhjkleim', mgts[0], mgts[1], mgts[2])

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[0] = l
                temp = tp.gcontract('aBbCcde,Bbf->afCcde', mgts[1], l)
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[1] = r
                temp = mgts[2]
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[1] = l
                temp = tp.gcontract('aBbCcde,Ccf->aBbfde', mgts[1], l)
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[0] = r

               # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[3])
                else:
                    cf = gts[0].shape[3][0], gts[0].shape[3][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                    pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

                    # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
 
                mgts[0] = tp.gcontract('fDd,abcDde->abcfe', pls[0], mgts[0])
                mgts[1] = tp.gcontract('aBbCcde,Bbf,Ccg->afgde', mgts[1], prs[0], prs[1])
                mgts[2] = tp.gcontract('fAa,Aabcde->fbcde', pls[1], mgts[2])

               # test_wf = tp.gcontract('abcDe,dD,fdghi,gG,Gjklm->abcfhjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check CAB wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[1]][1] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[1]][0] = (1.0/ss[1].max())*ss[1]

                # remove envs
                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                # CDB
                # 2*--<--*3, C,AD,B
                #        |
                #        v
                #        |
                #        *1
                cluster = [cy, cxy, cx]
                external_bonds = (0, 1, 3), (1, 2), (0, 2, 3)
                # te3_mpo = self.threebody_mpo_factorize(time_evo3, internal_flags=(1, 1))

                gts = [self._site_tensors[site] for site in cluster]
                envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
                envs_inv = []
                for i in range(3):
                    envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

                mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
                # time evo operations
                mgts[0] = tp.gcontract('ECe,abcde->abCcdE', te3_mpo[0], mgts[0])
                mgts[1] = tp.gcontract('AEDe,abcde->AabcDdE', te3_mpo[1], mgts[1])
                mgts[2] = tp.gcontract('BEe,abcde->aBbcdE', te3_mpo[2], mgts[2])

                # print(mgts[0].shape, mgts[1].shape, mgts[2].shape)
                # wf = tp.gcontract('abCcde,CcfgHhi,jHhklm->abdfgjkleim', mgts[0], mgts[1], mgts[2])
                # print(wf.shape)

                rs, ls = [None]*2, [None]*2
                temp = mgts[0]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                rs[0] = r
                temp = tp.gcontract('fAa,AabcDde->fbcDde', r, mgts[1])
                q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
                ls[1] = l
                temp = mgts[2]
                q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
                rs[1] = r
                temp = tp.gcontract('fDd,AabcDde->Aabcfe', r, mgts[1])
                q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
                ls[0] = l

                # set two kinds of cut-off
                if sort_weights:
                    cf = sum(gts[0].shape[2])
                else:
                    cf = gts[0].shape[2][0], gts[0].shape[2][1]

                ss = []
                prs, pls = [], []
                for i in range(2):
                    rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                    u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                    # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                    ss.append(s)
                    u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                    v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                    s_inv = tp.linalg.ginv(s)
                    pr = tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv)
                    pl = tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i])
                    prs.append(pr)
                    pls.append(pl)

                    # # check isometry of U and V
                    # iso = tp.gcontract('ab,bc->ac', u_dagger, u)
                    # print('Iso U:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())
                    # iso = tp.gcontract('ab,bc->ac', v, v_dagger)
                    # print('Iso V:', iso.dual)
                    # for q, t in iso.blocks().items():
                    #     print(q, t.diag())

                # q, r = tp.linalg.gtqr(mgts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
                # test = tp.gcontract('abcde,cFf->abFfde', q, r)
                # res = []
                # for key, val in mgts[0].blocks().items():
                #     print(key, (val-test.blocks()[key]).norm())
                #     res.append((val-test.blocks()[key]).norm())
                # print('mgts[0]', sum(res))
                # # q, l = tp.linalg.super_gtqr(mgts[1], group_dims=((2, 3, 4, 5, 6), (0, 1)), qr_dims=(0, 2))
                # # test = tp.gcontract('Ffa,abcDde->FfbcDde', l, q)
                # q, l = tp.linalg.super_gtqr(mgts[1], group_dims=((0, 1, 2, 3, 6), (4, 5)), qr_dims=(4, 2))
                # test = tp.gcontract('Ffd,Aabcde->AabcFfe', l, q)
                # res = []
                # for key, val in mgts[1].blocks().items():
                #     print(key, (val-test.blocks()[key]).norm())
                #     res.append((val-test.blocks()[key]).norm())
                # print('mgts[1]', sum(res))

                # rl = tp.gcontract('abc,bcd->ad', r, l)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                # u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                # v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                # s_inv = tp.linalg.ginv(s)
                # pr = tp.gcontract('Aab,bc,cd->Aad', l, v_dagger, s_inv)
                # pl = tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, r)
                # # id_test = tp.gcontract('Aab,bc,cDd->AaDd', pr, s, pl)
                # id_test = tp.gcontract('Aab,bc,cd,de,eFf->AaFf', l, v_dagger, s_inv, u_dagger, r)

                # idt = tp.gcontract('Aab,bc,cDd->AaDd', prs[0], ss[0], pls[0])
                # test_wf = tp.gcontract('abCcde,CcNn,NnfgHhi,jHhklm->abdfgjkleim', mgts[0], idt, mgts[1], mgts[2])
                # test_wf = tp.gcontract('abCcde,CcfgHhi->abdfgHhei', mgts[0], mgts[1])
                # test_wf = tp.gcontract('abCcde,CcJj,JjfgHhi->abdfgHhei', mgts[0], id_test, mgts[1])

                mgts[0] = tp.gcontract('abCcde,Ccf->abfde', mgts[0], prs[0])
                mgts[1] = tp.gcontract('gAa,fDd,AabcDde->gbcfe', pls[0], pls[1], mgts[1])
                mgts[2] = tp.gcontract('aBbcde,Bbf->afcde', mgts[2], prs[1])

                # idt = tp.gcontract('Aab,bc,cDd->AaDd', prs[1], ss[1], pls[1])
                # test_wf = tp.gcontract('abCcde,CcfgNni,HhNn,jHhklm->abdfgjkleim', mgts[0], mgts[1], idt, mgts[2])
                # test_wf = tp.gcontract('abCcde,CcfgHhi,jHhklm->abdfgjkleim', *mgts)
                # test_wf = tp.gcontract('abcde,cC,CfgHi,hH,jhklm->abdfgjkleim', mgts[0], ss[0], mgts[1], ss[1], mgts[2])
                # print('check CDB wf:')
                # res = []
                # for key, val in wf.blocks().items():
                #     print(key, (val-test_wf.blocks()[key]).norm())
                #     res.append((val-test_wf.blocks()[key]).norm())
                # print(sum(res))

                envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
                envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
                envs_inv[2][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

                # update 
                self._link_tensors[cluster[0]][0] = (1.0/ss[0].max())*ss[0]
                self._link_tensors[cluster[2]][1] = (1.0/ss[1].max())*ss[1]

                # remove envs
                bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
                for i, site in enumerate(cluster):
                    assert self._site_tensors[site].dual == bare_gts[i].dual
                    assert self._site_tensors[site].shape == bare_gts[i].shape
                    self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

                self.average_plquette_weights(c, mode='dominance', info='CAB,CDB')

        return 1

    def twobody_simple_update_2(self, time_evo: GTensor, sort_weights=False, average_weights=None, expand=None):
        r'''
        simple update
        average on 4 loops

        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        sort_weights: bool, 
            True: combine even and odd singular values together and sort then truncate
            False: truncation even and odd singular values seperately
        average_weights: string,
            'dominance', average by the dominance sector
            'parity', average by parity sectors
        expand: tuple[int], optional
            expand to larger D
        '''

        te_mpo = self.twobody_mpo_factorize(time_evo)

        # factorize to MPO
        # u, s, v = tp.linalg.gtsvd(time_evo, group_dims=((0, 2), (1, 3)), svd_dims=(1, 0))
        # se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
        # ss = GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
        # te_mpo = tp.gcontract('abc,bd->adc', u, ss), tp.gcontract('ab,bcd->acd', ss, v)

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            gts = [self._site_tensors[c], self._site_tensors[cx]]
            envs = [self.mixed_site_envs(c, ex_bonds=(0, 1, 3)), self.mixed_site_envs(cx, ex_bonds=(1, 2, 3))]
            # inverse of envs, for removing the envs later
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[2])
            else:
                cf = gts[0].shape[2][0], gts[0].shape[2][1]
                if isinstance(expand, tuple):
                    cf = expand
            # time evo operation
            gts[0] = tp.gcontract('ECe,abcde->abCcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('AEe,abcde->AabcdE', te_mpo[1], gts[1])
            # QR and LQ factorizations
            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # only left-conjugation of U and right-conjugation of V are valid under truncation
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            '''

            # check identity
            # print('check identity:')
            # print(l.shape, v_dagger.shape, s_inv.shape, u_dagger.shape, r.shape)
            # idt = tp.gcontract('abc,cd,de,ef,fg,gh,hij->abij', l, v_dagger, s_inv, s, s_inv, u_dagger, r)
            # for key, val in idt.blocks().items():
                # print(key)
                # print(val)
 
            # apply projectors
            gts[0] = tp.gcontract('abCcde,Ccf,fg,gh->abhde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fAa,Aabcde->hbcde', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cx].dual == gts[1].dual
            # assert self._link_tensors[c][0].dual == s.dual

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cx] = (1.0/gts[1].max())*gts[1]
            self._link_tensors[c][0] = (1.0/s.max())*s
            # print('X', self._link_tensors[c][0].blocks()[(0, 0)].diag(), self._link_tensors[c][0].blocks()[(1, 1)].diag())

            # Y-direction
            gts = [self._site_tensors[c], self._site_tensors[cy]]
            envs = [self.mixed_site_envs(c, ex_bonds=(0, 2, 3)), self.mixed_site_envs(cy, ex_bonds=(0, 1, 2))]
            envs_inv = [[tp.linalg.ginv(envs[0][j]) for j in range(4)], [tp.linalg.gpinv(envs[1][j]) for j in range(4)]]
            # absorb envs into GTensors
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(2)]
            # set two kinds of cut-off
            if sort_weights:
                cf = sum(gts[0].shape[1])
            else:
                cf = gts[0].shape[1][0], gts[0].shape[1][1]
                if isinstance(expand, tuple):
                    cf = expand
            # time evo
            gts[0] = tp.gcontract('EBe,abcde->aBbcdE', te_mpo[0], gts[0])
            gts[1] = tp.gcontract('DEe,abcde->abcDdE', te_mpo[1], gts[1])

            q, r = tp.linalg.gtqr(gts[0], group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            q, l = tp.linalg.super_gtqr(gts[1], group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            rl = tp.gcontract('abc,bcd->ad', r, l)
            u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
            # u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            # v_dagger = v.graded_conj(iso_dims=(0,), side=1)
            u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
            v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)

            s_inv = tp.linalg.ginv(s)

            '''
            # check isometry of U and V
            iso = tp.gcontract('ab,bc->ac', u_dagger, u)
            print('Iso U:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            iso = tp.gcontract('ab,bc->ac', v, v_dagger)
            print('Iso V:', iso.dual)
            for q, t in iso.blocks().items():
                print(q, t.diag())
            '''
 
            # apply projectors            
            gts[0] = tp.gcontract('aBbcde,Bbf,fg,gh->ahcde', gts[0], l, v_dagger, s_inv)
            gts[1] = tp.gcontract('hg,gf,fDd,abcDde->abche', s_inv, u_dagger, r, gts[1])
            # place identity on the connected bonds
            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=s.shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=s.shape)
            # remove envs
            gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs_inv[i]) for i in range(2)]
            # check and update
            # assert self._site_tensors[c].dual == gts[0].dual
            # assert self._site_tensors[cy].dual == gts[1].dual
            # assert self._link_tensors[c][1].dual == s.dual

            self._site_tensors[c] = (1.0/gts[0].max())*gts[0]
            self._site_tensors[cy] = (1.0/gts[1].max())*gts[1]
            self._link_tensors[c][1] = (1.0/s.max())*s
            # print('Y', self._link_tensors[c][1].blocks()[(0, 0)].diag(), self._link_tensors[c][1].blocks()[(1, 1)].diag())

            if expand is None and sort_weights is False:
                if 'parity' == average_weights:
                    # direct averge two sectors, naively
                    lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                    ses, sos = [], []
                    for t in lams:
                        ses.append(t.blocks()[(0, 0)].diag())
                        sos.append(t.blocks()[(1, 1)].diag())
                    se, so = 0.25*sum(ses), 0.25*sum(sos)
                    print('average parity:', se, so)
                    new_blocks = {(0, 0):torch.tensor(se).diag(), (1, 1):torch.tensor(so).diag()}
                    new_lam = GTensor(dual=(0, 1), shape=lams[0].shape, blocks=new_blocks, cflag=lams[0].cflag)
                    self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1] = tuple([new_lam]*4)

                if 'dominance' == average_weights:
                    # average by dominance part
                    lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                    sds, sms = [], []
                    flags = []
                    for t in lams:
                        se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                        print(se, so)
                        if se[0].item() > (1.0-1E-12):
                            sds.append(se)
                            sms.append(so)
                            flags.append(True)
                        elif so[0].item() > (1.0-1E-12):
                            sds.append(so)
                            sms.append(se)
                            flags.append(False)

                    # flags = [True, False, True, False]
                    # flags = [False, True, False, True]
                    # print(flags)
                    print('average dominance:', 0.25*sum(sds), 0.25*sum(sms))
                    if flags[0]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[c][0] = GTensor(dual=(0, 1), shape=lams[0].shape, blocks=new_blocks, cflag=lams[0].cflag)
                    if flags[1]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[cx][1] = GTensor(dual=(0, 1), shape=lams[1].shape, blocks=new_blocks, cflag=lams[1].cflag)
                    if flags[2]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[cy][0] = GTensor(dual=(0, 1), shape=lams[2].shape, blocks=new_blocks, cflag=lams[2].cflag)
                    if flags[3]:
                        se, so = 0.25*sum(sds), 0.25*sum(sms)
                    else:
                        se, so = 0.25*sum(sms), 0.25*sum(sds)
                    new_blocks = {(0, 0):se.diag(), (1, 1):so.diag()}
                    self._link_tensors[c][1] = GTensor(dual=(0, 1), shape=lams[3].shape, blocks=new_blocks, cflag=lams[3].cflag)

            elif sort_weights is True:
                lams = self._link_tensors[c][0], self._link_tensors[cx][1], self._link_tensors[cy][0], self._link_tensors[c][1]
                sds, sms = [], []
                for t in lams:
                    se, so = t.blocks()[(0, 0)].diag(), t.blocks()[(1, 1)].diag()
                    print(se, so)
                '''
                    if se[0].item() > (1.0-1E-12):
                        sds.append(se)
                        sms.append(so)
                        flags.append(True)
                    elif so[0].item() > (1.0-1E-12):
                        sds.append(so)
                        sms.append(se)
                        flags.append(False)

                # flags = [True, False, True, False]
                # flags = [False, True, False, True]
                # print(flags)
                print('average dominance:', 0.25*sum(sds), 0.25*sum(sms))
                '''

        return 1

    def threebody_cluster_update(self, time_evo_mpo: list, expand=None, fixed_dims=False, average_weights=None, ifprint=False):
        r'''
        Parameters
        ----------
        time_evo: GTensor, time evolution operator
        sort_weights: bool, 
            True: combine even and odd singular values together and sort then truncate
            False: truncation even and odd singular values seperately
        average_method:
            'plaquette', average around a plaquette
            'all', average all bonds
        average_weights: string,
        expand: tuple[int], optional
            expand to larger D
        '''

        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # three-body
            # starting from A
            # ABD
            #      *3
            #      |
            #      v
            #      |
            # 0*-<-*1,C,AB,D
            cluster = [c, cx, cxy]
            external_bonds = (0, 1, 3), (2, 3), (0, 1, 2)

            gts = [self._site_tensors[site] for site in cluster]

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[2]
                else:
                    cf = sum(gts[0].shape[2])
            else:
                cf = expand

            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            # time evo operation
            mgts[0] = tp.gcontract('ECe,abcde->abCcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('AEBe,abcde->AaBbcdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('DEe,abcde->abcDdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            # QR and LQ
            temp = mgts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[0] = r
            temp = tp.gcontract('fAa,AaBbcde->fBbcde', r, mgts[1])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[1] = r

            temp = mgts[2]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[1] = l
            temp = tp.gcontract('AaBbcde,Bbf->Aafcde', mgts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[0] = l

            ss = []
            # R: bond left; L: bond right
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('abCcde,Ccf->abfde', mgts[0], prs[0])
            mgts[1] = tp.gcontract('fAa,AaBbcde,Bbg->fgcde', pls[0], mgts[1], prs[1])
            mgts[2] = tp.gcontract('fDd,abcDde->abcfe', pls[1], mgts[2])

            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            new_lts = (1.0/ss[0].max())*ss[0], (1.0/ss[1].max())*ss[1]

            if ifprint and None == expand:
                diff_0, diff_1 = 0.0, 0.0
                for key, val in new_lts[0].blocks().items():
                    diff_0 += (self._link_tensors[cluster[0]][0].blocks()[key]-val).norm()
                for key, val in new_lts[1].blocks().items():
                    diff_1 += (self._link_tensors[cluster[1]][1].blocks()[key]-val).norm()
                print(c, 'ABD Lambda changing:', diff_0.item(), diff_1.item())

            # update 
            self._link_tensors[cluster[0]][0] = new_lts[0] 
            self._link_tensors[cluster[1]][1] = new_lts[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            # ACD
            # 2*-<-*3
            #  |
            #  v
            #  |
            #  *0,B,DC,A
            cluster = [c, cy, cxy]
            external_bonds = (0, 2, 3), (0, 1), (1, 2, 3)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[1]
                else:
                    cf = sum(gts[0].shape[1])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            mgts[0] = tp.gcontract('EBe,abcde->aBbcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('DECe,abcde->abCcDdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('AEe,abcde->AabcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[0] = r
            temp = tp.gcontract('fDd,abCcDde->abCcfe', r, mgts[1])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[1] = r

            temp = mgts[2]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[1] = l
            temp = tp.gcontract('abCcDde,Ccf->abfDde', mgts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[0] = l

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('aBbcde,Bbf->afcde', mgts[0], prs[0])
            mgts[1] = tp.gcontract('fDd,abCcDde,Ccg->abgfe', pls[0], mgts[1], prs[1])
            mgts[2] = tp.gcontract('fAa,Aabcde->fbcde', pls[1], mgts[2])

            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[0]][1] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[1]][0] = (1.0/ss[1].max())*ss[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            '''
            # average the bond weights in this plaquette
            if average_weights is not None:
                if 'plaquette' == average_method:
                    self.average_plquette_weights(c, mode=average_weights, info='ABD,ACD')
                elif 'all' == average_method:
                    self.average_all_weights(mode=average_weights)
                else:
                    raise ValueError('your average method is not valid')
            '''

            # starting from B
            # BAC
            #  2
            #  *
            #  |
            #  v
            #  |
            # 0*--<--*1, A,CB,D
            cluster = [cx, c, cy]
            external_bonds = (1, 2, 3), (0, 3), (0, 1, 2)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[3]
                else:
                    cf = sum(gts[0].shape[3])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            mgts[0] = tp.gcontract('EAe,abcde->AabcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('CEBe,abcde->aBbCcdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('DEe,abcde->abcDdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[0] = l
            temp = tp.gcontract('aBbCcde,Ccf->aBbfde', mgts[1], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[1] = r
                
            temp = mgts[2]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[1] = l
            temp = tp.gcontract('aBbCcde,Bbf->afCcde', mgts[1], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[0] = r

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('fAa,Aabcde->fbcde', pls[0], mgts[0])
            mgts[1] = tp.gcontract('aBbCcde,Ccf,Bbg->agfde', mgts[1], prs[0], prs[1])
            mgts[2] = tp.gcontract('fDd,abcDde->abcfe', pls[1], mgts[2])

            envs_inv[0][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[1]][0] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[1]][1] = (1.0/ss[1].max())*ss[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            # BDC
            # 2*--<--*3
            #        |
            #        v
            #        |
            #        *1,B,DA,C
            cluster = [cx, cxy, cy]
            external_bonds = (0, 2, 3), (1, 2), (0, 1, 3)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[2]
                else:
                    cf = sum(gts[0].shape[2])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            mgts[0] = tp.gcontract('EBe,abcde->aBbcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('DEAe,abcde->AabcDdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('CEe,abcde->abCcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[0] = r
            temp = tp.gcontract('fDd,AabcDde->Aabcfe', r, mgts[1])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[1] = l

            temp = mgts[2]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs [1] = r
            temp = tp.gcontract('fAa,AabcDde->fbcDde', r, mgts[1])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[0] = l

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('aBbcde,Bbf->afcde', mgts[0], prs[0])
            mgts[1] = tp.gcontract('gAa,fDd,AabcDde->gbcfe', pls[1], pls[0], mgts[1])
            mgts[2] = tp.gcontract('abCcde,Ccf->abfde', mgts[2], prs[1])

            envs_inv[0][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[0]][1] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[2]][0] = (1.0/ss[1].max())*ss[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # print(bare_gts[i].dual, bare_gts[i].shape)
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            '''
            if average_weights is not None:
                if 'plaquette' == average_method:
                    self.average_plquette_weights(c, mode=average_weights, info='BAC,BDC')
                elif 'all' == average_method:
                    self.average_all_weights(mode=average_weights)
                else:
                    raise ValueError('your average method is not valid')
            '''

            # starting from D
            # DBA
            #      *3,D,BA,C
            #      |
            #      v
            #      |
            # 0*-<-*1
            cluster = [cxy, cx, c]
            external_bonds = (0, 1, 2), (2, 3), (0, 1, 3)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[3]
                else:
                    cf = sum(gts[0].shape[3])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            mgts[0] = tp.gcontract('EDe,abcde->abcDdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('BEAe,abcde->AaBbcdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('CEe,abcde->abCcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[0] = l
            temp = tp.gcontract('AaBbcde,Bbf->Aafcde', mgts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[1] = l
            temp = mgts[2]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs [1] = r
            temp = tp.gcontract('fAa,AaBbcde->fBbcde', r, mgts[1])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[0] = r

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('fDd,abcDde->abcfe', pls[0], mgts[0])
            mgts[1] = tp.gcontract('gAa,AaBbcde,Bbf->gfcde', pls[1], mgts[1], prs[0])
            mgts[2] = tp.gcontract('abCcde,Ccf->abfde', mgts[2], prs[1])

            envs_inv[0][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[1]][1] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[2]][0] = (1.0/ss[1].max())*ss[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            # DCA
            # 2*-<-*3,A,CD,B
            #  |
            #  v
            #  |
            #  *0
            cluster = [cxy, cy, c]
            external_bonds = (1, 2, 3), (0, 1), (0, 2, 3)

            # set cut-off
            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[0]
                else:
                    cf = sum(gts[0].shape[0])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            mgts[0] = tp.gcontract('EAe,abcde->AabcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('CEDe,abcde->abCcDdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('BEe,abcde->aBbcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[0] = l
            temp = tp.gcontract('abCcDde,Ccf->abfDde', mgts[1], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[1] = l
            temp = mgts[2]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[1] = r
            temp = tp.gcontract('fDd,abCcDde->abCcfe', r, mgts[1])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[0] = r

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('fAa,Aabcde->fbcde', pls[0], mgts[0])
            mgts[1] = tp.gcontract('gDd,abCcDde,Ccf->abfge',  pls[1], mgts[1], prs[0])
            mgts[2] = tp.gcontract('aBbcde,Bbf->afcde', mgts[2], prs[1])

            envs_inv[0][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[1]][0] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[2]][1] = (1.0/ss[1].max())*ss[1]

            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            '''
            if average_weights is not None:
                if 'plaquette' == average_method:
                    self.average_plquette_weights(c, mode=average_weights, info='DBA,DCA')
                elif 'all' == average_method:
                    self.average_all_weights(mode=average_weights)
                else:
                    raise ValueError('your average method is not valid')
            '''

            # starting from C
            # CAB
            #  2,D,BC,A
            #  *
            #  |
            #  v
            #  |
            # 0*--<--*1
            cluster = [cy, c, cx]
            external_bonds = (0, 1, 2), (0, 3), (1, 2, 3)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[3]
                else:
                    cf = sum(gts[0].shape[3])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            # time evo operation
            mgts[0] = tp.gcontract('EDe,abcde->abcDdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('BECe,abcde->aBbCcdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('AEe,abcde->AabcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[0] = l
            temp = tp.gcontract('aBbCcde,Bbf->afCcde', mgts[1], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[1] = r
            temp = mgts[2]
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[1] = l
            temp = tp.gcontract('aBbCcde,Ccf->aBbfde', mgts[1], l)
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[0] = r

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                prs.append(tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv))
                pls.append(tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i]))

            mgts[0] = tp.gcontract('fDd,abcDde->abcfe', pls[0], mgts[0])
            mgts[1] = tp.gcontract('aBbCcde,Bbf,Ccg->afgde', mgts[1], prs[0], prs[1])
            mgts[2] = tp.gcontract('fAa,Aabcde->fbcde', pls[1], mgts[2])

            envs_inv[0][3] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][1] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][2] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][0] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[1]][1] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[1]][0] = (1.0/ss[1].max())*ss[1]

            # remove envs
            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            # CDB
            # 2*--<--*3, C,AD,B
            #        |
            #        v
            #        |
            #        *1
            cluster = [cy, cxy, cx]
            external_bonds = (0, 1, 3), (1, 2), (0, 2, 3)

            if expand is None:
                if fixed_dims:
                    cf = gts[0].shape[2]
                else:
                    cf = sum(gts[0].shape[2])
            else:
                cf = expand

            gts = [self._site_tensors[site] for site in cluster]
            envs = [self.mixed_site_envs(site, ex_bonds=ebs) for site, ebs in zip(cluster, external_bonds)]
            envs_inv = []
            for i in range(3):
                envs_inv.append([tp.linalg.ginv(envs[i][j]) for j in range(4)])

            mgts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', gts[i], *envs[i]) for i in range(3)]
            # time evo operations
            mgts[0] = tp.gcontract('ECe,abcde->abCcdE', time_evo_mpo[0], mgts[0])
            mgts[1] = tp.gcontract('AEDe,abcde->AabcDdE', time_evo_mpo[1], mgts[1])
            mgts[2] = tp.gcontract('BEe,abcde->aBbcdE', time_evo_mpo[2], mgts[2])

            rs, ls = [None]*2, [None]*2
            temp = mgts[0]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 1, 4, 5), (2, 3)), qr_dims=(2, 0))
            rs[0] = r
            temp = tp.gcontract('fAa,AabcDde->fbcDde', r, mgts[1])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((0, 1, 2, 5), (3, 4)), qr_dims=(3, 2))
            ls[1] = l
            temp = mgts[2]
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 3, 4, 5), (1, 2)), qr_dims=(1, 0))
            rs[1] = r
            temp = tp.gcontract('fDd,AabcDde->Aabcfe', r, mgts[1])
            q, l = tp.linalg.super_gtqr(temp, group_dims=((2, 3, 4, 5), (0, 1)), qr_dims=(0, 2))
            ls[0] = l

            ss = []
            prs, pls = [], []
            for i in range(2):
                rl = tp.gcontract('abc,bcd->ad', rs[i], ls[i])
                u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)), cut_off=cf)
                # u, s, v = tp.linalg.gtsvd(rl, group_dims=((0,), (1,)))
                ss.append(s)
                u_dagger = u.graded_conj(free_dims=(1,), side=0, reverse=True)
                v_dagger = v.graded_conj(free_dims=(0,), side=1, reverse=True)
                s_inv = tp.linalg.ginv(s)
                pr = tp.gcontract('Aab,bc,cd->Aad', ls[i], v_dagger, s_inv)
                pl = tp.gcontract('dc,cb,bAa->dAa', s_inv, u_dagger, rs[i])
                prs.append(pr)
                pls.append(pl)

            mgts[0] = tp.gcontract('abCcde,Ccf->abfde', mgts[0], prs[0])
            mgts[1] = tp.gcontract('gAa,fDd,AabcDde->gbcfe', pls[0], pls[1], mgts[1])
            mgts[2] = tp.gcontract('aBbcde,Bbf->afcde', mgts[2], prs[1])

            envs_inv[0][2] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][0] = GTensor.eye(dual=(0, 1), shape=ss[0].shape)
            envs_inv[1][3] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)
            envs_inv[2][1] = GTensor.eye(dual=(0, 1), shape=ss[1].shape)

            # update 
            self._link_tensors[cluster[0]][0] = (1.0/ss[0].max())*ss[0]
            self._link_tensors[cluster[2]][1] = (1.0/ss[1].max())*ss[1]

            # remove envs
            bare_gts = [tp.gcontract('abcde,Aa,bB,cC,Dd->ABCDe', mgts[i], *envs_inv[i]) for i in range(3)]
            for i, site in enumerate(cluster):
                # assert self._site_tensors[site].dual == bare_gts[i].dual
                # assert self._site_tensors[site].shape == bare_gts[i].shape
                self._site_tensors[site] = (1.0/bare_gts[i].max())*bare_gts[i]

            '''
            if average_weights is not None:
                if 'plaquette' == average_method:
                    self.average_plquette_weights(c, mode=average_weights, info='CAB,CDB')
                elif 'all' == average_method:
                    self.average_all_weights(mode=average_weights)
                else:
                    raise ValueError('your average method is not valid')
            '''

            # average
            if 'sort' == average_weights:
                self.sorted_average_weights(c, ifprint=ifprint)
            elif 'direct' == average_weights:
                self.direct_average_weights(c, ifprint=ifprint)

        return 1

    def simple_measurement_onebody(self, op: GTensor):
        r'''
        measure one-body operator by double tensors on the Beta lattice
        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, measured values
        '''

        mgts = self.merged_tensors()

        res = []
        for c in self._coords:

            # merge the Lambda weights as environment tensors
            pure_gt = tp.gcontract('aA,Bb,Cc,dD,ABCDE->abcdE', *self.site_envs(c), mgts[c])
            # and apply the operator
            impure_gt = tp.gcontract('aA,Bb,Cc,dD,eE,ABCDE->abcde', *self.site_envs(c), op, mgts[c])
            
            den = tp.gcontract('abcde,abcde->', mgts[c].conj(), pure_gt, bosonic_dims=('a', 'b', 'c', 'd', 'e'))
            num = tp.gcontract('abcde,abcde->', mgts[c].conj(), impure_gt, bosonic_dims=('a', 'b', 'c', 'd', 'e'))
            
            res.append(num / den)

        # return sum(res) / self._size
        return torch.tensor(res)

    def simple_measurement_twobody(self, op_0: GTensor, op_1: GTensor):
        r'''
        measure a two-body operator by double tensors on the Beta lattice

        Parameters
        ----------
        op_0: GTensor, the first operator
        op_1: GTensor, the second operator

        Returns
        -------
        res: tensor, measured values
        '''

        mgts = self.merged_tensors()

        meas = []
        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny

            # X-direction
            pure_dts, impure_dts = [], []
            # merge environments
            envs = self.site_envs(c)
            temp_gt = tp.gcontract('aA,Bb,dD,ABCDE->abCdE', envs[0], envs[1], envs[3], mgts[c])
            # build pure and impure double tensors
            pure_dt = tp.gcontract(
                    'abCde,abcde->Cc', mgts[c].conj(), temp_gt, 
                    bosonic_dims=('a', 'b', 'd', 'e'))
            # note that impure double tensor possibly break the Z2-parity symmetry
            # such as a single fermion operator
            impure_dt = tp.z2gcontract(
                    'abCdE,Ee,abcde->Cc', mgts[c].conj(), op_0, temp_gt,
                    bosonic_dims=('a', 'b', 'd', 'E'))
            pure_dts.append(pure_dt)
            impure_dts.append(impure_dt)

            envs = self.site_envs(cx)
            temp_gt = tp.gcontract('Bb,Cc,dD,ABCDE->AbcdE', envs[1], envs[2], envs[3], mgts[cx])
            pure_dt = tp.gcontract(
                    'Abcde,abcde->Aa', mgts[cx].conj(), temp_gt,
                    bosonic_dims=('b', 'c', 'd', 'e'))
            impure_dt = tp.z2gcontract(
                    'AbcdE,Ee,abcde->Aa', mgts[cx].conj(), op_1, temp_gt,
                    bosonic_dims=('b', 'c', 'd', 'E'))
            pure_dts.append(pure_dt)
            impure_dts.append(impure_dt)

            num = tp.z2gcontract('ab,ab->', *impure_dts)
            den = tp.gcontract('ab,ab->', *pure_dts)

            # print(num, den)
            meas.append(num / den)

            # Y-direction
            pure_dts, impure_dts = [], []

            envs = self.site_envs(c)
            temp_gt = tp.gcontract('aA,Cc,dD,ABCDE->aBcdE', envs[0], envs[2], envs[3], mgts[c])
            pure_dt = tp.gcontract(
                    'aBcde,abcde->Bb', mgts[c].conj(), temp_gt, 
                    bosonic_dims=('a', 'c', 'd', 'e'))
            impure_dt = tp.z2gcontract(
                    'aBcdE,Ee,abcde->Bb', mgts[c].conj(), op_0, temp_gt,
                    bosonic_dims=('a', 'c', 'd', 'E'))
            pure_dts.append(pure_dt)
            impure_dts.append(impure_dt)

            envs = self.site_envs(cy)
            temp_gt = tp.gcontract('aA,Bb,Cc,ABCDE->abcDE', envs[0], envs[1], envs[2], mgts[cy])
            pure_dt = tp.gcontract(
                    'abcDe,abcde->Dd', mgts[cy].conj(), temp_gt,
                    bosonic_dims=('a', 'b', 'c', 'e'))
            impure_dt = tp.z2gcontract(
                    'abcDE,Ee,abcde->Dd', mgts[cy].conj(), op_1, temp_gt,
                    bosonic_dims=('a', 'b', 'c', 'E'))
            pure_dts.append(pure_dt)
            impure_dts.append(impure_dt)

            num = tp.z2gcontract('ab,ab->', *impure_dts)
            den = tp.gcontract('ab,ab->', *pure_dts)
            
            # print(num, den)
            meas.append(num / den)

        return torch.tensor(meas)

    def simple_measurement_twobody_2(self, op_0: GTensor, op_1: GTensor):
        r'''
        measure a two-body operator by double tensors on the Beta lattice

        Parameters
        ----------
        op_0: GTensor, the first operator
        op_1: GTensor, the second operator

        Returns
        -------
        res: tensor, measured values
        '''

        mgts = self.merged_tensors()

        meas = []
        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny

            # X-direction
            pure_dts, impure_dts = [], []
            # merge environments
            envs = self.site_envs(c)
            temp_gt = tp.gcontract('aA,Bb,dD,ABCDE->abCdE', envs[0], envs[1], envs[3], mgts[c])

        return torch.tensor(res)

    def dt_measure_AB_onebody(self, op: GTensor):
        r'''
        measure 1-body operator by double tensors

        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, averaged values on each site
        '''

        '''
        def _convert_z2g(gt, dual=None):

            if dual is None:
                return Z2gTensor(gt.dual, gt.blocks())

            new_blocks = {}
            for qs, v in gt.blocks().items():
                # flip the dual and assign the sign
                sign = 1
                for i in range(gt.ndim):
                    if (gt.dual[i] != dual[i]) and 1 == qs[i]:
                        sign *= -1
                new_blocks[qs] = sign*v

            return Z2gTensor(dual, new_blocks)
        '''

        def _conj(gt, free_dims=None):

            if free_dims is None:
                free_dims = ()

            rank = len(gt.dual)
            dims = [i for i in range(rank)]
            dims.reverse()

            new_dual = [d ^ 1 for d in gt.dual]
            # reverse
            new_dual.reverse()
            new_shape = list(gt.shape)
            new_shape.reverse()
            # build new blocks
            new_blocks = {}
            for q, t in gt.blocks().items():
                # possible super trace sign should be considered
                sgns = [q[i]*gt.dual[i] for i in range(rank) if i not in free_dims]
                sign = (-1)**sum(sgns)
                new_q = list(q)
                new_q.reverse()
                new_blocks[tuple(new_q)] = sign*t.conj().permute(dims)

            # permute back to the original order
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks).permute(dims)

        mgts = self.merged_tensors()

        # dict: {site: measured_value}
        # measurements = dict.fromkeys(self._coords, 0.0)
        meas = {}
        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            # X-direction
            sites = c, cx
            gts, gts_dagger = {}, {}
            gts_envs = {}
            internal_dims = (2,), (0,)
            external_dims = (0, 1, 3), (1, 2, 3)
            pure_dt_strs = 'abcde,aA,Bb,dD,ABCDe->cC', 'abcde,Bb,Cc,dD,ABCDe->aA'
            impure_dt_strs = 'abcde,aA,Bb,dD,eE,ABCDE->cC', 'abcde,Bb,Cc,dD,eE,ABCDE->aA'

            for i, s in enumerate(sites):
                gts[s] = mgts[s]
                # gts_dagger[s] = _conj(gts[s], free_dims=internal_dims[i])
                gts_dagger[s] = _conj(gts[s], free_dims=internal_dims[i])
                # external environments
                envs = self.site_envs(s)
                o, p, q = external_dims[i]
                gts_envs[s] = envs[o], envs[p], envs[q]
            # build double tensors
            pure_dts, impure_dts = {}, {}
            for i, s in enumerate(sites):
                pure_dts[s] = tp.gcontract(pure_dt_strs[i], gts_dagger[s], *gts_envs[s], gts[s])
                impure_dts[s] = tp.gcontract(impure_dt_strs[i], gts_dagger[s], *gts_envs[s], op, gts[s])
            norm = tp.gcontract('aA,aA->', *pure_dts.values())

            for s in sites:
                temp = deepcopy(pure_dts)
                # repalce with a impure double tensor
                temp[s] = impure_dts[s]
                # one-body operator on each site
                value = tp.gcontract('aA,aA->', *temp.values())
                meas[s] = meas.get(s, 0.0)+(value/norm).item()

            # Y-direction
            sites = c, cy
            gts, gts_dagger = {}, {}
            gts_envs = {}
            internal_dims = (1,), (3,)
            external_dims = (0, 2, 3), (0, 1, 2)
            pure_dt_strs = 'abcde,aA,Cc,dD,ABCDe->bB', 'abcde,aA,Bb,Cc,ABCDe->dD'
            impure_dt_strs = 'abcde,aA,Cc,dD,eE,ABCDE->bB', 'abcde,aA,Bb,Cc,eE,ABCDE->dD'
            for i, s in enumerate(sites):
                gts[s] = mgts[s]
                gts_dagger[s] = _conj(gts[s], free_dims=internal_dims[i])
                # external environments
                envs = self.site_envs(s)
                o, p, q = external_dims[i]
                gts_envs[s] = envs[o], envs[p], envs[q]
            # build double tensors
            pure_dts, impure_dts = {}, {}
            for i, s in enumerate(sites):
                pure_dts[s] = tp.gcontract(pure_dt_strs[i], gts_dagger[s], *gts_envs[s], gts[s])
                impure_dts[s] = tp.gcontract(impure_dt_strs[i], gts_dagger[s], *gts_envs[s], op, gts[s])
            norm = tp.gcontract('aA,aA->', *pure_dts.values())
            # print(norm)

            for s in sites:
                temp = deepcopy(pure_dts)
                # repalce with a impure double tensor
                temp[s] = impure_dts[s]
                # one-body operator on each site
                value = tp.gcontract('aA,aA->', *temp.values())
                meas[s] = meas.get(s, 0.0)+(value/norm).item()

        # each site is measured fourth
        res = [0.25*v for v in meas.values()]

        return torch.tensor(res)

    def dt_measure_ABCD_twobody(self, op_0: Z2gTensor, op_1: Z2gTensor):
        r'''
        measure 1-body operator by double tensors

        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, averaged values on each site
        '''

        merged_gts = self.merged_tensors()

        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        external_dims = (0, 3), (2, 3), (0, 1), (1, 2)

        pure_dt_strs = 'abcde,aA,dD,ABCDe->bBcC', 'abcde,Cc,dD,ABCDe->aAbB', 'abcde,aA,Bb,ABCDe->cCdD', 'abcde,Bb,Cc,ABCDe->aAdD'
        impure_dt_strs = 'abcde,aA,dD,eE,ABCDE->bBcC', 'abcde,Cc,dD,eE,ABCDE->aAbB', 'abcde,aA,Bb,eE,ABCDE->cCdD', 'abcde,Bb,Cc,eE,ABCDE->aAdD'

        # loop through four UCs
        measurements = []
        for d in self._coords:
            dx = (d[0]+1) % self._nx, d[1]
            dy = d[0], (d[1]+1) % self._ny
            dxy = (d[0]+1) % self._nx, (d[1]+1) % self._ny

            cluster = d, dx, dy, dxy

            gts, gts_dagger = {}, {}
            gts_envs = {}
            for i, c in enumerate(cluster):
                gts[c] = merged_gts[c]
                gts_dagger[c] = gts[c].graded_conj(free_dims=internal_dims[i], side=0)
                envs = self.site_envs(c)
                p, q = external_dims[i]
                gts_envs[c] = envs[p], envs[q]

            pure_dts, impure_dts_0, impure_dts_1 = {}, {}, {}
            for i, c in enumerate(cluster):
                pure_dts[c] = tp.gcontract(pure_dt_strs[i], gts_dagger[c], *gts_envs[c], gts[c])
                impure_dts_0[c] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[c], *gts_envs[c], op_0, gts[c])
                impure_dts_1[c] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[c], *gts_envs[c], op_1, gts[c])
            
            norm = tp.gcontract('aAbB,bBcC,dDaA,dDcC->', *pure_dts.values())

            impure_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            impure_dts[d] = impure_dts_0[d]
            impure_dts[dx] = impure_dts_1[dx]
            value = tp.z2gcontract('aAbB,bBcC,dDaA,dDcC->', *impure_dts.values())
            measurements.append((value/norm).item())

            impure_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            impure_dts[d] = impure_dts_0[d]
            impure_dts[dy] = impure_dts_1[dy]
            value = tp.z2gcontract('aAbB,bBcC,dDaA,dDcC->', *impure_dts.values())
            measurements.append((value/norm).item())

        return torch.tensor(measurements)

    def dt_measure_ABCD_twobody_diag(self, op_0: Z2gTensor, op_1: Z2gTensor):
        r'''
        measure 1-body operator by double tensors

        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, averaged values on each site
        '''

        merged_gts = self.merged_tensors()

        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        external_dims = (0, 3), (2, 3), (0, 1), (1, 2)

        pure_dt_strs = 'abcde,aA,dD,ABCDe->bBcC', 'abcde,Cc,dD,ABCDe->aAbB', 'abcde,aA,Bb,ABCDe->cCdD', 'abcde,Bb,Cc,ABCDe->aAdD'
        impure_dt_strs = 'abcde,aA,dD,eE,ABCDE->bBcC', 'abcde,Cc,dD,eE,ABCDE->aAbB', 'abcde,aA,Bb,eE,ABCDE->cCdD', 'abcde,Bb,Cc,eE,ABCDE->aAdD'

        # loop through four UCs
        measurements = []
        for d in self._coords:
            dx = (d[0]+1) % self._nx, d[1]
            dy = d[0], (d[1]+1) % self._ny
            dxy = (d[0]+1) % self._nx, (d[1]+1) % self._ny

            cluster = d, dx, dy, dxy

            gts, gts_dagger = {}, {}
            gts_envs = {}
            for i, c in enumerate(cluster):
                gts[c] = merged_gts[c]
                gts_dagger[c] = gts[c].graded_conj(free_dims=internal_dims[i], side=0)
                envs = self.site_envs(c)
                p, q = external_dims[i]
                gts_envs[c] = envs[p], envs[q]

            pure_dts, impure_dts_0, impure_dts_1 = {}, {}, {}
            for i, c in enumerate(cluster):
                pure_dts[c] = tp.gcontract(pure_dt_strs[i], gts_dagger[c], *gts_envs[c], gts[c])
                impure_dts_0[c] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[c], *gts_envs[c], op_0, gts[c])
                impure_dts_1[c] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[c], *gts_envs[c], op_1, gts[c])
            
            norm = tp.gcontract('aAbB,bBcC,dDaA,dDcC->', *pure_dts.values())

            impure_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            impure_dts[d] = impure_dts_0[d]
            impure_dts[dxy] = impure_dts_1[dxy]
            value = tp.z2gcontract('aAbB,bBcC,dDaA,dDcC->', *impure_dts.values())
            measurements.append((value/norm).item())

            impure_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            impure_dts[dx] = impure_dts_0[dx]
            impure_dts[dy] = impure_dts_1[dy]
            value = tp.z2gcontract('aAbB,bBcC,dDaA,dDcC->', *impure_dts.values())
            measurements.append((value/norm).item())

        return torch.tensor(measurements)

    def dt_measure_AB_twobody(self, op_0: Z2gTensor, op_1: Z2gTensor):
        r'''
        measure 2-body operator by double tensors

        Parameters
        ----------
        op_0: Z2gTensor, one-body operator
        op_1: Z2gTensor, one-body operator

        Returns
        -------
        res: tensor, values on each bond of eight
        '''

        def _conj(gt, free_dims=None):

            if free_dims is None:
                free_dims = ()

            rank = len(gt.dual)
            dims = [i for i in range(rank)]
            dims.reverse()

            new_dual = [d ^ 1 for d in gt.dual]
            # reverse
            new_dual.reverse()
            new_shape = list(gt.shape)
            new_shape.reverse()
            # build new blocks
            new_blocks = {}
            for q, t in gt.blocks().items():
                # possible super trace sign should be considered
                sgns = [q[i]*gt.dual[i] for i in range(rank) if i not in free_dims]
                sign = (-1)**sum(sgns)
                new_q = list(q)
                new_q.reverse()
                new_blocks[tuple(new_q)] = sign*t.conj().permute(dims)

            # permute back to the original order
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks).permute(dims)

        ops = op_0, op_1

        mgts = self.merged_tensors()
        # dict: {bond: measured_value}
        # bonds = [i for i in range(2*self._size)]
        # bond_idxs = (0, 1, 4, 3), (2, 3, 6, 1), (4, 5, 0, 7), (6, 7, 2, 5)
        # measurements = dict.fromkeys(bonds, 0.0)
        # measurements = []
        meas = []
        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            # cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            sites = c, cx

            gts, gts_dagger = {}, {}
            gts_envs = {}
            internal_dims = (2,), (0,)
            external_dims = (0, 1, 3), (1, 2, 3)
            pure_dt_strs = 'abcde,aA,Bb,dD,ABCDe->cC', 'abcde,Bb,Cc,dD,ABCDe->aA'
            impure_dt_strs = 'abcde,aA,Bb,dD,eE,ABCDE->cC', 'abcde,Bb,Cc,dD,eE,ABCDE->aA'
            for i, s in enumerate(sites):
                gts[s] = mgts[s]
                gts_dagger[s] = _conj(gts[s], free_dims=internal_dims[i])
                # external environments
                envs = self.site_envs(s)
                o, p, q = external_dims[i]
                gts_envs[s] = envs[o], envs[p], envs[q]

            # build pure double tensors
            pure_dts, impure_dts = {}, {}
            for i, s in enumerate(sites):
                pure_dts[s] = tp.gcontract(pure_dt_strs[i], gts_dagger[s], *gts_envs[s], gts[s])
                impure_dts[s] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[s], *gts_envs[s], ops[i], gts[s])

            norm = tp.gcontract('aA,aA->', *pure_dts.values())
            value = tp.z2gcontract('aA,aA->', *impure_dts.values())
            meas.append((value/norm).item())

            # Y-direction
            sites = c, cy

            gts, gts_dagger = {}, {}
            gts_envs = {}
            internal_dims = (1,), (3,)
            external_dims = (0, 2, 3), (0, 1, 2)
            pure_dt_strs = 'abcde,aA,Cc,dD,ABCDe->bB', 'abcde,aA,Bb,Cc,ABCDe->dD'
            impure_dt_strs = 'abcde,aA,Cc,dD,eE,ABCDE->bB', 'abcde,aA,Bb,Cc,eE,ABCDE->dD'
            for i, s in enumerate(sites):
                gts[s] = mgts[s]
                gts_dagger[s] = _conj(gts[s], free_dims=internal_dims[i])
                # external environments
                envs = self.site_envs(s)
                o, p, q = external_dims[i]
                gts_envs[s] = envs[o], envs[p], envs[q]

            # build pure double tensors
            pure_dts, impure_dts = {}, {}
            for i, s in enumerate(sites):
                pure_dts[s] = tp.gcontract(pure_dt_strs[i], gts_dagger[s], *gts_envs[s], gts[s])
                impure_dts[s] = tp.z2gcontract(impure_dt_strs[i], gts_dagger[s], *gts_envs[s], ops[i], gts[s])

            norm = tp.gcontract('aA,aA->', *pure_dts.values())
            value = tp.z2gcontract('aA,aA->', *impure_dts.values())
            meas.append((value/norm).item())

        return torch.tensor(meas)

    def dt_measure_ABCD_onebody(self, op: GTensor):
        r'''
        measure 1-body operator by double tensors

        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, averaged values on each site
        '''

        merged_gts = self.merged_tensors()

        gts, gts_dagger = {}, {}
        gts_envs = {}
        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        external_dims = (0, 3), (2, 3), (0, 1), (1, 2)
        for i, c in enumerate(self._coords):
            gts[c] = merged_gts[c]
            gts_dagger[c] = gts[c].graded_conj(free_dims=internal_dims[i], side=0)
            # external bonds
            envs = self.site_envs(c)
            p, q = external_dims[i]
            gts_envs[c] = envs[p], envs[q]

        pure_dt_strs = 'abcde,aA,dD,ABCDe->bBcC', 'abcde,Cc,dD,ABCDe->aAbB', 'abcde,aA,Bb,ABCDe->cCdD', 'abcde,Bb,Cc,ABCDe->aAdD'
        impure_dt_strs = 'abcde,aA,dD,eE,ABCDE->bBcC', 'abcde,Cc,dD,eE,ABCDE->aAbB', 'abcde,aA,Bb,eE,ABCDE->cCdD', 'abcde,Bb,Cc,eE,ABCDE->aAdD'
        pure_dts, impure_dts = {}, {}
        for i, c in enumerate(self._coords):
            # print(i, c, pure_dt_strs[i], gts_envs[c])
            pure_dts[c] = tp.gcontract(pure_dt_strs[i], gts_dagger[c], *gts_envs[c], gts[c])
            impure_dts[c] = tp.gcontract(impure_dt_strs[i], gts_dagger[c], *gts_envs[c], op, gts[c])
        # norm
        norm = tp.gcontract('aAbB,bBcC,dDaA,dDcC->', *pure_dts.values())

        measurements = {}
        for i, c in enumerate(self._coords):
            temp_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            temp_dts[c] = impure_dts[c]
            value = tp.gcontract('aAbB,bBcC,dDaA,dDcC->', *temp_dts.values())
            measurements[c] = measurements.get(c, 0.0)+(value/norm).item()

        res = [v for v in measurements.values()]
        return torch.tensor(res)

    def bmps_left_canonical(self, mps: list):
        r'''
        left canonicalize a fermionic boundary MPS
        ancillary function for varitional_bmps()

        Parameters
        ----------
        mps: list[GTensor], the fermionic MPS
        '''

        new_mps = []
        temp = mps[0]
        q, r = tp.linalg.gtqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
        new_mps.append(q)
        for i in range(1, len(mps)):
            temp = tp.gcontract('ab,bcde->acde', r, mps[i])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            new_mps.append(q)

        return new_mps

    def bmps_right_canonical(self, mps: list):
        r'''
        right canonicalize a fermionic boundary MPS
        ancillary function for varitional_bmps()

        Parameters
        ----------
        mps: list[GTensor], the fermionic MPS
        '''

        new_mps = []
        temp = mps[-1]
        q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
        new_mps.append(q)
        for i in range(len(mps)-2, -1, -1):
            temp = tp.gcontract('abcd,be->aecd', mps[i], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            new_mps.append(q)

        new_mps.reverse()
        return new_mps

    def bmps_dagger(self, mps):
        r'''
        compute the conjugated MPS
        '''

        virtual_shape = mps[0].shape[0]
        # conjugated MPS
        mps_dagger = []
        for t in mps:
            mps_dagger.append(t.graded_conj(free_dims=(1,), side=0))

        # !a fermion parity operator should be replenished on the last open bond of MPS
        # fermion supertrace should be avoided here
        fpo = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=(virtual_shape, virtual_shape), cflag=True)
        mps_dagger[-1] = tp.gcontract('abcd,be->aecd', mps_dagger[-1], fpo)

        return mps_dagger

    def bmps_norm(self, mps):
        r'''
        compute the norm of a fermonic boundary MPS
        '''

        mps_dagger = self.bmps_dagger(mps)

        virtual_shape = mps[0].shape[0]
        left_env = tp.GTensor.eye(dual=(0, 1), shape=(virtual_shape, virtual_shape), cflag=True)
        for td, t in zip(mps_dagger, mps):
            # e--<--*--<--f
            #      |c|d
            # a-->--*-->--b
            left_env = tp.gcontract('ae,abcd,efcd->bf', left_env, td, t)

        return tp.gcontract('aa->', left_env)

    def bmps_up_cost(self, mpo, mps, left_fp, right_fp):
        r'''
        compute the cost for upper boundary MPS
        '''

        mps_dagger = self.bmps_dagger(mps)
        # |--<--a--<--*--<--g
        # |          |h|i
        # |-->--b-->--*-->--j
        # |--<--c--<--*--<--k
        # |          |l|m
        # |-->--d-->--*-->--n
        # |--<--e--<--*--<--o
        # |          |p|q
        # |--<--f--<--*--<--r
        left_fp = tp.gcontract(
                'abcde,aghi,bchijklm,delmnopq,frqp->gjknor', left_fp, mps[0], mpo[2], mpo[0], mps_dagger[0])
        # a--<--*--<--b--<--|
        #      |c|d         |
        # e-->--*-->--g-->--|
        # f--<--*--<--h--<--|
        #      |i|j         |
        # k-->--*-->--m-->--|
        # l--<--*--<--n--<--|
        #      |o|p         |
        # q--<--*--<--r--<--|
        right_fp = tp.gcontract(
                'abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps[1], mpo[3], mpo[1], mps_dagger[1], right_fp)

        return tp.gcontract('abcdef,abcdef->', left_fp, right_fp)

    def bmps_down_cost(self, mpo, mps, left_fp, right_fp):
        r'''
        compute the cost for lower boundary MPS
        '''

        mps_dagger = self.bmps_dagger(mps)

        left_fp = tp.gcontract(
                'abcde,aghi,bchijklm,delmnopq,frqp->gjknor', left_fp, mps_dagger[0], mpo[2], mpo[0], mps[0])
        right_fp = tp.gcontract(
                'abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps_dagger[1], mpo[3], mpo[1], mps[1], right_fp)

        return tp.gcontract('abcdef,abcdef->', left_fp, right_fp)

    def mv_left_fp(self, mpo, mps_u, mps_d, left_fp):
        r'''
        left fixed point multiplication

        Parameters
        ----------
        mpo: list[GTensor]
        '''

        # |--<--a--<--*--<--g
        # |          |h|i
        # |-->--b-->--*-->--j
        # |--<--c--<--*--<--k
        # |          |l|m
        # |-->--d-->--*-->--n
        # |--<--e--<--*--<--o
        # |          |p|q
        # |--<--f--<--*--<--r
        left_fp = tp.gcontract('abcde,aghi,bchijklm,delmnopq,frpq->gjknor', left_fp, mps_u[0], mpo[2], mpo[0], mps_d[0])
        left_fp = tp.gcontract('abcde,aghi,bchijklm,delmnopq,frpq->gjknor', left_fp, mps_u[1], mpo[3], mpo[1], mps_d[1])

        return left_fp

    def mv_right_fp(self, mpo, mps_u, mps_d, right_fp):
        r'''
        right fixed point multiplication

        Parameters
        ----------
        mpo: list[GTensor]
        '''

        # a--<--*--<--b--<--|
        #      |c|d         |
        # e-->--*-->--g-->--|
        # f--<--*--<--h--<--|
        #      |i|j         |
        # k-->--*-->--m-->--|
        # l--<--*--<--n--<--|
        #      |o|p         |
        # q--<--*--<--r--<--|
        right_fp = tp.gcontract('abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps_u[1], mpo[3], mpo[1], mps_d[1], right_fp)
        right_fp = tp.gcontract('abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps_u[0], mpo[2], mpo[0], mps_d[0], right_fp)

        return right_fp

    def bmps_solver(self, le: GTensor, re: GTensor, double_tensors: list, psi: GTensor, init_tensor=None):
        r'''
        upper boundary MPS solver

        Parameters
        ----------
        le: GTensor, left environment tensor
        re: GTensor, right environment tensor
        double_tensors: list[GTensor], daggered TPS and TPS tensors
        psi: GTensor, the referenerce state

        Returns
        -------
        '''

        # dual and shape for the MPS tensor
        v_dual = (0, 1, 1, 0)
        v_shape = (le.shape[0], re.shape[0], col_mpo[0].shape[2], col_mpo[0].shape[3])
        v_whole_shape = tuple([sum(d) for d in v_shape])

        def _mv(v):
            tv = torch.from_numpy(v.reshape(v_whole_shape)).cdouble()
            # convert to GTensor
            # print(tv.shape, v_shape)
            gtv = tp.GTensor.extract_blocks(tv, v_dual, v_shape)

            # --a--*--g--, A, as input
            #      |d
            # --b--*--e--, T^{\dagger}
            #      |f
            # --c--*--h--, \bar{M}

            # |--<--a    g h    q--<--|
            # |          | |          |
            # |-->--b-->--*-->--i-->--|
            # |--<--c--<--*--<--j--<--|
            # |          |k|l         |
            # |-->--d-->--*-->--m-->--|
            # |--<--e--<--*--<--n--<--|
            # |          |o|p         |
            # |--<--f           r--<--|
            print('mv')
            gtw = tp.gcontract('abcdef,bcghijkl,deklmnop,qijmnr,aqgh->frop', le, col_mpo[0], col_mpo[1], re, gtv)
            
            return gtw.push_blocks().numpy().flatten()

        init_v = None
        if init_tensor is not None:
            init_v = init_tensor.push_blocks().numpy().flatten()

        dim_op = math.prod(v_whole_shape)
        op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
        # with the largest magnitude
        vals, vecs = scipy.sparse.linalg.eigs(
            op, k=2, which='LM', v0=init_v, maxiter=None,
            return_eigenvectors=True)
        inds = abs(vals).argsort()[::-1]
        sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]

        print('MPS up solver:', sorted_vals[0])
        return sorted_vals[0], tp.GTensor.extract_blocks(torch.from_numpy(sorted_vecs[:, 0].reshape(v_whole_shape)), v_dual, v_shape)

    def bmps_down_solver(self, le, re, col_mpo, init_tensor=None):
        r'''
        lower boundary MPS solver

        Parameters
        ----------
        le: GTensor, left environment tensor
        re: GTensor, right environment tensor
        col_mpo: list[GTensor], column MPO tensors, order from up do down

        Returns
        -------
        '''

        # dual and shape for the MPS tensor
        v_dual = (0, 1, 1, 0)
        v_shape = (le.shape[0], re.shape[0], col_mpo[0].shape[2], col_mpo[0].shape[3])
        v_whole_shape = tuple([math.prod(d) for d in v_shape])

        def _mv(v):
            tv = torch.from_numpy(v.reshape(v_whole_shape)).cdouble()
            # convert to GTensor
            gtv = tp.GTensor.extract_blocks(tv, v_dual, v_shape)
            # |--<--a    g h    q--<--|
            # |          | |          |
            # |-->--b-->--*-->--i-->--|
            # |--<--c--<--*--<--j--<--|
            # |          |k|l         |
            # |-->--d-->--*-->--m-->--|
            # |--<--e--<--*--<--n--<--|
            # |          |o|p         |
            # |--<--f           r--<--|
            gtw = tp.gcontract('abcdef,bcghijkl,deklmnop,qijmnr,frop->aqgh', le, col_mpo[0], col_mpo[1], re, gtv)

            return gtw.push_blocks().numpy().flatten()

        init_v = None
        if init_tensor is not None:
            init_v = init_tensor.push_blocks().numpy().flatten()

        dim_op = math.prod(v_whole_shape)
        op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
        # with the largest magnitude
        vals, vecs = scipy.sparse.linalg.eigs(
            op, k=2, which='LM', v0=init_v, maxiter=None,
            return_eigenvectors=True)
        inds = abs(vals).argsort()[::-1]
        sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]

        return sorted_vals[0], tp.GTensor.extract_blocks(torch.from_numpy(sorted_vecs[:, 0].reshape(v_whole_shape)), v_dual, v_shape)

    def bmps_up_sweep(self, mpo, mps, left_fp, right_fp):
        r'''
        DMRG sweep for the upper boundary MPS
        '''

        err, cost = 1.0, 1.0
        n = 0
        while err > 1E-12 or n < 10:
            print(n, cost, err)
            # partition function density
            lams = []

            # 0
            left_env = left_fp
            # bring the MPS to right canonical
            q, l = tp.linalg.super_gtqr(mps[1], group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            mps[1] = q
            # TODO: possible simplification
            mps_dagger = self.bmps_dagger(mps)
            temp_gt = tp.gcontract('abcd,be->aecd', mps[0], l)
            # a--<--*--<--b--<--|
            #      |c|d         |
            # e-->--*-->--g-->--|
            # f--<--*--<--h--<--|
            #      |i|j         |
            # k-->--*-->--m-->--|
            # l--<--*--<--n--<--|
            #      |o|p         |
            # q--<--*--<--r--<--|
            print(mps[1].dual, mpo[3].dual, mpo[1].dual, mps_dagger[1].dual, right_fp.dual)
            print(mps[1].dtype, mpo[3].dtype, mpo[1].dtype, mps_dagger[1].dtype, right_fp.dtype)
            right_env = tp.gcontract(
                    'abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps[1], mpo[3], mpo[1], mps_dagger[1], right_fp)
            val, mps[0] = self.bmps_up_solver(left_env, right_env, col_mpo=(mpo[2], mpo[0]), init_tensor=temp_gt)
            lams.append(val)

            # 1
            right_env = right_fp
            q, r = tp.linalg.gtqr(mps[0], group_dims=((0, 2, 3), (1,)), qr_dims=(1, 0))
            mps[0] = q
            mps_dagger = self.bmps_dagger(mps)
            temp_gt = tp.gcontract('ab,bcde->acde', r, mps[1])
            left_env = tp.gcontract(
                    'abcde,aghi,bchijklm,delmnopq,frqp->gjknor', left_fp, mps[0], mpo[2], mpo[0], mps_dagger[0])
            val, mps[1] = self.bmps_up_solver(left_env, right_env, col_mpo=(mpo[3], mpo[1]), init_tensor=temp_gt)
            lams.append(val)

            new_cost = self.bmps_up_cost(mpo, mps, left_fp, right_fp).item()
            err = abs(new_cost-cost)
            cost = new_cost
            n += 1

        return lams, mps

    def test_mv(self, gts, gts_dagger, mps, left_fp, right_fp):


        # 0
        left_env = left_fp
        # bring the MPS to right canonical
        q, l = tp.linalg.super_gtqr(mps[1], group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
        mps[1] = q
        # TODO: possible simplification
        mps_dagger = self.bmps_dagger(mps)
        temp_gt = tp.gcontract('abcd,be->aecd', mps[0], l)
        # a--<--*--<--b--<--|
        #      |c|d         |
        # e-->--*-->--g-->--|
        #       |
        # f--<--*--<--h--<--|
        #      |i|j         |
        # k-->--*-->--m-->--|
        # l--<--*--<--n--<--|
        #      |o|p         |
        # q--<--*--<--r--<--|
        # print(mps[1].dual, mpo[3].dual, mpo[1].dual, mps_dagger[1].dual, right_fp.dual)
        # print(mps[1].dtype, mpo[3].dtype, mpo[1].dtype, mps_dagger[1].dtype, right_fp.dtype)
        # right_env = tp.gcontract(
                # 'abcd,efcdghij,klijmnop,qrop,bghmnr->aefklq', mps[1], mpo[3], mpo[1], mps_dagger[1], right_fp)
        right_env = right_fp

        for i in range(10):
            # |--<--a       g            s--<--|
            # |             |                  |
            # |-->--b,b-->--*-->--h,     h-->--|
            # |            i| \j |k            |
            # |--<--c,     c--<--*--<--l,l--<--|
            # |                  |m            |
            # |             |i                 |
            # |-->--d,d-->--*-->--n,     n-->--|
            # |            o| \p |m            |
            # |--<--e,     e--<--*--<--q,q--<--|
            # |                  |r            |
            # |--<--f                    t--<--|
            print('test mv:', i)
            # gtw = tp.gcontract('abcdef,bcghijkl,deklmnop,qijmnr,aqgh->frop', left_env, mpo[2], mpo[0], right_env, mps[0])
            gtw = tp.gcontract('abcdef,bghij,cklmj,dinop,emqrp,shlnqt,asgk->ftor', left_env, gts_dagger[2], gts[2], gts_dagger[0], gts[0], right_env, mps[0])

    def test_mv_onelayer(self, mpo, mps, rho):

        virtual_shape = rho // 2, rho // 2
        left_shape = virtual_shape, mpo[2].shape[0], mpo[2].shape[1], virtual_shape
        right_shape = virtual_shape, mpo[3].shape[4], mpo[3].shape[5], virtual_shape

        left_fp = tp.GTensor.rand(dual=(1, 0, 1, 0), shape=left_shape, cflag=True)
        right_fp = tp.GTensor.rand(dual=(0, 1, 0, 1), shape=right_shape, cflag=True)

        left_fp, right_fp = (1.0/left_fp.norm())*left_fp, (1.0/right_fp.norm())*right_fp

        # 0
        left_env = left_fp
        # bring the MPS to right canonical
        q, l = tp.linalg.super_gtqr(mps[1], group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
        mps[1] = q
        # TODO: possible simplification
        mps_dagger = self.bmps_dagger(mps)
        temp_gt = tp.gcontract('abcd,be->aecd', mps[0], l)
        # a--<--*--<--b--<--|
        #      |c|d         |
        # e-->--*-->--g-->--|
        # f--<--*--<--h--<--|
        #      |i|j         |
        # k--<--*--<--l--<--|
        right_env = tp.gcontract(
                'abcd,efcdghij,klij,bghl->aefk', mps[1], mpo[3], mps_dagger[1], right_fp)

        for i in range(24):
            # |--<--a    e f    k--<--|
            # |          | |          |
            # |-->--b-->--*-->--g-->--|
            # |--<--c--<--*--<--h--<--|
            # |          |i|j         |
            # |--<--d           l--<--|
            print('test 1-layer mv:', i)
            gtw = tp.gcontract('abcd,bcefghij,kghl,akef->dlij', left_env, mpo[2], right_env, mps[0])

    def variational_bmps(self, rho: int, init_mps=None, init_envs=None):
        r'''
        varitional boundary MPS method

        Parameters
        ----------
        rho: int, bond dimension of the boundary MPS
        init_mps: tuple[list], initial up- and down-MPS
        init_envs: tuple[GTensor], left- and right-environments
        '''

        merged_gts = self.merged_tensors()
        # tensors for the TPS
        tps_gts, tps_gts_dagger = [], []
        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        for i, c in enumerate(self._coords):
            tps_gts.append(merged_gts[c])
            tps_gts_dagger.append(merged_gts[c].graded_conj(free_dims=internal_dims[i]))

        # # double tensor as MPO
        # mpo = []
        # for c in self._coords:
        #     temp = tp.gcontract('abcde,ABCDe->aAbBcCdD', gts_dagger[c], gts[c])
        #     temp.cdouble()
        #     mpo.append(temp)

        mps_dual = (0, 1, 1, 0)
        virtual_shape = (rho // 2, rho // 2)

        # TODO: initial MPS build from bond matrices
        # initial fixed points for up and down MPS
        if init_mps is None:
            mps_u = []
            mps_u.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, tps_gts_dagger[2].shape[1], tps_gts[2].shape[1]), cflag=True))
            mps_u.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, tps_gts_dagger[2].shape[1], tps_gts[2].shape[1]), cflag=True))
            mps_d = []
            mps_d.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, tps_gts_dagger[0].shape[3], tps_gts[0].shape[3]), cflag=True))
            mps_d.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, tps_gts_dagger[0].shape[3], tps_gts[0].shape[3]), cflag=True))
        # fixed point tensors
        # |--<--
        # |
        # |-->--
        # |--<--
        # |
        # |-->--
        # |--<--
        # |
        # |-->--
        if init_envs is None:
            left_shape = virtual_shape, tps_gts_dagger[2].shape[0], tps_gts[2].shape[0], tps_gts_dagger[0].shape[0], tps_gts[0].shape[0], virtual_shape
            right_shape = virtual_shape, tps_gts_dagger[3].shape[2], tps_gts[3].shape[2], tps_gts_dagger[1].shape[2], tps_gts[1].shape[2], virtual_shape
            left_fp = tp.GTensor.rand(dual=(1, 0, 1, 0, 1, 0), shape=left_shape, cflag=True)
            right_fp = tp.GTensor.rand(dual=(0, 1, 0, 1, 0, 1), shape=right_shape, cflag=True)

            left_fp, right_fp = (1.0/left_fp.norm())*left_fp, (1.0/right_fp.norm())*right_fp

        num_fp_iter = 32
        
        # vals, mps_u = self.bmps_up_sweep(mpo, mps_u, left_fp, right_fp)
        self.test_mv(tps_gts, tps_gts_dagger, mps_u, left_fp, right_fp)
        # self.test_mv_onelayer(mpo, mps_u, rho)

        '''

        err_u, err_d = 1.0, 1.0
        while err_u < 1E-10 and err_d < 1E-10:
            mps_ulc, mps_urc = self.bmps_left_canonical(mps_u), self.bmps_right_canonical(mps_u)
            mps_dlc, mps_drc = self.bmps_left_canonical(mps_d), self.bmps_right_canonical(mps_d)

            # for i in range(num_fp_iter):
                # pass

            vals, mps_u = self.bmps_up_sweep(mpo, mps_u, left_fp, right_fp)
        '''


    def dt_measure_onebody_vbmps(self, op: GTensor):
        r'''
        measure 1-body operator by double tensors

        Parameters
        ----------
        op: GTensor, the one-body operator

        Returns
        -------
        res: tensor, averaged values on each site
        '''

        merged_gts = self.merged_tensors()

        gts, gts_dagger = {}, {}
        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        external_dims = (0, 3), (2, 3), (0, 1), (1, 2)
        for i, c in enumerate(self._coords):
            gts[c] = merged_gts[c]
            gts_dagger[c] = _conj(gts[c], free_dims=internal_dims[i])

        pure_dts, impure_dts = {}, {}
        for i, c in enumerate(self._coords):
            pure_dts[c] = tp.gcontract('abcde,ABCDe->aAbBcCdD', gts_dagger[c], gts[c])
            impure_dts[c] = tp.gcontract('abcde,eE,ABCDE->aAbBcCdD', gts_dagger[c], op, gts[c])

        measurements = {}
        for i, c in enumerate(self._coords):
            temp_dts = deepcopy(pure_dts)
            # repalce with an impure double tensor
            temp_dts[c] = impure_dts[c]
            value = tp.gcontract('aAbB,bBcC,dDaA,dDcC->', *temp_dts.values())
            measurements[c] = measurements.get(c, 0.0)+(value/norm).item()

        res = [v for v in measurements.values()]
        return torch.tensor(res)
