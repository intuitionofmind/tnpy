from copy import deepcopy
import itertools
import math
import numpy as np
import opt_einsum as oe
import pickle as pk

import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as tnf
torch.set_printoptions(precision=8)

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

    def coords(self) -> tuple:
        return self._coords

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
        build the tensors merged with square root of the environments

        Returns
        -------
        mgts: dict, {site: merged tensor}
        '''

        mgts = {}
        for c in self._coords:
            gt = self._site_tensors[c]
            envs = self.site_envs(c)
            half_envs = [self.sqrt_env(t) for t in envs]
            # merge
            mgts[c] = tp.gcontract('abcde,fa,bg,ch,id->fghie', gt, *half_envs)

        return mgts

    def simple_update_proj_loop(self, te_mpo: tuple, sort_weights=False, average_weights=None, expand=None):
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
            'dominant', average by the dominant sector
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

                if 'dominant' == average_weights:
                    # average by dominant part
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
                    print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
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
                print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
                '''

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
            'dominant', average by the dominant sector
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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

                if 'dominant' == average_weights:
                    # average by dominant part
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
                    print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
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

    def dt_fermion_parity_measure_ABCD_onebody(self, op: GTensor):
        r'''
        measure 1-body operator by double tensors
        use normal conjugation and insert fermion parity operator

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
        parity_strs = 'Aa,Dd,abcde->AbcDe', 'cC,Dd,abcde->abCDe', 'Aa,bB,abcde->ABcde', 'bB,cC,abcde->aBCde'
        for i, c in enumerate(self._coords):
            gts[c] = merged_gts[c]
            test_gt = gts[c].graded_conj(free_dims=internal_dims[i], side=0)

            temp = gts[c].conj()
            p, q = external_dims[i]
            # conjugated GTensor should be attached with an extra fermion parity operator if its dual is 0
            # namely recover the normal trace if a supertrace happens on this connected bonds
            if 0 == temp.dual[p]:
                fp = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=(gts[c].shape[p], gts[c].shape[p]))
            else:
                fp = tp.GTensor.eye(dual=(1, 0), shape=(gts[c].shape[p], gts[c].shape[p]))
            if 0 == temp.dual[q]:
                fq = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=(gts[c].shape[q], gts[c].shape[q]))
            else:
                fq = tp.GTensor.eye(dual=(1, 0), shape=(gts[c].shape[q], gts[c].shape[q]))
            gts_dagger[c] = tp.gcontract(parity_strs[i], fp, fq, temp)
            print(i, parity_strs[i])
            for key, val in gts_dagger[c].blocks().items():
                print(key, (val-test_gt.blocks()[key]).norm())
            # external bonds needs an extra environment tensor
            envs = self.site_envs(c)
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

    def bmps_up_solver(self, le, re, col_mpo, init_tensor=None):
        r'''
        upper boundary MPS solver

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
            # |--<--b-->--*-->--i-->--|
            # |--<--c--<--*--<--j--<--|
            # |          |k|l         |
            # |--<--d-->--*-->--m-->--|
            # |--<--e--<--*--<--n--<--|
            # |          |o|p         |
            # |--<--f           r--<--|
            gtw = tp.gcontract('abcde,bcghijkl,deklmnop,qijmnr,aqgh->frop', re, col_mpo[0], col_mpo[1], le, gtv)

            return gtw.push_blocks().numpy().flatten()

        init_v = None
        if init_tensor is not None:
            init_v = init_tensor.push_blocks().numpy().flatten()

        dim_op = math.prod(v_whole_shape)
        op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
        # with the largest magnitude
        vals, vecs = scipy.sparse.linalg.eigs(
            op, k=3, which='LM', v0=initial_v, maxiter=None,
            return_eigenvectors=True)
        inds = abs(vals).argsort()[::-1]
        sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]

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
            # |--<--b-->--*-->--i-->--|
            # |--<--c--<--*--<--j--<--|
            # |          |k|l         |
            # |--<--d-->--*-->--m-->--|
            # |--<--e--<--*--<--n--<--|
            # |          |o|p         |
            # |--<--f           r--<--|
            gtw = tp.gcontract('abcde,bcghijkl,deklmnop,qijmnr,frop->aqgh', re, col_mpo[0], col_mpo[1], le, gtv)

            return gtw.push_blocks().numpy().flatten()

        init_v = None
        if init_tensor is not None:
            init_v = init_tensor.push_blocks().numpy().flatten()

        dim_op = math.prod(v_whole_shape)
        op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
        # with the largest magnitude
        vals, vecs = scipy.sparse.linalg.eigs(
            op, k=3, which='LM', v0=initial_v, maxiter=None,
            return_eigenvectors=True)
        inds = abs(vals).argsort()[::-1]
        sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]

        return sorted_vals[0], tp.GTensor.extract_blocks(torch.from_numpy(sorted_vecs[:, 0].reshape(v_whole_shape)), v_dual, v_shape)

    def bmps_up_sweep(self, left_fp, right_fp, mpo, mps):

        err, cost = 1.0, 1.0
        n = 0
        while err > 1E-12 or n < 10:
            # bring the MPS to right canonical
            q, l = tp.linalg.super_gtqr(mps[1], group_dims=((1, 2, 3), (0,)), qr_dims=(0, 1))
            mps[1] = q
            old_gt = tp.gcontract('abcd,be->aecd', mps[0], l)
            left_env = left_fp

        return 1

    def varitional_bmps(self, rho: int, init_mps=None, init_envs=None):
        r'''
        varitional boundary MPS method

        Parameters
        ----------
        rho: int, bond dimension of the boundary MPS
        init_mps: tuple[list], initial up- and down-MPS
        init_envs: tuple[GTensor], left- and right-environments
        '''

        merged_gts = self.merged_tensors()

        gts, gts_dagger = {}, {}
        gts_envs = {}
        internal_dims = (1, 2), (0, 1), (2, 3), (0, 3)
        for i, c in enumerate(self._coords):
            gts[c] = merged_gts[c]
            gts_dagger[c] = gts[c].conj(free_dims=internal_dims[i])
            p, q = external_dims[i]
            gts_envs[c] = envs[p], envs[q]
        # double tensor as MPO
        mpo = []
        for c in self._coords:
            mpo.append(tp.gcontract('abcde,ABCDe->aAbBcCdD', gts_dagger[c], gts[c]))

        mps_dual = (0, 1, 1, 0)
        virtual_shape = (rho // 2, rho // 2)

        # TODO: initial MPS build from bond matrices
        # initial fixed points for up and down MPS
        if init_mps is None:
            mps_u = []
            mps_u.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, mpo[2].shape[2], mpo[2].shape[3]), cflag=True))
            mps_u.append(tp.GTensor.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, mpo[3].shape[2], mpo[3].shape[3]), cflag=True))
            mps_d = []
            mps_d.append(torch.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, mpo[0].shape[6], mpo[0].shape[7]), cflag=True))
            mps_d.append(torch.rand(dual=mps_dual, shape=(virtual_shape, virtual_shape, mpo[1].shape[6], mpo[1].shape[7]), cflag=True))
        # fixed point tensors
        # |--<--
        # |
        # |-->--
        # |--<--
        # |
        # |-->--
        # |--<--
        # |
        # |--<--
        if init_envs is None:
            left_shape = virtual_shape, mpo[2].shape[0], mpo[2].shape[1], mpo[0].shape[0], mpo[0].shape[1], virtual_shape
            right_shape = virtual_shape, mpo[3].shape[4], mpo[3].shape[5], mpo[1].shape[4], mpo[1].shape[5], virtual_shape
            left_fp = tp.GTensor.rand(dual=(1, 0, 1, 0, 1, 1), shape=left_shape, cflag=True)
            right_fp = tp.GTensor.rand(dual=(0, 1, 0, 1, 0, 0), shape=right_shape, cflag=True)

        num_fp_iter = 32

        err_u, err_d = 1.0, 1.0
        while err_u < 1E-10 and err_d < 1E-10:
            mps_ulc, mps_urc = self.bmps_left_canonical(mps_u), self.bmps_right_canonical(mps_u)
            mps_dlc, mps_drc = self.bmps_left_canonical(mps_d), self.bmps_right_canonical(mps_d)



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


class FermiTwoLayerSquareTPS(object):
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

    def coords(self) -> tuple:
        return self._coords

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
        build the tensors merged with square root of the environments

        Returns
        -------
        mgts: dict, {site: merged tensor}
        '''

        mgts = {}
        for c in self._coords:
            gt = self._site_tensors[c]
            envs = self.site_envs(c)
            half_envs = [self.sqrt_env(t) for t in envs]
            # merge
            mgts[c] = tp.gcontract('abcde,fa,bg,ch,id->fghie', gt, *half_envs)

        return mgts

    def simple_update_proj_loop(self, te_mpo: tuple, sort_weights=False, average_weights=None, expand=None):
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
            'dominant', average by the dominant sector
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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

                if 'dominant' == average_weights:
                    # average by dominant part
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
                    print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
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
                print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
                '''

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
            'dominant', average by the dominant sector
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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
            u_dagger = u.graded_conj(iso_dims=(1,), side=0)
            v_dagger = v.graded_conj(iso_dims=(0,), side=1)
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

                if 'dominant' == average_weights:
                    # average by dominant part
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
                    print('average dominant:', 0.25*sum(sds), 0.25*sum(sms))
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

