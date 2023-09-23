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

        def _site_envs(site, ex_bonds: tuple):

            envs = self.site_envs(site)
            for j in range(4):
                if j not in ex_bonds:
                    envs[j] = self.sqrt_env(envs[j])

            return envs

        # factorize to MPO
        # u, s, v = tp.linalg.gtsvd(time_evo, group_dims=((0, 2), (1, 3)), svd_dims=(1, 0))
        # se, so = s.blocks()[(0, 0)].sqrt(), s.blocks()[(1, 1)].sqrt()
        # ss = GTensor(dual=s.dual, shape=s.shape, blocks={(0, 0):se, (1, 1):so}, cflag=s.cflag)
        # te_mpo = tp.gcontract('abc,bd->adc', u, ss), tp.gcontract('ab,bcd->acd', ss, v)

        # se, so = [], []
        for c in self._coords:
            cx = (c[0]+1) % self._nx, c[1]
            cy = c[0], (c[1]+1) % self._ny
            cxy = (c[0]+1) % self._nx, (c[1]+1) % self._ny

            # X-direction
            gts = [self._site_tensors[c], self._site_tensors[cx]]
            envs = [_site_envs(c, ex_bonds=(0, 1, 3)), _site_envs(cx, ex_bonds=(1, 2, 3))]
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
            envs = [_site_envs(c, ex_bonds=(0, 2, 3)), _site_envs(cy, ex_bonds=(0, 1, 2))]
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

