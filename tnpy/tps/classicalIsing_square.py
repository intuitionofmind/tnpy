from copy import deepcopy
import itertools
import math
import numpy as np
import opt_einsum as oe
import pickle as pk
import scipy

import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as tnf

import tnpy as tp

class SquareClassicalIsing(object):
    r'''
    square lattice of 2D Ising model
    '''

    def __init__(self, beta: float, J: float):
        r'''

        Parameters
        ----------
        '''

        self._phys_dim = 2
        self._beta, self._J = beta, J
        # bond weight matrix
        w = torch.tensor([[np.exp(-beta*J), np.exp(beta*J)], [np.exp(beta*J), np.exp(-beta*J)]])
        u, s, v = torch.linalg.svd(w)
        m, mp = u@torch.sqrt(s).diag(), torch.sqrt(s).diag()@v

        # build the site tensor
        self._site_tensor = tp.contract('as,sb,sc,ds->abcd', mp, m, m, mp)

    def trg(self, num_scale: int, chi: int):
        r'''
        tensor renormalization group
        Levin and Nave, PRL 99, 120601 (2007)
        Gu, Levin, and Wen, PRB 78, 205116 (2008)

        Parameters
        ----------
        num_scale: int, number of RG scales
        '''

        def _svd_nesw(t, chi):
            r'''
            SVD along NorthEast and SouthWest direction
            '''

            #   1           |1
            #   |           S3-2
            # 0-*-2        /
            #   |       0-S1
            #   3         |3
            u, s, v = tp.linalg.tsvd(t, group_dims=((1, 2), (3, 0)), svd_dims=(2, 2), cut_off=chi)
            s_sqrt = torch.sqrt(s).diag()
            s3 = tp.contract('abc,cd->abd', u, s_sqrt)
            s1 = tp.contract('abc,cd->abd', v, s_sqrt)

            return s3, s1

        def _svd_nwse(t, chi: int):
            r'''
            SVD along NorthWest and SouthEest direction
            '''

            #   1         |1
            #   |       0-S2
            # 0-*-2         \
            #   |            S0-2 
            #   3            |3
            u, s, v = tp.linalg.tsvd(t, group_dims=((0, 1), (2, 3)), svd_dims=(2, 2), cut_off=chi)
            s_sqrt = torch.sqrt(s).diag()
            s2 = tp.contract('abc,cd->abd', u, s_sqrt)
            s0 = tp.contract('abc,cd->abd', v, s_sqrt)

            return s2, s0

        def _fuse_four_edges(edges):
            r'''
            contract four rank-3 tensors from SVD

            Parameters
            ----------
            edges: list of four edge tensor [s0, s1, s2, s3]

            Returns
            -------
            # a rank-4 tensor, rotated clockwisely by \pi/4
            # 0   1
            #  \ /
            #   *
            #  / \
            # 3   2
            '''

            # c        e
            #  \      /
            #   S0-a-S1
            #   |b   |d
            #   S3-f-S2
            #  /      \
            # h        g
            # Order: S0-abc, S1-dae, S2-fdg, S3-bfh
            # inner bonds are ordered clockwisely
            return tp.contract('abc,dae,fdg,bfh->cegh', *edges)

        lattice_const = 2
        # number of sites under this scale
        nx, ny = (2**num_scale), (2**num_scale)
        lenx, leny = lattice_const*nx, lattice_const*ny

        trg_layers, trg_norms = [], []
        # contruct the initial zeroth TRG layer
        initial_layer, initial_norm = {}, {}
        norm = torch.max(self._site_tensor.abs())
        for j, i in itertools.product(range(ny), range(nx)):
            c = lattice_const*i, lattice_const*j
            initial_layer[c] = self._site_tensor/norm
            initial_norm[c] = norm
        # all TRG layers shoud be stored
        trg_layers.append(initial_layer)
        trg_norms.append(initial_norm)

        # the scale loop
        # one scale iteration renormalizes the lattice from 2^{N}*2^{N} to 2^{N-1}*2^{N-1}
        for s in range(num_scale-1):
            # lattice spacing for the current scale and next scale
            curr_spacing, next_spacing = 2**(s+1), 2**(s+2)

            # firstly, TRG from even layers renormlizes the lattice from 2^{N}*2^{N} to 2^{N-1}*2^{N}
            l = 2*s
            nx, ny = nx//2, ny
            # the coordinate moving step size
            step = 2**s

            # the coordinate of the first site in current layer as a reference
            first_site = tuple(trg_layers[l].keys())[0]
            x0, y0 = first_site[0], first_site[1]
            # dict, to record the edges from SVD in next layer
            # (x, y): [s0, s1, s2, s3]
            next_edges = {}
            for j, i in itertools.product(range(ny), range(nx)):
                # x-direction is already renormalized, thus 'next_spacing'
                # a zigzag shift only for x-coordinate
                x = x0+step+next_spacing*i+curr_spacing*((j+1) % 2)
                # y-direction is NOT renormalized yet, still 'curr_spacing'
                y = y0+step+curr_spacing*j
                next_edges[(x, y)] = [0]*4

            ref = x0+y0
            for coord, tensor in trg_layers[l].items():
                # print(l, coord, np.sum(tensor))
                if ((coord[0]+coord[1]) % next_spacing) == ref:
                    s2, s0 = _svd_nwse(tensor, chi)
                    # determine the coordinates for s2 and s0
                    c2 = ((coord[0]-step) % lenx, (coord[1]+step) % leny)
                    c0 = ((coord[0]+step) % lenx, (coord[1]-step) % leny)
                    next_edges[c0][0] = s0
                    next_edges[c2][2] = s2
                elif ((coord[0]+coord[1]) % next_spacing) == (ref+curr_spacing):
                    s3, s1 = _svd_nesw(tensor, chi)
                    c3 = ((coord[0]+step) % lenx, (coord[1]+step) % leny)
                    c1 = ((coord[0]-step) % lenx, (coord[1]-step) % leny)
                    next_edges[c1][1] = s1
                    next_edges[c3][3] = s3

            # sort this dict according to the coordinate (x, y)
            # sorted_next_edges = dict(sorted(next_edges.items(), key=lambda x: (x[0][1], x[0][0])))
            # bulid the next layer, which is sorted already
            next_layer, next_norm = {}, {}
            for key, value in next_edges.items():
                new_tensor = _fuse_four_edges(value)
                norm = torch.max(new_tensor)
                next_layer[key] = new_tensor/norm
                next_norm[key] = norm

            trg_layers.append(next_layer)
            trg_norms.append(next_norm)

            # secondly, TRG from odd layers renormalizes the lattice from 2^{N-1}*2^{N} to 2^{N-1}*2^{N-1}
            l = 2*s+1
            nx, ny = nx, ny//2
            step = 2**(s+1)

            # use the coordinate of the first site as a reference
            first_site = tuple(trg_layers[l].keys())[0]
            x0, y0 = first_site[0], first_site[1]
            next_edges = {}
            for j, i in itertools.product(range(ny), range(nx)):
                #  y-direction is renormalized to next scale
                x = x0-step+next_spacing*i
                y = y0+next_spacing*j
                next_edges[(x, y)] = [0]*4

            # odd case only needs the y-coordinate as a reference
            ref = y0
            for coord, tensor in trg_layers[l].items():
                # print(l, coord, np.sum(tensor))
                # depends on the convension of the return from fuse_four_edges()
                # 0   1
                #  \ /
                #   *   -----NESW (s3, s1), x-direction
                #  / \
                # 3   2
                #   |
                #   | NWSE (s2, s0), y-direction
                if (coord[1] % (next_spacing)) == (ref+curr_spacing):
                    s2, s0 = _svd_nwse(tensor, chi)
                    c2 = (coord[0], (coord[1]+step) % leny)
                    c0 = (coord[0], (coord[1]-step) % leny)
                    next_edges[c0][0] = s0
                    next_edges[c2][2] = s2

                elif (coord[1] % (next_spacing)) == ref:
                    s3, s1 = _svd_nesw(tensor, chi)
                    c3 = ((coord[0]+step) % lenx, coord[1])
                    c1 = ((coord[0]-step) % lenx, coord[1])
                    next_edges[c1][1] = s1
                    next_edges[c3][3] = s3

            next_layer, next_norm = {}, {}
            for key, value in next_edges.items():
                # note that, two TRGs rotate the tensor clockwisely by \pi/2 (each \pi/4)
                # need to be restored, rotate counter-clockwisely by \pi/2
                #    a
                #    |
                # d--*--b
                #    |
                #    c
                # abcd -> dabc(3012)
                # new_tensor = Tensor(torch.permute(_fuse_four_edges(value), (3, 0, 1, 2)))
                new_tensor = _fuse_four_edges(value)
                norm = torch.max(new_tensor)
                next_layer[key] = new_tensor / norm
                next_norm[key] = norm

            trg_layers.append(next_layer)
            trg_norms.append(next_norm)

        # print(len(trg_layers))
        # for l in trg_layers:
        #    ts = tuple(l.values())
        #    print(len(ts))
            
        return trg_layers, trg_norms

    def exact_contract_2by2(self, trg_layers):
        r'''
        exactly contract the final layer of TRG consisting of 4 sites
        '''

        #    d   f
        #    |   |
        # g--*-h-*--g
        #   b|   |e
        # a--*-c-*--a
        #    |   |
        #    d   f
        tensors = tuple(trg_layers[-1].values())
        # print(tensors[0].shape, tensors[1].shape, tensors[2].shape, tensors[3].shape)

        return tp.contract('abcd,ceaf,gdhb,hfge', *tensors)

    def trg_value(self, trg_layers, trg_norms):
        r'''
        compute the value of this TN by TGR,
        trg_flow() should be run in the first place to build the TRG
        '''

        z = self.exact_contract_2by2(trg_layers)
        for layer in trg_norms:
            for value in layer.values():
                z *= value

        return z

    def trg_free_energy(self, trg_layers, trg_norms):
        r'''
        Helmholtz free energy from TRG
        '''

        z = self.exact_contract_2by2(trg_layers)
        res = torch.log(z)
        for layer in trg_norms:
            for key, value in layer.items():
                res += torch.log(value)

        return -res.item()

    def exact_free_energy(self):
        r'''
        compute the exact free energy of 2D Ising model in the themodynamic limit
        '''

        def fun(x):
            k = 2.0*np.sinh(2.0*self._K) / (np.cosh(2.0*self._K))**2
            return np.log(1.0+np.sqrt(1.0-(k**2)*(np.cos(x))**2))

        res = -0.5*np.log(2.0)-np.log(np.cosh(2.0*self._K))
        res = res - integrate.quad(fun, 0.0, np.pi)[0] / (2.0*np.pi)

        return res

    def vbmps_22(self, rho: int, left_fp=None, right_fp=None):
        r'''
        2*2 unit cell varitional boundary MPS
        '''
        
        def _mps_norm(mps):

            le = torch.eye(rho).cdouble()
            for i in range(2):
                le = tp.contract('ab,acd,bed->ce', le, mps[i], mps[i].conj())

            return tp.contract('aa', le)
        
        def _mps_u_cost(mpo, mps, lfp, rfp):

            lfp = tp.contract(
                'abcd,aef,bfgh,chij,dkj->egik',
                lfp, mps[0], mpo[2], mpo[0], mps[0].conj())
            rfp = tp.contract(
                'abc,dcef,gfhi,jki,behk->adgj',
                mps[1], mpo[3], mpo[1], mps[1].conj(), rfp)
            
            val = tp.contract('abcd,abcd', lfp, rfp)
            return val*val.conj()/_mps_norm(mps)
        
        def _mps_d_cost(mpo, mps, lfp, rfp):

            lfp = tp.contract(
                'abcd,aef,bfgh,chij,dkj->egik',
                lfp, mps[0].conj(), mpo[2], mpo[0], mps[0])
            rfp = tp.contract(
                'abc,dcef,gfhi,jki,behk->adgj',
                mps[1].conj(), mpo[3], mpo[1], mps[1], rfp)
            
            val = tp.contract('abcd,abcd', lfp, rfp)
            return val*val.conj()/_mps_norm(mps)

        def _mps_left_canonical(mps):

            new_mps = []
            # from left to right
            temp = mps[0]
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2), (1,)), qr_dims=(1, 0))
            new_mps.append(q)
            temp = tp.contract('ab,bcd->acd', r, mps[1])
            q, r = tp.linalg.tqr(temp, group_dims=((0, 2), (1,)), qr_dims=(1, 0))
            new_mps.append(q)

            return torch.stack(new_mps, dim=0)
        
        def _mps_right_canonical(mps):

            new_mps = []
            # from right to left
            temp = mps[1]
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
            new_mps.append(q)
            temp = tp.contract('abc,bd->adc', mps[0], l)
            q, l = tp.linalg.tqr(temp, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
            new_mps.append(q)

            new_mps = new_mps[::-1]
            return torch.stack(new_mps, dim=0)
        
        def _mv_l(mpo, mps_u, mps_d, lfp):

            # --a--*--e
            #      |f
            # --b--*--g
            #      |h
            # --c--*--i
            #      |j
            # --d--*--k
            lfp = tp.contract('abcd,aef,bfgh,chij,dkj->egik', lfp, mps_u[0], mpo[2], mpo[0], mps_d[0])
            lfp = tp.contract('abcd,aef,bfgh,chij,dkj->egik', lfp, mps_u[1], mpo[3], mpo[1], mps_d[1])
            
            return lfp

        def _mv_r(mpo, mps_u, mps_d, rfp):

            # a--*--b
            #    |c
            # d--*--e
            #    |f
            # g--*--h
            #    |i
            # j--*--k
            rfp = tp.contract('abc,dcef,gfhi,jki,behk->adgj', mps_u[1], mpo[3], mpo[1], mps_d[1], rfp)
            rfp = tp.contract('abc,dcef,gfhi,jki,behk->adgj', mps_u[0], mpo[2], mpo[0], mps_d[0], rfp)
            
            return rfp

        def _mps_u_solver(le, re, col_mpo, initial_tensor=None):
            r'''
            solve a MPS tensor, only work for 2*2 unit cell
            '''

            rho = le.shape[0]
            def _mv(v):
                t = torch.from_numpy(v.reshape(rho, rho, self._phys_dim)).cdouble()
                # --a     j--
                #      |e
                # --b--*--f--
                #      |g
                # --c--*--h--
                #      |i
                # --d     k--
                w = tp.contract('abcd,befg,cghi,jfhk,aje->dki', le, col_mpo[0], col_mpo[1], re, t)

                return w.flatten().numpy()

            initial_v = None
            if initial_tensor is not None:
                initial_v = initial_tensor.flatten().numpy()
            dim_op = (rho**2)*self._phys_dim
            op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
            vals, vecs = scipy.sparse.linalg.eigs(
                op, k=2, which='LM', v0=initial_v, maxiter=None,
                return_eigenvectors=True)
            inds = abs(vals).argsort()[::-1]
            sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]

            return sorted_vals[0], torch.from_numpy(sorted_vecs[:, 0]).reshape(rho, rho, self._phys_dim)
        
        def _mps_d_solver(le, re, col_mpo, initial_tensor=None):
            r'''
            solve a MPS tensor, only work for 2*2 unit cell
            '''

            rho = le.shape[0]
            def _mv(v):
                t = torch.from_numpy(v.reshape(rho, rho, self._phys_dim)).cdouble()
                # --a     j--
                #      |e
                # --b--*--f--
                #      |g
                # --c--*--h--
                #      |i
                # --d     k--
                w = tp.contract('abcd,befg,cghi,jfhk,dki->aje', le, col_mpo[0], col_mpo[1], re, t)

                return w.flatten().numpy()

            initial_v = None
            if initial_tensor is not None:
                initial_v = initial_tensor.flatten().numpy()
            dim_op = (rho**2)*self._phys_dim
            op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
            vals, vecs = scipy.sparse.linalg.eigs(
                op, k=2, which='LM', v0=initial_v, maxiter=None, return_eigenvectors=True)
            inds = abs(vals).argsort()[::-1]
            sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]
            # sorted_vals, sorted_vecs = vals, vecs

            return sorted_vals[0], torch.from_numpy(sorted_vecs[:, 0]).reshape(rho, rho, self._phys_dim)

        def _mps_u_sweep(lfp, rfp, mpo, mps):

            val, err = 1.0, 1.0
            cost = 1.0
            vals = []
            n = 0
            while err > 1E-12 or n < 10:
                vals = []
                # bring MPS to right canonical
                q, l = tp.linalg.tqr(mps[1], group_dims=((1, 2), (0,)), qr_dims=(0, 1))
                mps[1] = q
                temp = tp.contract('abc,bd->adc', mps[0], l)
                le = lfp
                re = tp.contract('abc,dcef,gfhi,jki,behk->adgj', q, mpo[3], mpo[1], q.conj(), rfp)
                v, mps[0] = _mps_u_solver(le, re, col_mpo=(mpo[2], mpo[0]), initial_tensor=temp)
                vals.append(v)
                q, r = tp.linalg.tqr(mps[0], group_dims=((0, 2), (1,)), qr_dims=(1, 0))
                mps[0] = q
                temp = tp.contract('ab,bcd->acd', r, mps[1])
                le = tp.contract('abcd,aef,bfgh,chij,dkj->egik', lfp, q, mpo[2], mpo[0], q.conj())
                re = rfp
                v, mps[1] = _mps_u_solver(le, re, col_mpo=(mpo[3], mpo[1]), initial_tensor=temp)
                vals.append(v)

                # err = abs(sum(vals)-val)
                new_cost = _mps_d_cost(mpo, mps, lfp, rfp).item()
                err = abs(cost-new_cost)
                cost = new_cost
                n += 1

                if n > 100:
                    break

            print('u sweep:', n, err, cost)
            return vals, mps

        def _mps_d_sweep(lfp, rfp, mpo, mps):

            val, err = 1.0, 1.0
            cost = 1.0
            vals = []
            n = 0
            while err > 1E-12 or n < 10:
                vals = []
                q, l = tp.linalg.tqr(mps[1], group_dims=((1, 2), (0,)), qr_dims=(0, 1))
                mps[1] = q
                temp = tp.contract('abc,bd->adc', mps[0], l)
                le = lfp
                re = tp.contract('abc,dcef,gfhi,jki,behk->adgj', q.conj(), mpo[3], mpo[1], q, rfp)
                v, mps[0] = _mps_d_solver(le, re, col_mpo=(mpo[2], mpo[0]), initial_tensor=temp)
                vals.append(v)

                q, r = tp.linalg.tqr(mps[0], group_dims=((0, 2), (1,)), qr_dims=(1, 0))
                mps[0] = q
                temp = tp.contract('ab,bcd->acd', r, mps[1])
                le = tp.contract('abcd,aef,bfgh,chij,dkj->egik', lfp, q.conj(), mpo[2], mpo[0], q)
                re = rfp
                v, mps[1] = _mps_d_solver(le, re, col_mpo=(mpo[3], mpo[1]), initial_tensor=temp)
                vals.append(v)

                new_cost = _mps_d_cost(mpo, mps, lfp, rfp).item()
                err = abs(cost-new_cost)
                cost = new_cost
                n += 1

                if n > 100:
                    break

            print('d sweep:', n, err, cost)
            return vals, mps
     
        mpo = [self._site_tensor.cdouble()]*4
        # randomly initialize up and down boundary MPS
        #    2
        #    |
        # 0--*--1
        mps = []
        for i in range(2):
            mps.append(torch.rand(rho, rho, self._phys_dim).cdouble())
        mps = torch.stack(mps, dim=0)
        # up and down MPS
        mps_u, mps_d = mps.clone(), mps.clone()
        # print(_mps_norm(mps))
        
        # initial left, right-fixed points  for up and down MPS
        if left_fp is None:
            fp_shape = (rho, self._phys_dim, self._phys_dim, rho)
            fp = torch.rand(fp_shape).cdouble()
            fp_ul, fp_ur = fp.clone(), fp.clone()
        else:
            fp_ul, fp_ur = left_fp, right_fp
        if right_fp is None:
            fp_shape = (rho, self._phys_dim, self._phys_dim, rho)
            fp = torch.rand(fp_shape).cdouble()
            fp_dl, fp_dr = fp.clone(), fp.clone()
        else:
            fp_dl, fp_dr = left_fp, right_fp

        fp_ul, fp_ur = fp_ul/fp_ul.norm(), fp_ur/fp_ur.norm()
        fp_dl, fp_dr = fp_dl/fp_dl.norm(), fp_dr/fp_dr.norm()
        # symmetrize
        # fp_ul, fp_ur = 0.5*(fp_ul+fp_ur.conj()), 0.5*(fp_ul.conj()+fp_ur)
        # fp_dl, fp_dr = 0.5*(fp_dl+fp_dr.conj()), 0.5*(fp_dl.conj()+fp_dr)


        # variational up and down MPS
        num_fp_iter = 10
        err_u, err_d = 1.0, 1.0
        cu, cd = 1.0, 1.0
        # for i in range(num_vmps):
        n = 0
        while err_u > 1E-10 or err_d > 1E-10 or n < 10:
            # power method to find a approximated fixed point
            # bring MPS to a left and right cannonical forms
            mps_ulc, mps_urc = _mps_left_canonical(mps_u), _mps_right_canonical(mps_u)
            mps_dlc, mps_drc = _mps_left_canonical(mps_d), _mps_right_canonical(mps_d)
            # iterate the environments
            for j in range(num_fp_iter):
                fp_ul, fp_ur = _mv_l(mpo, mps_ulc, mps_ulc.conj(), fp_ul), _mv_r(mpo, mps_urc, mps_urc.conj(), fp_ur)
                fp_ul, fp_ur = fp_ul/fp_ul.norm(), fp_ur/fp_ur.norm()
                fp_dl, fp_dr = _mv_l(mpo, mps_dlc.conj(), mps_dlc, fp_dl), _mv_r(mpo, mps_drc.conj(), mps_drc, fp_dr)
                fp_dl, fp_dr = fp_dl/fp_dl.norm(), fp_dr/fp_dr.norm()

            # fp_ul, fp_ur = 0.5*(fp_ul+fp_ur.conj()), 0.5*(fp_ul.conj()+fp_ur)
            # fp_dl, fp_dr = 0.5*(fp_dl+fp_dr.conj()), 0.5*(fp_dl.conj()+fp_dr)

            # solve the upper boundary MPS
            # sweep
            vals_u, mps_u = _mps_u_sweep(fp_ul, fp_ur, mpo, mps_u)
            vals_d, mps_d = _mps_d_sweep(fp_dl, fp_dr, mpo, mps_d)

            cost_u, cost_d = _mps_u_cost(mpo, mps_u, fp_ul, fp_ur).item(), _mps_d_cost(mpo, mps_d, fp_dl, fp_dr).item()
            print(n, err_u, err_d, cost_u, cost_d)
            # new_cu, new_cd = sum(vals_u), sum(vals_d)
            new_cu, new_cd = cost_u, cost_d
            # err_u, err_d = abs(new_cu.item()-cu), abs(new_cd.item()-cd)
            err_u, err_d = abs((new_cu-cu)/cu), abs((new_cd-cd)/cu)
            cu, cd = new_cu, new_cd
                # print(n, 'U cost:', cu, _mps_norm(mps_u), vals_u)
                # print(n, 'D cost:', cd, _mps_norm(mps_d), vals_d)
            n += 1

        # find the left and right environments
        # initial enironments
        mps_ulc, mps_urc = _mps_left_canonical(mps_u), _mps_right_canonical(mps_u)
        mps_dlc, mps_drc = _mps_left_canonical(mps_d), _mps_right_canonical(mps_d)
        err_l, err_r = 1.0, 1.0
        le, re = torch.rand(fp_shape).cdouble(), torch.rand(fp_shape).cdouble()
        # le, re = 0.5*(fp_ul+fp_dl), 0.5*(fp_ur+fp_dr)
        n = 0
        while err_l > 1E-14 or err_r > 1E-14 or n < 1E+2:
            new_le, new_re = _mv_l(mpo, mps_ulc, mps_dlc, le), _mv_r(mpo, mps_urc, mps_drc, re)
            new_le, new_re = new_le/new_le.norm(), new_re/new_re.norm()
            err_l, err_r = (new_le-le).norm(), (new_re-re).norm()
            if 0 == n % 10:
                print(n, err_l, err_r)
            le, re = new_le, new_re
            n += 1

            if n > 1E+3:
                break

        return mps_u, mps_d, le, re
    
    def itps_contraction(self, mpo, mps_u, mps_d, le, re):

        le = tp.contract('abcd,aef,bfgh,chij,dkj->egik', le, mps_u[0], mpo[2], mpo[0], mps_d[0])
        re = tp.contract('abc,dcef,gfhi,jki,behk->adgj', mps_u[1], mpo[3], mpo[1], mps_d[1], re)

        return tp.contract('abcd,abcd', le, re)
    
    def vbmps_bond_energy(self, mps_u, mps_d, le, re):

        # impure bond weight matrix
        w = [
            [np.exp(-self._beta*self._J), np.exp(self._beta*self._J)],
            [np.exp(self._beta*self._J), np.exp(-self._beta*self._J)]]
        u, s, v = torch.linalg.svd(torch.tensor(w))
        pure_m, pure_mp = u@torch.sqrt(s).diag(), torch.sqrt(s).diag()@v

        # impure bond weight matrix
        w = [
            [self._J*np.exp(-self._beta*self._J), -self._J*np.exp(self._beta*self._J)],
            [-self._J*np.exp(self._beta*self._J), self._J*np.exp(-self._beta*self._J)]]
        u, s, v = torch.linalg.svd(torch.tensor(w))
        impure_m, impure_mp = u@torch.sqrt(s).diag(), torch.sqrt(s).diag()@v

        pure_ts = [self._site_tensor.cdouble()]*4
        nor = self.itps_contraction(pure_ts, mps_u, mps_d, le, re)

        res = []
        impure_ts = deepcopy(pure_ts)
        impure_ts[0] = tp.contract('as,sb,sc,ds->abcd', pure_mp, pure_m, impure_m, pure_mp).cdouble()
        impure_ts[1] = tp.contract('as,sb,sc,ds->abcd', impure_mp, pure_m, pure_m, pure_mp).cdouble()
        val = self.itps_contraction(impure_ts, mps_u, mps_d, le, re)
        res.append(val/nor)
        impure_ts = deepcopy(pure_ts)
        impure_ts[2] = tp.contract('as,sb,sc,ds->abcd', pure_mp, pure_m, impure_m, pure_mp).cdouble()
        impure_ts[3] = tp.contract('as,sb,sc,ds->abcd', impure_mp, pure_m, pure_m, pure_mp).cdouble()
        val = self.itps_contraction(impure_ts, mps_u, mps_d, le, re)
        res.append(val/nor)
        impure_ts = deepcopy(pure_ts)
        impure_ts[0] = tp.contract('as,sb,sc,ds->abcd', pure_mp, impure_m, pure_m, pure_mp).cdouble()
        impure_ts[2] = tp.contract('as,sb,sc,ds->abcd', pure_mp, pure_m, pure_m, impure_mp).cdouble()
        val = self.itps_contraction(impure_ts, mps_u, mps_d, le, re)
        res.append(val/nor)
        impure_ts = deepcopy(pure_ts)
        impure_ts[1] = tp.contract('as,sb,sc,ds->abcd', pure_mp, impure_m, pure_m, pure_mp).cdouble()
        impure_ts[3] = tp.contract('as,sb,sc,ds->abcd', pure_mp, pure_m, pure_m, impure_mp).cdouble()
        val = self.itps_contraction(impure_ts, mps_u, mps_d, le, re)
        res.append(val/nor)

        return torch.tensor(res)
    
    def vbmps_free_energy(self, mps_u, mps_d, le, re):
        pass
