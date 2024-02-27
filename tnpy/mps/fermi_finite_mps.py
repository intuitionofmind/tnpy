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

class FermiMPS(object):
    r'''
    class of ferminoic matrix product state
    '''

    def __init__(self, tensors: list, info=None):
        r'''
        #    |
        #  --*--
        '''

        self._size = len(tensors)
        self._tensors = tensors

    @property
    def tensors(self):
        return self._tensors

    @property
    def size(self):
        return self._size

    def test(self):
        print('test')

        return 1

    @classmethod
    def rand(cls, n: int, dual: tuple, shape: tuple, cflag=True):
        r'''
        randomly generate a fMPS

        Parameters
        ----------
        n: int, size of the fMPS
        dual: tuple[int],
        shape, tuple[tuple],
        '''

        gts = []
        for i in range(n):
            gts.append(GTensor.rand(dual, shape, cflag))

        return cls(gts)

    @classmethod
    def rand_obc(cls, n: int, dual: tuple, max_shape: tuple, cflag=True):
        r'''
        randomly generate a fMPS with open boundary condition

        Parameters
        ----------
        n: int, size of the fMPS
        dual: tuple[int],
        max_shape, tuple[tuple], the max
        '''

        mid = n // 2
        shapes = []
        dim_phys = 2
        gts = []
        for i in range(n):
            if i < mid:
                dim_alpha = min(dim_phys**i, max_shape[0]), min(dim_phys**i, max_shape[1])
                dim_beta =  min(dim_phys**(i+1), max_shape[0]), min(dim_phys**(i+1), max_shape[1])
                gt_shape = dim_alpha, dim_beta, (2, 2)
            else:
                dim_alpha = min(dim_phys**(n-i), max_shape[0]), min(dim_phys**(n-i), max_shape[1])
                dim_beta =  min(dim_phys**(n-i-1), max_shape[0]), min(dim_phys**(n-i-1), max_shape[1])
                gt_shape = dim_alpha, dim_beta, (2, 2)
            gts.append(GTensor.rand(dual, shape=gt_shape, cflag=cflag))

        return cls(gts)

    def left_canonical(self):
        r'''
        left canonicalize a fermionic MPS

        Parameters
        ----------
        mps: list[GTensor], the fermionic MPS
        '''

        new_tensors = []
        temp = self._tensors[0]
        q, r = tp.linalg.gtqr(temp, group_dims=((0, 2), (1,)), qr_dims=(1, 0))
        new_tensors.append(q)
        for i in range(1, self._size):
            temp = tp.gcontract('ab,bcd->acd', r, self._tensors[i])
            q, r = tp.linalg.gtqr(temp, group_dims=((0, 2), (1,)), qr_dims=(1, 0))
            new_tensors.append(q)

        return FermiMPS(new_tensors)

    def right_canonical(self):
        r'''
        right canonicalize a fermionic MPS

        Parameters
        ----------
        mps: list[GTensor], the fermionic MPS
        '''

        new_tensors = []
        temp = self._tensors[-1]
        q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
        new_tensors.append(q)
        for i in range(self._size-2, -1, -1):
            temp = tp.gcontract('abc,be->aec', self._tensors[i], l)
            q, l = tp.linalg.super_gtqr(temp, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
            new_tensors.append(q)

        new_tensors.reverse()

        return FermiMPS(new_tensors)

    def dagger(self):
        r'''
        return the conjugated FermiMPS
        '''

        # conjugated MPS
        tensors_dagger = []
        for t in self._tensors:
            # graded conjugate
            # tensors_dagger.append(t.graded_conj(free_dims=(1,), side=0))
            tensors_dagger.append(t.conj())

        return FermiMPS(tensors_dagger)

    @staticmethod
    def inner_product(fmps_0, fmps_1):
        r'''
        inner product of two FermiMPSs: <Psi_0|Psi_1>
        --*--*--*-- <Psi_0|
          |  |  |
        --*--*--*-- |Psi_1>
        '''

        assert fmps_0.size == fmps_1.size
        size = fmps_0.size
        cflag = fmps_0.tensors[0].cflag

        # contract from left to right
        # if normal trace (1, 0)
        if 1 == fmps_0.tensors[0].dual[0] and 0 == fmps_1.tensors[0].dual[0]:
            temp = GTensor.eye(dual=(0, 1), shape=(fmps_0.tensors[0].shape[0], fmps_1.tensors[0].shape[0]), cflag=cflag)
        # if supertrace (0, 1)
        elif 0 == fmps_0.tensors[0].dual[0] and 1 == fmps_1.tensors[0].dual[0]:
            temp = GTensor.fermion_parity_operator(dual=(1, 0), shape=(fmps_0.tensors[0].shape[0], fmps_1.tensors[0].shape[0]), cflag=cflag)
        else:
            raise TypeError('MPS GTensors dual are not matched (left virtural bond)')

        for i in range(size):
            # if normal trace (1, 0)
            # a,a--*--c
            #      |d
            # b,b--*--e
            if 1 == fmps_0.tensors[i].dual[2] and 0 == fmps_1.tensors[i].dual[2]:
                # print(i, temp.dual, fmps_0.tensors[i].dual, fmps_1.tensors[i].dual)
                temp = tp.gcontract('ab,acd,bed->ce', temp, fmps_0.tensors[i], fmps_1.tensors[i])
            # if supertrace (0, 1)
            # a,a--*--c
            #      |d,e
            # b,b--*--f
            elif 0 == fmps_0.tensors[i].dual[2] and 1 == fmps_1.tensors[i].dual[2]:
                fp = GTensor.fermion_parity_operator(dual=(1, 0), shape=(fmps_0.tensors[i].shape[2], fmps_1.tensors[i].shape[2]), cflag=cflag)
                # print(i, temp.dual, fmps_0.tensors[i].dual, fmps_1.tensors[i].dual)
                temp = tp.gcontract('ab,acd,de,bfe->cf', temp, fmps_0.tensors[i], fp, fmps_1.tensors[i])
            else:
                raise TypeError('MPS GTensors dual are not matched (physical bond)')

        if 0 == temp.dual[0] and 1 == temp.dual[1]:
            fp = GTensor.fermion_parity_operator(dual=(1, 0), shape=(temp.shape[0], temp.shape[1]), cflag=cflag)
            res = tp.gcontract('ab,ab->', temp, fp)
        elif 1 == temp.dual[0] and 0 == temp.dual[1]:
            res = tp.gcontract('aa->', temp)

        return res

    @staticmethod
    def fidelity(fmps_0, fmps_1):
        r'''
        compute the fidelity between two fMPSs: <Psi_0|Psi_1>*<Psi_1|Psi_0>/<Psi_0|Psi_0>
        '''

        temp = fmps_0.inner_product(fmps_0.dagger(), fmps_1)

        return temp*temp.conj()/fmps_0.inner_product(fmps_0.dagger(), fmps_0)

    def onesite_solver(self, le, re, le_prime, re_prime, m, init_gt=None):
        r'''
        solver

        Parameters
        ----------
        le: GTensor, left environment
        re: GTensor, right environment
        le_prime: GTensor,
        re_prime: GTensor,
        m: GTensor, onsite reference tensor
        '''

        # shape and dual of A is determined by le, re and m
        a_shape = le.shape[1], re.shape[1], m.shape[2]
        a_whole_shape = tuple([sum(d) for d in a_shape])
        a_dual = le.dual[1] ^ 1, re.dual[1] ^ 1, m.dual[2] ^ 1

        def _mv(v):
            r'''
            define the tensor multiplication method
            '''

            t = torch.from_numpy(v.reshape(a_whole_shape)).cdouble()
            # build GTensor from such a dense tensor
            gt = tp.GTensor.extract_blocks(t, a_dual, a_shape)
            # ----b    d    e----, A
            #          |
            # ----a,a--*--c,c----, \bar{M}
            temp = tp.gcontract('ab,acd,ce,bed->', le, m.conj(), re, gt)
            # ----b,b--*--c,c----, M
            #          |d
            # ----a         e----, \bar{A}
            w = temp*tp.gcontract('ab,bcd,ec->aed', le_prime, m, re_prime)

            return w.push_blocks().numpy().flatten()

        # construct the linear operator
        dim_op = math.prod(a_whole_shape)
        print(dim_op)
        op = scipy.sparse.linalg.LinearOperator(shape=(dim_op, dim_op), matvec=_mv)
        if init_gt is None:
            init_v = tp.GTensor.rand(dual=a_dual, shape=a_shape).push_blocks().numpy().flatten()
        else:
            init_v = init_gt.push_blocks().numpy().flatten()

        vals, vecs = scipy.sparse.linalg.eigs(
            op, k=3, which='LM', v0=init_v, maxiter=None, return_eigenvectors=True)
        inds = abs(vals).argsort()[::-1]
        sorted_vals, sorted_vecs = vals[inds], vecs[:, inds]
        # construct GTensor from these results
        eig_t = torch.from_numpy(sorted_vecs[:, 0].reshape(a_whole_shape))
        eig_gt = tp.GTensor.extract_blocks(eig_t, a_dual, a_shape)

        return sorted_vals[0], eig_gt

    def sweep_lr(self, fmps):
        r'''
        sweep from left to right

        Parameters
        ----------
        fmps: fMPS, the reference MPS state to max the fidelity
        '''

        # build left environments in priority
        re_shape = fmps.tensors[-1].shape[1], self._tensors[-1].shape[1]
        # a fermion parity operator should be replendished
        re = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=re_shape, cflag=True)
        re_prime = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=re_shape, cflag=True)
        right_envs, right_envs_prime = [re,], [re_prime,]
        for s in range(self._size-2, -1, -1):
            # a--*--b, A
            #    |c
            # d--*--e, \bar{M}
            # print(s, self._tensors[s+1].dual, fmps.tensors[s+1].conj().dual, re.dual)
            re = tp.gcontract('abc,dec,eb->da', self._tensors[s+1], fmps.tensors[s+1].conj(), re)
            right_envs.append(re)
            # a--*--b, M
            #    |c
            # d--*--e, \bar{A}
            re_prime = tp.gcontract('abc,dec,eb->da', fmps.tensors[s+1], self._tensors[s+1].conj(), re_prime)
            right_envs_prime.append(re_prime)

        # reverse
        right_envs.reverse()
        right_envs_prime.reverse()

        le_shape = fmps.tensors[0].shape[0], self._tensors[0].shape[0]
        le = tp.GTensor.eye(dual=(0, 1), shape=le_shape, cflag=True)
        le_prime = tp.GTensor.eye(dual=(0, 1), shape=le_shape, cflag=True)
        for s in range(self._size):
            re, re_prime = right_envs[s], right_envs_prime[s]
            # solve the effective eigenvalue problem
            val, gt = self.onesite_solver(le, re, le_prime, re_prime, fmps.tensors[s], init_gt=self._tensors[s])
            if s < (self._size-1):
                q, r = tp.linalg.gtqr(gt, group_dims=((0, 2), (1,)), qr_dims=(1, 0))
                self._tensors[s] = q
                self._tensors[s+1] = tp.gcontract('ab,bcd->acd', r, self._tensors[s+1])
            else:
                self._tensors[s] = gt

            # update left environments
            le = tp.gcontract('ab,acd,bed->ce', le, self._tensors[s], fmps.tensors[s].conj())
            le_prime = tp.gcontract('ab,acd,bed->ce', le_prime, fmps.tensors[s], self._tensors[s].conj())

        return 1

    def sweep_rl(self, fmps):
        r'''
        sweep from left to right

        Parameters
        ----------
        fmps: fMPS, the reference MPS state
        '''

        # build left environments in the first place
        le_shape = fmps.tensors[0].shape[0], self._tensors[0].shape[0]
        le = tp.GTensor.eye(dual=(0, 1), shape=le_shape, cflag=True)
        le_prime = tp.GTensor.eye(dual=(0, 1), shape=le_shape, cflag=True)
        left_envs, left_envs_prime = [le,], [le_prime,]
        for s in range(1, self._size):
            # b--*--b, A
            #    |c
            # a--*--e, \bar{M}
            le = tp.gcontract('ab,acd,bed->ce', le, self._tensors[s], fmps.tensors[s].conj())
            left_envs.append(le)
            le_prime = tp.gcontract('ab,acd,bed->ce', le_prime, fmps.tensors[s], self._tensors[s].conj())
            left_envs_prime.append(le_prime)

        re_shape = fmps.tensors[-1].shape[1], self._tensors[-1].shape[1]
        re = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=re_shape, cflag=True)
        re_prime = tp.GTensor.fermion_parity_operator(dual=(1, 0), shape=re_shape, cflag=True)
        for s in range(self._size-2, -1, -1):
            le, le_prime = left_envs[s], left_envs_prime[s]
            val, gt = self.onesite_solver(self, le, re, le_prime, re_prime, fmps.tensors[s], init_gt=self._tensors[s])
            if s > 0:
                q, l = tp.linalg.super_gtqr(gt, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
                self._tensors[s] = q
                self._tensors[s-1] = tp.gcontract('abc,bd->adc', self._tensors[s-1], l)
            else:
                self._tensors[s] = gt

    def max_fidelity(self, fmps, err=1E-10):
        r'''
        max the fidelity of this fMPS with another reference one

        Parameters
        ----------
        fmps: fMPS, the reference MPS state
        '''

        fid = self.fidelity(self, fmps)
        print(fid)
        diff = 1.0
        n = 0
        while diff > err and n < 10:
            self.sweep_lr(fmps)
            self.sweep_rl(fmps)
            new_fid = self.fidelity(self, fmps)
            diff = abs(new_fid-fid)
            fid = new_fid
            print(n, fid)
            n += 1

        return 1
