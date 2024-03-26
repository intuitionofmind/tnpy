import torch
torch.set_default_dtype(torch.float64)
from torch import Tensor

import pickle
import copy

import scipy
import pytn

class OpenFiniteMPS(object):
    r'''
    class of the finite size MPS
    major applications:
    1. solve a quantum model written in MPO as to converge to its ground state
    2. measure physical quantites written in MPO

    '''

    def __init__(self, size: int, chi_max: int, trun_err: float, c_flag=False):
        r'''

        Parameters
        ----------
        mpo: list, Hamiltonian MPO of a model
        chi_max: int, max bond dimension
        trun_err: float, truncation error
        '''

        self._size = size
        self._chi_max = chi_max
        self._trun_err = trun_err
        self._c_flag = c_flag

        self._mps = []
        self._bond_dims = []

    def rand_right_canonical_mps(self, dim_phys: int):
        r'''
        randomly generate a right cannoical MPS

        Parameters
        ----------
        dim_phys: int, dimension of local physical Hilbert space
        '''

        mid = self._size // 2
        mps = []
        dims = [1,]
        for s in range(self._size):
            #        s
            #        |
            # alpha--*--beta
            # dertermin the bond dimensons
            if s < mid:
                dim_alpha, dim_beta = min(self._chi_max, dim_phys**s), min(self._chi_max, dim_phys**(s+1))
            else:
                dim_alpha, dim_beta = min(self._chi_max, dim_phys**(self._size-s)), min(self._chi_max, dim_phys**(self._size-1-s))

            if self._c_flag:
                u_mat = scipy.stats.unitary_group.rvs(dim_phys*dim_beta)[:dim_alpha]
            else:
                u_mat = scipy.stats.ortho_group.rvs(dim_phys*dim_beta)[:dim_alpha]

            u_ten = u_mat.reshape(dim_alpha, dim_phys, dim_beta)
            mps.append(torch.tensor(u_ten))
            dims.append(dim_beta)

        self._mps = mps
        self._bond_dims = dims

        return 1

    def mps(self):
        return self._mps

    def write_mps(self, file_name: str):
        r'''
        write the MPS list to disk
        '''

        f = open(file_name, 'wb')
        pickle.dump(self._mps, f)
        f.close()

        return 1

    def load_mps(self, file_name: str):
        r'''
        load the MPS
        '''

        try:
            f = open(file_name, 'rb')
            mps = pickle.load(f)
        except Exception:
            print('MPS file does not exist')

        assert self._size == len(mps), 'length of the loaded MPS is not correct'
        self._mps = mps

        return 1

    def import_mps(self, mps: list):
        r'''
        import MPS
        '''
        assert self._size == len(mps), 'length of the imported MPS is not correct'
        self._mps = mps

        return 1

    def bond_dims(self) -> list:
        r'''
        return all the virtual bond dimensions of MPS
        '''

        return self._bond_dims

    def truncation_err(self) -> list:
        r'''
        return the truncation err
        '''

        mps = copy.deepcopy(self._mps)
        dim_phys = mps[0].shape[1]
        errs = []
        for s in range(self._size-1):
            #    1
            #    |
            # 0--*--2
            u, sd, v = pytn.linalg.tensor_svd(mps[s], group_dims=((0, 1), (2,)))
            errs.append(sd[-1].item())
            r = pytn.contract('ab,bc', sd.diag(), v)
            mps[s+1] = pytn.contract('ab,bcd', r, mps[s+1])
        
        return errs

    def onesite_lanczos_solver(self, le: Tensor, mpo_tensor: Tensor, re: Tensor, v_init=None):
        r'''
        solve the local effective Hamitonian given by left-env, mpo and right-env
        return one unrenormalized MPS tensor like
          |
        --*--

        '''

        dim_phys = mpo_tensor.shape[1]
        dim_alpha = le.shape[0]
        dim_beta = re.shape[0]
        dim = dim_alpha*dim_phys*dim_beta

        # inner functions
        def _inner_product(a, b):
            return torch.real(torch.tensordot(a.conj(), b, dims=([0, 1, 2], [0, 1, 2])))

        def _mv(mps_tensor: Tensor):
            r'''
            # local effective tensor operates on a MPS tensor
            # ----a    d    g----
            #          |
            # ----b,b--*--e,e----
            #          |
            # ----c    f,f  h----
            #          |
            #       c--*--h
            '''

            tn = pytn.contract('abc,bdef,geh,cfh', le, mpo_tensor, re, mps_tensor)

            return tn

        if v_init is None:
            v_init = torch.rand(dim_alpha, dim_phys, dim_beta)

        v0 = v_init
        v0 = v0 / torch.sqrt(_inner_product(v0, v0))
        w0_prime = _mv(v0)
        alpha = _inner_product(w0_prime, v0)
        w0 = w0_prime-alpha*v0

        # lanczos basis
        basis_lanczos = []
        basis_lanczos.append(v0)
        dim_lanczos = 1

        # Lanczos triadiagonal matrix
        vec_alpha, vec_beta = [], []
        vec_alpha.append(alpha)
        err = 1.0
        gs_val = 0.0
        while ((err > 1.e-14) and (dim_lanczos < dim)):
            # Gram-Schimit orthogonalization
            for j in range(dim_lanczos):
                w0 -= _inner_product(basis_lanczos[j], w0)*basis_lanczos[j]

            beta = torch.sqrt(_inner_product(w0, w0))
            vec_beta.append(beta)
            v1 = w0 / beta
            basis_lanczos.append(v1)
            dim_lanczos += 1
            w1_prime = _mv(v1)
            alpha = _inner_product(w1_prime, v1)
            vec_alpha.append(alpha)
            w1 = w1_prime-alpha*v1-beta*v0

            # tridiagonal Lanczos matrix
            tri_mat = torch.diag(torch.tensor(vec_alpha))+torch.diag(torch.tensor(vec_beta), 1)+torch.diag(torch.tensor(vec_beta), -1)
            eigvals_lanczos, eigvecs_lanczos = torch.linalg.eigh(tri_mat)
            err = torch.abs(eigvals_lanczos[0]-gs_val)

            # update to next iteration
            gs_val = eigvals_lanczos[0]
            v0 = v1
            w0 = w1

        # ground state
        gs_tensor = torch.zeros(dim_alpha, dim_phys, dim_beta)
        for l in range(dim_lanczos):
            gs_tensor += eigvecs_lanczos[l, 0]*basis_lanczos[l]

        return gs_val, gs_tensor

    def onesite_arpack_solver(self, le: Tensor, mpo_tensor: Tensor, re: Tensor, init_tensor=None):

        dim_phys = mpo_tensor.shape[1]
        dim_alpha = le.shape[0]
        dim_beta = re.shape[0]
        dim = dim_alpha*dim_phys*dim_beta
        if init_tensor is None:
            init_tensor = torch.rand(dim_alpha, dim_phys, dim_beta)

        def _mv(v):
            mps_tensor = torch.reshape(v, (dim_alpha, dim_phys, dim_beta))
            # local effective tensor operates on a MPS tensor
            # ----a    d    g----
            #          |
            # ----b,b--*--e,e----
            #          |
            # ----c    f,f  h----
            #          |
            #       c--*--h
            tn = pytn.contract('abc,bdef,geh,cfh', le, mpo_tensor, re, mps_tensor)

            return torch.flatten(tn)

        op = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=_mv)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
                op, k=2, v0=init_tensor.flatten(),
                which='SA', return_eigenvectors=True)
        gs_tensor = torch.reshape(eigvecs[:, 0], (dim_alpha, dim_phys, dim_beta))

        return eigvals[0], gs_tensor

    def onesite_sweep_lr(self, ham_mpo: list):
        r'''
        DMRG sweep from left to right by one site

        Parameters
        ----------
        ham_mpo: list[Tensor]
        '''

        dim_phys = ham_mpo[0].shape[1]
        # build the right environment tensors in priority
        re = torch.tensor([1.0]).reshape(1, 1, 1)
        right_envs = [re]
        for r in range(self._size-1, 0, -1):
            # iteratively contract mps, mpo and right_env
            # a--*--c,c----
            #    |b,b
            # d--*--e,e----
            #    |f,f
            # g--*--h,h----
            re = pytn.contract(
                    'abc,dbef,gfh,ceh',
                    self._mps[r].conj(), ham_mpo[r], self._mps[r], re)
            right_envs.append(re)

        # left environment can be build during the sweep
        le = torch.tensor([1.0]).reshape(1, 1, 1)
        # begin sweeping
        for s in range(self._size-1):
            k = self._size-s-1
            # old wavefunction as the trial wavefunction
            trial_wf = self._mps[s]

            eig_val, eig_ten = self.onesite_lanczos_solver(
                le, ham_mpo[s], right_envs[k], v_init=trial_wf)
            '''
            # Arpack solver
            eig_val, eig_ten = self.one_site_arpack_solver(
                le, ham_mpo[site], right_envs[k], init_tensor=trial_wf)
            '''
            # QR decomposition to find the isometric part
            #    1
            #    |        |
            # 0--*--2 = --*-- --*--
            q, r = pytn.linalg.tensor_qr(eig_ten, group_dims=((0, 1), (2,)))
            # update this site
            self._mps[s] = q
            # merge the residual R to the next
            self._mps[s+1] = pytn.contract('ab,bcd', r, self._mps[s+1])
            # update the left environment
            # ----a,a--*--e
            #          |d,d
            # ----b,b--*--f
            #          |h,h
            # ----c,c--*--i
            le = pytn.contract(
                'abc,ade,bdfh,chi',
                le, torch.conj(self._mps[s]), ham_mpo[s], self._mps[s])

        return 1

    def onesite_sweep_rl(self, ham_mpo: list):
        r'''
        DMRG sweep from right to left by one site

        Parameters
        ----------
        ham_mpo: list[Tensor]
        '''

        dim_phys = ham_mpo[0].shape[1]
        # build the left environment tensors in priority
        le = torch.tensor([1.0]).reshape(1, 1, 1)
        left_envs = [le]
        for l in range(0, self._size-1):
            # ----a,a--*--e
            #          |d,d
            # ----b,b--*--f
            #          |h,h
            # ----c,c--*--i
            le = pytn.contract(
                'abc,ade,bdfh,chi',
                le, torch.conj(self._mps[l]), ham_mpo[l], self._mps[l])
            left_envs.append(le)

        re = torch.tensor([1.0]).reshape(1, 1, 1)
        for s in range(self._size-1, 0, -1):
            k = s
            trial_wf = self._mps[s]
            eig_val, eig_ten = self.onesite_lanczos_solver(
                left_envs[k], ham_mpo[s], re, v_init=trial_wf)
            '''
            eig_val, eig_ten = self.onesite_arpack_solver(
                left_envs[k], self._mpo._tensors[site], re, init_tensor=trial_wf)
            '''
            # QR decomposition to find the isometric part
            #    1
            #    |              |
            # 0--*--2 = --*-- --*--
            q, r = pytn.linalg.tensor_qr(eig_ten, group_dims=((1, 2), (0,)), qr_dims=(0, 1))
            self._mps[s] = q
            # merge the residual part R to the next
            self._mps[s-1] = pytn.contract('abc,cd', self._mps[s-1], r)
            # update the right environment
            # a--*--c,c----
            #    |b,b
            # d--*--e,e----
            #    |f,f
            # g--*--h,h----
            re = pytn.contract(
                'abc,dbef,gfh,ceh',
                torch.conj(self._mps[s]), ham_mpo[s], self._mps[s], re)

        return 1

    def twosite_lanczos_solver(self, le: Tensor, mpo_tensor_0: Tensor, mpo_tensor_1: Tensor, re: Tensor, v_init=None):
        r'''
        solve the local effective Hamitonian given by left-env, two MPOs and right-env

        Returns
        -------
        gs_tensor: Tensor,
        #    1  2
        #    |  |
        # 0--*--*--3

        gs_value: float,
        '''

        def _twosite_inner_product(a: Tensor, b: Tensor):
            return torch.real(torch.tensordot(a.conj(), b, dims=([0, 1, 2, 3], [0, 1, 2, 3])))

        def _twosite_mv(twosite_mps: Tensor):
            # local effective tensor operates on a MPS tensor
            # local effective tensor
            # ----a    d       g    j----
            #          |       |
            # ----b,b--*--e,e--*--h,h----
            #          |       |
            # ----c    f,f     i,i  k----
            #          |       |
            #       c--*-------*--k
            tn = pytn.contract(
                'abc,bdef,eghi,jhk,cfik',
                le, mpo_tensor_0, mpo_tensor_1, re, twosite_mps)

            return tn

        dim_phys = mpo_tensor_0.shape[1]
        dim_alpha = le.shape[0]
        dim_beta = re.shape[0]
        dim = dim_alpha*dim_phys*dim_phys*dim_beta

        if v_init is None:
            v_init = torch.rand(dim_alpha, dim_phys, dim_phys, dim_beta)

        v0 = v_init
        v0 = v0/torch.sqrt(_twosite_inner_product(v0, v0))
        w0_prime = _twosite_mv(v0)
        alpha = _twosite_inner_product(w0_prime, v0)
        w0 = w0_prime-alpha*v0

        # lanczos basis
        basis_lanczos = []
        basis_lanczos.append(v0)
        dim_lanczos = 1
        # Lanczos triadiagonal matrix
        vec_alpha, vec_beta = [], []
        vec_alpha.append(alpha)
        err, gs_val = 1.0, 0.0
       
        # eigvals_lanczos, eigvecs_lanczos = 0, 0
        while ((err > 1E-14) and (dim_lanczos < dim)):
            # print('Lanczos', err, dim_lanczos, dim)
            # Gram-Schimit orthogonalization
            for j in range(dim_lanczos):
                w0 -= _twosite_inner_product(basis_lanczos[j], w0)*basis_lanczos[j]
            beta = torch.sqrt(_twosite_inner_product(w0, w0))
            vec_beta.append(beta)
            v1 = w0/beta
            basis_lanczos.append(v1)
            dim_lanczos += 1
            w1_prime = _twosite_mv(v1)
            alpha = _twosite_inner_product(w1_prime, v1)
            vec_alpha.append(alpha)
            w1 = w1_prime-alpha*v1-beta*v0

            # tridiagonal Lanczos matrix
            tri_mat = torch.diag(torch.tensor(vec_alpha))+torch.diag(torch.tensor(vec_beta), 1)+torch.diag(torch.tensor(vec_beta), -1)
            eigvals_lanczos, eigvecs_lanczos = torch.linalg.eigh(tri_mat)
            err = torch.abs(eigvals_lanczos[0]-gs_val).item()
            # update to next iteration
            gs_val = eigvals_lanczos[0]
            v0 = v1
            w0 = w1

        # print(dim_lanczos, dim, dim_alpha, dim_phys, dim_beta)
        # ground state
        gs_tensor = torch.zeros(dim_alpha, dim_phys, dim_phys, dim_beta)
        for l in range(dim_lanczos):
            gs_tensor += eigvecs_lanczos[l, 0]*basis_lanczos[l]

        return gs_val, gs_tensor

    def twosite_arpack_solver(self, le: Tensor, mpo_tensor_0: Tensor, mpo_tensor_1: Tensor, re: Tensor, init_tensor=None):

        dim_phys = mpo_tensor_0.shape[1]
        dim_alpha = le.shape[0]
        dim_beta = re.shape[0]
        dim = dim_alpha*dim_phys*dim_phys*dim_beta

        if init_tensor is None:
            init_tensor = torch.rand(dim_alpha, dim_phys, dim_phys, dim_beta)

        # inner function
        def _mv(mps_tensors: Tensor):
            # local effective tensor operates on a MPS tensor
            # local effective tensor
            # ----a    d       g    j----
            #          |       |
            # ----b,b--*--e,e--*--h,h----
            #          |       |
            # ----c    f,f     i,i  k----
            #          |       |
            #       c--*-------*--k
            tn = pytn.contract(
                'abc,bdef,eghi,jhk,cfik',
                le, mpo_tensor_0, mpo_tensor_1, re,
                mps_tensors.reshape(dim_alpha, dim_phys, dim_phys, dim_beta))

            return tn.flatten()

        op = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=_mv)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
                op, k=2, v0=init_tensor.flatten(),
                which='SA', return_eigenvectors=True)
        gs_tensor = torch.reshape(torch.tensor(eigvecs[:, 0]), (dim_alpha, dim_phys, dim_phys, dim_beta))

        return eigvals[0], gs_tensor

    def twosite_sweep_lr(self, ham_mpo: list):
        r'''
        two-site sweep from left to right
        '''

        dim_phys = ham_mpo[0].shape[1]
        # build the right environment tensors in priority
        re = torch.tensor([1.0]).reshape(1, 1, 1)
        right_envs = [re]
        for r in range(self._size-1, 1, -1):
            # iteratively contract mps, mpo and right_env
            # a--*--c,c----
            #    |b,b
            # d--*--e,e----
            #    |f,f
            # g--*--h,h----
            re = pytn.contract(
                'abc,dbef,gfh,ceh',
                torch.conj(self._mps[r]), ham_mpo[r], self._mps[r], re)
            right_envs.append(re)

        # left environment can be build during the sweep
        le = torch.tensor([1.0]).reshape(1, 1, 1)
        # begin sweeping
        for s in range(self._size-1):
            # print(s)
            k = self._size-s-2
            # prepare the initial trial tensor
            #    b       d
            #    |       |
            # a--*--c,c--*--e
            trial_ten = pytn.contract('abc,cde', self._mps[s], self._mps[s+1])
            re = right_envs[k]
            eig_val, eig_ten = self.twosite_lanczos_solver(
                le, ham_mpo[s], ham_mpo[s+1], re, v_init=trial_ten)
            eig_ten = torch.nan_to_num(eig_ten)
            u, sd, v = pytn.linalg.tensor_svd(eig_ten, group_dims=((0, 1), (2, 3)))
            # truncated the Schimidt spectrum at the center
            if s == self._size // 2:
                trun_sd = [v for v in sd if v >= self._trun_err]
                dim_trun = min(self._chi_max, len(trun_sd))
                dim_trun = max(dim_phys, dim_trun)
            else:
                dim_trun = self._bond_dims[self._size // 2]
            # print('twosite sweep', s, len(sd), dim_trun, sd)
            # update tensors
            self._mps[s] = u[:, :, :dim_trun]
            r = pytn.contract('ab,bcd', sd.diag(), v)
            self._mps[s+1] = r[:dim_trun, :, :]
            # update bond dims
            self._bond_dims[s+1] = self._mps[s].shape[2]
            # update left environment
            # ----a,a--*--e
            #          |d,d
            # ----b,b--*--f
            #          |g,g
            # ----c,c--*--h
            le = pytn.contract(
                'abc,ade,bdfg,cgh',
                le, torch.conj(self._mps[s]), ham_mpo[s], self._mps[s])

        return 1

    def twosite_sweep_rl(self, ham_mpo: list):
        r'''
        two-site sweep from right to left
        '''

        dim_phys = ham_mpo[0].shape[1]
        le = torch.tensor([1.0]).reshape(1, 1, 1)
        left_envs = [le]
        for l in range(0, self._size-2):
            # ----a,a--*--e
            #          |d,d
            # ----b,b--*--f
            #          |h,h
            # ----c,c--*--i
            le = pytn.contract(
                'abc,ade,bdfh,chi',
                le, torch.conj(self._mps[l]), ham_mpo[l], self._mps[l])
            left_envs.append(le)

        re = torch.tensor([1.0]).reshape(1, 1, 1)
        for s in range(self._size-1, 0, -1):
            k = s-1
            # prepare the initial trial tensor
            #    b       d
            #    |       |
            # a--*--c,c--*--e
            trial_ten = pytn.contract('abc,cde', self._mps[s-1], self._mps[s])
            eig_val, eig_ten = self.twosite_lanczos_solver(
                left_envs[k], ham_mpo[s-1], ham_mpo[s], re, v_init=trial_ten)
            eig_ten = torch.nan_to_num(eig_ten)
            u, sd, v = pytn.linalg.tensor_svd(eig_ten, group_dims=((0, 1), (2, 3)))
            # truncate the Schmidt spectrum
            if s == self._size // 2:
                trun_sd = [v for v in sd if v >= self._trun_err]
                dim_trun = min(self._chi_max, len(trun_sd))
                dim_trun = max(dim_phys, dim_trun)
            else:
                dim_trun = self._bond_dims[self._size // 2]
            # trun_sd = [v for v in sd if v >= self._trun_err]
            # dim_trun = min(self._chi_max, len(trun_sd))
            # update tensors
            self._mps[s] = v[:dim_trun, :, :]
            l = pytn.contract('abc,cd', u, sd.diag())
            self._mps[s-1] = l[:, :, :dim_trun]
            # update bond dims
            self._bond_dims[s] = self._mps[s].shape[0]
            # update right environment
            # a--*--c,c----
            #    |b,b
            # d--*--e,e----
            #    |f,f
            # g--*--h,h----
            re = pytn.contract(
                'abc,dbef,gfh,ceh',
                torch.conj(self._mps[s]), ham_mpo[s], self._mps[s], re)

        return 1

    def onesite_sweep(self, ham_mpo: list):

        self.onesite_sweep_lr(ham_mpo)
        self.onesite_sweep_rl(ham_mpo)

        return 1

    def twosite_sweep(self, ham_mpo: list):

        self.twosite_sweep_lr(ham_mpo)
        self.twosite_sweep_rl(ham_mpo)

        return 1

    def measure(self, mpo: list) -> float:
        r'''
        measure the physical quantity expressed by MPO under the MPS:
        \langle M^{2} \rangle

        Parameters
        ----------
        mpo: list[Tensor]
        '''

        tn = torch.tensor([1.0]).reshape(1, 1, 1)
        for s in range(self._size):
            # contract tn, mps and mpo
            # ----a,a--*--e 
            #          |d,d
            # ----b,b--*--f
            #          |g,g
            # ----c,c--*--h
            tn = pytn.contract(
                'abc,ade,bdfg,cgh',
                tn, torch.conj(self._mps[s]), mpo[s], self._mps[s])
        
        return torch.real(tn).item()

    def measure_square(self, mpo: list) -> float:
        r'''
        measure the physical quantity M^{2} expressed by MPO under the MPS:
        \langle M^{2} \rangle

        Parameters
        ----------
        mpo: list[Tensor]
        '''

        tn = torch.tensor([1.0]).reshape(1, 1, 1, 1)
        for s in range(self._size):
            # ----a,a--*--f
            #          |e,e
            # ----b,b--*--g
            #          |h,h
            # ----c,c--*--i
            #          |j,j
            # ----d,d--*--k
            tn = pytn.contract(
                'abcd,aef,begh,chij,djk',
                tn, torch.conj(self._mps[s]), mpo[s], mpo[s], self._mps[s])

        return torch.real(tn).item()

    def variance(self, mpo: list):
        r'''
        compute the variance of H expressed by MPO:
        v=\langle{H}^{2}\rangle-\langle{H}\rangle^{2}

        Parameters
        ----------
        mpo: list[Tensor]
        '''

        var = self.measure_square(mpo)-self.measure(mpo)**2

        return var

    def schmidt_spectrum(self):
        r'''
        compute Schmidt spectrum
        as to bring the MPS into the mixed cannoical form

        Returns
        -------
        schmidt_specs: list[Tensor]
        '''

        mps = copy.deepcopy(self._mps)
        dim_phys = mps[0].shape[1]
        schmidt_specs = []
        for s in range(self._size-1):
            #    1
            #    |
            # 0--*--2
            u, sd, v = pytn.linalg.tensor_svd(mps[s], group_dims=((0, 1), (2,)))
            # print(s, sd)
            schmidt_specs.append(sd)
            r = pytn.contract('ab,bc', sd.diag(), v)
            # merged a unitary V, the next site tensor is still unitary/isometric
            mps[s+1] = pytn.contract('ab,bcd', r, mps[s+1])
        
        return schmidt_specs
    
    def entanglement_entropy(self):
        r'''
        compute entanglement entropy

        Returns
        -------
        ees: list[float]
        '''

        schmidt_specs = self.schmidt_spectrum()
        ees = []
        for s in schmidt_specs: 
            sq = s**2
            # set a cut-off
            for j, v in enumerate(sq):
                if v.item() < 1E-99:
                    sq[j] = 1E-99
            e = -1.0*torch.dot(sq, torch.log(sq))
            ees.append(e.item())

        return ees
