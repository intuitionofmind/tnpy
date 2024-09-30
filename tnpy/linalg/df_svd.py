'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577 
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import scipy
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A):
        # U, S, V = torch.svd(A)
        U, S, V = torch.linalg.svd(A, full_matrices=False)

        ctx.save_for_backward(U, S, V)

        return U, S, V

    @staticmethod
    def backward(ctx, dU, dS, dV):

        U, S, V = ctx.saved_tensors

        V = V.t()
        dV = dV.t()

        Vt = V.t()
        Ut = U.t()

        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        # print((F-G).shape, VdV.shape, VdV.t().shape)
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + dS.diag()) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)

        return dA

class diff_svd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a: torch.tensor):

        # u, s, v = torch.linalg.svd(a, full_matrices=False)
        try:
            u, s, v = torch.linalg.svd(mat, full_matrices=False)

        except torch._C._LinAlgError:
            u, s, v = scipy.linalg.svd(mat, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
            u, s, v = torch.tensor(u), torch.tensor(s), torch.tensor(v)

        ctx.save_for_backward(u, s, v)

        return u, s, v

    @staticmethod
    def backward(ctx, u_bar: torch.tensor, s_bar: torch.tensor, v_bar: torch.tensor):

        u, s, v = ctx.saved_tensors

        # build F matrix
        F = torch.zeros(s.diag().shape)

        for i, j in itertools.product(range(s.shape[0]), range(s.shape[0])):
            if torch.abs(s[i]-s[j]).item() > 1E-12:
                F[i, j] = 1.0 / (s[i]-s[j])

        return u.conj() @ (s_bar.diag()+torch.mul(F.t(), (u.t() @ u_bar)-(v_bar @ v.t()))) @ v.conj()
