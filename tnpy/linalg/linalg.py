import math
import torch
torch.set_default_dtype(torch.float64)
import scipy

from .df_svd import SVD, diff_svd

svd = diff_svd.apply

def _svd(mat, full_matrices=False):
    r'''
    robust SVD
    also refer to: https://tenpy.github.io/reference/tenpy.linalg.svd_robust.html
    '''

    flag = False 
    if full_matrices is not None:
        flag = full_matrices

    try:
        u, s, v = torch.linalg.svd(mat, full_matrices=flag)

    except torch._C._LinAlgError:
        u, s, v = scipy.linalg.svd(mat, full_matrices=flag, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
        u, s, v = torch.tensor(u), torch.tensor(s), torch.tensor(v)

    return u, s, v

def tsvd(input_ten, group_dims: tuple, svd_dims=None, cut_off=None) -> tuple:
    r'''
    tensor SVD
    SVD a tensor T to T = A S B

    Parameters
    ----------
    input_ten: tensor
    group_dims: tuple[tuple], tuple[0] bonds going to A; tuple[1] bonds going to B
    svd_dims: tuple[int], positions of new SVD bonds in A and B
            if svd_dims[0] > A.ndim, then it will be place at the end
            similar to B
    cut_off: int, optional, if a trunction on singular value spectrum is required

    Returns
    -------
    u_ten: tensor, isometric
    s: 1d tensor, singular values
    v_ten: tensor, isometric

    Examples
    --------
    #    b         d            e
    #    |         |            |
    # a--*--d = c--*--P,P--Q,Q--*-b
    #   / \        |           
    #  d   c       a

    T_{abcde} -> \sum_{PQ} A_{cdPa} S_{PQ} B_{ebQ}

    Indeed we have three steps to accomplish this:
    step #0: T_{abcde} -> T_{cdaeb}
    step #1: T_{cdaeb} -> \sum_{PQ} A_{cdaP} S_{PQ} B_{Qeb}
    step #2: A_{cdaP} -> A_{cdPa}; B_{Qeb} -> B_{ebQ}

    Jupyter notebook:

    t = torch.rand(4, 8, 16, 32, 16)
    %timeit -r 5 -n 100 a, s, b = tensor_svd(t, group_dims=((2, 3, 0), (4, 1)), svd_dims=(2, 2))
    print(a.shape, s.shape, b.shape)
    res = pytn.contract('abcd,cg,efg->dfabe', a, s.diag(), b)
    print('Test tensor SVD:', torch.linalg.norm(t-res))
    '''

    # permute to the operation order
    temp_ten = torch.permute(input_ten, group_dims[0]+group_dims[1])
    new_shape = temp_ten.shape
    # divide
    divide = len(group_dims[0])
    # reshape to a matrix
    # Python>=3.8
    mat_shape = math.prod(new_shape[:divide]), math.prod(new_shape[divide:])
    mat = torch.reshape(temp_ten, mat_shape)
    # SVD
    u, s, v = svd(mat)
    svd_dim = min(mat_shape)
    # if SVD truncation is needed
    if (cut_off is not None) and (cut_off < svd_dim):
        svd_dim = cut_off
    # new shapes appended with SVD indices
    # column vectors of U and rows of V are orthogonal
    u_shape, v_shape = new_shape[:divide]+(svd_dim,), (svd_dim,)+new_shape[divide:]
    u_ten, v_ten = torch.reshape(u[:, :svd_dim], u_shape), torch.reshape(v[:svd_dim, :], v_shape)

    if svd_dims is not None:
        u_dims, v_dims = list(range(u_ten.ndim)), list(range(v_ten.ndim))
        u_dims.insert(svd_dims[0], u_dims.pop(-1))
        v_dims.insert(svd_dims[1], v_dims.pop(0))
        u_ten = torch.permute(u_ten, u_dims)
        v_ten = torch.permute(v_ten, v_dims)

    return u_ten, s[:svd_dim], v_ten

def tqr(input_ten, group_dims: tuple, qr_dims=None) -> tuple:
    r''' 
    QR decomposition of a tensor
    T = QR
    Q is the isometric tensor, R is the residual tensor

    Parameters
    ----------
    input_ten: torch.tensor
    group_dims: tuple[tuple], tuple[0]: bonds of Q; tuple[1]: bonds of R
    qr_dims: tuple[int], optional, where the new QR bonds in Q and R
        Defalut: put two new QR bonds at the tail of Q and head of R, respectively

    Returns
    -------
    q_ten: tensor, isometric tensor
    r_ten: tensor, residual tensor
    '''

    # permute to the operation order
    temp_ten = torch.permute(input_ten, group_dims[0]+group_dims[1])
    new_shape = temp_ten.shape
    divide = len(group_dims[0])
    # Python>=3.8
    mat_shape = math.prod(new_shape[:divide]), math.prod(new_shape[divide:])
    mat = torch.reshape(temp_ten, mat_shape)
    q, r = torch.linalg.qr(mat, mode='reduced')
    qr_dim = min(mat_shape)
    q_shape, r_shape = new_shape[:divide]+(qr_dim,), (qr_dim,)+new_shape[divide:]
    q_ten, r_ten = torch.reshape(q, q_shape), torch.reshape(r, r_shape)

    # put the qr_dims in the desired order
    if qr_dims is not None:
        q_dims, r_dims = list(range(q_ten.ndim)), list(range(r_ten.ndim))
        q_dims.insert(qr_dims[0], q_dims.pop(-1))
        r_dims.insert(qr_dims[1], r_dims.pop(0))
        q_ten, r_ten = torch.permute(q_ten, q_dims), torch.permute(r_ten, r_dims)

    return q_ten, r_ten
