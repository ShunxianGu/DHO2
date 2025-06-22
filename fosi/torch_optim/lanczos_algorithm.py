import torch
from torchopt import pytree
from torch.autograd import forward_ad
import torch.distributed as dist
from time import time

lanczos_precision = torch.float64
vec_tree = None  # Place holder


def lanczos_alg(order, loss, k_largest, l_smallest=0, return_precision='32', device=None,  precondition_world_size = None, worker_world_size = None, group = None, rank = None):
    """
    Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
    This function returns a function that performs the Lanczos iterations and can be jitted.
    Args:
        order: an integer corresponding to the number of Lanczos steps to take.
        loss: the loss function used to build the hvp operator (loss function to derive).
        k_largest: an integer corresponding to the required number of the largest eigenvalues and eigenvectors.
        l_smallest: an integer corresponding to the required number of the smallest eigenvalues and eigenvectors.
        return_precision: the algorithm must run in high precision; however, after extracting the eigenvalues and
            eigenvectors we can cast it back to 32/64 bit according to the precision required in return_precision.
    Returns:
        lanczos_alg_jitted: a function that performs the Lanczos algorith and can be jitted.
    """

    # The algorithm must run in high precision; however, after extracting the eigenvalues and eigenvectors we can
    # cast it back to 32 bit.

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    global vec_tree
    vec_tree = None

    # Forward mode.
    # TODO: using torch.nn.MSELoss inside the loss function causes 'RuntimeError: ZeroTensors are immutable' when
    #  calling torch.autograd.grad here.
    def hvp_forward_ad(params, vec, batch):
        global vec_tree
        torch.nn.utils.vector_to_parameters(vec, vec_tree)

        with forward_ad.dual_level():
            dual_params = pytree.tree_map(lambda a, b: forward_ad.make_dual(a, b), params, vec_tree)
            loss_val = loss(dual_params, batch)
            gradient = torch.autograd.grad(loss_val, dual_params)
            # dist.all_reduce(gradient, group = group, op=dist.reduce_op.SUM, async_op=False)
            # gradient /= precondition_world_size
            _, hvp = zip(*[forward_ad.unpack_dual(g) for g in gradient])

        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hessian_vec_prod

    # Fast but inaccurate approximation. Using 32 bit precision results in lanczos_algorithm_sanity.py failure.
    # Using 64 precision in the test works.
    def hvp_finite_diff(params, vec, batch) -> torch.Tensor:
        epsilon = 1e-6
        global vec_tree
        torch.nn.utils.vector_to_parameters(vec, vec_tree)
        input = pytree.tree_map(lambda a, b: a + epsilon * b, params, vec_tree)
        grad_plus = torch.autograd.grad(loss(input, batch), input)
        input = pytree.tree_map(lambda a, b: a - epsilon * b, params, vec_tree)
        grad_minus = torch.autograd.grad(loss(input, batch), input)
        hessian_vec_prod = pytree.tree_map(lambda a, b: (a - b) / (2 * epsilon), grad_plus, grad_minus)
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod])
        return hessian_vec_prod

    # Slow due to backward mode AD constraints. see https://github.com/pytorch/pytorch/issues/24004
    def hvp_backward_ad(params, vec, batch) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        Should not use functorch as it does not support batch norm: https://pytorch.org/functorch/stable/batch_norm.html
        """
        # Compute original gradient, tracking computation graph
        loss_val = loss(params, batch)
        grad_dict = torch.autograd.grad(loss_val, params, create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])

        # Take the second gradient and mult with vec, Hv
        hessian_vec_prod_dict = torch.autograd.grad(grad_vec, params, grad_outputs=vec)
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])
        return hessian_vec_prod

    def orthogonalization(vecs, w, tridiag, i, init_vec):
        # Full reorthogonalization.
        # Note that orthogonalization here includes all vectors in vecs, and not just vectors j s.t. j <= i.
        # Since all vectors j s.t. j > i are zeros (vecs is initialized to zeros), there is no impact on w if iteration
        # continues for j > i.
        # However, using the iteration on all the vectors enables us to use jit over this function.
        # Otherwise, we will have to iterate/slice by i, which is not supported by jit.

        # The operation (vecs.matmul(w)).matmul(vecs) is equivalent to multiply (scale) each vector in its own coeff,
        # where coeffs = vecs.matmul(w) is an array of coeffs (scalars) with shape (order,), and then sum all the
        # scaled vectors.
        w1 =  (vecs.matmul(w))  # single vector with the shape of w
        # repeat the full orthogonalization for stability
        dist.all_reduce(w1, group = group, op=dist.ReduceOp.SUM, async_op=False)
        w -= w1.matmul(vecs) 
        w1 =  (vecs.matmul(w))
        dist.all_reduce(w1, group = group, op=dist.ReduceOp.SUM, async_op=False)
        w -= w1.matmul(vecs)

        init_vec.zero_()
        init_vec[int((rank - int(worker_world_size)) * (split_num)):int((rank - int(worker_world_size)+1) * (split_num))] =  w 
        dist.all_reduce(init_vec, group = group, op=dist.ReduceOp.SUM, async_op=False)

        beta = torch.linalg.norm(init_vec)

        tridiag[i, i + 1] = beta
        tridiag[i + 1, i] = beta

        return (tridiag, vecs, (init_vec / beta).squeeze())

    def lanczos_iteration(i, args, params, batch):

        # init_vec should also be the space to communicate
        vecs, tridiag, init_vec = args

    
        # Get last two vectors
        # v = vecs[i]
        v = init_vec

        # Assign to w the Hessian vector product Hv. Uses forward-over-reverse mode for computing Hv.
        # We assume here that the default precision is 32 bit.
        v_ = v if return_precision == '64' else v.type(torch.float32)
        # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)
        if spare_num == 0:
            w = hvp_forward_ad(params, v_.clone().detach(), batch).detach()
        else:
            # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)
            w = hvp_forward_ad(params, v_[:-spare_num].clone().detach(), batch).detach()
        # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)
        # dist.all_reduce(w, group = group, op=dist.reduce_op.SUM, async_op=False)
        # w /= precondition_world_size
        dist.broadcast(w,src=0)
        # print(w)
        w = torch.hstack((w,torch.zeros((spare_num)).to(device)))
        w = w.to(dtype=lanczos_precision)
        # print(w)


        # assert 0

        # Evaluates alpha and update tridiag diagonal with alpha
        alpha = torch.dot(w, v)
        tridiag[i, i] = alpha

        w = w[int((rank - int(worker_world_size)) * (split_num)):int((rank - int(worker_world_size)+1) * (split_num))]

        # For every iteration except the last one, perform full orthogonalization on w and normalized it (beta is w's
        # norm). Update tridiag secondary diagonals with beta and update vecs with the normalized orthogonal w.
        if i + 1 < order:
            tridiag, vecs, init_vec = orthogonalization(vecs, w, tridiag, i, init_vec)
            vecs[i+1] = init_vec[int((rank - int(worker_world_size)) * (split_num)):int((rank - int(worker_world_size)+1) * (split_num))]

        return (vecs, tridiag, init_vec)

    def lanczos_alg_jitted(params, batch):
        """
        Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
        The first time the function is called it is compiled, which can take ~30 second for 10,000,000 parameters
        and order (m) 100.
        Args:
            params: values of the model/function parameters. The gradient of the loss at this params value is used
                in the hvp operator.
            batch: a batch of samples that determines the actual loss function (each loss_i is determined by a
                batch_i of samples, and we use a specific loss_i).
        Returns:
            k_largest_eigenvals: approximate k largest eigenvalues of the Hessian of loss_i at the point params.
            k_largest_eigenvecs: approximate k eigenvectors corresponding to the largest eigenvalues.
            l_smallest_eigenvals: approximate l smallest eigenvalues of the Hessian of loss_i at the point params.
            l_smallest_eigenvecs: approximate l eigenvectors corresponding to the smallest eigenvalues.
        """

        # Initialization
        params_flatten = torch.nn.utils.parameters_to_vector(params)
        global num_params
        num_params = params_flatten.shape[0]
        tridiag = torch.zeros((order, order), dtype=lanczos_precision).to(device)


        #use model parallism
        # assert (num_params % precondition_world_size == 0)
        global spare_num, split_num
        spare_num = precondition_world_size - (num_params % precondition_world_size)

        assert ((num_params + spare_num) % precondition_world_size == 0)
        split_num = (num_params + spare_num) // precondition_world_size
        # print(spare_num)
        # assert 0
        # print(split_num)
        # assert 0

        vecs = torch.zeros((order, split_num), dtype=lanczos_precision).to(device)
        # assert 0
        # print(spare_num)
        # print(split_num)
        # print(rank)

        init_vec = torch.normal(mean=0.0, std=1.0, size=(num_params,), dtype=lanczos_precision).to(device)
        init_vec = init_vec / torch.linalg.norm(init_vec)
        init_vec = torch.hstack((init_vec,torch.zeros((spare_num)).to(device)))

        dist.broadcast(init_vec, int(worker_world_size), group = group)

        vecs[0] = init_vec[int((rank - int(worker_world_size)) * (split_num)):int((rank - int(worker_world_size)+1) * (split_num))]

        # if rank == 0:
        #     print(split_num)
        #     print(spare_num)
        #     print(num_params)
        #     assert 0
        # print(worker_world_size)
        # print(int((rank - int(worker_world_size)) * (split_num)))
        # if rank == 7:
        #     print(int((rank - int(worker_world_size)+1) * (split_num)))
        # print(init_vec[int((rank - int(worker_world_size)) * (split_num)):int((rank - int(worker_world_size)+1) * (split_num))])
        # print(init_vec[-2:])

        # print(init_vec.shape)

        # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)


        # vecs[0] = init_vec
        global vec_tree

        if vec_tree is None:
            vec_tree = pytree.tree_map(lambda t: torch.zeros_like(t), params)

        lanczos_iter = lambda i, args: lanczos_iteration(i, args, params, batch)
        # Lanczos iterations.
        for i in range(order):
            # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)
            vecs, tridiag, init_vec = lanczos_iter(i, (vecs, tridiag, init_vec))
            # print('Memory allocated:', torch.cuda.memory_allocated()/(1024**2)/1024)
        # if rank == 15:
        #     print(vecs)
        #     assert 0

        eigs_tridiag, eigvecs_triag = torch.linalg.eigh(tridiag)  # eigs_tridiag are also eigenvalues of  the Hessian
        # print(eigvecs_triag.shape)

        precision = torch.float64 if return_precision == '64' else torch.float32
        k_largest_eigenvals = eigs_tridiag[order-k_largest:].type(precision)
        k_largest_eigenvecs = (eigvecs_triag.T[order-k_largest:] @ vecs).type(precision)
        l_smallest_eigenvals = eigs_tridiag[:l_smallest].type(precision)
        l_smallest_eigenvecs = (eigvecs_triag.T[:l_smallest] @ vecs).type(precision)

        # print(k_largest_eigenvecs.shape)
        # assert 0

        del vecs, tridiag, eigs_tridiag, eigvecs_triag, params_flatten, init_vec
        return k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs

    return lanczos_alg_jitted
