import torch as th


def wasserstein_loss(repr, t, lamb=1., iter=10):
    """
    Implementation of the Wasserstein loss using Sinkhorn distances as approximations as described in
        "Estimating individual treatment effect: generalization bounds and algorithms" (https://arxiv.org/abs/1606.03976)
    which makes use of Algorithm 3 from "Fast Computation of Wasserstein Barycenters" (https://arxiv.org/abs/1310.4375)

    :param repr:    feature representation where we want to minimise the Wasserstein distance between treated and control
    :param t:       assigned treatments t
    :param lamb:    scalar parameter for additionally weighting the calculated distances
    :param iter:    number of iterations for running the approximation algorithm
    :return:        approximated Wasserstein distance between treated and control samples
    """
    repr_0 = repr[t == 0, :]
    repr_1 = repr[t == 1, :]
    Mij = th.cdist(repr_0, repr_1)  # n x m distances between control samples and treated samples as input for algorithm
    Kij = th.exp(-lamb * Mij)

    nt = th.sum(t, dtype=int)
    nc = t.shape[0] - nt

    a = th.ones(nc) / nc
    b = th.ones(nt) / nt
    Ktil = th.matmul(th.diag(1/a), Kij)

    u = th.ones(nc) / nc
    for _ in range(iter):
        temp1 = th.matmul(th.transpose(Kij, 0, 1), u)
        temp2 = b /temp1
        u = 1.0 / (th.matmul(Ktil, temp2))

    v = b / th.matmul(th.transpose(Kij, 0, 1), u)
    T = (th.matmul(th.diag(u), th.matmul(Kij, th.diag(v)))).detach()  # gradients only through Mij, not also T

    return th.sum(T * Mij)