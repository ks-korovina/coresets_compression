"""

Imlements compression strategies:
1. described in https://arxiv.org/pdf/1804.05345.pdf:

- CoreNet
- CoreNet+
- CoreNet++

2. our own proposed for CNNs

"""

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

from datasets import sample_from_dataset
from constants import DEVICE


def sparsify_neuron_corenet(mask, row, eps, delta, inp,
                             eta, eta_star):
    """
    [Desc]
    
    Arguments:
        mask {torch.FloatTensor} -- mask of active connections, (n_input,)
        row {torch.FloatTensor} -- vector of shape (n_input,)
                        (will be (n_output, n_input) though, later)
        eps {float} -- [description]
        delta {float} -- [description]
        inp {torch.FloatTensor} -- input to current layer ell,
                                   evaluated on set S, hence it
                                   is a vector of size (len(S),)
        eta, eta_star -- from enclosing scope
    
    Returns:
        w_new {torch.FloatTensor} -- sparsified connections to neuron
    """
    # is mask is empty, there are no active connections, return zero?
    if 0 == torch.sum(mask):
        return torch.zeros_like(row)

    # compute sensitivity weights for all j in inds:
    # For every active input connection, compute its max (normalized) activation
    assert 0 < torch.sum(mask), mask
    sensit = (inp * row * mask).detach().numpy()  # (bs, n_in), (n_in) - broadcast ok

    assert np.all(np.abs(sensit.sum(axis=1)) > 0), sensit.sum(axis=1)
    sensit = (sensit.T/sensit.sum(axis=1)).max(axis=1)  # [0] - for pytorch
    sum_sensit = sensit.sum()
    q = sensit / sum_sensit

    # Issue: this m is too huge
    print(sum_sensit, np.log(eta*eta_star),np.log(8*eta/delta), eps**(-2))
    m = int(np.ceil(8 * sum_sensit * np.log(eta*eta_star)*np.log(8*eta/delta) / eps**2))
    m = 100
    # sample a multiset of neurons with probs q
    # TODO: len(row)
    w_inds = np.random.choice(np.arange(len(row)), size=m, p=q)
    w_new = torch.zeros_like(row)
    # for each ind of neuron, update corresponding w_new_j
    for ind in w_inds:
        w_new[ind] += row[ind]/(m*q[ind])
    return w_new


def sparsify_corenet(model, train, eps=0.5, delta=0.5):
    """Base CoreNet function

    TODO:
    * currently implementation does not care about
      numeric issues or efficiency

    """
    sparse_model = deepcopy(model)
    sparse_model.zero_weights()

    # compute parameters
    n_hidden = len(model.sizes_list)  # = L-1
    eps_prime = 0.5 * eps / n_hidden
    eta = max(model.sizes_list)
    eta_star = max(model.sizes_list[:-1])
    lambda_star = np.log(eta * eta_star) / 2
    sq_ = np.sqrt(2 * lambda_star)
    kappa = sq_ * (1 + sq_ * np.log(8*eta*eta_star/delta))

    # sample the set S from Dataset train
    subset_size = int(np.ceil(np.log(8*eta*eta_star/delta)*np.log(eta*eta_star)))
    print("Subset S of size {}".format(subset_size))
    S = sample_from_dataset(train, subset_size).to(DEVICE)

    input_activations = S  # (batch_size, num_input_features)

    # 1. triangle-setting forward
    triangles = []
    for m in model.layers():
        print("First layer:", m)
        if isinstance(m, nn.Linear):
            # shape (batch_size, n_out)
            triangle_num = input_activations.abs().matmul(m.weight.transpose(0,1).abs())
            triangle_denom = input_activations.matmul(m.weight.transpose(0,1)).abs()

            # Issue: if all previous activations died, div by denom leads to nans
            assert torch.all(triangle_denom.abs() > 0), input_activations
            triangle = (triangle_num / triangle_denom).mean(dim=0).max() + kappa
            triangles.append(triangle.detach().numpy())

            input_activations = m(input_activations)
        else:
            # TODO: add a check that this is an activation
            input_activations = m(input_activations)

        # else:
        #     raise NotImplementedError("{} is not supported yet".format(m))

    print("Triangles:", triangles)

    # 2. sparsification forward
    # reset
    input_activations = S  # (batch_size, num_input_features)
    ell = 0
    for m, m_sp in zip(model.layers(), sparse_model.layers()):
        print("Second loop:", m)
        if isinstance(m, nn.Linear):
            # shape (batch_size, n_out)
            triangle_forward = np.prod(triangles[ell:])
            eps_ell = eps_prime / triangle_forward

            n_out = m.out_features
            # for every neuron do (TODO: vectorize this loop,
            # most likely it will take nothing more then removing for)
            for i in range(n_out):
                incoming_conn = m.weight[i, :]  # all connections to i-th neuron
                pos_mask, neg_mask = incoming_conn.ge(0).float(), incoming_conn.le(0).float()
                # sparsify both
                W_pos = sparsify_neuron_corenet(pos_mask, incoming_conn,
                        eps_ell, delta, input_activations, eta, eta_star)
                W_neg = sparsify_neuron_corenet(neg_mask, -incoming_conn, 
                        eps_ell, delta, input_activations, eta, eta_star)
                # set i-th row of m_sp
                m_sp.weight[i, :] = W_pos - W_neg

            input_activations = m(input_activations)
            ell += 1

        else:
            # TODO: add a check that this is an activation
            input_activations = m(input_activations)

        # else:
        #     raise NotImplementedError("{} is not supported yet".format(m))
    
    return sparse_model


if __name__ == "__main__":
    # Testing on an very simple model,
    # so that everything is computable by hand
    from models import get_model
    from datasets import get_debug

    model = get_model("debug")
    data = get_debug() 

    # For a small MNIST:
    #model = get_model("debug2")
    #get_mnist(True)

    sparse_model = sparsify_corenet(model, data)

    print("Nonzero weights in unsparsified model:\t{}".format(model.count_nnz()))
    print("Nonzero weights in sparsified model:\t{}".format(sparse_model.count_nnz()))

