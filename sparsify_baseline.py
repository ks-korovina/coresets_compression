"""

Implements compression strategies based on SVD:


"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from copy import deepcopy
from constants import DEVICE


def sparsify_layer(weight_matrix, sparsity_param, var_based, verbose=False):
    """Sparsify connections in a given layer.

    Arguments:
        weight_matrix -- layer's weight matrix to be sparsified
        sparsity_param {float} -- sparsity parameter
        var_based {bool} -- identifier of whether sparsity parameter 
                            identifies number of parameters to keep or
                            amount of variance to keep

    Returns:
        w_new {torch.FloatTensor} -- sparsified layer
    """
    layer_svd = np.linalg.svd(weight_matrix, full_matrices=False)

    num_of_sing_values = layer_svd[1].shape[0] # get minimum of two sizes of weight_matrix

    sum_of_sing_val_sq = np.linalg.norm(layer_svd[1]) ** 2 # to assess the quality of approximation

    # calculate number of principal components needed
    number_of_components = 0 
    current_approximation = 0
    if verbose: print(var_based)
    if var_based: # case: provided parameter specifies the desired approximation level in Frobenius norm squared  
        # calculate number of directions to achieve desired approximation 
        while current_approximation < sparsity_param: 
            number_of_components += 1
            current_approximation = np.linalg.norm(layer_svd[1][:number_of_components]) ** 2 / \
                                                                       sum_of_sing_val_sq
        if verbose:
            print ("{} principal directions will be used out of total {}".format(number_of_components, \
                                                                    num_of_sing_values))
    else: # case: provided parameter specifies the desired proportion of parameters to keep
        if verbose: print(sparsity_param , num_of_sing_values)
        number_of_components = np.round(sparsity_param * num_of_sing_values, decimals=0).astype(int) 
        # calculate corresponding number of directions
        num_of_comp_var = 0 # additionally calculate number of principal components required to achieve
        # 95 percent approximation for comparison 
        while current_approximation < 0.95: 
            # check number of components to achieve 95 percent approximation 
            num_of_comp_var += 1
            current_approximation = np.linalg.norm(layer_svd[1][:num_of_comp_var]) ** 2 / \
                                                                       sum_of_sing_val_sq
        
        if verbose: 
            print (("{} principal directions out of total {} required to get "
                    "0.95 approximation").format(num_of_comp_var, num_of_sing_values))
        if verbose:
            print("{} principal directions used instead".format(number_of_components) )
        used_approx =  np.linalg.norm(layer_svd[1][:number_of_components]) ** 2 / \
                                                                       sum_of_sing_val_sq
        if verbose: print("With corresponding approximation {}".format(used_approx))
    
    #provide sparsifications levels for a layer

    nnz_before = np.count_nonzero(layer_svd[0]) + np.count_nonzero(layer_svd[2])
    nnz_after = np.count_nonzero(layer_svd[0][:number_of_components]) + \
                                         np.count_nonzero(layer_svd[2][:number_of_components])

    if verbose: print ("Total number of non-zero parameters in layer before {}".format(nnz_before))
    # print ("Total number of non-zero parameters in layer after {}".format(nnz_after))

    #calculate corresponding approximation

    approx_weight = layer_svd[0][:,:number_of_components].dot(\
                    np.diag(layer_svd[1][:number_of_components])).dot(\
                        layer_svd[2][:number_of_components,:])

    return approx_weight
  


def sparsify_svd(model, sparsity_parameter=0.3, variance_based=False, verbose=False):
    """ SVD sparsifier.

    Arguments:
        model {torch Module} -- [description]
        variance_based {bool} -- parameter specifying whether compression is done based on
                                 desired approximation level or number of parameters to keep
            sparsity_parameter {float} -- parameter either


    Returns:
        [torch Module] -- sparsified Model
    """
    sparse_model = deepcopy(model)
    sparse_model.zero_weights()

    cur_Layer = 1

    for m, m_sp in zip(model.layers(), sparse_model.layers()):
        if isinstance(m, nn.Linear):
            if verbose: print("Sparsification of the {} linear layer".format(cur_Layer))
            cur_Layer += 1
            #print(m.weight)
            m_sp.weight = torch.nn.Parameter(torch.tensor(sparsify_layer(m.weight.detach().numpy(),\
                                                                 sparsity_parameter, variance_based)))

    return sparse_model

