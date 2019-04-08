"""
Basic utilities for the project
@author: kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import sample_from_dataset
from train import validate
from constants import *


def evaluate_coverage(model, sparse_model, dataset, sample_rate=0.2, use_log=False):
	""" Returns delta_hat - percentage of cases in dataset where
		sparse model's outputs are within
		[(1-value)*y_hat, (1+value)*y_hat],
		where y_hat are original model's outputs
	"""
	# this cap depends on GPU capacity:
	sample_size = min(500, int(sample_rate * len(dataset)))
	S = sample_from_dataset(dataset, sample_size).to(DEVICE)

	sm = nn.LogSoftmax(dim=1) if use_log else (lambda x: x)
	model_out  = sm(model(S)).detach().numpy()
	sparse_out = sm(sparse_model(S)).detach().numpy()

	ratio = np.exp(sparse_out - model_out) if use_log else sparse_out/model_out
	max_dev_from_one = np.max(np.abs(ratio - 1), axis=1)
	# hist =  np.histogram(max_dev_from_one)
	return max_dev_from_one  #  (max_dev_from_one<value).mean()

def evaluate_val_acc(model, data):
	crit = nn.CrossEntropyLoss()
	loss, acc =  validate(model, data, crit)
	return acc

def estimated_sparsity(model_config, eps, delta):
    raise NotImplementedError("ImplementMe")


def compute_nnz_svd(model):
	nnz = 0
	tol = 1e-5 # tolerance level for threshold
	for m in model.layers():
		if isinstance(m, nn.Linear):
			m_svd = np.linalg.svd(m.weight.detach().numpy())
			nnz += (m.weight.shape[0] + m.weight.shape[1]) * ((m_svd[1] > tol).sum())
			nnz += m.bias.shape[0]
		
	return nnz
