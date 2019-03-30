"""
Basic utilities for the project:

- Values of upper bounds on sparsity
- Values of upper bounds on generalization
- Boring utility functions of little interest

"""

import torch
import torch.nn as nn
import numpy as np
from datasets import sample_from_dataset
from train import validate
from constants import *


def evaluate_coverage(value, model, sparse_model, dataset, sample_rate=0.2):
	""" Returns delta_hat - percentage of cases in dataset where
		sparse model's outputs are within
		[(1-value)*y_hat, (1+value)*y_hat],
		where y_hat are original model's outputs
	"""
	# this cap depends on GPU capacity:
	sample_size = min(500, int(sample_rate * len(dataset)))
	S = sample_from_dataset(dataset, sample_size).to(DEVICE)
	model_out = model(S).detach().numpy()
	sparse_out = sparse_model(S).detach().numpy()
	max_dev_from_one = np.max(np.abs(sparse_out/model_out - 1), axis=1)
	return (max_dev_from_one < value).mean()

def evaluate_val_acc(model, data):
	crit = nn.CrossEntropyLoss()
	loss, acc =  validate(model, data, crit)
	return acc

def estimated_sparsity(model_config, eps, delta):
    raise NotImplementedError("ImplementMe")
