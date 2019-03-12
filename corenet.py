"""

Imlements compression strategies:
1. described in https://arxiv.org/pdf/1804.05345.pdf:

- CoreNet
- CoreNet+
- CoreNet++

2. our own proposed for CNNs

"""

from copy import deepcopy


def sparsify_corenet(model, eps, delta):
	"""Base CoreNet function"""
	raise NotImplementedError("ImplementMe")

	# TODO:
	sparse_model = deepcopy(model)
	return sparse_model
