"""Architecture definitions.

This module contains:
- Feedforward nets: [...]
- Convolutional nets: TBA

"""

from itertools import chain
from functools import partial

import torch
import torch.nn as nn

act_decode = {"relu": partial(nn.ReLU, inplace=True),
			  "tanh": partial(nn.Tanh, inplace=True),
			  "sigmoid": partial(nn.Sigmoid, inplace=True)
			  "id": lambda x: x,
			  }

def FullyConnectedNN(nn.Module):
	def __init__(self, input_size, size_list, activations_list):
		"""FF-FCN
		
		A class for all feed-forward FCNs with
		custom layer sizes and activations.
		
		Arguments:
			input_size {int} -- size of input vector
			size_list {list[int]} -- sizes of hidden neurons
			activations_list {list[string]} -- types of activation functions
			                                   (can be identity)
		"""
		assert len(size_list) == len(activations_list)
		in_sizes = [input_size] + size_list[:-1]
		out_sizes = size_list
		layers = [nn.Linear(n_in, n_out) for (n_in, n_out) in zip(in_sizes, out_sizes)]
		activations = [act_decode[act] for act in activations_list]
		self.net = nn.Sequential(itertools.chain(*zip(layers, activations)))
		self.initialize()

	def initialize(self):
		for m in self.modules():
			nn.init.normal_(m.weight, 0, 0.01)
	        nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.net(x)
