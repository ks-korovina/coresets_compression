"""Architecture definitions.

This module contains:
- Feedforward nets: [...]
- Convolutional nets: TBA

"""

import os
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from constants import DEVICE


def get_model(model_name):
    model_config = model_settings[model_name]
    return FullyConnectedNN(**model_config)

act_decode = {"relu": partial(nn.ReLU, inplace=True),
              "tanh": partial(nn.Tanh, inplace=True),
              "sigmoid": partial(nn.Sigmoid, inplace=True),
              "id": (lambda x: x),
              }

model_settings = {
                "debug": {"input_size": 5,
                         "sizes_list": [3, 2],
                         "activations_list": ["relu"]},
                "debug2": {"input_size": 28 * 28,
                         "sizes_list": [200, 100, 10],
                         "activations_list": ["relu", "relu"]},
                "large_max": {"input_size": None,
                              "sizes_list": [None],
                              "activations_list": [None]},
                "equal_sizes": {"input_size": None,
                              "sizes_list": [None],
                              "activations_list": [None]},
                "long": {"input_size": None,
                        "sizes_list": [None],
                        "activations_list": [None]}
                }

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, sizes_list, activations_list):
        """FF-FCN
        
        A class for all feed-forward FCNs with
        custom layer sizes and activations.
        
        Arguments:
            input_size {int} -- size of input vector
            sizes_list {list[int]} -- sizes of hidden neurons
            activations_list {list[string]} -- types of activation functions
                                               (can be identity)
        """
        super(FullyConnectedNN, self).__init__()

        assert len(sizes_list) == len(activations_list)+1, "last output should not be activated"

        # faster to compute constants in corenet
        self.input_size = input_size
        self.sizes_list = sizes_list

        in_sizes = [input_size] + sizes_list[:-1]
        out_sizes = sizes_list
        layers = [nn.Linear(n_in, n_out) for (n_in, n_out) in zip(in_sizes, out_sizes)]
        activations = [act_decode[act]() for act in activations_list]

        self.net = nn.Sequential(layers[0], *chain(*zip(activations, layers[1:])))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def zero_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def layers(self):
        """ Returns an generator over layers """
        layer_iter = self.net.children()
        return layer_iter

    def count_nnz(self):
        """ Counting both weights and biases """
        count = 0
        for m in self.layers():
            # tensor.nonzero().size(0) is convenient for .nnz()
            if isinstance(m, nn.Linear):
                count += m.weight.nonzero().size(0)
                count += m.bias.nonzero().size(0)
        return count

    def forward(self, x):
        x = x.squeeze(1).view(-1, self.input_size)
        res = self.net(x)
        return res

    def save(self, check_name, model_dir):
        checkpoint = self.state_dict()
        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint, model_dir+"/{}.pth".format(check_name))

    def load(self, check_name, model_dir):
        path = model_dir+"/{}.pth".format(check_name)
        checkpoint = torch.load(path, map_location='cpu')
        self.to(DEVICE)
        self.load_state_dict(checkpoint)

