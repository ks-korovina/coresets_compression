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


def get_model(model_name):
    model_config = model_settings[model_name]
    return FullyConnectedNN(**model_config)


act_decode = {"relu": partial(nn.ReLU, inplace=True),
              "tanh": partial(nn.Tanh, inplace=True),
              "sigmoid": partial(nn.Sigmoid, inplace=True),
              "id": (lambda x: x),
              }


model_settings = {
                "debug": {"input_size": 28 * 28,
                         "size_list": [100, 10],
                         "activations_list": ["relu", "relu"]},
                "large_max": {"input_size": None,
                              "size_list": [None],
                              "activations_list": [None]},
                "equal_sizes": {"input_size": None,
                              "size_list": [None],
                              "activations_list": [None]},
                "long": {"input_size": None,
                        "size_list": [None],
                        "activations_list": [None]}
                }

class FullyConnectedNN(nn.Module):
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
        super(FullyConnectedNN, self).__init__()
        assert len(size_list) == len(activations_list)
        self.input_size = input_size
        in_sizes = [input_size] + size_list[:-1]
        out_sizes = size_list
        layers = [nn.Linear(n_in, n_out) for (n_in, n_out) in zip(in_sizes, out_sizes)]
        activations = [act_decode[act]() for act in activations_list]

        self.net = nn.Sequential(*chain(*zip(layers, activations)))
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

    def forward(self, x):
        x = x.squeeze(1).view(-1, self.input_size)
        return self.net(x)

    def save(self, check_name, model_dir):
        checkpoint = self.state_dict()
        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint, model_dir+"/{}.pth".format(check_name))


