"""
Experiment sets.
@author: kkorovin@cs.cmu.edu
"""

from train import train_model
from sparsify import evaluate_sparsifier, display_results

# Experiment definitions ------------------------------------------------------
experiments = [
    {'model_name': 'small_mnist', 'dataset': 'mnist'},
    {'model_name': 'long_mnist', 'dataset': 'mnist'},
    {'model_name': 'small_cifar', 'dataset': 'cifar10'},
]
# Modify these arguments depending on settings
default_train_settings = {'batch_size': 64, 'lr': 1e-2, 'n_epochs': 2,
                          'check_name': 'default', 'model_dir': './checkpoints'}

for exp in experiments:
    exp = exp.update(default_train_settings)

# Experiment runner -----------------------------------------------------------
if __name__ == "__main__":
    for exp_setting in experiments:
        train_model(**exp_setting)
        for sparse in [0.1, 1, 2]:
            exp_setting['sparse'] = sparse
            results = evaluate_sparsifier(**exp_setting)
            exp_header = f"{exp_setting['model_name']}, {exp_setting['dataset']}, {exp_setting['sparse']}"
            display_results(exp_header, results, logfile='./results/log')
