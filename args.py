"""
Argument parser
@author: kkorovin@cs.cmu.edu
"""

import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--model_name", type=str,
        help="Model config name from models.py")
    parser.add_argument("-e", "--n_epochs", default=10, type=int)
    parser.add_argument("-l", "--lr", default=1e-2, type=float)
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    # checkpoints
    parser.add_argument("-c", "--check_name", default="default", type=str,
        help="Model checkpoint name to save. Defaults to model_name_dataset")
    parser.add_argument("--model_dir", default="./checkpoints", type=str)
    return parser.parse_args()

def sparsify_args():
    # add s_sparse, coverage evaluation etc
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--model_name", type=str,
        help="Model config name from models.py")
    parser.add_argument("-s", "--sparse", type=float,
        help="Sparsification rate")
    # checkpoints
    parser.add_argument("-c", "--check_name", default="default", type=str,
        help="Model checkpoint name to save. Defaults to model_name_dataset")
    parser.add_argument("--model_dir", default="./checkpoints", type=str)
    return parser.parse_args()
