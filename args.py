"""

Argument parser

TODO: implement

"""

import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_dir", default="./checkpoints", type=str)
    return parser.parse_args()

def sparsify_args():
    # add s_sparse, coverage evaluation etc
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model_dir", default="./checkpoints", type=str)
    return parser.parse_args()
