"""

Evaluate different sparsity-related things on trained models

TODO:
* sparsify_args()

"""

import matplotlib.pyplot as plt

from args import sparsify_args
from models import get_model
from datasets import get_data_loader, get_dataset
from corenet import sparsify_corenet
from utils import evaluate_coverage, evaluate_val_acc


if __name__=="__main__":
    args = sparsify_args()
    model = get_model(args.model_name)
    model.load("debug", args.model_dir)
    train_data = get_dataset("mnist", is_train=True)
    val_data   = get_dataset("mnist", is_train=False)
    val_loader = get_data_loader("mnist", is_train=False)

    sparse_model = sparsify_corenet(model, train_data, 
                                    s_sparse=1.)

    pre_nnz = model.count_nnz()
    post_nnz = sparse_model.count_nnz()
    print("Nonzero weights in unsparsified model:\t{}".format(pre_nnz))
    print("Nonzero weights in sparsified model:\t{}".format(post_nnz))
    print("Sparsification rate:\t{:.3f}".format(post_nnz/pre_nnz))

    val = 0.5
    max_dev = evaluate_coverage(val, model, sparse_model, val_data, 0.5)
    cov = (max_dev < val).mean()
    print("Percentage of sampled val pts within {:.3f} relative range: {}".format(val, cov))
    plt.hist(max_dev, bins=100, range=[0, 10])
    plt.savefig("./results/max_dev")

    pre_acc = evaluate_val_acc(model, val_loader)
    post_acc = evaluate_val_acc(sparse_model, val_loader)
    print("Unsparsified model val acc:\t{:.3f}".format(pre_acc))
    print("Sparsified model val acc:\t{:.3f}".format(post_acc))

    # according to theoretical bounds
    # est = estimated_sparsity()