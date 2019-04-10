"""
Evaluate different sparsity-related things on trained models
@author: kkorovin@cs.cmu.edu
"""

import matplotlib.pyplot as plt

from args import sparsify_args
from models import get_model
from datasets import get_data_loader, get_dataset
from corenet import sparsify_corenet
from utils import evaluate_coverage, evaluate_val_acc, compute_nnz_svd
from sparsify_baseline import sparsify_svd

def display_results(exp_header, res, logfile=None):
    """Display results of sparsification experiment

    Arguments:
        exp_header {string} -- description of experiment
        res {dict} -- results

    Keyword Arguments:
        logfile {string} -- file to save readable description (default: {None})
    """
    def display_results_for_method(exp_header, res, logfile):
        pre_nnz = res['sparsification']['pre_nnz']
        post_nnz = res['sparsification']['post_nnz']
        max_dev = res['coverage']
        val = 0.5
        cov = (max_dev < val).mean()
        pre_acc = res['accuracy']['pre_acc']
        post_acc = res['accuracy']['post_acc']

        if logfile is None:
            print(exp_header)
            print("Nonzero weights in unsparsified model:\t{}".format(pre_nnz))
            print("Nonzero weights in sparsified model:\t{}".format(post_nnz))
            print("Sparsification rate:\t{:.3f}".format(post_nnz/pre_nnz))
            print("Percentage of sampled val pts within {:.3f} relative range: {}".format(val, cov))
            print("Unsparsified model val acc:\t{:.3f}".format(pre_acc))
            print("Sparsified model val acc:\t{:.3f}".format(post_acc))
        else:
            with open(logfile, 'a+') as f:  # append to existing file
                f.write('\t' + exp_header + '\n')
                f.write("Nonzero weights in unsparsified model:\t{}\n".format(pre_nnz))
                f.write("Nonzero weights in sparsified model:\t{}\n".format(post_nnz))
                f.write("Sparsification rate:\t{:.3f}\n".format(post_nnz/pre_nnz))
                f.write("Percentage of sampled val pts within {:.3f} relative range: {}\n".format(val, cov))
                f.write("Unsparsified model val acc:\t{:.3f}\n".format(pre_acc))
                f.write("Sparsified model val acc:\t{:.3f}\n".format(post_acc))
        plt.style.use('ggplot')
        plt.hist(max_dev, bins=100, range=[0, 10], label=exp_header, alpha = 0.6)

    if logfile is None:
        print("-" * 60)
        print(exp_header)
    else:
        with open(logfile, 'a+') as f:  # append to existing file
            f.write("-" * 60)
            f.write("\t" + exp_header + "\n")

    if 'corenet' in res:  # several-method results
        display_results_for_method('corenet', res['corenet'], logfile)
        display_results_for_method('svd', res['svd'], logfile)
        plt.legend()
        plt.savefig(f"./results/max_dev_{exp_header}")
        plt.clf()
    else:
        display_results_for_method(exp_header, res, logfile)


def evaluate_sparsifier(model_name, dataset, check_name, model_dir, sparse,
                        method, variance_based=False, **kwargs):
    """ Run a single sparsification eval and return the result.
        TODO: add a custom sparsification caller.
    """
    if check_name == "default":
        check_name = f"{model_name}_{dataset}"

    model = get_model(model_name)
    model.load(check_name, model_dir)
    train_data = get_dataset(dataset, is_train=True)
    val_data   = get_dataset(dataset, is_train=False)
    val_loader = get_data_loader(dataset, is_train=False)
    
    if method == "corenet": # two cases to account for different nnz parameters computation 
        sparse_model = sparsify_corenet(model, train_data, 
                                    s_sparse=sparse)
        pre_nnz = model.count_nnz()
        post_nnz = sparse_model.count_nnz()
    elif method == "svd":
        #print(variance_based)
        sparse_model = sparsify_svd(model, sparse, variance_based)
        pre_nnz = compute_nnz_svd(model)
        post_nnz = compute_nnz_svd(sparse_model)
    else:
        raise ValueError(f"Method {method} not available")

    max_dev = evaluate_coverage(model, sparse_model, val_data, 0.5)

    pre_acc = evaluate_val_acc(model, val_loader)
    post_acc = evaluate_val_acc(sparse_model, val_loader)

    res = {
        'sparsification': {'pre_nnz': pre_nnz, 'post_nnz': post_nnz},
        'accuracy': {'pre_acc': pre_acc,  'post_acc': post_acc},
        'coverage': max_dev
    }

    return res


if __name__=="__main__":
    args = sparsify_args()
    results = evaluate_sparsifier(**vars(args))
    exp_header = f"{args.model_name}, {args.dataset}, {args.sparse}"
    display_results(exp_header, results)

