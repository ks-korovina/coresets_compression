"""

Evaluate different sparsity-related things on trained models

"""

from args import parse_args
from models import get_model
from datasets import get_data_loader, get_mnist
from corenet import sparsify_corenet
# from utils import estimated_sparsity

if __name__=="__main__":
    args = parse_args()
    model = get_model(args.model_name)
    model.load("debug", args.model_dir)
    data = get_mnist(True)
    sparse_model = sparsify_corenet(model, data)

    print("Nonzero weights in unsparsified model:\t{}".format(model.count_nnz()))
    print("Nonzero weights in sparsified model:\t{}".format(sparse_model.count_nnz()))

    # according to theoretical bounds
    # est = estimated_sparsity()