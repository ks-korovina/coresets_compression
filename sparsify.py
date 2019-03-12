"""

Evaluate different sparsity-related things on trained models

"""

from corenet import sparsify_corenet
from utils import estimate_sparsity


if __name__=="__main__":
	args = parse_args()  # TODO
	# load model from args.path
	sparsify_corenet(model)
	est = estimated_sparsity()
