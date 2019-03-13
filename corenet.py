"""

Imlements compression strategies:
1. described in https://arxiv.org/pdf/1804.05345.pdf:

- CoreNet
- CoreNet+
- CoreNet++

2. our own proposed for CNNs

"""

from copy import deepcopy


def sparsify_neurons_corenet(inds, w, eps_l, delta, inp):
	"""[summary]
	
	[description]
	
	Arguments:
		inds {[type]} -- indices of active neurons
		w {[type]} -- vector of shape (n_input,)
		eps_l {[type]} -- [description]
		delta {[type]} -- [description]
		inp {[type]} -- input to current layer ell,
						evaluated on set S, hence it
						is a vector of size (len(S),)
	
	Returns:
		[type] -- [description]
	"""
	# compute importance weights for all j in inds
	importance = <tensor product of vectors w and inp, shp (inds, S) -> max>
	sum_importances = importance.sum()
	q = importance/sum_importances
	m = np.ceil(<EXPR>)
	# TODO: sample a multiset of neurons with probs q
	w_new = <zeros>
	# for each ind of neuron, update corresponding w_new_j
	return w_new


def sparsify_corenet(model, train, eps, delta):
	"""Base CoreNet function

	TODO:
	* currently implementation does not care about
	  numeric issues or efficiency

	"""
	sparse_model = deepcopy(model)
	sparse_model.zero_weights()

	# compute parameters
	L = len(model.modules())
	eps_prime = eps / (L - 1)
	eta = <TODO>
	eta_star = <TODO>
	lambda_star = np.log(eta * eta_star) / 2
	sq_ = np.sqrt(2 * lambda_star)
	kappa = sq_ * (1 + sq_ * np.log(8*eta*eta_star))

	# sample the set S
	subset_size = np.ceil(np.log(8*eta*eta_star/delta)*np.log(eta*eta_star))
	S = train.sample(subset_size)

	input_activations = S  # (batch_size, num_input_features)
	captured_activation = (lambda x: x)
	
	for m, m_sp in zip(model.modules(), sparse_model.modules()):
		if isinstance(m, FC):
			input_activations = m(input_activations)
			input_activations = captured_activation(input_activations)

			n_out = m.fan_out
			triangle = <TODO>
			# for every neuron
			for i in range(n_out):
				W[i, :]  # all connections to i-th neuron
				W_p, W_m
				# sparsify both
				W_sp[i, :] = <set i-th row of m_sp>
		elif isinstance(m, activation):
			captured_activation = m
		else:
			raise NotImplementedError("What is going on, I only work on FF-FCN")
	
	return sparse_model

