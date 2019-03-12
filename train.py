"""
Training loop, saves the model into ./saved_models

"""


def train_epoch(model, data):
	raise NotImplementedError("ImplementMe")

def validate(model, data):
	raise NotImplementedError("ImplementMe")


if __name__=="__main__":
	# TODO: run training loop
	args = parse_args()
	model = <TODO>  # smth like from_model_name(args.model_name)
	train_dataloader = <TODO>  # add val_dataloader

	for epoch in n_epochs:
		train(model, train_dataloader)
		val(model, val_dataloader)
		# TODO: save model

	print("Model {} has been saved to {}".format(model_name, model_dir))
