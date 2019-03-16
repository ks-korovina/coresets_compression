"""
Training loop, saves the model into ./saved_models

"""

from args import parse_args
from models import get_model
from datasets import get_data_loader

from constants import DEVICE

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def train_epoch(model, data, crit, opt, scheduler):
    running_loss = 0.
    for (xs, ys) in data:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        opt.zero_grad()
        logits = model(xs)
        loss = crit(logits, ys)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    return running_loss / len(data)


def validate(model, data, crit, opt, scheduler):
    pass


if __name__=="__main__":
    args = parse_args()
    model = get_model(args.model_name)
    train_dataloader = get_data_loader(args.dataset, True, 
                                       batch_size=args.batch_size)
    val_dataloader = get_data_loader(args.dataset, False, 
                                     batch_size=args.batch_size)

    # stuff that could be adjusted
    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, patience=3, threshold=0.1, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        loss = train_epoch(model, train_dataloader, criterion, opt, scheduler)
        print("Finished epoch {}, avg loss {:.3f}".format(epoch+1, loss))
        validate(model, val_dataloader, criterion, opt, scheduler)
        model.save("debug", args.model_dir)

    print("Model {} has been saved to {}".format(args.model_name, args.model_dir))

