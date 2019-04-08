"""
Training loop, saves the model into ./saved_models

"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from args import train_args
from models import get_model
from datasets import get_data_loader
from constants import DEVICE


def train_epoch(model, data, crit, opt, scheduler):
    model.train()
    running_loss = 0.
    for (xs, ys) in data:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        logits = model(xs)
        loss = crit(logits, ys)
        loss.backward()
        opt.step()
        opt.zero_grad()
        running_loss += loss.item()
    return running_loss / len(data)

def validate(model, data, crit, scheduler=None):
    model.eval()
    running_loss = 0.
    accs = []
    for (xs, ys) in data:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        logits = model(xs)
        # accuracy
        y_pred = logits.argmax(dim=1)
        batch_acc = (y_pred == ys).float().mean().item()
        accs.append(batch_acc)
        # val loss
        loss = crit(logits, ys)
        running_loss += loss.item()
    running_loss /= len(data)
    return running_loss, np.mean(accs)

def train_model(model_name, dataset, batch_size, lr, n_epochs, check_name, model_dir, **kwargs):
    if check_name == "default":
        check_name = f"{model_name}_{dataset}"

    model = get_model(model_name)
    train_dataloader = get_data_loader(dataset, True, 
                                       batch_size=batch_size)
    val_dataloader = get_data_loader(dataset, False, 
                                     batch_size=batch_size)

    # stuff that could be adjusted
    opt = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, patience=3, threshold=0.1, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        loss = train_epoch(model, train_dataloader, criterion, opt, scheduler)
        print("Finished epoch {}, avg loss {:.3f}".format(epoch+1, loss))
        val_loss, acc = validate(model, val_dataloader, criterion, scheduler)
        print("Validation loss: {}, accuracy: {}".format(val_loss, acc))
        model.save(check_name, model_dir)

    print("Model {} has been saved to {}".format(model_name, model_dir))


if __name__=="__main__":
    args = train_args()
    train_model(**vars(args))

