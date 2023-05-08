# -*- coding: UTF-8 -*-
import torch


def get_optim(model, optim, lr=0.1, momentum=0):
    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        exit("Unknown Optimizer!")


def get_loss(loss):
    if loss == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss == "MSE":
        return torch.nn.MSELoss()
    else:
        exit("Unknown Loss")
