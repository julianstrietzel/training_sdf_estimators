import os

import torch
from torch import nn


def defensive_programming_opt_input_checks_datasets(opt):
    """
    Checks if the opt object has some necessary attributes for the dataset to work.
    :param opt: The opt object to check for
    """
    if not hasattr(opt, "dataroot"):
        raise ValueError("Dataroot not specified")
    if not os.path.exists(opt.dataroot):
        raise ValueError("Dataroot does not exist")


def loss_factory(loss_name):
    """
    Returns a loss function given a loss name.
    Choose from:
    - mse
    - mae (l1)
    :param loss_name: The name of the loss function
    """
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "l1": nn.L1Loss,
    }
    if loss_name not in losses.keys():
        raise ValueError("Loss [%s] not recognized." % loss_name)
    return losses.get(loss_name)()


def optimizer_factory(optimizer_id, model, lr) -> torch.optim.Optimizer:
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if optimizer_id not in optimizers.keys():
        raise ValueError("Optimizer [%s] not recognized." % optimizer_id)
    return optimizers.get(optimizer_id)(model.parameters(), lr=lr)


def lr_scheduler_init(scheduler_id, optimizer, opt) -> torch.optim.lr_scheduler:
    if scheduler_id == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma
        )
    elif scheduler_id == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=opt.lr_decay_gamma,
            patience=opt.lr_decay_patience,
            verbose=True,
        )
    elif scheduler_id == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=opt.lr_decay_gamma
        )
    else:
        raise ValueError("Scheduler [%s] not recognized." % scheduler_id)
