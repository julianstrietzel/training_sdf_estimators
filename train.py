import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import dataset_factory
from models import model_factory
from options.train_options import TrainOptions
from utils import loss_factory, optimizer_factory, lr_scheduler_init


def train():
    opt = TrainOptions().parse()
    # init dataset
    dataset = dataset_factory(opt.dataloader, opt)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads
    )
    # init model
    model = model_factory(opt.model_name, opt)
    model.train()

    # init loss
    loss_fn = loss_factory(opt.loss)

    # init optimizer
    optimizer = optimizer_factory(opt.optimizer, model, opt.lr)

    # lr scheduler
    lr_scheduler_init(opt.lr_scheduler, optimizer, opt)

    # tensorboard
    writer = SummaryWriter(log_dir="logs")

    # training loop
    for epoch in range(opt.epochs):
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            model_input, target = batch
            model_output = model(model_input)
            loss = loss_fn(model_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss/train", loss.item(), epoch * len(dataloader) + batch_idx
            )
        print("epoch: %d, loss: %f" % (epoch, loss.item()))
        # save model checkpoint
        if epoch % opt.save_epoch_freq == 0:
            torch.save(
                model.state_dict(), f"{opt.checkpoints_dir}/model_{epoch}.pth",
            )

    torch.save(model.state_dict(), f"{opt.checkpoints_dir}/model_final.pth")
    writer.close()


def optimizer_factory(model, opt, optimizer_id):
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if optimizer_id not in optimizers.keys():
        raise ValueError("Optimizer [%s] not recognized." % optimizer_id)
    return optimizers.get(optimizer_id)(model.parameters(), lr=opt.lr)


if __name__ == "__main__":
    train()
