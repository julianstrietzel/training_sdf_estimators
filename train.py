import torch
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
    model = model.train()
    model.init_weights()

    # init loss
    loss_fn = loss_factory(opt.loss)

    # init optimizer
    optimizer = optimizer_factory(opt.optimizer, model, opt.lr)

    # lr scheduler
    scheduler = lr_scheduler_init(opt.lr_scheduler, optimizer, opt)

    # tensorboard
    writer = SummaryWriter("./runs/" + opt.expr_dir.split("/")[-1])

    def logging(context, global_step, loss):
        writer.add_scalar(
            opt.loss + "_loss/train/" + context, loss.item(), global_step,
        )
        writer.add_scalar(
            "learning_rate/" + context, optimizer.param_groups[0]["lr"], global_step,
        )

    # training loop
    for epoch in range(opt.epochs):
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            model_input, target = batch
            model_output = model(model_input)
            loss = loss_fn(model_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging("batch", epoch * len(dataloader) + batch_idx, loss)
        if epoch > 25:
            scheduler.step()
        print(
            "epoch: %d, loss: %f, lr: %f"
            % (epoch, loss.item(), optimizer.param_groups[0]["lr"])
        )
        # save model checkpoint
        if epoch % opt.save_epoch_freq == 0:
            torch.save(
                model.state_dict(), f"{opt.expr_dir}/model_{epoch}.pth",
            )
        logging("epoch", epoch, loss)

    torch.save(model.state_dict(), f"{opt.expr_dir}/model_final.pth")
    writer.close()


if __name__ == "__main__":
    train()
