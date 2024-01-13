import abc
import os
from typing import Optional

import torch
import torch.nn as nn


class AbsRegressionModel(nn.Module):
    def __init__(self, opt):
        super(AbsRegressionModel, self).__init__()
        self.opt = opt

    def init_weights(self):
        print("No weight initialization implemented for this model.")

    def load_weights(self, path: Optional[str] = None):
        if path is not None:
            self.load_state_dict(torch.load(path))
        else:
            # check if expr_dir is absolute path
            expr_dir = (
                self.opt.expr_dir
                if self.opt.expr_dir.startswith(".")
                else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    self.opt.expr_dir,
                )
            )

            self.load_state_dict(
                torch.load(
                    os.path.join(
                        expr_dir,
                        f"model_{self.opt.which_epoch if hasattr(self.opt, 'which_epoch') else 'final'}.pth",
                    )
                )
            )


class SimpleRegressionModel(AbsRegressionModel):
    def __init__(self, opt):
        super(SimpleRegressionModel, self).__init__(opt)
        hidden_sizes = opt.hlayer_sizes
        input_size = opt.kdtree_num_samples * 6
        self.fc_layers = [nn.Linear(input_size, hidden_sizes[0])] + [
            nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ]
        self.fc_out = nn.Linear(hidden_sizes[-1], 1)
        self.relu = nn.ReLU()

        self.fc_layers = [layer.to("cuda") for layer in self.fc_layers]
        self.fc_out = self.fc_out.to("cuda")

    def init_weights(self):
        init_weights(
            self.fc_layers + [self.fc_out],
            init_type=self.opt.init_type,
            gain=self.opt.init_gain,
        )

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        x = self.fc_out(x)
        return x


class ConvoRegressionModel(AbsRegressionModel):
    def __init__(self, opt):
        super(ConvoRegressionModel, self).__init__(opt)
        # input_size will be (batch_size
        self.cv1 = nn.Conv1d(6, 12, 1)
        self.act1 = nn.LeakyReLU()
        self.cv2 = nn.Conv1d(12, 16, 1)
        self.act2 = nn.LeakyReLU()
        self.cv3 = nn.Conv1d(16, 8, 3)
        self.act3 = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16, 8)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.act5 = nn.ReLU()

    def init_weights(self):
        init_weights(
            [self.cv1, self.cv2, self.cv3, self.fc1, self.fc2,],
            init_type=self.opt.init_type,
            gain=self.opt.init_gain,
        )

    def forward(self, x):
        x = self.act1(self.cv1(x))
        x = self.act2(self.cv2(x))
        x = self.act3(self.cv3(x))
        x = self.flatten(x)
        x = self.act4(self.fc1(x))
        x = self.act5(self.fc2(x))
        return x


def init_weights(layers, init_type="normal", gain=0.02):
    """
    Initialize network weights.
    :param init_type: normal | xavier | kaiming | orthogonal
    :param gain: scaling factor for normal, xavier and orthogonal.
    :return:
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    map(init_func, layers)
