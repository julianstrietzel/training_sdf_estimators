import abc

import torch.nn as nn


class AbsRegressionModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, opt):
        super(AbsRegressionModel, self).__init__()
        self.opt = opt

    @abc.abstractmethod
    def init_weights(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleRegressionModel(AbsRegressionModel):
    def __init__(self, opt):
        super(SimpleRegressionModel, self).__init__(opt)
        hidden_sizes = opt.hlayer_sizes
        input_size = opt.kdtree_num_samples * 3
        self.fc_layers = [nn.Linear(input_size, hidden_sizes[0])] + [
            nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ]

        self.fc_out = nn.Linear(hidden_sizes[-1], 1)
        self.relu = nn.ReLU()

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
