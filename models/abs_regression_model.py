import torch.nn as nn


class SimpleRegressionModel(nn.Module):
    def __init__(self, opt):
        super(SimpleRegressionModel, self).__init__()
        hidden_sizes = opt.hlayer_sizes
        input_size = opt.num_closest_points * 3
        self.fc_layers = [nn.Linear(input_size, hidden_sizes[0])] + [
            nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes - 1))
        ]
        self.fc_out = nn.Linear(hidden_sizes[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        x = self.fc_out(x)
        return x
