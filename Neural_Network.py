import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as ttensor
from torch import float as tfloat
from torch import relu
from math import ceil


class Net(nn.Module):

    def __init__(self, layers, cnn_layers, cnn_kernels, img_size):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.hidden = nn.ModuleList()
        self.cnn = nn.ModuleList()
        self.maxpooling = nn.ModuleList()
        self.bn = nn.ModuleList()

        i = 0
        temp = img_size
        for input, output in zip(cnn_layers, cnn_layers[1:]):

            self.cnn.append(nn.Conv2d(input, output, kernel_size=cnn_kernels[i], stride=1))
            temp = (temp - cnn_kernels[i] + 1)

            self.maxpooling.append(nn.MaxPool2d(kernel_size=2, stride=2))
            temp = int(ceil((temp - 2 + 1) / 2))

            self.bn.append(nn.BatchNorm2d(output))
            i += 1


        # Number of nodes after convolution and maxpooling
        layers[0] = (temp ** 2) * output

        for input, output in zip(layers, layers[1:]):
            layer = nn.Linear(input, output)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu") # "He" method
            self.hidden.append(layer)


        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        self.fc_loc = nn.Sequential(
            nn.Linear(160, img_size),
            nn.ReLU(True),
            nn.Linear(img_size, 3 * 2)
            )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(ttensor([1, 0, 0, 0, 1, 0], dtype=tfloat))


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 160)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


    def forward(self, x):
        x = self.stn(x)
        
        l = len(self.maxpooling)
        for i, layer in zip(range(l), self.cnn):
            x = F.leaky_relu(layer(x))
            x = self.maxpooling[i](x)
            x = self.bn[i](x)

        x = x.view(x.size(0), -1)
        l = len(self.hidden)
        for i, layer in zip(range(l), self.hidden):
            if i < l - 1:
                x = self.dropout(x)
                x = relu(layer(x))
            else:
                x = nn.functional.log_softmax(layer(x), dim=1)

        return x
