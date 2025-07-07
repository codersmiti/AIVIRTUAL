import torch
import torch.nn as nn
import torch.nn.functional as functional

from .functions import inplace_abn, inplace_abn_sync, ACT_RELU, ACT_LEAKY_RELU, ACT_ELU, ACT_NONE

class ABN(nn.Module):
    """Activated Batch Normalization (ABN)"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, activation="leaky_relu", slope=0.01):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias,
            self.training, self.momentum, self.eps
        )

        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def forward(self, x):
        x, _, _ = inplace_abn(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.training, self.momentum,
            self.eps, self.activation, self.slope
        )
        return x


class InPlaceABNSync(ABN):
    """InPlace ABN with cross-device sync (fallbacks to single-device on CPU)"""

    def forward(self, x):
        x, _, _ = inplace_abn_sync(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.training, self.momentum,
            self.eps, self.activation, self.slope
        )
        return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)



