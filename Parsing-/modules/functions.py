import torch
import torch.nn.functional as F

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"

class InPlaceABN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Store stats
        ctx.save_for_backward(x, weight, bias)

        x = F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)

        if activation == ACT_LEAKY_RELU:
            x = F.leaky_relu(x, negative_slope=slope, inplace=True)
        elif activation == ACT_RELU:
            x = F.relu(x, inplace=True)
        elif activation == ACT_ELU:
            x = F.elu(x, inplace=True)

        return x, running_mean, running_var

    @staticmethod
    def backward(ctx, grad_output, *_):
        # Only return grad_input, rest are None
        return grad_output, None, None, None, None, None, None, None, None, None

# Use same fallback for sync version (single GPU / CPU)
inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABN.apply

