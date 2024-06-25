from .functional import revgrad
import torch
from torch import nn

class GradientReversal(nn.Module):
    """
    Gradient reversal layer from the paper "Unsupervised Domain Adaptation by Backpropagation"
        available at https://arxiv.org/abs/1409.7495

    Existing implementation from https://github.com/tadeephuy/GradientReversal/tree/5d9857d63fae504b712b3280f0bed71c7503e0c2
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)