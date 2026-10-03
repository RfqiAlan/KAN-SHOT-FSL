import torch.nn as nn
from torch import Tensor
import argparse
from typing import Tuple

class FSmethod(nn.Module):
    """
    Abstract class for few-shot methods
    """
    def __init__(self, args: argparse.Namespace):
        super(FSmethod, self).__init__()
        self.args = args

    def forward(
        self,
        x_s: Tensor,
        x_q: Tensor,
        y_s: Tensor,
        y_q: Tensor,
        model: nn.Module
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
