
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable

from models.LCNN import LCNN
from sandbox.util_frontend import LFCC


class Model(nn.Module):
    def __init__(self, d_args):
        super(Model, self).__init__()

        self.front_end = LFCC(320, 160, 512, 16000, 20, with_energy=True)
        self.back_end = LCNN()

    def forward(self, x, Freq_aug=None):
        x = self.front_end.forward(x)

        # x size = (batch_size, frame_num, frame_feat_dim)
        x = torch.unsqueeze(x, dim=1)
        # x size = (batch_size, 1, 404, 60)
        # print(x.size())

        output = self.back_end.forward(x)
        return None, output
