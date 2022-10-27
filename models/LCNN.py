import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable

import sandbox.block_nn as nii_nn
import core_modules.a_softmax as nii_a_softmax


class LCNN(nn.Module):
    def __init__(self):
        super(LCNN, self).__init__()
         
        v_feat_dim = 128
        nb_classes = 2

        self.lcnn = nn.Sequential(
            nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
            nii_nn.MaxFeatureMap2D(),
            torch.nn.MaxPool2d([2, 2], [2, 2]),

            nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            nii_nn.MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
            nii_nn.MaxFeatureMap2D(),

            torch.nn.MaxPool2d([2, 2], [2, 2]),
            nn.BatchNorm2d(48, affine=False),

            nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
            nii_nn.MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
            nii_nn.MaxFeatureMap2D(),

            torch.nn.MaxPool2d([2, 2], [2, 2]),

            nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
            nii_nn.MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
            nii_nn.MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),

            nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            nii_nn.MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
            nii_nn.MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]))

        lfcc_dim = 20

        self.nb_classes = 2
        self.fc_in_dim = 25 * 3
        
        self.fc = nn.Sequential(
                    nn.Dropout(0.7),
                    nn.Linear(32 * self.fc_in_dim, 160),
                    nii_nn.MaxFeatureMap2D(),
                    nn.Linear(80, self.nb_classes)
                )


    def forward(self, x):
        
        x = self.lcnn(x)

        
        # print(x.size())

        x = torch.flatten(x, start_dim=1)

        # print(x.size())

        x = self.fc(x)

        x = F.softmax(x, dim=1)

        return x
