import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

import sandbox.block_nn as nii_nn
import torch.nn.functional as F

from sandbox.util_frontend import LFCC



class Model(nn.Module):

    def __init__(self, d_args=None):
        super(Model, self).__init__()

        # this is just for loading the model
        self.input_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.input_std = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_std = nn.Parameter(torch.zeros(1), requires_grad=False)
        # this is useless

        self.v_out_class = 2

        self.m_transform = []

        self.m_before_pooling = []
        # 2nd part of the classifier
        self.m_output_act = []
        # front-end
        self.m_frontend = []
        
        # confidence predictor
        self.m_conf = []

        lfcc_dim = 20 * 3

        self.m_transform.append(
            nn.Sequential(
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
            nn.MaxPool2d([2, 2], [2, 2]),
            
            nn.Dropout(0.7))
        )


        self.m_before_pooling.append(
            nn.Sequential(
                nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32),
                nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32)
            )
        )

        self.m_output_act.append(nn.Linear((lfcc_dim // 16) * 32, self.v_out_class))

        self.m_frontend.append(LFCC(fl=320, fs=160, fn=512, sr=16000, filter_num=20, with_energy=True))

        self.m_frontend = nn.ModuleList(self.m_frontend)
        self.m_transform = nn.ModuleList(self.m_transform)
        self.m_output_act = nn.ModuleList(self.m_output_act)
        self.m_before_pooling = nn.ModuleList(self.m_before_pooling)
        # output 
        

    def forward(self, x, Freq_aug=None):
        batch_size = x.size()[0]

        # x size (batch_size, data_length)

        # (batch, frame_num, feat_dim)
        x = self.m_frontend[0](x)

        # x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim) 

        hidden_features = self.m_transform[0](x.unsqueeze(1))
        # unsqueeze to (batch, 1, frame_length, fft_bin)

        #  3. (batch, channel, frame//N, feat_dim//N) ->
        #     (batch, frame//N, channel * feat_dim//N)
        #     where N is caused by conv with stride
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.view(batch_size, frame_num, -1)
        
        #  4. pooling
        #  4. pass through LSTM then summing
        hidden_features_lstm = self.m_before_pooling[0](hidden_features)
        
        out_emb = (hidden_features_lstm + hidden_features).mean(1)

        m_out = self.m_output_act[0](out_emb)

        return None, m_out


if __name__ == "__main__":
    m = Model()