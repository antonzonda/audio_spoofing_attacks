import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

import sandbox.block_nn as nii_nn
import torch.nn.functional as F


import fairseq


class SSLModel():
    def __init__(self, cp_path, ssl_orig_output_dim):
        """ SSLModel(cp_path, ssl_orig_output_dim)
        
        Args:
          cp_path: string, path to the pre-trained SSL model
          ssl_orig_output_dim: int, dimension of the SSL model output feature
        """
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.out_dim = ssl_orig_output_dim
        return

    def extract_feat(self, input_data):
        """ feature = extract_feat(input_data)
        Args:
          input_data: tensor, waveform, (batch, length)
        
        Return:
          feature: tensor, feature, (batch, frame_num, feat_dim)
        """
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.eval()

        emb = self.model(input_data, mask=False, features_only=True)['x']
        return emb

# not good way, but the path is fixed
ssl_path = './models/_w2v/wav2vec_small.pt'
# This model produces 768 output feature dimensions (per frame)
ssl_orig_output_dim = 768
# SSL model is declared as a global var since it is fixed
g_ssl_model = SSLModel(ssl_path, ssl_orig_output_dim)


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

        v_feat_dim = 128

        self.m_before_pooling.append(
            nn.Sequential(
                nii_nn.BLSTMLayer((v_feat_dim//16) * 32, (v_feat_dim//16) * 32),
                nii_nn.BLSTMLayer((v_feat_dim//16) * 32, (v_feat_dim//16) * 32)
            )
        )

        self.m_output_act.append(nn.Linear((v_feat_dim // 16) * 32, self.v_out_class))

        self.m_frontend.append(nn.Linear(g_ssl_model.out_dim, v_feat_dim))

        self.m_frontend = nn.ModuleList(self.m_frontend)
        self.m_transform = nn.ModuleList(self.m_transform)
        self.m_output_act = nn.ModuleList(self.m_output_act)
        self.m_before_pooling = nn.ModuleList(self.m_before_pooling)
        # output 
        
        self.m_frontend_model = g_ssl_model.extract_feat


    def forward(self, x, Freq_aug=None):
        batch_size = x.size()[0]

        # x size (batch_size, data_length)
        x = g_ssl_model.extract_feat(x)

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

        m_out = F.softmax(m_out, dim=1)

        return None, m_out


