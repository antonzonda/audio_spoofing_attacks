
# preparation:
# 1. install speechbrain
# 2. 定位speechbrain的源码，把lobes/features.py和lobes/augment.py中的torch.no_grad()删掉

import torch.nn as nn
import torch
from speechbrain.processing.speech_augmentation import AddNoise

class NOISE(nn.Module):
    
    def __init__(self, csv_file=None, sorting="random", snr_low=0, snr_high=5, pad=False) -> None:
        super().__init__()

        self.noise_apply = AddNoise(csv_file=csv_file, sorting=sorting, snr_low=snr_low, snr_high=snr_high, pad_noise=pad)
    
    def forward(self, waveforms, avg=1):
        # waveforms： shape: (Batch, 1, Time)
        lengths = torch.ones(waveforms.shape[0], device=waveforms.device, dtype=torch.float)
        new_wav = torch.zeros_like(waveforms, device=waveforms.device, dtype=torch.float)
        for i in range(avg):
            new_wav += self.noise_apply(waveforms.squeeze(1), lengths)
        return new_wav / avg
