
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
    
    def forward(self, waveforms):
        # waveforms： shape: (Batch, 1, Time)
        lengths = torch.ones(waveforms.shape[0], device=waveforms.device, dtype=torch.float)
        return self.noise_apply(waveforms.squeeze(1), lengths)

 
 class DropC(nn.Module): # random
        
    # def __init__(self, drop_length_low=5000, drop_length_high=5000, drop_count_low=4, drop_count_high=5) -> None:
    def __init__(self, drop_length_low=2000, drop_length_high=4000, drop_count_low=0, drop_count_high=5) -> None:
        super().__init__()

        drop_start = 0
        drop_end = None

        drop_prob = 1
        
        # noise_factor = 0.0
        drop_chunk_noise_factor = 0
        noise_factor = drop_chunk_noise_factor

        self.drop_chunk_instance = DropChunk(drop_length_low=drop_length_low, drop_length_high=drop_length_high,
            drop_count_low=drop_count_low, drop_count_high=drop_count_high,
            drop_start=drop_start, drop_end=drop_end, drop_prob=drop_prob,
            noise_factor=noise_factor)
    
    def forward(self, audios, length=None):
        length = length if length is not None else \
        torch.tensor([1.0] * audios.shape[0], device=audios.device, dtype=torch.long)
        audios = audios.squeeze(1)
