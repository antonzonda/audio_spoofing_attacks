from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn
from attacks.MI_FGSM_ensemble import MI_FGSM_ensemble
from attacks.Noise import NOISE


class MI_FGSM_noise_iter(MI_FGSM_ensemble):
    def __init__(self, models, attack_config) -> None:
        
        super().__init__(models, attack_config)
        self.noise = NOISE()
        self.noise_avg = attack_config['noise_avg']

    def attack(self, x, y):

        x = x.clone() # avoid influencing
        # x.requires_grad = True

        # x.requires_grad = True

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        g = torch.zeros_like(x).to(self.device)
        adv_x = x.clone().detach()

        for t in range(self.max_iter):
            # print(t)

            for model in self.models:
                model.eval()
                adv_x.requires_grad = True

                # out_list = []
                loss = torch.zeros([1], device=self.device)
                for s in range(self.noise_avg):
                    adv_x_noise = self.noise(adv_x)
                    _, temp_out = model(adv_x_noise)
                    temp_loss = self.loss_fn(temp_out, y)
                    # print(temp_loss)
                    loss = loss + temp_loss

                loss = loss / self.noise_avg

                # print(loss)
                g, adv_x = self.attack_method(adv_x, x, loss, g)

        return adv_x
