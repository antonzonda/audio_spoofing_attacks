from torch import nn
import torch

from attacks.loss import get_loss_fn
from attacks.I_FGSM_ensemble import I_FGSM_ensemble
from attacks.Noise import NOISE

# iterative FGSM ensemble attack
class I_FGSM_noise_iter(I_FGSM_ensemble):
    def __init__(self, models, attack_config) -> None:
        
        super().__init__(models, attack_config)
        self.noise = NOISE()

    def attack(self, x, y):

        x = x.clone() # avoid influencing
        # x.requires_grad = True

        # x.requires_grad = True

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        adv_x = x.clone().detach()

        for t in range(self.max_iter):
            # print(t)
            adv_x.requires_grad = True

            for model in self.models:
                model.eval()

                # print(adv_x.size())
                adv_x = self.noise(adv_x)
                # print("add noise ", adv_x.size())
                _, out = model(adv_x)

                loss = self.loss_fn(out, y)

                adv_x = self.attack_method(adv_x, x, loss)

        return adv_x
