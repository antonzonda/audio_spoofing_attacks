from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn
from attacks.MI_FGSM_ensemble import MI_FGSM_ensemble

class MI_FGSM_ensemble_iter(MI_FGSM_ensemble):
    def __init__(self, models, attack_config) -> None:
        
        super().__init__(models, attack_config)

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

                _, out = model(adv_x)
            
                loss = self.loss_fn(out, y)

                g, adv_x = self.attack_method(adv_x, x, loss, g)

        return adv_x
