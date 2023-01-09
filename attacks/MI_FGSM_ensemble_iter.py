from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn
from attacks.Iter_ensemble import Iter_ensemble

class MI_FGSM_ensemble_iter(Iter_ensemble):
    def __init__(self, models, attack_config):
        super().__init__(models, attack_config)
        
        self.mu = attack_config['decay_factor']

    def attack_method(self, delta, y, model):
        delta.requires_grad = True

        model.eval()
        _, out = model(delta)

        loss = self.loss_fn(out, y)

        data_grad = torch.autograd.grad(loss, delta,
                                    retain_graph=False, create_graph=False)[0]

        # x.grad.zero_()
        # Create the adversarial audio

        self.g = self.mu * self.g + nn.functional.normalize(data_grad)

        delta = delta.detach() + self.alpha * self.g.sign()

        delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon) 

        return delta

    def attack(self, x, y):
 
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        delta = (torch.rand_like(x).to(self.device) - 0.5) * 2 * self.epsilon
        delta.requires_grad = True

        self.g = torch.zeros_like(x).to(self.device)

        for t in range(self.max_iter):

            for model in self.models:

                delta = self.attack_method(delta, y, model)

        adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()

        return adv_x