from torch import nn
import torch

from attacks.loss import get_loss_fn

# iterative Ensemble Method
class Iter_ensemble():
    def __init__(self, models, attack_config) -> None:
        
        self.loss_fn = get_loss_fn( attack_config['loss'] )
        self.epsilon = attack_config['epsilon']

        self.models = models
        self.num_models = len(models)

        self.max_iter = attack_config['max_iter']
        self.alpha = attack_config['alpha'] # step size
        # self.alpha = self.epsilon / self.num_models
        self.device = "cuda"
        
    def attack_method(self, delta, y, model):
        return delta

    def attack(self, x, y):

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        delta = (torch.rand_like(x).to(self.device) - 0.5) * 2 * self.epsilon
        delta.requires_grad = True

        for t in range(self.max_iter):

            for model in self.models:

                delta = self.attack_method(delta, y, model)

        adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()

        return adv_x
