from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn

class MI_FGSM_ensemble_iter():
    def __init__(self, models, attack_config) -> None:
        
        self.loss_fn = get_loss_fn( attack_config['loss'] )
        self.epsilon = attack_config['epsilon']

        self.models = models
        self.num_models = len(models)

        self.max_iter = attack_config['max_iter']
        self.alpha = attack_config['alpha']

        self.mu = attack_config['decay_factor']
        self.device = "cuda"

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

                # Calculate gradients of model in backward pass
                
                # Collect datagrad
                data_grad = torch.autograd.grad(loss, adv_x,
                                        retain_graph=False, create_graph=False)[0]

                g = self.mu * g + nn.functional.normalize(data_grad, p=1)

                # Create the adversarial audio
                adv_x = adv_x.detach() + self.alpha * g.sign() 

                delta = torch.clamp(adv_x - x, min=-self.epsilon, max=self.epsilon)

                # we need to clamp the data in (-1, 1)
                adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()

        return adv_x
