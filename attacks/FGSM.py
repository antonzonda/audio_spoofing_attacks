from torch import nn
import torch

from attacks.loss import get_loss_fn

class FGSM():
    def __init__(self, model, attack_config) -> None:
        
        self.loss_fn = get_loss_fn( attack_config['loss'] )
        self.epsilon = attack_config['epsilon']
        self.model = model
        self.device = 'cuda'
         

    def attack(self, x, y):

        # print(torch.max(x))

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        self.model.eval()
        x.requires_grad = True

        _, out = self.model(x)

        loss = self.loss_fn(out, y)

        self.model.zero_grad()
        # Create the adversarial audio


        # we need to clamp the data in [-1, 1]
        
        grad = torch.autograd.grad(loss, x,
                                   retain_graph=False, create_graph=False)[0]

        adver_x = x + self.epsilon * grad.sign()
        adver_x = torch.clamp(adver_x, min=-1, max=1).detach()

        return adver_x
