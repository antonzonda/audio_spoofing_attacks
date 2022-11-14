from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn

class SRS_ensemble():
    def __init__(self, models, attack_config) -> None:


        self.epsilon = attack_config['epsilon']

        self.models = models
        self.num_models = len(models) # k

        self.max_iter = attack_config['max_iter']
        # self.alpha = attack_config['alpha'] # step size
        self.alpha = self.epsilon / self.max_iter
        # mu = attack_config['decay_factor']
        self.device = "cuda"

        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.loss_fn =  nn.CrossEntropyLoss(weight=weight, reduction='none')

        # # mu := 0, sig := 1
        # mu = torch.zeros(self.batch_size, self.num_models).to(self.device)
        # sig = torch.ones(self.batch_size, self.num_models).to(self.device)


    def attack(self, x, y):

        x = x.clone() # avoid influencing
        # x.requires_grad = True

        # x.requires_grad = True

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        adv_x = x.clone().detach()

        batch_size = x.size()[0]

        mu = torch.zeros(batch_size, self.num_models).to(self.device)
        sig = torch.ones(batch_size, self.num_models).to(self.device)

        for t in range(1, self.max_iter+1):
            # print(t)
            
            adv_x.requires_grad = True

            loss_list = []
            for model in self.models:
                model.eval()
                _, out = model(adv_x)
                # out = F.softmax(out)

                loss_list.append(self.loss_fn(out, y))
            
            # (batch_size, num_models)
            f = torch.stack(loss_list, dim=1)
            # print(f.size())

            mu = mu + (f - mu) / t
            sig = sig + ((f - mu)**2 - sig) / t
            
            loss = torch.sum((f - mu) / torch.sqrt(sig))

            # print(loss.size())

            # Collect datagrad
            data_grad = torch.autograd.grad(loss, adv_x,
                                       retain_graph=False, create_graph=False)[0]
            
            # print('grad', data_grad.size())

            # Create the adversarial audio
            adv_x = adv_x.detach() + self.alpha * data_grad.sign() 

            delta = torch.clamp(adv_x - x, min=-self.epsilon, max=self.epsilon)

            # we need to clamp the data in (-1, 1)
            adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()

        return adv_x
