from torch import nn
import torch

from attacks.loss import get_loss_fn

# iterative FGSM ensemble attack
class I_FGSM_ensemble():
    def __init__(self, models, attack_config) -> None:
        
        self.loss_fn = get_loss_fn( attack_config['loss'] )
        self.epsilon = attack_config['epsilon']

        self.models = models
        self.num_models = len(models)

        self.max_iter = attack_config['max_iter']
        # self.alpha = attack_config['alpha']
        self.alpha = self.epsilon / 5 # step size
        self.device = "cuda"

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

            # out_list = []
            # for model in self.models:
            #     model.eval()
            #     _, out = model(adv_x)
            #     out_list.append(out)
            
            # # print(out_list[0].size())

            # # ensemble logits
            # out_mean = torch.mean(torch.stack(out_list), dim=0)

            # loss = self.loss_fn(out_mean, y)

            # ensemble losses

            loss_list = []
            for model in self.models:
                model.eval()
                _, out = model(adv_x)

                loss_x = self.loss_fn(out, y)
                loss_list.append(loss_x)
            
            loss = torch.mean( torch.stack(loss_list))

            # self.model.zero_grad()
            # Calculate gradients of model in backward pass
            # loss.backward()
            
            # Collect datagrad
            data_grad = torch.autograd.grad(loss, adv_x,
                                       retain_graph=False, create_graph=False)[0]

            # x.grad.zero_()
            # Create the adversarial audio
            adv_x = adv_x.detach() + self.alpha * data_grad.sign()

            # we need to clamp the data in (-1, 1)
            delta = torch.clamp(adv_x - x, min=-self.epsilon, max=self.epsilon)
            adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()

        return adv_x
