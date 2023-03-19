from torch import nn
import torch

import torch.nn.functional as F

from attacks.loss import get_loss_fn

# becasue the difference of losses among different models are high
# we adopt new methods 

class MI_FGSM_ensemble_avg():
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

        loss_avg = torch.zeros(self.num_models, device=x.device, dtype=x.dtype)
        loss_var = torch.zeros(self.num_models, device=x.device, dtype=x.dtype)

        for t in range(self.max_iter):

            adv_x.requires_grad = True
            all_grads_max = []
            all_grads_norm_max = []
            all_loss = []
            all_loss_norm = []
            
            for model_id, model in enumerate(self.models):
                
                model.eval()
                _, out = model(adv_x)
                # out_list.append(out)
                loss_i = self.loss_fn(out, y)
                grad_i = torch.autograd.grad(loss_i, adv_x, retain_graph=False, create_graph=False)[0]
                all_grads_max.append(round(grad_i.max().item(), 3))
                all_loss.append(round(loss_i.item(), 3))
                
                with torch.no_grad():

                    loss_avg[model_id] = loss_avg[model_id] + (1.0 / (t+1)) * (loss_i.item() - loss_avg[model_id])
                    loss_var[model_id] = loss_var[model_id] + (1.0 / (t+1)) * ((loss_i.item() - loss_avg[model_id]) ** 2 - loss_var[model_id])
                    
                    if t > 0:
                        if loss_var[model_id] == 0.:
                            loss_var_eps = 1e-8
                        else:
                            loss_var_eps = loss_var[model_id] 
                        loss_i.data = (loss_i  - loss_avg[model_id]) / (loss_var_eps ** 0.5)
                        grad_i.data = grad_i / (loss_var_eps ** 0.5)
                    else:
                        loss_i.data = loss_i - loss_avg[model_id]   

                    if model_id == 0:
                        data_grad = grad_i
                    else:
                        data_grad += grad_i
                    
                    all_grads_norm_max.append(round(grad_i.max().item(), 3)) 
                    all_loss_norm.append(round(loss_i.item(), 3))
                  
                
            g = self.mu * g + nn.functional.normalize(data_grad, p=1)

            # Create the adversarial audio
            adv_x = adv_x.detach() + self.alpha * g.sign() 

            delta = torch.clamp(adv_x - x, min=-self.epsilon, max=self.epsilon)

            # we need to clamp the data in (-1, 1)
            adv_x = torch.clamp(x + delta, min=-(1-2**(-15)), max=1-2**(-15)).detach()
            
            print(t, all_grads_max, all_grads_norm_max, all_loss, all_loss_norm)

        return adv_x
