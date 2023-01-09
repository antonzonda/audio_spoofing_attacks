from torch import nn
import torch

from attacks.loss import get_loss_fn
from attacks.Iter_ensemble import Iter_ensemble

# iterative FGSM ensemble attack
# using the iterative ensemble methods

class I_FGSM_ensemble_iter(Iter_ensemble):

    def attack_method(self, delta, y, model):
        delta.requires_grad = True

        model.eval()
        _, out = model(delta)

        loss = self.loss_fn(out, y)

        data_grad = torch.autograd.grad(loss, delta,
                                    retain_graph=False, create_graph=False)[0]

        # x.grad.zero_()
        # Create the adversarial audio
        delta = delta.detach() + self.alpha * data_grad.sign()

        delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)

        return delta