
from logging import raiseExceptions
import torch 
import torch.nn as nn


def get_loss_fn(loss_fn_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")


    if loss_fn_name == 'CrossEntropy':
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise "No such loss function"
