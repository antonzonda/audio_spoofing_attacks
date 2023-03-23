
from logging import raiseExceptions
import torch 
import torch.nn as nn


def get_loss_fn(loss_fn_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")


    if loss_fn_name == 'CrossEntropy':
        # weight = torch.FloatTensor([0.1, 0.9]).to(device)
        return nn.CrossEntropyLoss()
    elif loss_fn_name == 'BCE':
        
        def binaryCE(x, y):
            BCE = nn.BCELoss()
            m = nn.Sigmoid()
            
            return BCE(m(x[:,1]), y.float())
        
        return binaryCE
        
    elif loss_fn_name == 'Logits':
        
        def logitLoss(x, y):
            return sum(x[:,1])
        
        return logitLoss
    else:
        raise "No such loss function"
