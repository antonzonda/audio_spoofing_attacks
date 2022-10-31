import argparse
from ast import Param
import json
import os
import sys
import warnings

from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import metrics
import numpy as np


from utils import create_optimizer, seed_worker, set_seed, str_to_bool


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    
    print("Model arch is:", model_config["architecture"])
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)

    # for param in model.parameters():
    #     print(param)
    nb_params = sum([param.nelement() for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []

    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    # print(trial_path)
    # print(len(trial_lines), len(fname_list), len(score_list) )

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)

        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss


# evaluate the development set, and returns the equal error rate 
def dev_epoch(dev_loader: DataLoader, model, device: torch.device):

    model.eval()
        
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)

    score_list = []
    label_list = []

    for batch_x, batch_y in dev_loader:

        batch_x = batch_x.to(device)
        _, batch_out = model(batch_x, None)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        score_list.extend(batch_score)
        label_list.extend(batch_y.numpy().ravel())

    return compute_eer(np.array(label_list), np.array(score_list))


def compute_eer(y, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
    fnr = 1 - tpr

    t = np.nanargmin(np.abs(fnr-fpr))
    eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
    eer = (eer_low+eer_high)*0.5

    return eer
