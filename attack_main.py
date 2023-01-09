import argparse
from distutils import core
from email.mime import audio
import json
from operator import ge
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from scipy.io.wavfile import write

import attacks

import soundfile as sf


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


from data_utils import (Dataset_ASVspoof2019_attack, genSpoof_list)

from main_utils import get_model


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    
    # define database related paths
    output_dir = Path(args.output_dir)

    database_path = Path(config["database_path"])
    eval_database_path = Path(config["eval_database_path"])

    # track == "toy_example"
    eval_trial_path = (database_path / "cm_protocols/eval.txt")

    # define model related paths
    model_tag = config["model_tag"]
    model_tag = output_dir / model_tag
    
    # model_save_path = model_tag / "pretained_weights"
    writer = SummaryWriter(model_tag)


    # define path for adversarial audio
    adv_path =  model_tag / 'eval'

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # case 1: eval 
    if args.eval:
        # load the new data first
        print('reading from {}'.format(eval_database_path))

        models = get_pretrained_models(config["blackbox_model"], device)

        adv_loader = get_adv_loader(eval_database_path, config, eval=True)

        print("Start evaluation...")
        adv_path = os.path.join(eval_database_path, 'eval')
        attack_evaluation(adv_loader, models, device)

        print("DONE.")

        sys.exit(0)
        # exit after evaluation

    # case 2: attack

    os.makedirs(model_tag, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # attack evaluation
    attack_eval = args.attack_eval

    # load model

    model = get_pretrained_models(config["whitebox_models"], device)

    blackbox_models = get_pretrained_models(config["blackbox_model"], device) if attack_eval else None
    if attack_eval:
        for blackbox_model in blackbox_models:
            blackbox_model.eval()

    # load attack configure
    with open(config["attack_config"], "r") as f_json:
        attack_config = json.loads(f_json.read())

    attack_model = get_attack_model(attack_config, model)

    # define dataloaders
    eval_loader = get_adv_loader(
        database_path, config, eval=False)

    print("Start attacks...")
    attack(eval_loader, attack_model, blackbox_models, adv_path, device, attack_eval)

    return 0 # now we first make it run


def get_pretrained_models(model_config_path: list, device):
    if len(model_config_path) == 0:
        return [attacks.no_attack.NoAttack]

    models = []    
    for m in model_config_path:
        with open(m, "r") as f_json:
            config = json.loads(f_json.read())
            # model_configs.append(config)

        model = get_model(config["model_config"], device)

        checkpoint = torch.load(config["model_path"], map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded : {}".format(config["model_path"]))

        model.eval()
        models.append(model)

    return models


def get_attack_model(attack_config, model):
    attack_module = import_module("attacks.{}".format(attack_config["attack_type"]))
    _model = getattr(attack_module, attack_config["attack_type"])
    attack_model = _model(model, attack_config)
    return attack_model


# we only care about the success rate
def attack_evaluation(data_loader: DataLoader, models, device):

    count = 0

    for model in models:

        model.eval()
        correct = 0
        total_len = 0
        count += 1

        for index, (origin, label, utt_id)  in enumerate(data_loader):

            origin = origin.to(device)
            # print(origin)
            label = label.view(-1).type(torch.int64).to(device)
            #print(label)

            total_len += label.size()[0]
            
            _, out1 = model(origin)

            pred1 = out1.argmax(1, keepdim=True).view(-1) # get the index of the max log-probability

            correct += ((pred1 == label).sum())

        success_rate = (1 - correct / float(total_len))
        print("model is: ", count)
        print("Success Rate is ", success_rate * 100)


def attack(data_loader: DataLoader, attack_model, blackbox_models, adver_dir, device: torch.device, attack_eval: bool):
    
    flac_path = os.path.join(adver_dir, 'flac')
    os.makedirs(flac_path, exist_ok=True)
    print('is saving to path ', flac_path)
    # note the dir shoule be something like 
    # attack_result/toy_example_aasist_attack_ep100_bs24/adv_audio/flac
    torch.backends.cudnn.enabled = False

    # attack_iter_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    attack_iter_list = [1]


    num_models = len(blackbox_models)
    num_iter = len(attack_iter_list)

    # if blackbox_models:
    #     blackbox_model = blackbox_models[0]

    # correct = [0] * num_models
    # total_len = 1000
    correct = torch.zeros(num_iter, num_models).to(device)

    for index, (origin, label, utt_id) in enumerate(data_loader):
        # print(torch.min(origin, dim=1))
        origin = origin.to(device)
        label = label.view(-1).type(torch.int64).to(device)
        # print(label)
        
        des_path = os.path.join(adver_dir, utt_id[0] + '.flac')
        if os.path.exists(des_path):
            print('*' * 40, index, utt_id[0], 'Exists, SKip', '*' * 40)
        
        adver_audio = origin.clone()

        # attack_model.get_origin(origin)

        for k, iter in enumerate(attack_iter_list):
            
            # print(k)
            # attack_model.change_max_iter(5)

            adver_audio = attack_model.attack(adver_audio, label)
            adver_audio = adver_audio.clone()

            if attack_eval:
                # total_len += label.size()[0]
                
                for i, model in enumerate(blackbox_models):
                    _, out1 = model(adver_audio)
                    pred1 = out1.argmax(1, keepdim=True).view(-1) # get the index of the max log-probability
                    correct[k, i] += ((pred1 == label).sum())

                print("All the adversarial examples in the batch has been evaluated")

            else: 

                for adv, id in zip(adver_audio, utt_id):
                    fs = 16000 # sampling rate of LA is 16k
                    adv_path = os.path.join(adver_dir, 'flac', id + '.pt') # test wav
                    adv = adv.cpu().detach()
                    #sf.write(adv_path, adv, samplerate=fs)

                    torch.save(adv, adv_path)

                print('All adversarial audio in the batch are saved!')

    if attack_eval:
        print(1000 - correct)
        # for k, iter in enumerate(attack_iter_list):
        #     for i in range(num_models):
        #         success_rate = 1 - correct[k, i] / 1000.0
        #         print("num iter: ", iter)
        #         print("model: ", i)
        #         print("The success rate is:", success_rate * 100)


def get_adv_loader(
        database_path: str,
        config: dict,
        eval) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    # if track == 'toy_example':
    print('USING toy_example')
    eval_database_path = database_path / "eval"

    if eval: 
        eval_trial_path = "dataset/Adver_eval/cm_protocols/eval.txt"
    else:
        # pass 
        eval_trial_path = database_path / "cm_protocols/eval.txt"
    # eval_trial_path = "dataset/toy_example/cm_protocols/eval.txt"

    d_label_eval, file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=False)
    
    print("Looking at database", eval_database_path)
    print('No. of files: ', len(file_eval))

    eval_set = Dataset_ASVspoof2019_attack(list_IDs=file_eval,
                                        labels=d_label_eval,
                                        base_dir=eval_database_path,
                                        eval=eval)

    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return eval_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./attack_result",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument(
        "--attack_eval",
        action="store_true",
        help="when this flag is given, attack and evaluate but do not save the audio")
    main(parser.parse_args())
