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

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF", "toy_example"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    
    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    eval_db_path = Path(config["eval_database_path"])

    if track == "toy_example":
        eval_trial_path = (database_path / "cm_protocols/eval.txt")
    else:
        eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # load attack configure
    with open(args.attack_config, "r") as f_json:
        attack_config = json.loads(f_json.read())

    # define model related paths
    model_tag = "{}_{}_{}".format(
        track,
        config["model_name"],
        attack_config["attack_type"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    
    model_tag = output_dir / model_tag
    
    # model_save_path = model_tag / "pretained_weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_tag, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # define path for adversarial audio
    adv_path =  model_tag / 'eval'

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)


    # attack with the pretrained model
    checkpoint = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded : {}".format(config["model_path"]))

    # case 1: eval 
    if args.eval:
        # load the new data first
        print('reading from {}'.format(database_path))
        adv_loader = get_adv_loader(database_path, config, eval=False)

        print("Start evaluation...")
        adv_path = os.path.join(eval_db_path, 'eval')
        attack_evaluation(adv_loader, model, device, adv_path)
        # attack(adv, None, model,  adv_path, device)

        print("DONE.")

        sys.exit(0)
        # exit after evaluation

    # case 2: attack
    # get attack 
    attack_model = get_attack_model(attack_config, model)

    # define dataloaders
    eval_loader = get_adv_loader(
        database_path, config, eval=False)

    print("Start attacks...")
    attack(eval_loader, attack_model, model,  adv_path, device)


    # start to evaluate the attack

    return 0 # now we first make it run



def get_attack_model(attack_config, model):
    attack_module = import_module("attacks.{}".format(attack_config["attack_type"]))
    _model = getattr(attack_module, attack_config["attack_type"])
    attack_model = _model(model, attack_config)
    return attack_model


# we only care about the success rate
def attack_evaluation(data_loader: DataLoader, model, device, adver_dir):

    model.eval()

    correct = 0
    total_len = 0

    for index, (origin, label, utt_id)  in enumerate(data_loader):

        origin = origin.to(device)
        # print(origin)
        label = label.view(-1).type(torch.int64).to(device)
        #print(label)

        total_len += label.size()[0]
        
        _, out1 = model(origin)

        pred1 = out1.argmax(1, keepdim=True).view(-1) # get the index of the max log-probability


        correct += ((pred1 == label).sum())

    success_rate = correct / float(total_len)
    print("Success Rate is ", success_rate)


def attack(data_loader: DataLoader, attack_model, model, adver_dir, device: torch.device):
    
    flac_path = os.path.join(adver_dir, 'flac')
    os.makedirs(flac_path, exist_ok=True)
    print('is saving to path ', flac_path)
    # note the dir shoule be something like 
    # attack_result/toy_example_aasist_attack_ep100_bs24/adv_audio/flac

    correct = 0
    ori_correct = 0
    total_len = 0

    model.eval()
    torch.backends.cudnn.enabled = False

    for index, (origin, label, utt_id) in enumerate(data_loader):
        # print(torch.min(origin, dim=1))
        origin = origin.to(device)
        label = label.view(-1).type(torch.int64).to(device)
        # print(label)
        
        des_path = os.path.join(adver_dir, utt_id[0] + '.flac')
        if os.path.exists(des_path):
            print('*' * 40, index, utt_id[0], 'Exists, SKip', '*' * 40)
        
        # print('id size', len(utt_id)) # utt id list of batch size
        # print('origin size', origin.size()) # batch_size * data_size 
        # print('label size', label.size()) # batch_size
        
        _, ori_out = model(origin)
        model.zero_grad()

        adver_audio = attack_model.attack(origin, label)
        adver_audio = adver_audio.clone()

        _, output = model(adver_audio)
        model.zero_grad()

        print(adver_audio - origin)

        # print(output)
 
        final_pred = output.argmax(1, keepdim=True).view(-1) # get the index of the max log-probability
        ori_pred = ori_out.argmax(1, keepdim=True).view(-1)
        # print(final_pred)

        # print("label", label)
        # print(final_pred)
        # print(ori_pred)
        
        correct += ((final_pred == label).sum())
        ori_correct += ((ori_pred == label).sum())
        total_len += label.size()[0]

        for adv, id in zip(adver_audio, utt_id):
            fs = 16000 # sampling rate of LA is 16k
            adv_path = os.path.join(adver_dir, 'flac', id + '.pt') # test wav
            adv = adv.cpu().detach()
            #sf.write(adv_path, adv, samplerate=fs)

            torch.save(adv, adv_path)    

        print('All adversarial audio in the batch are saved!')


    success_rate = correct / float(total_len)
    ori_sr = ori_correct / float(total_len)
    print("Success rate ", success_rate)
    print("Origin Success Rate ", ori_sr)

def get_adv_loader(
        database_path: str,
        config: dict,
        eval) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    track = config["track"] 
    prefix_2019 = "ASVspoof2019.{}".format(track)

    if track == 'toy_example':
        print('USING toy_example')
        eval_database_path = database_path / "eval"

        # eval_trial_path = (database_path / "cm_protocols/eval.txt")
        eval_trial_path = "dataset/toy_example/cm_protocols/eval.txt"

    else: # track == LA

        eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)
        eval_trial_path = (
            database_path /
            "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                track, prefix_2019))

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
    parser.add_argument("--attack_config",
                        dest="attack_config",
                        type=str)
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
    
    main(parser.parse_args())
