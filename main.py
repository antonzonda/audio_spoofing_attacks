"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
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
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from main_utils import  dev_epoch, get_model, produce_evaluation_file, train_epoch

warnings.filterwarnings("ignore", category=FutureWarning)


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

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    if track == "toy_example":
        eval_trial_path = (database_path / "cm_protocols/eval.txt")
    else:
        eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "pretained_weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # evaluates pretrained model and exit script
    if args.eval:

        _, _, eval_loader = get_loader(database_path, args.seed, config, is_eval=True)
        checkpoint = torch.load(config["model_path"], map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        # eval_eer, eval_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=eval_score_path,
        #     asv_score_file=database_path / config["asv_score_path"],
        #     output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # define dataloaders
    trn_loader, dev_loader, _ = get_loader(database_path, args.seed, config, is_eval=False)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, _ = create_optimizer(model.parameters(), optim_config)

    best_dev_eer = 1.

    loss_log = []

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    save_every = 5

    # Training
    start_epoch = 0

    if args.load_checkpoint:
        model, optimizer, start_epoch, best_dev_eer, loss_log = load_checkpoint(model, optimizer, model_save_path / "checkpoint.pth")

    for epoch in range(start_epoch, config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   config)

        loss_log.append(running_loss)

        dev_eer = dev_epoch(dev_loader, model, device=device)

        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(running_loss, dev_eer))

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), model_save_path / "curr_best.pth".format(epoch, dev_eer))

            print("Saving epoch {}".format(epoch))

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

        # we need to save the model every n (5) epoch
        if (epoch % save_every == 0) and (epoch > 0):
            save_checkpoint(epoch, model, optimizer, loss_log, best_dev_eer, model_save_path / "checkpoint.pth")

    print("Start final evaluation")
    epoch += 1

    print(loss_log)

    print("saving model to", model_save_path)
    torch.save(model.state_dict(), model_save_path / "final.pth")


def get_loader(
        database_path: str,
        seed: int,
        config: dict,
        is_eval: bool) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    
    track = config["track"] 
    prefix_2019 = "ASVspoof2019.{}".format(track)

    if track == 'toy_example':
        print('USING toy_example')
        trn_database_path = database_path / "train"
        dev_database_path = database_path / "dev"
        eval_database_path = database_path / "eval"

        trn_list_path = (database_path / "cm_protocols/train.txt" )
        dev_trial_path = (database_path / "cm_protocols/dev.txt")
        eval_trial_path = (database_path / "cm_protocols/eval.txt")

    else: # track == LA
        trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
        dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
        eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

        trn_list_path = (database_path /
                        "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                            track, prefix_2019))
        dev_trial_path = (database_path /
                        "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                            track, prefix_2019))
        eval_trial_path = (
            database_path /
            "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                track, prefix_2019))

    if is_eval:

        file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
        eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                                base_dir=eval_database_path)

        eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
        
        return None, None, eval_loader

    else:

        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                                is_train=True,
                                                is_eval=False)
        print("no. training files:", len(file_train))

        train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                            labels=d_label_trn,
                                            base_dir=trn_database_path)
        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)

        d_label_dev, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                    is_train=False,
                                    is_eval=False)
        print("no. validation files:", len(file_dev))

        dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev, labels=d_label_dev, base_dir=dev_database_path)

        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

        return trn_loader, dev_loader, None


def load_checkpoint(model, optimizer, filename):

    start_epoch = 0
    dev_eer = 1.
    losslogger = []
    
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
        dev_eer = checkpoint["best_dev_eer"]
    else:
        print("No checkpoint!")

    return model, optimizer, start_epoch, dev_eer, losslogger

def save_checkpoint(epoch, model, optimizer, logger, best_dev_eer, filename):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': logger, 'best_dev_eer': best_dev_eer}

    print("Saving checkpoint at epoch", epoch)

    torch.save(state, filename)



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
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
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
    parser.add_argument("--load_checkpoint", action="store_true")
    main(parser.parse_args())

