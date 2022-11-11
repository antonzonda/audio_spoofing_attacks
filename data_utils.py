import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

        # load the whole training set into memory
        self.x_list = []
        
        for key in self.list_IDs:

            X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
            X_pad = pad_random(X, self.cut)
            x_inp = Tensor(X_pad)
            # y = self.labels[key]

            self.x_list.append(x_inp)


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        # X_pad = pad_random(X, self.cut)
        # x_inp = Tensor(X_pad)

        x_inp = self.x_list[index]
        key = self.list_IDs[index]
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

        self.x_list = []
        
        for key in list_IDs:
            # key = self.list_IDs[index]
            X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
            self.x_list.append(x_inp)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        # X_pad = pad(X, self.cut)
        # x_inp = Tensor(X_pad)
        key = self.list_IDs[index]
        x_inp = self.x_list[index]

        return x_inp, key



class Dataset_ASVspoof2019_attack(Dataset):
    def __init__(self, list_IDs, labels, base_dir, eval):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.eval = eval

        self.x_list = []
        for key in list_IDs:

            if self.eval:

                # # testing saving tensor object

                x_inp = torch.load(str(self.base_dir / f"flac/{key}.pt"))
                self.x_list.append(x_inp)
            else:
            # if True:
                X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))

                X_pad = pad_random(X, self.cut)
                X_pad = pad(X, self.cut)
                x_inp = Tensor(X_pad)
                self.x_list.append(x_inp)


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # if self.eval:
        #     # X, _ = sf.read(str(self.base_dir / f"flac/{key}.wav"))
        #     # testing saving tensor object

        #     x_inp = torch.load(str(self.base_dir / f"flac/{key}.pt"))
        # else:
        #     X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))

        #     X_pad = pad_random(X, self.cut) #???
        #     X_pad = pad(X, self.cut)
        #     x_inp = Tensor(X_pad)

        x_inp = self.x_list[index]
        y = self.labels[key]
        return x_inp, y, key
