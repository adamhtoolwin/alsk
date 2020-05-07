import torch
from torch.utils.data import Dataset
import glob
import pickle
import random


class DEAP_DATASET(Dataset):
    __WHOLE_DATA = {}
    __TRAIN_DATA = {"data": [], "labels": []}
    __TEST_DATA = {"data": [], "labels": []}
    __CURR_PART_ID = 0

    def __init__(self, path: str, train: bool, part_id: int, cross_val_id: int):
        self.train = train
        self.EEG_FILES = glob.glob(str(path) + "/data_preprocessed_python/*.dat")
        self.cross_val_id = cross_val_id
        self.set_participant_id(part_id)
        self.set_cross_id(cross_val_id)

    def set_participant_id(self, i):
        self.__CURR_PART_ID = i
        self.__WHOLE_DATA = pickle.load(open(self.EEG_FILES[i], 'rb'), encoding='iso-8859-1')

    def set_cross_id(self, c):
        self.cross_val_id = c
        start = (c - 1) * 8
        end = (c * 8) - 1
        for i, (data, label) in enumerate(zip(self.__WHOLE_DATA['data'], self.__WHOLE_DATA['labels'])):
            if start <= i <= end:  # If in range of choosen, put it to test set
                self.__TEST_DATA['data'].append(data)
                self.__TEST_DATA['labels'].append(label)
            else:
                self.__TRAIN_DATA['data'].append(data)
                self.__TRAIN_DATA['labels'].append(label)

    def __len__(self):
        if self.train:
            return len(self.__TRAIN_DATA['labels'])
        else:
            return len(self.__TEST_DATA['labels'])

    def __getitem__(self, i):
        if self.train:
            eeg = self.__TRAIN_DATA['data'][i][0:32]
            label = self.__TRAIN_DATA['labels'][i][0:2]
        else:
            eeg = self.__TEST_DATA['data'][i][0:32]
            label = self.__TEST_DATA['labels'][i][0:2]
        return torch.tensor(eeg, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def set_train(self, is_train: bool):
        self.train = is_train

    # TESTING METHOD
    def test_get_item(self, i):
        return self.__getitem__(i)

    def test_len(self):
        return self.__len__()


class ModularDeapDataset(Dataset):
    def __init__(self, path: str, train=True):
        if train:
            self.files = glob.glob(str(path) + "modular/train/*.dat")
        else:
            self.files = glob.glob(str(path) + "modular/test/*.dat")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = pickle.load(open(self.files[i], 'rb'), encoding='iso-8859-1')

        # select only first 32 channels
        eeg = data['data'][0:32]
        label = data['label']
        return torch.tensor(eeg, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class CombinedDeapDataset(Dataset):

    def __init__(self, path: str):
        eeg_file_path = str(path) + "eegs.dat"
        labels_file_path = str(path) + "labels.dat"

        self.labels = pickle.load(open(labels_file_path, 'rb'), encoding='iso-8859-1')
        self.eegs = pickle.load(open(eeg_file_path, 'rb'), encoding='iso-8859-1')

        assert len(self.eegs) == len(self.labels)

    def __len__(self):
        assert len(self.eegs) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, i):
        eeg = self.eegs[i]
        label = self.labels[i]
        return torch.tensor(eeg, dtype=torch.float), torch.tensor(label, dtype=torch.float)
