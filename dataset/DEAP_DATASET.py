import torch
from torch.utils.data import Dataset
import glob
import pickle
import random


class DEAP_DATASET(Dataset):
    __WHOLE_DATA = []
    __CURR_PART_ID = 0

    def __init__(self, path: str):
        EEG_FILES = glob.glob(str(path) + "*.dat")
        for each_path in EEG_FILES:
            dat = pickle.load(open(each_path, 'rb'), encoding='iso-8859-1')
            self.__WHOLE_DATA.append(dat)

    def set_participant_id(self, i):
        self.__CURR_PART_ID = i

    def __len__(self):
        return len(self.__WHOLE_DATA[self.__CURR_PART_ID]['labels'])

    def __getitem__(self, i):
        eeg = self.__WHOLE_DATA[self.__CURR_PART_ID]['data'][i]
        label = self.__WHOLE_DATA[self.__CURR_PART_ID]['labels'][i]
        return torch.tensor(eeg, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    # TESTING METHOD
    def test_get_item(self, i):
        return self.__getitem__(i)

    def test_len(self):
        return self.__len__()
