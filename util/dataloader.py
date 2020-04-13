import pickle
import glob


def test():
    path = '../dataset/data_preprocessed_python/s01.dat'
    eeg_files_m = glob.glob("../dataset/data_preprocessed_python/*.dat")

    data = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

    print("Video Num", len(data['data']))

    for each_eeg_seq, each_label in zip(data['data'], data['labels']):
        print(each_eeg_seq.shape, each_label)

    for each in eeg_files_m:
        print(each)


from dataset.DEAP_DATASET import DEAP_DATASET, CombinedDeapDataset, ModularDeapDataset

DATA_SET_PATH = '../dataset/'

data_set = ModularDeapDataset(DATA_SET_PATH)

import pdb; pdb.set_trace()