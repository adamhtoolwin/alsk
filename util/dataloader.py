import pickle
import glob

path = '../dataset/data_preprocessed_python/s01.dat'
eeg_files_m = glob.glob(r"../dataset/data_preprocessed_python/*.dat")

data = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

print("Video Num", len(data['data']))

for each_eeg_seq, each_label in zip(data['data'], data['labels']):
    print(each_eeg_seq.shape, each_label)
