PARTICIPANT_NUM = 32  # This is constant... this must not be changed
CROSS_VAL = 5  # This is constant... this must not be changed

from dataset.DEAP_DATASET import DEAP_DATASET
from tqdm.auto import trange
import torch

# Initialize CUDA Device
CUDA = True
gpu_id = '0'
batch_size = 128
device = torch.device("cuda:" + gpu_id if CUDA and torch.cuda.is_available() else "cpu")
print("[SYS] Using", device)
print("")


DATA_SET_PATH = "../dataset"
dataset = DEAP_DATASET(DATA_SET_PATH, train=True, part_id=1, cross_val_id=1)
eeg, label = dataset.test_get_item(1)


for p in trange(1, PARTICIPANT_NUM + 1, desc="On participant"):
    dataset.set_participant_id(p - 1)

    for c in trange(1, CROSS_VAL + 1, desc="On cross val"):
        dataset.set_cross_id(c + 1)
