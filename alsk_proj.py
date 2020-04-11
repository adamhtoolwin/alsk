from dataset.DEAP_DATASET import DEAP_DATASET
from model.simple_rnn import SIMPLE_RNN
from util.train import *
from torch.utils.data import DataLoader

DATA_SET_PATH = '../dataset/data_preprocessed_python/'

deap_dataset = DEAP_DATASET(DATA_SET_PATH)

# MODEL_CONFIG
INPUT_SIZE = 40
HIDDEN_SIZE = 160
OUTPUT_SIZE = 4
model = SIMPLE_RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

deap_loader = DataLoader(deap_dataset, shuffle=True, batch_size=1)

deap_dataset.set_participant_id(0)
train(model, deap_loader)
