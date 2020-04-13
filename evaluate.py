from dataset.DEAP_DATASET import DEAP_DATASET, CombinedDeapDataset
from models.simple_rnn import SIMPLE_RNN
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim
import matplotlib.pyplot as plt


DATA_SET_PATH = 'dataset/data_preprocessed_python/'

deap_dataset_test = DEAP_DATASET(DATA_SET_PATH)
deap_dataset_test.set_participant_id(1)
deap_test_loader = DataLoader(deap_dataset_test, shuffle=True, batch_size=1)

# MODEL_CONFIG
INPUT_SIZE = 40
HIDDEN_SIZE = 160
OUTPUT_SIZE = 4
model = SIMPLE_RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-5
EPCH = 100
optim = optim.Adam(model.parameters(), lr=LR)

PATH = 'models/saved_weights/simple_rnn.pth'

model.load_state_dict(torch.load(PATH))
model.eval()

val_loss = eval(model, CRITERION, deap_test_loader, eval_size=5)



