from dataset.DEAP_DATASET import DEAP_DATASET
from models.simple_rnn import SIMPLE_RNN
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_SET_PATH = 'dataset/data_preprocessed_python/'

deap_dataset = DEAP_DATASET(DATA_SET_PATH)

# MODEL_CONFIG
INPUT_SIZE = 40
HIDDEN_SIZE = 160
OUTPUT_SIZE = 4
model = SIMPLE_RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
deap_dataset.set_participant_id(0)
deap_loader = DataLoader(deap_dataset, shuffle=True, batch_size=1)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-5
EPCH = 100
optim = optim.Adam(model.parameters(), lr=LR)

print("Starting training...")
loss_hist = []
for i in tqdm(range(EPCH)):
    avg_loss = train(model, optim, CRITERION, deap_loader)
    loss_hist.append(avg_loss)

plt.plot(loss_hist)
plt.show()
