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
deap_dataset.set_participant_id(0)
deap_loader = DataLoader(deap_dataset, shuffle=True, batch_size=1)

deap_dataset_test = DEAP_DATASET(DATA_SET_PATH)
deap_dataset_test.set_participant_id(1)
deap_test_loader = DataLoader(deap_dataset, shuffle=True, batch_size=1)

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
EXPORT_PATH = 'models/saved_weights/simple_rnn.pth'

print("Starting training...")
loss_hist = []
val_loss_hist = []
for i in tqdm(range(EPCH)):
    avg_loss = train(model, optim, CRITERION, deap_loader)
    loss_hist.append(avg_loss)
    val_loss = eval(model, CRITERION, deap_test_loader, eval_size=5)
    export_or_not(val_loss, val_loss_hist, model, EXPORT_PATH)
    val_loss_hist.append(val_loss)
    if i % 1 == 0:
        plt.plot(loss_hist, label="Training loss")
        plt.plot(val_loss_hist, label="Validation loss")
        plt.legend()
        plt.show()
