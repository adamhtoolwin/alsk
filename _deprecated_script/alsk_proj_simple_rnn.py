from dataset.DEAP_DATASET import DEAP_DATASET, CombinedDeapDataset
from models.simple_rnn import SIMPLE_RNN
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm



DATA_SET_PATH = '../dataset/'

CUDA = True
gpu_id = '1'
device = torch.device("cuda:" + gpu_id if CUDA and torch.cuda.is_available() else "cpu")
print("Using ", device)
print("")

deap_dataset = CombinedDeapDataset(DATA_SET_PATH)

train_size = int(0.8 * len(deap_dataset))
test_size = len(deap_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(deap_dataset, [train_size, test_size])

deap_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
deap_test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)

# MODEL_CONFIG
INPUT_SIZE = 40
HIDDEN_SIZE = 160
OUTPUT_SIZE = 4
model = SIMPLE_RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.to(device)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-5
EPCH = 20
optim = optim.Adam(model.parameters(), lr=LR)
EXPORT_PATH = 'models/saved_weights/simple_rnn_v2.pth'

print("==============================")
print("Starting training...")
loss_hist = []
val_loss_hist = []
for i in tqdm(range(EPCH)):
    avg_loss = train(model, optim, CRITERION, deap_train_loader, device)
    loss_hist.append(avg_loss)
    val_loss = eval(model, CRITERION, deap_test_loader, device, eval_size=99999)
    export_or_not(val_loss, val_loss_hist, model, EXPORT_PATH)
    val_loss_hist.append(val_loss)
    if i % 1 == 0:
        plt.plot(loss_hist, label="Training loss")
        plt.plot(val_loss_hist, label="Validation loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.show()
