from dataset.DEAP_DATASET import ModularDeapDataset
from models.tcn import EEG_TCN
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_SET_PATH = 'dataset/'

CUDA = True
gpu_id = '0'
batch_size = 128
device = torch.device("cuda:" + gpu_id if CUDA and torch.cuda.is_available() else "cpu")
print("[SYS] Using", device)
print("")

deap_train_dataset = ModularDeapDataset(DATA_SET_PATH, train=True)
deap_test_dataset = ModularDeapDataset(DATA_SET_PATH, train=False)

# import pdb; pdb.set_trace()

deap_train_loader = DataLoader(deap_train_dataset, shuffle=True, batch_size=batch_size)
deap_test_loader = DataLoader(deap_test_dataset, shuffle=True, batch_size=batch_size)

# MODEL_CONFIG
CHAN_LIST = [64, 32, 2]  # The list of each convolutional layers
KERN_SIZE = 5
DROP_OUT = 0.2
EXPORT_PATH = 'models/saved_weights/tcn_v0.pth'

model = EEG_TCN(CHAN_LIST, KERN_SIZE, DROP_OUT)
model.to(device)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-4
EPCH = 6000
optim = optim.Adam(model.parameters(), lr=LR)

# TRAINING VISUALIZE CONFIG
PLOT_EVERY = 5

print("==============================")
print("Starting training TCN model...")
loss_hist = []
val_loss_hist = []
for i in tqdm(range(EPCH)):
    avg_loss = train_tcn(model, optim, CRITERION, deap_train_loader, device)
    loss_hist.append(avg_loss)
    val_loss = eval_tcn(model, CRITERION, deap_test_loader, device, eval_size=99999)
    if not DBG:
        export_or_not(val_loss, val_loss_hist, model, EXPORT_PATH)
    val_loss_hist.append(val_loss)
    # print(val_loss - avg_loss)
    if i % PLOT_EVERY == 0 or i == EPCH - 1:
        plt.clf()
        plt.plot(loss_hist, label="Training loss")
        plt.plot(val_loss_hist, label="Validation loss")
        plt.legend()
        plt.savefig("./results/loss.png")
        plt.show()
