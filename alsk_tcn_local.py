from dataset.DEAP_DATASET import ModularDeapDataset
from models.tcn import EEG_TCN
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_SET_PATH = 'dataset/'

batch_size = 128
device = "cuda:1"
print("[SYS] Using", device)

deap_train_dataset = ModularDeapDataset(DATA_SET_PATH, train=True)
deap_test_dataset = ModularDeapDataset(DATA_SET_PATH, train=False)

deap_train_loader = DataLoader(deap_train_dataset, shuffle=True, batch_size=batch_size)
deap_test_loader = DataLoader(deap_test_dataset, shuffle=True, batch_size=batch_size)

# MODEL_CONFIG
CHAN_LIST = [32, 24, 16, 10, 6, 2]  # The list of each convolutional layers
KERN_SIZE = 5
DROP_OUT = 0.2
EXPORT_PATH = 'models/saved_weights/tcn_deeper_small_kern.pth'

model = EEG_TCN(CHAN_LIST, KERN_SIZE, DROP_OUT)
model.to(device)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-4
EPCH = 6000
optim = optim.Adam(model.parameters(), lr=LR)
RESUME = False

# TRAINING VISUALIZE CONFIG
PLOT_EVERY = 10

print("===========[TCN INFO REPORT]===========")
print("<I> Using model config")
print("\tModel feature list :", CHAN_LIST)
print("\tModel kernel size :", KERN_SIZE)
print("\tModel drop out:", DROP_OUT)
print("\tExport path :", EXPORT_PATH)
print("<I> Using training config")
print("\tBatch size :", batch_size)
print("\tLearning Rate :", LR)
print("\tEpochs :", EPCH)
print("\tOptimizer :", "Adam")

if RESUME:
    print("<I> Resume the model training...")
    model.load_state_dict(torch.load(EXPORT_PATH, map_location=device))
    model.to(device)
else:
    print("<W> Resume has not set")
    input("\tPress ENTER to proceed.")

print("Starting training model...")

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
        plt.savefig("./results/loss_tcn_small_kern.png")
        plt.show()
