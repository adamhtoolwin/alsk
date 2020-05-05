from dataset.DEAP_DATASET import DEAP_DATASET, CombinedDeapDataset, ModularDeapDataset
from models.simple_rnn import SIMPLE_RNN
from models.gru import DEAP_GRU
from util.train import *
from torch.utils.data import DataLoader
import torch
from torch import optim as optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_SET_PATH = 'dataset/'

CUDA = True
gpu_id = '0'
batch_size = 128
device = torch.device("cuda:" + gpu_id if CUDA and torch.cuda.is_available() else "cpu")
print("[SYS] Using", device, "\n")

deap_train_dataset = ModularDeapDataset(DATA_SET_PATH, train=True)
deap_test_dataset = ModularDeapDataset(DATA_SET_PATH, train=False)

deap_train_loader = DataLoader(deap_train_dataset, shuffle=True, batch_size=batch_size)
deap_test_loader = DataLoader(deap_test_dataset, shuffle=True, batch_size=batch_size)

# MODEL_CONFIG
FEATURE_LIST = [64, 32]
EXPORT_PATH = 'models/saved_weights/gru__dropout.pth'

model = DEAP_GRU(FEATURE_LIST, batch_size)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-5
EPCH = 6000
optim = optimizer.Adam(model.parameters(), lr=LR)
RESUME = False

if RESUME:
    print("<I> Resume the model training...")
    model.load_state_dict(torch.load(EXPORT_PATH, map_location=device))
    model.to(device)
else:
    print("<W> Resume has not set")
    input("\tPress ENTER to proceed.")

# TRAINING VISUALIZE CONFIG
PLOT_EVERY = 10

print("===========[INFO REPORT]===========")
print("<I> Using model config")
print("\tModel feature list :", FEATURE_LIST)
print("\tExport path :", EXPORT_PATH)
print("<I> Using training config")
print("\tBatch size :", batch_size)
print("\tLearning Rate :", LR)
print("\tEpochs :", EPCH)
print("\tOptimizer :", "Adam")
print("Starting training GRU model...")
loss_hist = []
val_loss_hist = []
for i in tqdm(range(EPCH)):
    avg_loss = train_lstm_gru(model, optim, CRITERION, deap_train_loader, device)
    loss_hist.append(avg_loss)
    val_loss = eval_lstm_gru(model, CRITERION, deap_test_loader, device, eval_size=99999)
    if not DBG:
        export_or_not(val_loss, val_loss_hist, model, EXPORT_PATH)
    val_loss_hist.append(val_loss)
    # print(val_loss - avg_loss)
    if i % PLOT_EVERY == 0 or i == EPCH - 1:
        plt.plot(loss_hist, label="Training loss")
        plt.plot(val_loss_hist, label="Validation loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.show()
