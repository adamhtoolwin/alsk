PARTICIPANT_NUM = 3  # This is constant... this must not be changed
CROSS_VAL = 5  # This is constant... this must not be changed

from dataset.DEAP_DATASET import DEAP_DATASET
from tqdm.auto import trange
from torch.utils.data import DataLoader
import torch.optim as optimizer
import matplotlib.pyplot as plt

from models.tcn import *
from util.train import *
import numpy as np
import os

# Initialize CUDA Device
CUDA = True
gpu_id = '1'
batch_size = 8
device = torch.device("cuda:" + gpu_id if CUDA and torch.cuda.is_available() else "cpu")
print("[SYS] Using", device)
print("")

# MODEL_CONFIG
CHAN_LIST = [32, 24, 16, 10, 6, 2]  # The list of each convolutional layers
KERN_SIZE = 15
DROP_OUT = 0.2
INPUT_SIZE = 32
# model = EEG_TCN(CHAN_LIST, KERN_SIZE, DROP_OUT)
# model.load_state_dict(torch.load(EXPORT_PATH, map_location=device))
# model.to(device)

# PATH initialize
EXPORT_PATH_DIR = 'models/saved_weights/tcn/bigKernel/'
mkdir(EXPORT_PATH_DIR)

# TRAINING_CONFIG
CRITERION = torch.nn.MSELoss()
LR = 1e-4
EPCH = 1000

print("===========[INFO REPORT]===========")
print("<I> Using model config")
print("\tChannels size :", CHAN_LIST)
print("\tKernel size: ", KERN_SIZE)
print("\tDropout: ", DROP_OUT)
print("\tExport path :", EXPORT_PATH_DIR)
print("<I> Using training config")
print("\tBatch size :", batch_size)
print("\tLearning Rate :", LR)
print("\tEpochs :", EPCH)
print("\tOptimizer :", "Adam")

print("Please check config...")
input("\tPress ENTER to proceed.")

print("Starting training TCN BIG kernel model...")

# TRAINING VISUALIZE CONFIG
PLOT_EVERY = 10

DATA_SET_PATH = "./dataset"
train_dataset = DEAP_DATASET(DATA_SET_PATH, train=True, part_id=1, cross_val_id=1)
test_dataset = DEAP_DATASET(DATA_SET_PATH, train=False, part_id=1, cross_val_id=1)

train_MSE_Loss_buffer = []
test_MSE_Loss_buffer = []
for p in range(1, PARTICIPANT_NUM + 1):
    print("Participant:", p)
    train_dataset.set_participant_id(p - 1)
    test_dataset.set_participant_id(p - 1)
    for c in range(1, CROSS_VAL + 1):
        model = EEG_TCN(CHAN_LIST, KERN_SIZE, DROP_OUT)
        # model.load_state_dict(torch.load(EXPORT_PATH, map_location=device))
        model.to(device)

        optim = optimizer.Adam(model.parameters(), lr=LR)
        print("Cross val:", c)
        # Directory preparation
        EXPORT_PATH = EXPORT_PATH_DIR + "s" + str(p) + "/"
        mkdir(EXPORT_PATH)

        train_dataset.set_cross_id(c)
        test_dataset.set_cross_id(p - 1)

        deap_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        deap_test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

        loss_hist = []
        val_loss_hist = []
        EXPORT_PATH_FILE = ""
        for i in trange(EPCH, desc="Epoch"):
            avg_loss = train_tcn(model, optim, CRITERION, deap_train_loader, device)
            loss_hist.append(avg_loss)
            val_loss = eval_tcn(model, CRITERION, deap_test_loader, device, eval_size=99999)
            if not DBG:
                EXPORT_PATH_FILE = EXPORT_PATH + "c" + str(c) + ".pth"
                export_or_not(val_loss, val_loss_hist, model, EXPORT_PATH_FILE)
            val_loss_hist.append(val_loss)
            # print(val_loss - avg_loss)
            if i % PLOT_EVERY == 0 or i == EPCH - 1:
                plt.clf()
                plt.plot(loss_hist, label="Training loss")
                plt.plot(val_loss_hist, label="Validation loss")
                plt.title("On participant " + str(p) + "cross id " + str(c))
                plt.legend()
                plt.savefig(EXPORT_PATH + "loss_" + str(p) + "_" + str(c) + ".png")
                plt.show()

        # After finish training, load the best model
        print(">> Loading previous model from : " + EXPORT_PATH_FILE)
        model.load_state_dict(torch.load(EXPORT_PATH_FILE, map_location=device))
        model.to(device)
        train_loss = eval_tcn(model, CRITERION, deap_train_loader, device, eval_size=99999)
        val_loss = eval_tcn(model, CRITERION, deap_test_loader, device, eval_size=99999)

        train_MSE_Loss_buffer.append(train_loss)
        test_MSE_Loss_buffer.append(val_loss)

print("train_MSE_loss :", np.mean(train_MSE_Loss_buffer))
print("val_MSE_loss :", np.mean(test_MSE_Loss_buffer))
