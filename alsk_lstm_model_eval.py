from models.lstm import EEGLSTM
from dataset.DEAP_DATASET import ModularDeapDataset
from torch.utils.data import DataLoader
import torch
from util.train import select_last_elm

DATA_SET_PATH = 'dataset/'
deap_val = ModularDeapDataset(DATA_SET_PATH, train=False)
loader = DataLoader(deap_val, shuffle=True, batch_size=1)

INPUT_SIZE = 40
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 4

MDL_PATH = 'models/saved_weights/lstm_v2_dropout.pth'

model = EEGLSTM(HIDDEN_SIZE1, HIDDEN_SIZE2, 1)

model.load_state_dict(torch.load(MDL_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    for i, (signal, label) in enumerate(loader):
        hidden = model.initHidden()
        output = model(signal, hidden)
        output = select_last_elm(output, "cpu")
        print("output:", output.numpy(), "label:", label.numpy())
