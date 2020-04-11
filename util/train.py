from torch.utils.data import DataLoader


def train(model, data_loader):
    for i, (signal, label) in enumerate(data_loader):

