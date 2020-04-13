from numpy import mean
import torch
from tqdm import tqdm

DBG = False


def train_lstm(model, optim, criterion, data_loader, device):
    hidden = model.initHidden()
    hidden = [h.to(device) for h in hidden]

    loss_hist = []
    model.train()
    for i, (signal, label) in tqdm(enumerate(data_loader)):
        signal = signal.to(device)
        label = label.to(device)

        output = None
        # Just add the option to skip the training if its too long
        if DBG and i > 5:
            break

        optim.zero_grad()

        output, hidden = model(signal, hidden)

        # print(output)
        # print(output, label)
        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        print(" Training Loss: ", loss.item())
        optim.step()
        loss_hist.append(loss.item())
    return mean(loss_hist)


def eval_lstm(model, criterion, data_loader, device, eval_size):
    hidden = model.initHidden()
    hidden = [h.to(device) for h in hidden]

    loss_hist = []
    model.eval()
    for i, (signal, label) in tqdm(enumerate(data_loader)):
        signal = signal.to(device)
        label = label.to(device)

        output = None
        if i >= eval_size:
            break

        output, hidden = model(signal, hidden)

        # print("Predicted: ", output)
        # print("Label: ", label)

        loss = criterion(output, label)
        print(" Validation Loss: ", loss.item())
        # print("Loss: ", loss.item())
        loss_hist.append(loss.item())
    return mean(loss_hist)


def eval(model, criterion, data_loader, device, eval_size):
    hidden = model.initHidden()
    hidden = hidden.to(device)

    loss_hist = []
    model.eval()
    for i, (signal, label) in enumerate(data_loader):
        signal = signal.to(device)
        label = label.to(device)

        output = None
        if i >= eval_size:
            break
        for j in range(signal.shape[2]):
            pts_signal = signal[:, :, j]
            output, hidden = model(pts_signal, hidden)
        # print("Predicted: ", output)
        # print("Label: ", label)

        loss = criterion(output, label)
        # print("Loss: ", loss.item())
        loss_hist.append(loss.item())
    return mean(loss_hist)


def train(model, optim, criterion, data_loader, device):
    hidden = model.initHidden()
    hidden = hidden.to(device)

    loss_hist = []
    model.train()
    for i, (signal, label) in enumerate(data_loader):
        signal = signal.to(device)
        label = label.to(device)

        output = None
        # Just add the option to skip the training if its too long
        if DBG and i > 5:
            break
        optim.zero_grad()
        for j in range(signal.shape[2]):
            pts_signal = signal[:, :, j]
            output, hidden = model(pts_signal, hidden)
        # print(output)
        # print(output, label)
        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        # print(loss.item())
        optim.step()
        loss_hist.append(loss.item())
    return mean(loss_hist)


def export_or_not(val_loss, val_loss_hist, model, path):
    if len(val_loss_hist) == 0:
        torch.save(model.state_dict(), path)
    elif val_loss < val_loss_hist[len(val_loss_hist) - 1]:
        torch.save(model.state_dict(), path)
