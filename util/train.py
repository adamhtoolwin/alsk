from numpy import mean
import torch

DBG = False


def eval(model, criterion, data_loader, eval_size):
    hidden = model.initHidden()
    loss_hist = []
    model.eval()
    for i, (signal, label) in enumerate(data_loader):
        output = None
        if i > eval_size:
            break
        for j in range(signal.shape[2]):
            pts_signal = signal[:, :, j]
            output, hidden = model(pts_signal, hidden)
        # print(output)
        # print(output, label)
        loss = criterion(output, label)
        loss_hist.append(loss.item())
    return mean(loss_hist)


def train(model, optim, criterion, data_loader):
    hidden = model.initHidden()
    loss_hist = []
    model.train()
    for i, (signal, label) in enumerate(data_loader):
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
