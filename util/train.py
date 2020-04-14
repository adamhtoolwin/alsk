from numpy import mean
import torch
from tqdm import tqdm

DBG = False


# This function is used for select last element of the sequence.
def select_last_elm(output, device):
    last_seq_index = output.shape[1] - 1
    sel_tensor = torch.tensor(last_seq_index).to(device)
    output = torch.index_select(output, 1, sel_tensor).squeeze(1)
    return output


def train_lstm(model, optim, criterion, data_loader, device):
    loss_hist = []
    model.train()
    for i, (signal, label) in enumerate(data_loader):
        # Init the hidden input for single time
        hidden = model.initHidden()
        hidden = [h.to(device) for h in hidden]

        signal = signal.to(device)  # It increase for 100 MB (Single time)
        label = label.to(device)

        output = None
        # Just add the option to skip the training if its too long
        if DBG and i > 5:
            break

        optim.zero_grad()

        output = model(signal, hidden)  # Expected shape: [128,8064,4]
        output = select_last_elm(output, device)  # Now we gonna select last element [128,4]

        # print(output)
        # print(output, label)
        loss = criterion(output, label)
        loss.backward()
        # print(" Training Loss: ", loss.item())
        optim.step()
        loss_hist.append(loss.item())

    return mean(loss_hist)


def eval_lstm(model, criterion, data_loader, device, eval_size):
    loss_hist = []
    model.eval()
    for i, (signal, label) in enumerate(data_loader):
        hidden = model.initHidden()
        hidden = [h.to(device) for h in hidden]

        signal = signal.to(device)
        label = label.to(device)

        output = None
        if i >= eval_size:
            break

        output = model(signal, hidden)
        output = select_last_elm(output, device)

        # print("Predicted: ", output)
        # print("Label: ", label)

        loss = criterion(output, label)
        # print(" Validation Loss: ", loss.item())
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
