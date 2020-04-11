from numpy import mean


def train(model, optim, criterion, data_loader):
    hidden = model.initHidden()
    loss_hist = []
    for i, (signal, label) in enumerate(data_loader):
        output = None
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
