import torch
import torch.nn as nn


class EEGLSTM(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(EEGLSTM, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.lstm1 = nn.LSTM(8064, self.hidden_size1, batch_first=True)
        self.reduce = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.lstm2 = nn.LSTM(self.hidden_size1, self.hidden_size2, batch_first=True)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(self.hidden_size2, 4)

    def forward(self, x, hidden):
        # Input Size (1, 40, 8064), Hidden/Cell Size (1, 1, hidden_size1) (1 eeg sequence for 1 video)
        x, hidden = self.lstm1(x, (hidden[0], hidden[1]))
        x = self.relu(x)

        # Hidden/Cell Size (1, 1, hidden_size1)
        reduced_hidden = (self.reduce(hidden[0]), self.reduce(hidden[1]))

        # (1, 40, hidden_size1), (1, 1, hidden_size2)
        x, hidden = self.lstm2(x, (reduced_hidden[0], reduced_hidden[1]))

        output = self.fc(x)

        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

