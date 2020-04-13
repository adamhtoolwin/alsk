import torch
import torch.nn as nn


class EEGLSTM(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, batch_size):
        super(EEGLSTM, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(40, self.hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size1, self.hidden_size2, batch_first=True)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(8064 * self.hidden_size2, 4)

    def forward(self, x, hidden):
        x = torch.transpose(x, 1, 2)
        
        # Input Size (1, 8064, 40), Hidden/Cell Size (1, 1, hidden_size1) (1 eeg sequence for 1 video)
        x, hidden1 = self.lstm1(x, (hidden[0], hidden[1]))
        x = self.relu(x)

        # (1, 8064, hidden_size1), (1, 1, hidden_size2)
        x, hidden2 = self.lstm2(x, (hidden[2], hidden[3]))

        # x = x.contiguous()

        x = x.reshape(self.batch_size, -1)
        next_hidden = [hidden1[0], hidden1[1], hidden2[0], hidden2[1]]

        output = self.fc(x)

        return output, next_hidden

    def initHidden(self):
        return [torch.zeros(1, self.batch_size, self.hidden_size1), torch.zeros(1, self.batch_size, self.hidden_size1),
                torch.zeros(1, self.batch_size, self.hidden_size2), torch.zeros(1, self.batch_size, self.hidden_size2)]

