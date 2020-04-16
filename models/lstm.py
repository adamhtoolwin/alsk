import torch
import torch.nn as nn


class EEGLSTM_V2(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, batch_size):
        super(EEGLSTM_V2, self).__init__()

        self.hidden_size1 = hidden_size1  # 64
        self.hidden_size2 = hidden_size2  # 32
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(40, self.hidden_size1, batch_first=True)
        # Put dropout here later :D
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(self.hidden_size1, self.hidden_size2, batch_first=True)

        self.relu = nn.ReLU()

        # self.fc = nn.Linear(8064 * self.hidden_size2, 4) # It should
        self.fc = nn.Linear(self.hidden_size2, 4)

    # Do 1 cnn dimensional << Let's try
    # increase number of layer

    def forward(self, x, hidden):
        x = torch.transpose(x, 1, 2)

        # Input Size (1, 8064, 40), Hidden/Cell Size (1, 1, hidden_size1) (1 eeg sequence for 1 video)
        x, _ = self.lstm1(x, (hidden[0], hidden[1]))  # Mem_inc [1st: 500MB]

        # (1, 8064, hidden_size1), (1, 1, hidden_size2)
        x, _ = self.lstm2(x, (hidden[2], hidden[3]))  # Mem_inc [1st: 200MB, 2nd: 200MB]

        x = self.fc(x)
        output = self.relu(x)

        return output  # , next_hidden

    def initHidden(self):
        return [torch.zeros(1, self.batch_size, self.hidden_size1), torch.zeros(1, self.batch_size, self.hidden_size1),
                torch.zeros(1, self.batch_size, self.hidden_size2), torch.zeros(1, self.batch_size, self.hidden_size2)]


class EEGLSTM(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, batch_size):
        super(EEGLSTM, self).__init__()

        self.hidden_size1 = hidden_size1  # 64
        self.hidden_size2 = hidden_size2  # 32
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(40, self.hidden_size1, batch_first=True)
        # Put dropout here later :D
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(self.hidden_size1, self.hidden_size2, batch_first=True)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size2, 4)

    def forward(self, x, hidden):
        x = torch.transpose(x, 1, 2)

        # Input Size (1, 8064, 40), Hidden/Cell Size (1, 1, hidden_size1) (1 eeg sequence for 1 video)
        x, _ = self.lstm1(x, (hidden[0], hidden[1]))
        x = self.dropout(x)
        # (1, 8064, hidden_size1), (1, 1, hidden_size2)
        x, _ = self.lstm2(x, (hidden[2], hidden[3]))
        # may be we add dropout here?
        x = self.fc(x)
        output = self.relu(x)
        return output

    def initHidden(self):
        return [torch.zeros(1, self.batch_size, self.hidden_size1), torch.zeros(1, self.batch_size, self.hidden_size1),
                torch.zeros(1, self.batch_size, self.hidden_size2), torch.zeros(1, self.batch_size, self.hidden_size2)]
