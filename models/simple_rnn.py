import torch
from torch import nn


class SIMPLE_RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SIMPLE_RNN, self).__init__()

        self.hidden_size = hidden_size

        # it is input_size + hidden_size because it is a feed back from the past
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.relu_act = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=len(input.shape) - 1)
        # Input to hidden
        hidden = self.relu_act(self.i2h(combined))
        # Input to output
        output = self.relu_act(self.i2o(combined))
        return output, hidden

    # Init the hidden vector (All zeroes)
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
