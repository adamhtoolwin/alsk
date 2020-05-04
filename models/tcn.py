from torch import nn
from models.component.TCN import TemporalConvNet


class EEG_TCN(nn.Module):
    def __init__(self, num_channels, kernel_size, dropout):
        super(EEG_TCN, self).__init__()
        self.tcn = TemporalConvNet(40, num_channels, kernel_size, dropout=dropout)
        #self.linear = nn.Linear(num_channels[-1], 2)
        self.sigsig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x).transpose(1, 2)
        #output = self.linear(output).double()
        # return self.sigsig(output) * 10
        return output
