import torch
import torch.nn as nn


class DEAP_GRU(nn.Module):
    # We use all of the features in the dataset
    def __init__(self, fea_list: list, batch_size, input_size):
        super(DEAP_GRU, self).__init__()
        self.gru_unit_list = nn.ModuleList()
        self.bs = batch_size
        self.fea_list = fea_list + [2]
        self.input_size = input_size

        print("ARCH:[ ", end="")
        print(input_size, "--> ", end="")
        self.gru_unit_list.append(nn.GRU(input_size=self.input_size, hidden_size=fea_list[0], batch_first=True))

        for i, each_fea in enumerate(fea_list):

            if len(fea_list) == 1:
                break

            print(fea_list[i], "--> ", end="")
            self.gru_unit_list.append(nn.GRU(input_size=each_fea, hidden_size=fea_list[i + 1], batch_first=True))

            if i == len(fea_list) - 2:
                break

        print(fea_list[-1], "-->", 2, "]")
        self.gru_unit_list.append(nn.GRU(input_size=fea_list[-1], hidden_size=2, batch_first=True))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, hidden_l):
        x = torch.transpose(x, 1, 2)
        for i, each_gru in enumerate(self.gru_unit_list):
            # print("Passing in layer ["+str(i)+"], Hidden =", self.fea_list[i], hidden_l[i].shape)
            x, _ = each_gru(x, hidden_l[i])
            if i != len(self.gru_unit_list) - 2: # Only apply dropout after first 4 units
                x = self.dropout(x)
        return x

    def initHidden(self):
        hidden_list = []
        # print("Init hidden:[ ", end="")
        for i, each_fea in enumerate(self.fea_list):
            # print(each_fea, end=" ")
            hidden_list.append(torch.randn(1 , self.bs, each_fea))
        # print("]")
        return hidden_list


def sample_code():
    #LAYER_NUM = 2  # Will use 1
    #HIDDEN_SIZE = 2
    #INPUT_FEATURES = 10
    #rnn = nn.GRU(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=LAYER_NUM, batch_first=True)

    INPUT_FEATURES = 40
    BATCH_SIZE = 3
    SEQ_LEN = 300
    input = torch.randn(BATCH_SIZE, INPUT_FEATURES, SEQ_LEN)

    #h0 = torch.randn(LAYER_NUM, BATCH_SIZE, HIDDEN_SIZE)
    #output, hn = rnn(input, h0)

    model = DEAP_GRU([32, 16, 8, 4], batch_size=BATCH_SIZE)
    print(len(model.gru_unit_list))
    print(input.shape)
    h_l = model.init_hidden()
    output = model(input, h_l)
    print(output.shape)
