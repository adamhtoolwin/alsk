import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, layer_list, batch_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.layer_list = layer_list + [2]
        self.modules = []

        print("ARCH:[ ", end="")
        print(input_size, "--> ", end="")
        self.modules.append(nn.Linear(input_size, self.layer_list[0]))

        for i, layer in enumerate(layer_list):
            module = nn.Linear(layer, self.layer_list[i+1])
            self.modules.append(module)

            print(layer_list[i], "--> ", end="")

            if i == len(layer_list) - 2:
                break

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)
