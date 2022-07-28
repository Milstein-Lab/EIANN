import torch
import torch.nn as nn

params_dict = {'layer_0':
                   {'E':
                        {'n': 7,
                         'bias': False,
                         'activation': 'relu',
                         'projections': []}},
               'layer_1':
                   {'E':
                        {'n': 100,
                         'bias': False,
                         'activation': 'relu',
                         'projections': ['layer_0']}},
               'layer_2':
                   {'E':
                        {'n': 10,
                         'bias': False,
                         'activation': 'relu',
                         'projections': ['layer_1']}}
               }


class universalNet(nn.Module):
    def __init__(self, params_dict):
        super().__init__()

        self.net = nn.ModuleList()

        for layer in params_dict:
            for population in params_dict[layer]:
                out_size = params_dict[layer][population]['n']
                for projection in params_dict[layer][population]['projections']:
                    if layer != 'layer_0':
                        in_size = params_dict[projection][population]['n']
                        params_dict[layer][population]['projections'][0] = nn.Linear(in_size, out_size, bias=False)
                        self.net.append(params_dict[layer][population]['projections'][0])

    def forward(self, x):
        for layer in params_dict:
            for population in params_dict[layer]:
                for projection in params_dict[layer][population]['projections']:
                    x = projection(x)
        return x


class Projection(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        layer = nn.Linear(in_size, out_size, bias=False)

