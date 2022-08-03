import torch
import torch.nn as nn


'''
params_dict = {'layer_0':
                   {'E':
                        {'n': 7,
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

model = universalNet(params_dict)
'''
'''
hparams = {'tau': 1,
           'seed': 42}

params_dict = {'layer_0':
                   {'E':
                        {'n': 7,
                         'projections': {}}},
               'layer_1':
                   {'E':
                        {'n': 100,
                         'bias': False,
                         'activation': 'relu',
                         'projections': 
                             {'layer_0':{'E':[]}}}},
               'layer_2':
                   {'E':
                        {'n': 10,
                         'bias': False,
                         'activation': 'relu',
                         'projections':
                             {'layer_1':{'E':[]}}}}
               }
'''


class universalNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = Layer(populations={'E':7, 'I':1})

        self.layer_1 = Layer(populations={'E':7, 'I':1})
        self.layer_1.E.activation = 'softplus'

        self.layer_2 = Layer(populations={'E':5})
        self.layer_2.E.activation = 'softplus'

        self.layer_2.E.add_projections([self.input_layer.E, self.layer_1.E])
        # self.layer_2.E.add_projections(['input_layer.E', 'layer_1.E'])


#
#
# class universalNet(nn.Module):
#     def __init__(self, params_dict, hparams):
#         super().__init__()
#         self.tau = hparams['tau']
#         self.seed = hparams['seed']
#
#         # self.net = nn.Sequential()
#         # self.net.add_module('name',nn.Linear)
#         # self.nn_modules = nn.ModuleList()
#
#         self.net = Net()
#
#         for layer in params_dict:
#             self.net.__dict__[layer] = Layer(name=layer, )
#
#             for population in params_dict[layer]:
#                 post_size = params_dict[layer][population]['n']
#
#                 self.net.__dict__[layer].__dict__[population] = Population(size=post_size, activation='softplus', bias=False)
#
#                 for incoming_layer in params_dict[layer][population]['projections']:
#                     for pre_population in params_dict[layer][population]['projections'][incoming_layer]:
#                         pre_size = params_dict[incoming_layer][pre_population]['n']
#                         self.net.__dict__[layer].__dict__[population].__dict__[incoming_layer] = Projection(pre_size, post_size)
#
#
#     def forward(self, x):
#         for layer in self.net:
#             return


class Net(object):
    def __iter__(self):
        for value in self.__dict__.values():
            yield value


class Layer(nn.Module):
    def __init__(self, populations):
        super().__init__()
        self.populations = populations
        for pop,size in populations.items():
            self.__dict__[pop] = Population(size)


class Population(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.activation = 'linear'
        self.bias_rule = None
        self.bias = torch.zeros(size)
        self.learn_bias = False

        self.activity = torch.zeros(self.size)
        self.prev_activity = torch.zeros(self.size)
        self.activity_history = None
        self.state = torch.zeros(self.size)
        self.projections = {}

    def add_projections(self, pre_populations):
        self.inputs = pre_populations
        for pre_pop in pre_populations:
            self.__dict__[pre_pop] = nn.Linear(pre_pop.size, self.size, bias=False)


# class universalNet1(nn.Module):
#     def __init__(self, params_dict):
#         super().__init__()
#         self.params_dict = params_dict
#
#         self.nn_modules = nn.ModuleList()
#         self.net = nn.Sequential()
#
#         self.input_layer = Layer(n=7,)
#
#
#
#         self.net.add_module()
#
#         for layer in params_dict:
#             self.net.layer = Layer(param_args)
#
#             for population in params_dict[layer]:
#                 self.net.layer.population = Population(args)
#
#                 out_size = params_dict[layer][population]['n']
#                 for projection_layer in params_dict[layer][population]['projections']:
#                     self.net[][].population.projection = Projection(args)
#
#                     for pre_population in params_dict[layer][population]['projections'][projection_layer]:
#                         in_size = params_dict[projection_layer][pre_population]['n']
#                         params_dict[layer][population]['projections'][projection_layer][pre_population] = nn.Linear(in_size, out_size, bias=False)
#                         self.net.append(params_dict[layer][population]['projections'][projection_layer][pre_population])
#
#         # self.net = nn.ModuleList()
#         # for layer in params_dict:
#         #     for population in params_dict[layer]:
#         #         out_size = params_dict[layer][population]['n']
#         #         for projection_layer in params_dict[layer][population]['projections']:
#         #             for pre_population in params_dict[layer][population]['projections'][projection_layer]:
#         #                 in_size = params_dict[projection_layer][pre_population]['n']
#         #                 params_dict[layer][population]['projections'][projection_layer][pre_population] = nn.Linear(in_size, out_size, bias=False)
#         #                 self.net.append(params_dict[layer][population]['projections'][projection_layer][pre_population])
#
#     def forward(self, x):
#         for layer in self.params_dict:
#             for population in self.params_dict[layer]:
#                 for projection_layer in self.params_dict[layer][population]['projections']:
#                     for projection in self.params_dict[layer][population]['projections'][projection_layer].values():
#                         x = projection(x)
#         return x