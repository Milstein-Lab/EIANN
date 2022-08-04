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


activation_dict = {'linear': lambda x: x,
                   'relu': nn.ReLU(),
                   'sigmoid': nn.Sigmoid(),
                   'softplus': nn.Softplus(beta=4)}


class universalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = 1
        self.seed = 42
        self.nn_modules = nn.ModuleList()
        self.net = Net()

        self.net.input_layer = Layer('input_layer', populations={'E':10})

        self.net.layer_1 = Layer('layer_1', populations={'E':7, 'I':2})
        self.net.layer_1.E.activation = 'softplus'
        self.net.layer_1.E.add_projections(self, [self.net.input_layer.E])

        self.net.layer_2 = Layer('layer_2', populations={'E':5})
        self.net.layer_2.E.activation = 'softplus'
        self.net.layer_2.E.add_projections(self, [self.net.input_layer.E, self.net.layer_1.E])

    def forward(self, input_pattern):
        for i, layer in enumerate(self.net):
            for j, population in enumerate(layer):
                if i==0 and j==0: # set activity for the first population of the input layer
                    population.activity = input_pattern

                delta_state = -population.state
                for projection in population:
                    delta_state += projection(projection.pre.activity)
                population.state += delta_state / self.tau
                population.activity = population.activation(population.state)


# class universalNet(nn.Module):
#     def __init__(self, params_dict, hparams):
#         super().__init__()
#         self.tau = hparams['tau']
#         self.seed = hparams['seed']
#
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


class Net(object):
    def __iter__(self):
        for key,value in self.__dict__.items():
            yield value

    def __repr__(self):
        items = ", ".join([name for name in self.__dict__])
        return f'{type(self)} :\n\t({items})'


class Layer(nn.Module):
    def __init__(self, name, populations):
        super().__init__()
        self.name = name
        self.populations = populations
        for pop,size in populations.items():
            self.__dict__[pop] = Population(layer=self.name, name=pop, size=size)

    def __iter__(self):
        for key,value in self.__dict__.items():
            if callable(value): #only iterate over Populations
                yield value


class Population(nn.Module):
    def __init__(self, layer, name, size):
        super().__init__()

        # Hyperparameters
        self.layer = layer
        self.name = name
        self.fullname = self.layer + self.name
        self.size = size
        self.activation = 'linear'
        self.bias_rule = 'backprop'
        self.learn_bias = False
        self.inputs = []

        # State variables
        self.state = torch.zeros(self.size)
        self.activity = torch.zeros(self.size)
        self.bias = torch.zeros(size)

        self.prev_activity = torch.zeros(self.size)
        self.activity_history = None

    def add_projections(self, network, pre_populations):
        for pre_pop in pre_populations:
            name = pre_pop.layer + pre_pop.name
            self.inputs.append(name)
            self.__dict__[name] = nn.Linear(pre_pop.size, self.size, bias=False)

            # Set projection attributes
            projection = self.__dict__[name]
            projection.name = f'{name}_to_{self.fullname}'
            projection.pre = pre_pop
            projection.learning_rule = 'backprop'
            projection.direction = 'FF'
            projection.lr = 0.01

            # Add to ModuleList to make projection a trainable parameter
            network.nn_modules.append(projection)

    def __iter__(self):
        for key,value in self.__dict__.items():
            if isinstance(value, nn.Linear): #only iterate over projections
                yield value

    # Automatically turn activation string into a callable function
    @property
    def activation(self):
        return activation_dict[self._act_name]

    @activation.setter
    def activation(self, act_name):
        self._act_name = act_name



# class universalNet1(nn.Module):
#     def __init__(self, params_dict):
#         super().__init__()
#         self.params_dict = params_dict
#         self.nn_modules = nn.ModuleList()
#
#         self.input_layer = Layer(n=7,)
##
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