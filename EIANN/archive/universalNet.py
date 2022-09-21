
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from universalNet_utils import plot_summary

# from typing import Any
# from torch import Tensor
# from torch.nn.parameter import Parameter, UninitializedParameter
# from torch.nn import functional as F
# from torch.nn import init
# from torch.nn.modules.module import Module


# hparams = {'seed': 42,
#            'dales_law': True}
#
# params_dict = {'layer0':
#                    {'E': {'n': 7}},
#                'layer1':
#                    {'E': {'n': 5,
#                          'activation': 'softplus',
#                          'inputs': ['layer0.E']},
#                     'I': {'n': 1}},
#                'layer2':
#                    {'E': {'n': 21,
#                          'bias': True,
#                          'inputs': ['layer0.E', 'layer1.E'],
#                          'learning_rule': 'Oja'}}}
#
# model = universalNet(params_dict, **hparams)

activation_dict = {'linear': nn.Identity(),
                   'relu': nn.ReLU(),
                   'sigmoid': nn.Sigmoid(),
                   'softplus': nn.Softplus(beta=4)}

# TODO: store order of presented patterns
# TODO: add kwargs as dict to population activation params eg softmax, {beta=4}
# TODO: store list/set of rules for every projection in a population (different rule for each proj)
# TODO: add weight bounds

def Hebb(pre, post):
    delta_W = torch.outer(post, pre)
    return delta_W

def Oja(pre, post, W):
    delta_W = torch.outer(post, pre) - W * (post**2).unsqueeze(1)
    return delta_W

def BCM(pre, post, theta):
    delta_W = torch.outer(post, pre) * (post - theta).unsqueeze(1)
    return delta_W


class universalNet(nn.Module):
    def __init__(self, params_dict, seed=42, dales_law=True):
        super().__init__()
        self.seed = seed
        # torch.manual_seed(self.seed)
        self.dales_law = dales_law

        self.nn_modules = nn.ModuleDict()

        # Create layer & population objects from params dict
        for layer_name in params_dict:
            populations = {}
            for population in params_dict[layer_name]:
                populations[population] = params_dict[layer_name][population].pop('n')
            self._modules[layer_name] = self._modules[layer_name] = Layer(self, layer_name, populations=populations)

            # Add projections and apply optional parameters (bias, learning rules, etc.) to the population
            for population in params_dict[layer_name]:
                # if params_dict[layer_name][population]:
                self._modules[layer_name].__dict__[population].update(self, **params_dict[layer_name][population])

        self.output_pop = self._modules[layer_name].E # E pop of final layer

        # self.input_layer = Layer('input_layer', populations={'E':7})
        # self.layer1 = Layer('layer1', populations={'E':5, 'I':1})
        # self.layer1.E.update(self, activation='softplus',
        #                      inputs=['input_layer.E'])
        # self.layer2 = Layer('layer2', populations={'E':21})
        # self.layer2.E.update(self, activation='softplus', bias=True,
        #                      inputs=['input_layer.E', 'layer1.E'], learning_rule = 'Oja')

    def forward(self, all_patterns, track_activity=False):
        for p, pattern in enumerate(self.all_patterns):
            self.reset_units()
            for t in range(self.num_timesteps):
                self.forward_single(pattern)
                for layer in self:
                    for populaltion in layer:
                        populaltion.all_pattern_activities[t,p,:] = populaltion.activity.detach()
        return self.output_pop.all_pattern_activities.clone()

    def forward_single(self, input_pattern, track_activity=False, training=False):
        for i, layer in enumerate(self):
            for j, population in enumerate(layer):
                if i==0 and j==0: # pass pattern to first population of input layer
                    population.activity = input_pattern
                else:
                    population.prev_activity = population.activity
                    population.state = population.state - population.state/population.tau
                    for projection in population:
                        if projection.direction == 'FF':
                            population.state = population.state + projection(projection.pre.activity)/population.tau
                        elif projection.direction == 'FB':
                            population.state = population.state + projection(projection.pre.prev_activity)/population.tau
                    population.activity = population.activation(population.state + population.bias)

                if track_activity:
                    population.activity_history_ls.append(population.activity.detach())

                if training:
                    population.activity_train_history[self.epoch, self.t, self.pattern_idx, :] = population.activity.detach()

    def train(self, num_epochs, all_patterns, all_targets, lr, num_timesteps=1, num_BPTT_steps=1, plot=False):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.all_patterns = all_patterns
        num_patterns = all_patterns.shape[0]
        self.num_timesteps = num_timesteps
        loss_history = []
        for layer in self:
            for population in layer:
                population.activity_train_history = torch.zeros(num_epochs, num_timesteps, num_patterns, population.size)
                population.all_pattern_activities = torch.zeros(num_timesteps, num_patterns, population.size)

        self.initial_activity = self.forward(all_patterns)

        for epoch in tqdm(range(num_epochs)):
            self.epoch = epoch
            for pattern_idx in torch.randperm(num_patterns):
                self.pattern_idx = pattern_idx
                input_pattern = all_patterns[pattern_idx]
                target = all_targets[pattern_idx]

                self.reset_units()
                for t in range(num_timesteps):
                    self.t = t
                    if t >= (num_timesteps - num_BPTT_steps):  # truncate BPTT to only evaluate n steps from the end
                        track_grad = True
                    else:
                        track_grad = False

                    with torch.set_grad_enabled(track_grad):
                        self.forward_single(input_pattern, training=True)

                output = self.output_pop.activity
                loss = self.criterion(output,target)
                loss_history.append(loss.detach())

                self.update_params(loss)

                if self.dales_law:
                    self.rectify_weights()

                # Save weights & biases & activity
                for layer in self:
                    for population in layer:
                        if hasattr(population,'bias'):
                            population.bias_history_ls.append(population.bias.detach().clone())
                        for projection in population:
                            projection.weight_history_ls.append(projection.weight.detach().clone())

        self.loss_history = torch.tensor(loss_history)
        self.final_activity = self.forward(all_patterns)

        if plot:
            plot_summary(self)

    def rectify_weights(self):
        for layer in self:
            for population in layer:
                for projection in population:
                    projection.rectify_weights()

    def reset_units(self):
    # Zero states and activities
        for layer in self:
            for population in layer:
                population.state = torch.zeros(population.size)
                population.activity = torch.zeros(population.size)

    def update_params(self, loss):
        if any([param.requires_grad for param in self.parameters()]):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for layer in self:
            for population in layer:
                # population.step_bias()
                # TODO: add theta update conditional on population projections
                # delta_theta = (-population.theta_BCM + population.activity ** 2 / population.theta_k) / population.theta_tau
                # population.theta_BCM += delta_theta
                for projection in population:
                    if projection.learning_rule == 'Hebb':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity)
                        projection.weight.data += projection.lr * delta_W

                    elif projection.learning_rule == 'Oja':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity,
                                                     W = projection.weight)
                        projection.weight.data += projection.lr * delta_W

                    elif projection.learning_rule == 'BCM':
                        delta_W = projection.delta_W(pre = projection.pre.activity,
                                                     post = population.activity,
                                                     theta = population.theta_BCM)
                        delta_theta = (-population.theta_BCM + population.activity**2/population.theta_k) / population.theta_tau
                        population.theta_BCM += delta_theta
                        projection.weight.data += projection.lr * delta_W

    def __iter__(self):
        for key,value in self._modules.items():
            if isinstance(value, Layer):
                yield value


class Layer(nn.Module):
    def __init__(self, network, name, populations):
        super().__init__()
        self.name = name
        self.populations = populations
        for pop,size in populations.items():
            self.__dict__[pop] = Population(network, layer=name, name=pop, size=size)

    def __iter__(self):
        for key,value in self.__dict__.items():
            if callable(value): #only iterate over Populations
                yield value

    def __repr__(self):
        ls = []
        for name,value in self.populations.items():
            if name[0] != '_':
                ls.append(f'{name}: {value}')
        items = ", ".join(ls)
        return f'{type(self)} :\n\t({items})'


class Population(nn.Module):
    def __init__(self, network, layer, name, size):
        super().__init__()
        # Hyperparameters
        self.layer = layer
        self.name = name
        self.fullname = self.layer + self.name
        self.size = size
        self.inputs = []
        # self.theta_tau = 1 # for BCM learning
        # self.theta_k = 1 # for BCM learning

        # State variables
        self.state = torch.zeros(self.size)
        self.activity = torch.zeros(self.size)
        self.prev_activity = torch.zeros(self.size)
        self.activity_history_ls = []
        self.bias = torch.zeros(self.size)
        self.bias_history_ls = [self.bias.detach().clone()]

        # register to ModuleDict to make bias a backprop-trainable parameter
        network.nn_modules[self.fullname] = self

    def update(self, network, activation='linear', tau=1, bias=None, bias_rule='backprop', inputs=None):
        self.activation = activation
        self.learn_bias = bias
        self.bias_rule = bias_rule
        self.tau = tau

        if bias:
            if bias_rule == 'backprop':
                self.bias = nn.Parameter(self.bias)

        if inputs:
            self.add_projections(network, inputs)

    def step_bias(self):
        # TODO: add bias update (as nn.Parameter? or through nn.Linear?)
        return

    def add_projections(self, network, inputs: dict):
        for proj_origin in inputs:
            pre_layer, pre_pop = proj_origin.split('.')
            pre_pop = network._modules[pre_layer].__dict__[pre_pop] # get population object from string name
            self.inputs.append(pre_pop.fullname)

            # TODO: add theta during projection init for BCM layers
            # TODO: create set of learning rules and separate backward passes for each
            self.__dict__[pre_pop.fullname] = Projection(pre_pop, self, dales_law=network.dales_law, **inputs[proj_origin])

            # register to ModuleDict to make projection a backprop-trainable parameter
            projection_name = self.__dict__[pre_pop.fullname].name
            network.nn_modules[projection_name] = self.__dict__[pre_pop.fullname]

    def __iter__(self):
        for key,value in self.__dict__.items():
            if isinstance(value, Projection): #only iterate over projections
                yield value

    def __repr__(self):
        ls = []
        for name in self.__dict__:
            if name[0] != '_':
                ls.append(name)
        items = ", ".join(ls)
        return f'{type(self)} :\n\t({items})'

    @property
    def activation(self):
        # Automatically turn activation string into a callable function
        return activation_dict[self._act_name]
    @activation.setter
    def activation(self, act_name):
        self._act_name = act_name

    @property
    def activity_history(self):
        return torch.stack(self.activity_history_ls)

    @property
    def bias_history(self):
        return torch.stack(self.bias_history_ls)


class Projection(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Attributes:
        weight: the learnable weights of the module of shape
    """
    def __init__(self, pre_population, post_population, name=None, distribution='kaiming_uniform', bounds=(-0.5,0.5),
                 dales_law=True, direction='FF', learning_rule='backprop'):
        super().__init__()
        self.pre = pre_population
        self.post = post_population
        self.direction = direction
        self.bounds = bounds
        self.learning_rule = learning_rule
        if name:
            self.name = name
        else:
            self.name = f'{self.post.fullname}_{self.pre.fullname}'

        self.weight = nn.Parameter(torch.empty(self.post.size, self.pre.size))
        self.initialize_weights(distribution, dales_law, bounds)
        self.weight_history_ls = [self.weight.detach().clone()]

        if learning_rule != 'backprop':
            self.weight.requires_grad = False
            self.delta_W = lambda **args: globals()[learning_rule](**args)

    def forward(self, input):
        return F.linear(input, self.weight)

    def initialize_weights(self, distribution, dales_law, bounds):
        if distribution == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            distribution += '_'
            distribute_weights = getattr(nn.init, distribution) # get function for desired initial weight distribution
            distribute_weights(self.weight, *bounds)

        if dales_law:
            self.rectify_weights()

    def rectify_weights(self):
        if self.pre.name == 'E':
            self.weight.data = self.weight.data.clamp(min=0, max=None)
        elif self.pre.name == 'I':
            self.weight.data = self.weight.data.clamp(min=None, max=0)
        else:
            raise RuntimeWarning("Population name must be 'E' or 'I' when using Dale's law. Weights not rectified")

    @property
    def weight_history(self):
        return torch.stack(self.weight_history_ls)
