import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
from torch.optim import Adam, SGD
from torch.nn import MSELoss

import itertools
import numpy as np
from tqdm import tqdm


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.populations = {}

    def append_population(self, population):
        self.populations[population.name] = population


class Input(object):
    def __init__(self, network, layer, name, size):
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size
        self.activity = torch.zeros(self.size)
        self.prev_activity = torch.zeros(self.size)
        self.temp_activity_history = []
        self.activity_history = None
        self.projections = {}

    def reinit(self):
        self.activity = torch.zeros(self.size)
        self.temp_activity_history = []
        self.prev_activity = torch.zeros(self.size)


class Population(object):
    def __init__(self, network, layer, name, size, activation, activation_kwargs=None, include_bias=False,
                 learn_bias=False, bias_init=None, bias_init_args=None, bias_bounds=None, bias_rule=None,
                 bias_rule_kwargs=None):
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size
        if not (activation in globals() and callable(globals()[activation])):
            raise RuntimeError \
                ('Population: callable for activation: %s must be imported' % activation)
        if activation_kwargs is None:
            activation_kwargs = {}
        self.activation_kwargs = activation_kwargs
        self.activation = lambda x: globals()[activation](x, **activation_kwargs)
        self.include_bias = include_bias
        self.learn_bias = learn_bias
        self.bias_init = bias_init
        self.bias_init_args = bias_init_args
        self.bias_bounds = bias_bounds
        self.bias_rule = bias_rule
        self.bias_rule_kwargs = bias_rule_kwargs
        if self.bias_rule is None or self.bias_rule == 'backprop':
            self.bias_update = lambda population: None
        elif not (self.bias_rule in globals() and callable(globals()[self.bias_rule])):
            raise RuntimeError \
                ('Population: callable for bias_rule: %s must be imported' % bias_rule)
        else:
            if self.bias_rule_kwargs is None:
                self.bias_rule_kwargs = {}
            self.bias_update = lambda population: globals()[self.bias_rule](population, **self.bias_rule_kwargs)
        self.activity = torch.zeros(self.size)
        self.prev_activity = torch.zeros(self.size)
        self.temp_activity_history = []
        self.activity_history = None
        self.state = torch.zeros(self.size)
        self.projections = {}

    def append_projection(self, pre_pop, weight_init=None, weight_init_args=None, weight_bounds=None, direction='FF',
                          learning_rule='backprop', learning_rule_kwargs=None, backward='backprop_backward'):
        if not self.projections:
            include_bias = self.include_bias
        else:
            include_bias = False
        projection = nn.Linear(pre_pop.size, self.size, bias=include_bias)

        if not self.projections and self.include_bias:
            if self.learn_bias and self.bias_rule == 'backprop':
                projection.bias.requires_grad = True
            else:
                projection.bias.requires_grad = False
            if self.bias_init is not None:
                if not hasattr(projection.bias.data, self.bias_init):
                    raise RuntimeError(
                        'Population.append_projection: callable for bias_init: %s must be a method of Tensor' %
                        bias_init)
                if self.bias_init_args is None:
                    self.bias_init_args = ()

        projection.learning_rule = learning_rule
        if learning_rule_kwargs is None:
            learning_rule_kwargs = {}
        projection.learning_rule_kwargs = learning_rule_kwargs
        if learning_rule is None or learning_rule == 'backprop':
            projection.weight_update = lambda projection: self.network.optimizer.step()
        elif not (learning_rule in globals() and callable(globals()[learning_rule])):
            raise RuntimeError \
                ('Population.append_projection: callable for learning_rule: %s must be imported' % learning_rule)
        else:
            projection.weight_update  = lambda projection: globals()[learning_rule](projection, **learning_rule_kwargs)
        if learning_rule == 'backprop':
            projection.weight.requires_grad = True
        else:
            projection.weight.requires_grad = False

        projection.weight_init = weight_init
        if weight_init is not None and not hasattr(projection.weight.data, weight_init):
            raise RuntimeError \
                ('Population.append_projection: callable for weight_init: %s must be a method of Tensor' % weight_init)
        if weight_init_args is None:
            weight_init_args = ()
        projection.weight_init_args = weight_init_args
        projection.weight_bounds = weight_bounds

        if direction not in ['FF', 'FB']:
            raise RuntimeError('Population.append_projection: direction (%s) must be either FF or FB' % direction)
        projection.direction = direction
        projection.pre = pre_pop
        projection.post = self
        if pre_pop.layer.name not in self.projections:
            self.projections[pre_pop.layer.name] = {}
        self.projections[pre_pop.layer.name][pre_pop.name] = projection

        if backward is not None:
            if backward in globals() and callable(globals()[backward]):
                self.network.backward_methods.add(globals()[backward])
            else:
                raise RuntimeError('Population.append_projection: backward (%s) must be imported and callable' %
                                   backward)
        self.network.module_list.append(projection)

    @property
    def bias(self):
        if self.include_bias:
            pre_pops = next(iter(self.projections.values()))
            projection = next(iter(pre_pops.values()))
            return projection.bias
        else:
            return torch.zeros(self.size)

    def reinit(self):
        self.activity = torch.zeros(self.size)
        self.temp_activity_history = []
        self.prev_activity = torch.zeros(self.size)
        self.state = torch.zeros(self.size)


class FBI_RNN(nn.Module):
    def __init__(self, input_size, output_size, fbi_size, learning_rate, optimizer=SGD, optimizer_kwargs=None,
                 criterion=MSELoss, criterion_kwargs=None, seed=None, tau=1, forward_steps=1, backward_steps=1):
        """

        :param input_size:
        :param output_size:
        :param fbi_size:
        :param learning_rate: float; applies to weights and biases in absence of projection-specific learning rates
        :param optimizer: callable
        :param optimizer_kwargs: dict
        :param criterion: callable
        :param criterion_kwargs: dict
        :param seed: int or sequence of int
        :param tau: int
        :param forward_steps: int
        :param backward_steps: int
        """
        super().__init__()
        self.learning_rate = learning_rate
        if not callable(criterion):
            raise RuntimeError('EIANN: criterion (%s) must be imported and callable' % criterion)
        if criterion_kwargs is None:
            criterion_kwargs = {}
        self.criterion_kwargs = criterion_kwargs
        self.criterion = criterion(**criterion_kwargs)

        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.tau = tau
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps

        self.backward_methods = set()
        self.module_list = nn.ModuleList()

        self.layers = {}
        input_layer = Layer('Input')
        self.layers[input_layer.name] = input_layer
        input_pop = Input(self, input_layer, 'E', input_size)
        input_layer.append_population(input_pop)

        output_layer = Layer('Output')
        self.layers[output_layer.name] = output_layer
        E_pop = Population(self, output_layer, 'E', output_size, 'softplus', {'beta': 4.}, include_bias=False)
        output_layer.append_population(E_pop)
        FBI_pop = Population(self, output_layer, 'FBI', fbi_size, 'softplus', {'beta': 4.}, include_bias=False)
        output_layer.append_population(FBI_pop)
        output_layer.populations['E'].append_projection(input_layer.populations['E'], 'uniform_', (0, 1), (0, 100),
                                                        'FF', 'backprop')
        output_layer.populations['E'].append_projection(output_layer.populations['FBI'], 'fill_', (-3.838023E+00,),
                                                        None, 'FB', learning_rule=None)
        output_layer.populations['FBI'].append_projection(output_layer.populations['E'], 'fill_', (1,), None,
                                                          'FF', learning_rule=None)

        if optimizer is not None:
            if not callable(optimizer):
                raise RuntimeError('EIANN: optimizer (%s) must be imported and callable' % optimizer)
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            self.optimizer_kwargs = optimizer_kwargs
            optimizer = optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.optimizer = optimizer
        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        for i, post_layer in enumerate(self.layers.values()):
            if i > 0:
                for post_pop in post_layer.populations.values():
                    if post_pop.include_bias and post_pop.bias_init is not None:
                        getattr(post_pop.bias.data, post_pop.bias_init)(*post_pop.bias_init_args)
                    for pre_layer_name in post_pop.projections:
                        for projection in post_pop.projections[pre_layer_name].values():
                            if projection.weight_init is not None:
                                getattr(projection.weight.data, projection.weight_init)(*projection.weight_init_args)

    def clamp_weights_and_biases(self):
        for i, post_layer in enumerate(self.layers.values()):
            if i > 0:
                for post_pop in post_layer.populations.values():
                    if post_pop.learn_bias and post_pop.bias_bounds is not None:
                        post_pop.bias.data = post_pop.bias.data.clamp(*post_pop.bias_bounds)
                    for pre_layer_name in post_pop.projections:
                        for projection in post_pop.projections[pre_layer_name].values():
                            if projection.weight_bounds is not None:
                                projection.weight.data = projection.weight.data.clamp(*projection.weight_bounds)

    def reset_history(self):
        for layer in self.layers.values():
            for population in layer.populations.values():
                population.reinit()
                population.activity_history = None

    def forward(self, sample, store_history=False):

        for i, layer in enumerate(self.layers.values()):
            for pop in layer.populations.values():
                pop.reinit()
            if i == 0:
                input_pop = next(iter(layer.populations.values()))
                input_pop.activity = sample

        for t in range(self.forward_steps):
            if t >= self.forward_steps - self.backward_steps:
                track_grad = True
            else:
                track_grad = False
            with torch.set_grad_enabled(track_grad):
                for i, post_layer in enumerate(self.layers.values()):
                    for post_pop in post_layer.populations.values():
                        if i > 0:
                            delta_state = -post_pop.state + post_pop.bias
                            for pre_layer_name in post_pop.projections:
                                for pre_pop_name in post_pop.projections[pre_layer_name]:
                                    pre_pop = self.layers[pre_layer_name].populations[pre_pop_name]
                                    projection = post_pop.projections[pre_layer_name][pre_pop_name]
                                    if projection.direction == 'FF':
                                        delta_state += projection(pre_pop.activity)
                                    elif projection.direction == 'FB':
                                        delta_state += projection(pre_pop.prev_activity)
                            post_pop.state = post_pop.state + delta_state / self.tau
                            post_pop.activity = post_pop.activation(post_pop.state)
                            post_pop.prev_activity = post_pop.activity.clone()
                        if store_history:
                            post_pop.temp_activity_history.append(post_pop.activity.detach().clone())
        if store_history:
            for layer in self.layers.values():
                for pop in layer.populations.values():
                    temp_activity_history = torch.stack(pop.temp_activity_history)
                    if pop.activity_history is None:
                        pop.activity_history = torch.unsqueeze(temp_activity_history, 0)
                    else:
                        pop.activity_history = torch.cat \
                            ((pop.activity_history, torch.unsqueeze(temp_activity_history, 0)))

        return next(iter(layer.populations.values())).activity

    def train(self, dataset, target, epochs, store_history=False, shuffle=True, status_bar=False):
        """

        :param dataset: 2d array or tensor of float
        :param target: 2d array or tensor of float
        :param epochs: int
        :param store_history: bool
        :param shuffle: bool
        :param status_bar: bool
        """
        num_samples = dataset.shape[0]
        self.sample_order = []
        self.sorted_sample_indexes = []
        self.loss_history = []

        if status_bar:
            epoch_iter = tqdm(range(epochs))
        else:
            epoch_iter = range(epochs)
        for epoch in epoch_iter:
            sample_indexes = torch.randperm(num_samples)
            self.sample_order.extend(sample_indexes)
            self.sorted_sample_indexes.extend(np.add(epoch * num_samples, np.argsort(sample_indexes)))
            for sample_idx in sample_indexes:
                sample = dataset[sample_idx]
                sample_target  = target[sample_idx]
                output = self.forward(sample, store_history)
                self.loss = self.criterion(output, sample_target)
                self.loss_history.append(self.loss.detach())
                for backward in self.backward_methods:
                    backward(self, output, sample_target)

                for i, post_layer in enumerate(self.layers.values()):
                    if i > 0:
                        for post_pop in post_layer.populations.values():
                            if post_pop.learn_bias:
                                post_pop.bias_update(post_pop)
                            for pre_layer_name in post_pop.projections:
                                for projection in post_pop.projections[pre_layer_name].values():
                                    projection.weight_update(projection)
                self.clamp_weights_and_biases()

        self.sample_order = torch.LongTensor(self.sample_order)
        self.sorted_sample_indexes = torch.LongTensor(self.sorted_sample_indexes)
        self.loss_history = torch.Tensor(self.loss_history)

        return self.loss


def n_choose_k(n,k):
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n,length):
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def Hebb(projection, learning_rate):
    delta_W = projection.post.activity * projection.pre.activity * learning_rate
    projection.weight += delta_W


def BCM(projection, learning_rate, tau, k):
    delta_W = projection.bias


def backprop_backward(network, output, target):
    loss = network.criterion(output, target)
    network.optimizer.zero_grad()
    loss.backward()

