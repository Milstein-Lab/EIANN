import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
from torch.optim import Adam, SGD
from torch.nn import MSELoss

import itertools
import numpy as np
from tqdm import tqdm


class AttrDict(dict):
    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key)
        if found is None:
            raise KeyError
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.populations = {}

    def append_population(self, population):
        self.populations[population.name] = population
        self.__dict__[population.name] = population

    def __iter__(self):
        for population in self.populations.values():
            yield population


class Population(object):
    def __init__(self, network, layer, name, size, activation, activation_kwargs=None, include_bias=False,
                 learn_bias=False, bias_init=None, bias_init_args=None, bias_bounds=None, bias_learning_rule='backprop',
                 bias_learning_rule_kwargs=None):
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
        if learn_bias:
            include_bias = True
        self.include_bias = include_bias
        self.learn_bias = learn_bias
        self.bias_init = bias_init
        if bias_init_args is None:
            bias_init_args = ()
        self.bias_init_args = bias_init_args
        self.bias_bounds = bias_bounds
        self.bias_learning_rule = bias_learning_rule
        if bias_learning_rule_kwargs is None:
            bias_learning_rule_kwargs = {}
        self.bias_learning_rule_kwargs = bias_learning_rule_kwargs
        if self.bias_learning_rule is None or self.bias_learning_rule == 'backprop':
            self.bias_update = lambda population: None
        elif not (self.bias_learning_rule in globals() and callable(globals()[self.bias_learning_rule])):
            raise RuntimeError \
                ('Population: callable for bias_learning_rule: %s must be imported' % bias_learning_rule)
        else:
            self.bias_update = \
                lambda population: globals()[self.bias_learning_rule](population, **self.bias_learning_rule_kwargs)
        self.activity_history_list = []
        self.projections = {}
        self.reinit()

    def reinit(self):
        self.activity = torch.zeros(self.size)
        self.state = torch.zeros(self.size)
        self.sample_activity = []

    def append_projection(self, pre_pop, weight_init=None, weight_init_args=None, weight_constraint=None,
                          weight_constraint_kwargs=None, weight_bounds=None, direction='FF', learning_rule='backprop',
                          learning_rule_kwargs=None, backward=None):
        if not self.projections:
            include_bias = self.include_bias
        else:
            include_bias = False
        projection = nn.Linear(pre_pop.size, self.size, bias=include_bias)

        if not self.projections and self.include_bias:
            if self.learn_bias and self.bias_learning_rule == 'backprop':
                projection.bias.requires_grad = True
                if backward is None:
                    backward = 'backprop_backward'
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
            projection.weight_update = lambda projection: None
        elif not (learning_rule in globals() and callable(globals()[learning_rule])):
            raise RuntimeError \
                ('Population.append_projection: callable for learning_rule: %s must be imported' % learning_rule)
        else:
            projection.weight_update  = lambda projection: globals()[learning_rule](projection, **learning_rule_kwargs)
        if learning_rule == 'backprop':
            if backward is None:
                backward = 'backprop_backward'
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

        projection.weight_constraint = weight_constraint
        if weight_constraint_kwargs is None:
            weight_constraint_kwargs = {}
        projection.weight_constraint_kwargs = weight_constraint_kwargs
        if weight_constraint is not None:
            if not(weight_constraint in globals() and callable(globals()[weight_constraint])):
                raise RuntimeError \
                    ('Population.append_projection: weight_constraint: %s must be imported and callable' %
                     weight_constraint)
            projection.constrain_weight = \
                lambda projection: \
                    globals()[weight_constraint](projection, **projection.weight_constraint_kwargs)

        if direction not in ['FF', 'FB']:
            raise RuntimeError('Population.append_projection: direction (%s) must be either FF or FB' % direction)
        projection.direction = direction
        projection.pre = pre_pop
        projection.post = self
        if pre_pop.layer.name not in self.projections:
            self.projections[pre_pop.layer.name] = {}
            self.__dict__[pre_pop.layer.name] = AttrDict()
        self.projections[pre_pop.layer.name][pre_pop.name] = projection
        self.__dict__[pre_pop.layer.name][pre_pop.name] = projection

        if backward is not None:
            if backward in globals() and callable(globals()[backward]):
                self.network.backward_methods.add(globals()[backward])
            else:
                raise RuntimeError('Population.append_projection: backward (%s) must be imported and callable' %
                                   backward)
        self.network.module_list.append(projection)

    def __iter__(self):
        for projections in self.projections.values():
            for projection in projections.values():
                yield projection

    @property
    def bias(self):
        if self.include_bias:
            projection = next(iter(self))
            return projection.bias
        else:
            return torch.zeros(self.size)

    @property
    def activity_history(self):
        return torch.cat([torch.unsqueeze(torch.stack(sample_activity), 0)
                          for sample_activity in self.activity_history_list], 0)


class Input(Population):
    def __init__(self, network, layer, name, size, *args, **kwargs):
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size
        self.activity_history_list = []
        self.projections = {}
        self.reinit()


class EIANN(nn.Module):
    def __init__(self, layer_config, projection_config, learning_rate, optimizer=SGD, optimizer_kwargs=None,
                 criterion=MSELoss, criterion_kwargs=None, seed=None, tau=1, forward_steps=1, backward_steps=1):
        """

        :param layer_config: nested dict
        :param projection_config: nested dict
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
        for i, (layer_name, pop_config) in enumerate(layer_config.items()):
            layer = Layer(layer_name)
            self.layers[layer_name] = layer
            self.__dict__[layer_name] = layer
            for j, (pop_name, pop_kwargs) in enumerate(pop_config.items()):
                if i == 0 and j == 0:
                    pop = Input(self, layer, pop_name, **pop_kwargs)
                else:
                    pop = Population(self, layer, pop_name, **pop_kwargs)
                layer.append_population(pop)

        for post_layer_name in projection_config:
            post_layer = self.layers[post_layer_name]
            for post_pop_name in projection_config[post_layer_name]:
                post_pop = layer.populations[post_pop_name]
                for pre_layer_name in projection_config[post_layer_name][post_pop_name]:
                    pre_layer = self.layers[pre_layer_name]
                    for pre_pop_name, projection_kwargs in \
                            projection_config[post_layer_name][post_pop_name][pre_layer_name].items():
                        pre_pop = pre_layer.populations[pre_pop_name]
                        post_pop.append_projection(pre_pop, **projection_kwargs)

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
        for i, post_layer in enumerate(self):
            if i > 0:
                for post_pop in post_layer:
                    if post_pop.include_bias and post_pop.bias_init is not None:
                        getattr(post_pop.bias.data, post_pop.bias_init)(*post_pop.bias_init_args)
                    for projection in post_pop:
                        if projection.weight_init is not None:
                            getattr(projection.weight.data, projection.weight_init)(*projection.weight_init_args)
        self.constrain_weights_and_biases()

    def constrain_weights_and_biases(self):
        for i, post_layer in enumerate(self):
            if i > 0:
                for post_pop in post_layer:
                    if post_pop.include_bias and post_pop.bias_bounds is not None:
                        post_pop.bias.data = post_pop.bias.data.clamp(*post_pop.bias_bounds)
                    for projection in post_pop:
                        if projection.weight_bounds is not None:
                            projection.weight.data = projection.weight.data.clamp(*projection.weight_bounds)
                        if projection.weight_constraint is not None:
                            projection.constrain_weight(projection)

    def reset_history(self):
        for layer in self:
            for population in layer:
                population.reinit()
                population.activity_history_list = []

    def forward(self, sample, store_history=False):

        for i, layer in enumerate(self):
            for pop in layer:
                pop.reinit()
            if i == 0:
                input_pop = next(iter(layer))
                input_pop.activity = sample

        for t in range(self.forward_steps):
            if t >= self.forward_steps - self.backward_steps:
                track_grad = True
            else:
                track_grad = False
            with torch.set_grad_enabled(track_grad):
                for post_layer in self:
                    for post_pop in post_layer:
                        post_pop.prev_activity = post_pop.activity
                for i, post_layer in enumerate(self):
                    for post_pop in post_layer:
                        if i > 0:
                            delta_state = -post_pop.state
                            for projection in post_pop:
                                pre_pop = projection.pre
                                if projection.direction == 'FF':
                                    delta_state = delta_state + projection(pre_pop.activity)
                                elif projection.direction == 'FB':
                                    delta_state = delta_state + projection(pre_pop.prev_activity)
                            post_pop.state = post_pop.state + delta_state / self.tau
                            post_pop.activity = post_pop.activation(post_pop.state)
                        if store_history:
                            post_pop.sample_activity.append(post_pop.activity.detach().clone())

        if store_history:
            for layer in self:
                for pop in layer:
                    pop.activity_history_list.append(pop.sample_activity)

        return next(iter(layer)).activity

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

                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.learn_bias:
                                post_pop.bias_update(post_pop)
                            for projection in post_pop:
                                projection.weight_update(projection)
                self.constrain_weights_and_biases()

        self.sample_order = torch.LongTensor(self.sample_order)
        self.sorted_sample_indexes = torch.LongTensor(self.sorted_sample_indexes)
        self.loss_history = torch.Tensor(self.loss_history)

        return self.loss

    def __iter__(self):
        for layer in self.layers.values():
            yield layer

def n_choose_k(n,k):
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n,length):
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def normalize_weight(projection, scale, autapses=False, axis=1):
    projection.weight.data /= torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    projection.weight.data *= scale
    if not autapses and projection.pre == projection.post:
        for i in range(projection.post.size):
            projection.weight.data[i,i] = 0.


def gjorgieva_hebb(projection, sign, learning_rate=None):
    if learning_rate is None:
        learning_rate = projection.post.network.learning_rate
    pre_activity = projection.pre.activity
    post_activity = projection.post.activity
    delta_weight = torch.outer(post_activity, pre_activity)
    if sign > 0:
        projection.weight.data += learning_rate * delta_weight
    else:
        projection.weight.data -= learning_rate * delta_weight


def bcm(projection, theta_init, theta_tau, k, learning_rate=None):
    if learning_rate is None:
        learning_rate = projection.post.network.learning_rate
    if not hasattr(projection, 'theta'):
        projection.theta = torch.ones(projection.post.size) * theta_init
    pre_activity = projection.pre.activity
    post_activity = projection.post.activity

    delta_weight = torch.outer(post_activity, pre_activity) * (post_activity - projection.theta).unsqueeze(1)
    projection.weight.data += learning_rate * delta_weight

    delta_theta = (-projection.theta + post_activity ** 2. / k) / theta_tau
    projection.theta += delta_theta


def oja(projection, learning_rate=None):
    if learning_rate is None:
        learning_rate = projection.post.network.learning_rate
    pre_activity = projection.pre.activity
    post_activity = projection.post.activity

    delta_weight = torch.outer(post_activity, pre_activity) - projection.weight * (post_activity ** 2).unsqueeze(1)
    projection.weight.data += learning_rate * delta_weight


def backprop_backward(network, output, target):
    loss = network.criterion(output, target)
    network.optimizer.zero_grad()
    loss.backward()
    network.optimizer.step()


def get_diag_argmax_row_indexes(data):
    """
    Sort the rows of a square matrix such that whenever row argmax and col argmax are equal, that value appears
    on the diagonal. Returns row indexes.
    :param data: 2d array; square matrix
    :return: array of int
    """
    data = np.array(data)
    if data.shape[0] != data.shape[1]:
        raise Exception('get_diag_argmax_row_indexes: data must be a square matrix')
    dim = data.shape[0]
    avail_row_indexes = list(range(dim))
    avail_col_indexes = list(range(dim))
    final_row_indexes = np.empty_like(avail_row_indexes)
    while(len(avail_col_indexes) > 0):
        row_selectivity = np.zeros_like(avail_row_indexes)
        row_max = np.max(data[avail_row_indexes, :][:, avail_col_indexes], axis=1)
        row_mean = np.mean(data[avail_row_indexes,:][:,avail_col_indexes], axis=1)
        nonzero_indexes = np.where(row_mean > 0)
        row_selectivity[nonzero_indexes] = row_max[nonzero_indexes] / row_mean[nonzero_indexes]

        row_index = avail_row_indexes[np.argsort(row_selectivity)[-1]]
        col_index = avail_col_indexes[np.argmax(data[row_index,avail_col_indexes])]
        final_row_indexes[col_index] = row_index
        avail_row_indexes.remove(row_index)
        avail_col_indexes.remove(col_index)
    return final_row_indexes


def test_EIANN_config(network, dataset, target, epochs):
    import matplotlib.pyplot as plt

    for sample in dataset:
        network.forward(sample, store_history=True)

    plt.figure()
    plt.imshow(network.Output.E.Input.E.weight.data)
    plt.colorbar()
    plt.xlabel('Input unit ID')
    plt.ylabel('Output unit ID')
    plt.title('Initial weights\nOutput_E <- Input_E')

    Output_E_activity_history = network.Output.E.activity_history
    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    im = this_axis.imshow(Output_E_activity_history[-dataset.shape[0]:, -1, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Initial activity\nOutput_E')
    this_axis = axes[1]
    Output_FBI_activity_history = network.Output.FBI.activity_history
    im = this_axis.imshow(Output_FBI_activity_history[-dataset.shape[0]:, -1, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Initial activity\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    for i in range(network.Output.E.size):
        this_axis.plot(torch.mean(Output_E_activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_E')
    this_axis = axes[1]
    for i in range(network.Output.FBI.size):
        this_axis.plot(torch.mean(Output_FBI_activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)

    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=True)

    Output_E_activity_history = network.Output.E.activity_history

    final_output = Output_E_activity_history[network.sorted_sample_indexes, -1, :][-dataset.shape[0]:, :].T
    if final_output.shape[0] == final_output.shape[1]:
        sorted_idx = get_diag_argmax_row_indexes(final_output)
    else:
        sorted_idx = np.arange(final_output.shape[0])

    sorted_loss_history = []
    for i in range(Output_E_activity_history.shape[0]):
        sample_idx = network.sample_order[i]
        sample_target = target[sample_idx,:]
        output = Output_E_activity_history[i,-1,sorted_idx]
        loss = network.criterion(output, sample_target)
        sorted_loss_history.append(loss)
    sorted_loss_history = torch.tensor(sorted_loss_history)

    fig = plt.figure()
    plt.plot(network.loss_history, label='Unsorted')
    plt.plot(sorted_loss_history, label='Sorted')
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    fig.tight_layout()
    fig.show()

    fig = plt.figure()
    plt.imshow(network.Output.E.Input.E.weight.data[sorted_idx, :])
    plt.colorbar()
    plt.xlabel('Input unit ID')
    plt.ylabel('Output unit ID')
    plt.title('Final weights\nOutput_E <- Input_E')
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    im = this_axis.imshow(final_output[sorted_idx, :])
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Final activity\nOutput_E')
    this_axis = axes[1]
    Output_FBI_activity_history = network.Output.FBI.activity_history
    im = this_axis.imshow(Output_FBI_activity_history[network.sorted_sample_indexes, -1, :][
                   -dataset.shape[0]:, :].T)
    plt.colorbar(im, ax=this_axis)
    this_axis.set_xlabel('Input pattern ID')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Final activity\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    fig, axes = plt.subplots(1, 2)
    this_axis = axes[0]
    for i in range(network.Output.E.size):
        this_axis.plot(torch.mean(Output_E_activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_E')
    this_axis = axes[1]
    for i in range(network.Output.FBI.size):
        this_axis.plot(torch.mean(Output_FBI_activity_history[-dataset.shape[0]:, :, i], axis=0))
    this_axis.set_xlabel('Equilibration time steps')
    this_axis.set_ylabel('Output unit ID')
    this_axis.set_title('Mean activity dynamics\nOutput_FBI')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, 'bias:', population.bias)