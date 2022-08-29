import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
from torch.optim import Adam, SGD
from torch.nn import MSELoss
import numpy as np


class AttrDict(dict):
    '''
    Dict class for storing population attributes (?)
    '''
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
                 bias_init=None, bias_init_args=None, bias_bounds=None, bias_learning_rule=None,
                 bias_learning_rule_kwargs=None):
        """
        Class for population of neurons
        :param network:
        :param layer:
        :param name:
        :param size:
        :param activation:
        :param activation_kwargs:
        :param include_bias:
        :param bias_init:
        :param bias_init_args:
        :param bias_bounds:
        :param bias_learning_rule:
        :param bias_learning_rule_kwargs:
        """
        # Constants
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size

        # Set callable activation function
        if not (activation in globals() and callable(globals()[activation])):
            raise RuntimeError \
                ('Population: callable for activation: %s must be imported' % activation)
        if activation_kwargs is None:
            activation_kwargs = {}
        self.activation_kwargs = activation_kwargs
        self.activation = lambda x: globals()[activation](x, **activation_kwargs)

        # Set bias parameters
        self.bias_init = bias_init
        if bias_init_args is None:
            bias_init_args = ()
        self.bias_init_args = bias_init_args
        self.bias_bounds = bias_bounds
        if bias_learning_rule_kwargs is None:
            bias_learning_rule_kwargs = {}
        if bias_learning_rule is None:
            self.bias_learning_rule_class = BiasLearningRule
        else:
            include_bias = True
            if bias_learning_rule == 'Backprop':
                self.bias_learning_rule_class = BackpropBias
            elif bias_learning_rule in globals() and issubclass(globals()[bias_learning_rule], BiasLearningRule):
                self.bias_learning_rule_class = globals()[bias_learning_rule]
            else:
                raise RuntimeError \
                    ('Population: bias_learning_rule: %s must be imported and subclass of BiasLearningRule' %
                     bias_learning_rule)
        self.include_bias = include_bias
        self.bias_learning_rule = self.bias_learning_rule_class(self, **bias_learning_rule_kwargs)
        self.network.backward_methods.add(self.bias_learning_rule_class.backward)

        # Initialize storage containers
        self.activity_history_list = []
        self._activity_history = None
        self.projections = {}
        self.backward_projections = []
        self.reinit()

    def reinit(self):
        '''
        Method for resetting state variables of a population
        '''
        self.activity = torch.zeros(self.size)
        self.state = torch.zeros(self.size)
        self.sample_activity = []

    def append_projection(self, pre_pop, weight_init=None, weight_init_args=None, weight_constraint=None,
                          weight_constraint_kwargs=None, weight_bounds=None, direction='forward', compartment=None,
                          learning_rule='Backprop', learning_rule_kwargs=None):
        """

        :param pre_pop: :class:'Population'
        :param weight_init: str
        :param weight_init_args: tuple
        :param weight_constraint: str
        :param weight_constraint_kwargs: dict
        :param weight_bounds: tuple of float
        :param direction: str in ['forward', 'backward', 'recurrent', 'F', 'B', 'R']
        :param compartment: None or str in ['soma', 'dend']
        :param learning_rule: str
        :param learning_rule_kwargs: dict
        """
        # Create projection object
        if not self.projections:
            include_bias = self.include_bias
        else:
            include_bias = False
        projection = nn.Linear(pre_pop.size, self.size, bias=include_bias)
        projection.pre = pre_pop
        projection.post = self

        # Specify bias, stored as an attribute of the projection class
        if not self.projections and self.include_bias:
            if self.bias_learning_rule_class != BackpropBias:
                projection.bias.requires_grad = False
            if self.bias_init is not None:
                if not hasattr(projection.bias.data, self.bias_init):
                    raise RuntimeError(
                        'Population.append_projection: callable for bias_init: %s must be a method of Tensor' %
                        bias_init)
                if self.bias_init_args is None:
                    self.bias_init_args = ()

        # Set learning rule as callable of the projection
        projection.weight_bounds = weight_bounds

        # Set learning rule as callable of the projection
        if learning_rule_kwargs is None:
            learning_rule_kwargs = {}
        if learning_rule is None:
            projection.learning_rule_class = LearningRule
        elif learning_rule in globals() and issubclass(globals()[learning_rule], LearningRule):
            projection.learning_rule_class = globals()[learning_rule]
        else:
            raise RuntimeError \
                ('Population.append_projection: learning_rule: %s must be imported and instance of LearningRule' %
                 learning_rule)
        projection.learning_rule  = projection.learning_rule_class(projection, **learning_rule_kwargs)
        if projection.learning_rule_class != Backprop:
            projection.weight.requires_grad = False
        self.network.backward_methods.add(projection.learning_rule_class.backward)

        # Set projection parameters
        projection.weight_init = weight_init
        if weight_init is not None and not hasattr(projection.weight.data, weight_init):
            raise RuntimeError \
                ('Population.append_projection: callable for weight_init: %s must be a method of Tensor' % weight_init)
        if weight_init_args is None:
            weight_init_args = ()
        projection.weight_init_args = weight_init_args

        projection.weight_constraint = weight_constraint
        if weight_constraint_kwargs is None:
            weight_constraint_kwargs = {}
        projection.weight_constraint_kwargs = weight_constraint_kwargs
        if weight_constraint is not None:
            if not (weight_constraint in globals() and callable(globals()[weight_constraint])):
                raise RuntimeError \
                    ('Population.append_projection: weight_constraint: %s must be imported and callable' %
                     weight_constraint)
            projection.constrain_weight = \
                lambda projection: \
                    globals()[weight_constraint](projection, **projection.weight_constraint_kwargs)

        if direction not in ['forward', 'backward', 'recurrent', 'F', 'B', 'R']:
            raise RuntimeError('Population.append_projection: direction (%s) must be forward, backward, or recurrent' %
                               direction)
        projection.direction = direction
        if projection.direction in ['backward', 'B']:
            self.backward_projections.append(projection)

        if compartment is not None and compartment not in ['soma', 'dend']:
            raise RuntimeError('Population.append_projection: compartment (%s) must be None, soma, or dend' %
                               compartment)
        projection.compartment = compartment

        if pre_pop.layer.name not in self.projections:
            self.projections[pre_pop.layer.name] = {}
            self.__dict__[pre_pop.layer.name] = AttrDict()
        self.projections[pre_pop.layer.name][pre_pop.name] = projection
        self.__dict__[pre_pop.layer.name][pre_pop.name] = projection

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
        if self._activity_history is None:
            if self.activity_history_list:
                self._activity_history = torch.stack([torch.stack(sample_activity)
                                                      for sample_activity in self.activity_history_list])
                self.activity_history_list = []
        else:
            if self.activity_history_list:
                self._activity_history = torch.cat([self._activity_history,
                                                    torch.stack([torch.stack(sample_activity)
                                                                 for sample_activity in self.activity_history_list])])
                self.activity_history_list = []

        return self._activity_history


class Input(Population):
    def __init__(self, network, layer, name, size, *args, **kwargs):
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size
        self.activity_history_list = []
        self._activity_history = None
        self.projections = {}
        self.reinit()


class EIANN(nn.Module):
    def __init__(self, layer_config, projection_config, learning_rate, optimizer=SGD, optimizer_kwargs=None,
                 criterion=MSELoss, criterion_kwargs=None, seed=None, tau=1, forward_steps=1, backward_steps=1,
                 verbose=False):
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
        :param verbose: bool
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
                post_pop = post_layer.populations[post_pop_name]
                for pre_layer_name in projection_config[post_layer_name][post_pop_name]:
                    pre_layer = self.layers[pre_layer_name]
                    for pre_pop_name, projection_kwargs in \
                            projection_config[post_layer_name][post_pop_name][pre_layer_name].items():
                        pre_pop = pre_layer.populations[pre_pop_name]
                        post_pop.append_projection(pre_pop, **projection_kwargs)
                        if verbose:
                            print('EIANN: appending a projection from %s %s -> %s %s' %
                                  (pre_pop.layer.name, pre_pop.name, post_pop.layer.name, post_pop.name))

        if optimizer is not None:
            if not callable(optimizer):
                raise RuntimeError('EIANN: optimizer (%s) must be imported and callable' % optimizer)
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            self.optimizer_kwargs = optimizer_kwargs
            optimizer = optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        self.optimizer = optimizer
        self.init_weights_and_biases()
        self.sample_order = []
        self.sorted_sample_indexes = []
        self.loss_history = []

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
        self.sample_order = []
        self.sorted_sample_indexes = []
        self.loss_history = []
        for layer in self:
            for population in layer:
                population.reinit()
                population.activity_history_list = []
                population._activity_history = None

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
                                if projection.direction in ['forward', 'F']:
                                    delta_state = delta_state + projection(pre_pop.activity)
                                elif projection.direction in ['recurrent', 'R']:
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

        if status_bar:
            from tqdm import tqdm
            epoch_iter = tqdm(range(epochs))
        else:
            epoch_iter = range(epochs)

        for epoch in epoch_iter:
            sample_indexes = torch.randperm(num_samples)
            self.sample_order.extend(sample_indexes)
            self.sorted_sample_indexes.extend(np.add(epoch * num_samples, np.argsort(sample_indexes)))
            for sample_idx in sample_indexes:
                sample = dataset[sample_idx]
                sample_target = target[sample_idx]
                output = self.forward(sample, store_history)

                self.loss = self.criterion(output, sample_target)
                self.loss_history.append(self.loss.detach())

                for backward in self.backward_methods:
                    backward(self, output, sample_target)

                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.step()
                            for projection in post_pop:
                                projection.learning_rule.step()
                self.constrain_weights_and_biases()

        self.sample_order = torch.LongTensor(self.sample_order)
        self.sorted_sample_indexes = torch.LongTensor(self.sorted_sample_indexes)
        self.loss_history = torch.Tensor(self.loss_history)

        return self.loss

    def __iter__(self):
        for layer in self.layers.values():
            yield layer


def normalize_weight(projection, scale, autapses=False, axis=1):
    projection.weight.data /= torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    projection.weight.data *= scale
    if not autapses and projection.pre == projection.post:
        for i in range(projection.post.size):
            projection.weight.data[i, i] = 0.


class LearningRule(object):
    def __init__(self, projection, learning_rate=None):
        self.projection = projection
        if learning_rate is None:
            learning_rate = self.projection.post.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target):
        pass


class BiasLearningRule(object):
    def __init__(self, population, learning_rate=None):
        self.population = population
        if learning_rate is None:
            learning_rate = self.population.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target):
        pass


class Backprop(LearningRule):
    @classmethod
    def backward(cls, network, output, target):
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class BackpropBias(BiasLearningRule):
    backward = Backprop.backward


class Oja(LearningRule):
    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) - \
                       self.projection.weight * (self.projection.post.activity ** 2).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight


class BCM(LearningRule):
    def __init__(self, projection, theta_init, theta_tau, k, learning_rate=None):
        """

        :param projection:
        :param theta_init: float
        :param theta_tau: float
        :param k: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size) * theta_init

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) * \
                       (self.projection.post.activity - self.projection.post.prev_theta).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target):
        for i, layer in enumerate(network):
            if i > 0:
                for population in layer:
                    for projection in population:
                        if projection.learning_rule_class == cls:
                            population.prev_theta = population.theta.clone()
                            delta_theta = (-population.theta + population.activity ** 2. /
                                           projection.learning_rule.k) / projection.learning_rule.theta_tau
                            population.theta += delta_theta
                            # only update theta once per population
                            break


class GjorgievaHebb(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


def get_scaled_rectified_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return np.vectorize(
        lambda xi:
        (target_amp / amp) * (1. / (1. + np.exp(-slope * (max(min(xi, x[-1]), x[0]) - th))) - start_val) + ylim[0])


class BTSP(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]

    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)

        delta_weight = IS * ((self.w_max - self.projection.weight) * ET - \
                       self.projection.weight * self.dep_ratio * self.q_dep(ET))

        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer):
        """
        Update activity and dendritic state for populations in the layer that receive backward projections.
        :param layer:
        """
        # update activity
        for pop in layer:
            if pop.backward_projections:
                pop.delta_state = torch.zeros(pop.size)
                if hasattr(pop, 'dend_to_soma'):
                    pop.delta_state += pop.dend_to_soma
                for projection in pop.backward_projections:
                    if projection.compartment in [None, 'soma']:
                        pop.delta_state += projection(projection.pre.activity)
                pop.activity = pop.activation(pop.prev_state + pop.delta_state)

        # update dendritic state
        for pop in layer:
            count = 0
            for projection in pop.backward_projections:
                if projection.compartment == 'dend':
                    if count == 0:
                        pop.dendritic_state = torch.zeros(pop.size)
                        count += 1
                    pop.dendritic_state += projection(projection.pre.activity)
            if count > 0:
                pop.dendritic_state = torch.clamp(pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = next(iter(output_layer))
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.prev_state = output_pop.state.detach().clone()
        output_pop.dend_to_soma = torch.zeros(output_pop.size)
        for projection in output_pop:
            if projection.learning_rule_class == cls:
                output_pop.plateau = torch.zeros(output_pop.size)
                for i in range(output_pop.size):
                    if output_pop.dendritic_state[i] >  projection.learning_rule.pos_loss_th:
                        output_pop.plateau[i] = output_pop.dendritic_state[i]
                        output_pop.dend_to_soma[i] = output_pop.dendritic_state[i]
                        output_pop.activity[i] = \
                            output_pop.activation(output_pop.prev_state[i] + output_pop.dend_to_soma[i])
                    elif output_pop.dendritic_state[i] < projection.learning_rule.neg_loss_th:
                        output_pop.plateau[i] = output_pop.dendritic_state[i]
                break

        # initialize cells that receive backward projections
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections:
                    pop.prev_state = pop.state.detach().clone()
            cls.backward_update_layer_activity(layer)

        # update dendritic state variables
        for layer in reversed_layers:
            for pop in layer:
                for projection in pop.backward_projections:
                    if projection.learning_rule_class == cls:
                        pop.plateau = torch.zeros(pop.size)
                        pop.dend_to_soma = torch.zeros(pop.size)
                        # sort cells by dendritic state
                        pop_indexes = torch.argsort(pop.dendritic_state, descending=True)
                        for i in pop_indexes:
                            if pop.dendritic_state[i] > projection.learning_rule.pos_loss_th:
                                pop.plateau[i] = pop.dendritic_state[i]
                                pop.dend_to_soma[i] = pop.dendritic_state[i]
                                cls.backward_update_layer_activity(layer)
                            elif pop.dendritic_state[i] < projection.learning_rule.neg_loss_th:
                                pop.plateau[i] = pop.dendritic_state[i]
                        break


class DendriticLossBias(BiasLearningRule):
    def step(self):
        # self.population.bias.data += self.learning_rate * self.population.dendritic_state
        self.population.bias.data += self.learning_rate * self.population.plateau

    backward = BTSP.backward


class DendriticLoss(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        #self.projection.weight.data += self.sign * self.learning_rate * \
        #                               torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.plateau, self.projection.pre.activity)

    backward = BTSP.backward