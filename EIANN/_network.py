import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.nn.functional import softplus, relu
from torch.optim import Adam, SGD
import numpy as np

from .utils import half_kaining_init, scaled_kaining_init
import EIANN.rules as rules
import EIANN.external as external


class Network(nn.Module):
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
        if isinstance(criterion, str):
            if criterion in globals():
                criterion = globals()[criterion]
            elif hasattr(external, criterion):
                criterion = getattr(external, criterion)
        if not callable(criterion):
            raise RuntimeError('Network: criterion (%s) must be imported and callable' % criterion)
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
        self.parameter_list = nn.ParameterList()

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
                        projection = Projection(pre_pop, post_pop, **projection_kwargs)
                        post_pop.append_projection(projection)
                        if verbose:
                            print('Network: appending a projection from %s %s -> %s %s' %
                                  (pre_pop.layer.name, pre_pop.name, post_pop.layer.name, post_pop.name))

        if optimizer is not None:
            if isinstance(optimizer, str):
                if optimizer in globals():
                    optimizer = globals()[optimizer]
                elif hasattr(external, optimizer):
                    optimizer = getattr(external, optimizer)
            if not callable(optimizer):
                raise RuntimeError('Network: optimizer (%s) must be imported and callable' % optimizer)
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            optimizer = optimizer(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)
        self.optimizer = optimizer
        self.init_weights_and_biases()
        self.sample_order = []
        self.sorted_sample_indexes = []
        self.loss_history = []

    def init_weights_and_biases(self):
        for i, post_layer in enumerate(self):
            if i > 0:
                for post_pop in post_layer:
                    total_fan_in = 0
                    for projection in post_pop:
                        fan_in = projection.pre.size
                        total_fan_in += fan_in
                        if projection.weight_init is not None:
                            if projection.weight_init == 'half_kaining':
                                 half_kaining_init(projection.weight.data, fan_in, *projection.weight_init_args,
                                                   bounds=projection.weight_bounds)
                            elif projection.weight_init == 'scaled_kaining':
                                scaled_kaining_init(projection.weight.data, fan_in, *projection.weight_init_args)
                            else:
                                try:
                                    getattr(projection.weight.data,
                                            projection.weight_init)(*projection.weight_init_args)
                                except:
                                    raise RuntimeError('Network.init_weights_and_biases: callable for weight_init: %s '
                                                       'must be half_kaining, scaled_kaining, or a method of Tensor' %
                                                       projection.weight_init)
                    if post_pop.include_bias:
                        if post_pop.bias_init is None:
                            scaled_kaining_init(post_pop.bias.data, total_fan_in)
                        elif post_pop.bias_init == 'scaled_kaining':
                            scaled_kaining_init(post_pop.bias.data, total_fan_in, *post_pop.bias_init_args)
                        else:
                            try:
                                getattr(post_pop.bias.data, post_pop.bias_init)(*post_pop.bias_init_args)
                            except:
                                raise RuntimeError('Network.init_weights_and_biases: callable for bias_init: %s '
                                                   'must be scaled_kaining, or a method of Tensor' % post_pop.bias_init)

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
                        if projection.constrain_weight is not None:
                            projection.constrain_weight()

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
                input_pop.activity = torch.squeeze(sample)

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
                            delta_state = -post_pop.state + post_pop.bias
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

    def train(self, dataloader, epochs, store_history=False, status_bar=False):
        """

        :param dataloader: :class:'DataLoader';
            returns index (int), sample_data (tensor of float), and sample_target (tensor of float)
        :param epochs: int
        :param store_history: bool
        :param status_bar: bool
        """
        self.num_samples = len(dataloader)

        # Save weights & biases & activity
        if store_history:
            self.param_history = [self.state_dict()]

        if status_bar:
            from tqdm.autonotebook import tqdm

        if status_bar:
            epoch_iter = tqdm(range(epochs), desc='Epochs')
        else:
            epoch_iter = range(epochs)

        for epoch in epoch_iter:
            epoch_sample_order = []
            if status_bar:
                dataloader_iter = tqdm(dataloader, desc='Samples', leave=epoch == epochs - 1)
            else:
                dataloader_iter = dataloader

            for sample_idx, sample_data, sample_target in dataloader_iter:
                sample_data = torch.squeeze(sample_data)
                sample_target = torch.squeeze(sample_target)
                epoch_sample_order.append(sample_idx)
                output = self.forward(sample_data, store_history)

                loss = self.criterion(output, sample_target)
                self.loss_history.append(loss.detach())

                # Update state variables required for weight and bias updates
                for backward in self.backward_methods:
                    backward(self, output, sample_target)

                # Step weights and biases
                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.step()
                            for projection in post_pop:
                                projection.learning_rule.step()

                self.constrain_weights_and_biases()

                 # Save weights & biases & activity
                if store_history:
                    self.param_history.append(self.state_dict())

            epoch_sample_order = torch.concat(epoch_sample_order)
            self.sample_order.extend(epoch_sample_order)
            self.sorted_sample_indexes.extend(torch.add(epoch * self.num_samples, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.stack(self.loss_history)
        return loss.detach()

    def __iter__(self):
        for layer in self.layers.values():
            yield layer


class AttrDict(dict):
    '''
    Enables object attribute references for Layers, Populations, and Projections.
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

    def __repr__(self):
        ls = []
        for pop_name in self.populations.keys():
            ls.append(pop_name)
        items = ", ".join(ls)
        return f'{type(self)} :\n\t({items})'


class Population(object):
    def __init__(self, network, layer, name, size, activation, activation_kwargs=None, include_bias=False,
                 bias_init=None, bias_init_args=None, bias_bounds=None, bias_learning_rule=None,
                 bias_learning_rule_kwargs=None):
        """
        Class for population of neurons
        :param network: :class:'Network'
        :param layer: :class:'Layer'
        :param name: str
        :param size: int
        :param activation: str; name of imported callable
        :param activation_kwargs: dict
        :param include_bias: bool
        :param bias_init: str; name of imported callable
        :param bias_init_args: dict
        :param bias_bounds: tuple of float
        :param bias_learning_rule: str; name of imported callable
        :param bias_learning_rule_kwargs: dict
        """
        # Constants
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size

        # Set callable activation function
        if isinstance(activation, str):
            if activation in globals():
                activation = globals()[activation]
            elif hasattr(external, activation):
                activation = getattr(external, activation)
        if not callable(activation):
            raise RuntimeError \
                ('Population: callable for activation: %s must be imported' % activation)
        if activation_kwargs is None:
            activation_kwargs = {}
        self.activation = lambda x: activation(x, **activation_kwargs)

        # Set bias parameters
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.bias_init = bias_init
        if bias_init_args is None:
            bias_init_args = ()
        self.bias_init_args = bias_init_args
        self.bias_bounds = bias_bounds
        if bias_learning_rule_kwargs is None:
            bias_learning_rule_kwargs = {}
        if bias_learning_rule is None:
            bias_learning_rule_class = rules.BiasLearningRule
        else:
            include_bias = True
            if bias_learning_rule == 'Backprop':
                bias_learning_rule_class = rules.BackpropBias
                self.bias.requires_grad = True
            else:
                try:
                    if isinstance(bias_learning_rule, str):
                        if hasattr(rules, bias_learning_rule):
                            bias_learning_rule_class = getattr(rules, bias_learning_rule)
                        elif hasattr(external, bias_learning_rule):
                            bias_learning_rule_class = getattr(external, bias_learning_rule)
                    elif callable(bias_learning_rule):
                        bias_learning_rule_class = bias_learning_rule
                    if not issubclass(bias_learning_rule_class, rules.BiasLearningRule):
                        raise Exception
                except:
                    raise RuntimeError \
                        ('Population: bias_learning_rule: %s must be imported and subclass of BiasLearningRule' %
                         bias_learning_rule)

        self.include_bias = include_bias
        self.network.parameter_list.append(self.bias)
        self.bias_learning_rule = bias_learning_rule_class(self, **bias_learning_rule_kwargs)
        self.network.backward_methods.add(bias_learning_rule_class.backward)

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

    def append_projection(self, projection):
        """
        Register Projection parameters as Network module parameters. Enables convenient attribute access syntax.
        :param projection: :class:'Projection'
        """
        self.network.backward_methods.add(projection.learning_rule.__class__.backward)

        if projection.pre.layer.name not in self.projections:
            self.projections[projection.pre.layer.name] = {}
            self.__dict__[projection.pre.layer.name] = AttrDict()
        self.projections[projection.pre.layer.name][projection.pre.name] = projection
        self.__dict__[projection.pre.layer.name][projection.pre.name] = projection

        self.network.module_list.append(projection)

    def __iter__(self):
        for projections in self.projections.values():
            for projection in projections.values():
                yield projection

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


class Projection(nn.Linear):
    def __init__(self, pre_pop, post_pop, weight_init=None, weight_init_args=None, weight_constraint=None,
                 weight_constraint_kwargs=None, weight_bounds=None, direction='forward', compartment=None,
                 learning_rule='Backprop', learning_rule_kwargs=None, device=None, dtype=None):
        """

        :param pre_pop: :class:'Population'
        :param post_pop: :class:'Population'
        :param weight_init: str
        :param weight_init_args: tuple
        :param weight_constraint: str
        :param weight_constraint_kwargs: dict
        :param weight_bounds: tuple of float
        :param direction: str in ['forward', 'backward', 'recurrent', 'F', 'B', 'R']
        :param compartment: None or str in ['soma', 'dend']
        :param learning_rule: str
        :param learning_rule_kwargs: dict
        :param device:
        :param dtype:
        """
        super().__init__(pre_pop.size, post_pop.size, bias=False, device=device, dtype=dtype)

        self.pre = pre_pop
        self.post = post_pop

        self.weight_init = weight_init
        if weight_init_args is None:
            weight_init_args = ()
        self.weight_init_args = weight_init_args

        if weight_constraint is None:
            self.constrain_weight = None
        else:
            if isinstance(weight_constraint, str):
                if hasattr(rules, weight_constraint):
                    weight_constraint = getattr(rules, weight_constraint)
                elif hasattr(external, weight_constraint):
                    weight_constraint = getattr(external, weight_constraint)
            if not callable(weight_constraint):
                raise RuntimeError \
                    ('Projection: weight_constraint: %s must be imported and callable' %
                     weight_constraint)
            if weight_constraint_kwargs is None:
                weight_constraint_kwargs = {}
            self.constrain_weight = \
                lambda projection=self, kwargs=weight_constraint_kwargs: \
                    weight_constraint(projection, **weight_constraint_kwargs)

        self.weight_bounds = weight_bounds

        if direction not in ['forward', 'backward', 'recurrent', 'F', 'B', 'R']:
            raise RuntimeError('Projection: direction (%s) must be forward, backward, or recurrent' %
                               direction)
        self.direction = direction
        if self.direction in ['backward', 'B']:
            self.post.backward_projections.append(self)

        if compartment is not None and compartment not in ['soma', 'dend']:
            raise RuntimeError('Projection: compartment (%s) must be None, soma, or dend' %
                               compartment)
        self.compartment = compartment

        # Set learning rule as callable of the projection
        if learning_rule_kwargs is None:
            learning_rule_kwargs = {}
        if learning_rule is None:
            learning_rule_class = rules.LearningRule
        else:
            try:
                if isinstance(learning_rule, str):
                    if hasattr(rules, learning_rule):
                        learning_rule_class = getattr(rules, learning_rule)
                    elif hasattr(external, learning_rule):
                        learning_rule_class = getattr(external, learning_rule)
                elif callable(learning_rule):
                    learning_rule_class = learning_rule
                if not issubclass(learning_rule_class, rules.LearningRule):
                    raise Exception
            except:
                raise RuntimeError \
                    ('Projection: learning_rule: %s must be imported and instance of LearningRule' % learning_rule)
        self.learning_rule  = learning_rule_class(self, **learning_rule_kwargs)
        if learning_rule_class != rules.Backprop:
            self.weight.requires_grad = False