import torch
import torch.nn as nn
from torch.nn import MSELoss, BCELoss
from torch.nn.functional import softplus, relu, sigmoid, elu
from torch.optim import Adam, SGD
import sys
import os
import shutil
import pickle
import datetime
from copy import deepcopy
import time

from EIANN.utils import half_kaining_init, scaled_kaining_init, linear
import EIANN.rules as rules
import EIANN.external as external


class Network(nn.Module):
    def __init__(self, layer_config, projection_config, learning_rate, optimizer=SGD, optimizer_kwargs=None,
                 criterion=MSELoss, criterion_kwargs=None, seed=None, device='cpu', tau=1, forward_steps=1,
                 backward_steps=1, verbose=False):
        """

        :param layer_config: nested dict
        :param projection_config: nested dict
        :param learning_rate: float; applies to weights and biases in absence of projection-specific learning rates
        :param optimizer: callable
        :param optimizer_kwargs: dict
        :param criterion: callable
        :param criterion_kwargs: dict
        :param seed: int or sequence of int
        :param device: str
        :param tau: int
        :param forward_steps: int
        :param backward_steps: int
        :param verbose: bool
        """
        super().__init__()
        self.device = device

        # Load loss criterion
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

        # Load other learning hyperparameters
        self.learning_rate = learning_rate
        self.tau = tau
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.params_to_save = []

        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.backward_methods = set()
        self.module_dict = nn.ModuleDict()
        self.parameter_dict = nn.ParameterDict()
        self.optimizer_params_list = []

        # Build network populations
        self.output_pop = None
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

        # if no output_pop is designated, default to the first population specified in the final layer
        if self.output_pop is None:
            output_layer = list(self)[-1]
            self.output_pop = next(iter(output_layer))

        # Build network projections
        for post_layer_name in projection_config:
            post_layer = self.layers[post_layer_name]
            for post_pop_name in projection_config[post_layer_name]:
                post_pop = post_layer.populations[post_pop_name]
                for pre_layer_name in projection_config[post_layer_name][post_pop_name]:
                    pre_layer = self.layers[pre_layer_name]
                    for pre_pop_name, projection_kwargs in \
                            projection_config[post_layer_name][post_pop_name][pre_layer_name].items():
                        pre_pop = pre_layer.populations[pre_pop_name]
                        projection = Projection(pre_pop, post_pop, device=device, **projection_kwargs)
                        post_pop.append_projection(projection)
                        post_pop.incoming_projections[projection.name] = projection
                        pre_pop.outgoing_projections[projection.name] = projection
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
            # optimizer = optimizer(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)
            optimizer = optimizer(self.optimizer_params_list, **optimizer_kwargs)

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
                population.reinit(self.device)
                population.reset_history()

    def forward(self, sample, store_history=False):

        for i, layer in enumerate(self):
            for pop in layer:
                pop.reinit(self.device)
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
                                if projection.update_phase in ['forward', 'all', 'F', 'A']:
                                    if projection.direction in ['forward', 'F']:
                                        delta_state = delta_state + projection(pre_pop.activity)
                                    elif projection.direction in ['recurrent', 'R']:
                                        delta_state = delta_state + projection(pre_pop.prev_activity)
                            post_pop.state = post_pop.state + delta_state / post_pop.tau
                            post_pop.activity = post_pop.activation(post_pop.state)
                        post_pop.forward_steps_activity.append(post_pop.activity.detach().clone())

        if store_history:
            for layer in self:
                for pop in layer:
                    pop.activity_history_list.append(pop.forward_steps_activity)

        return self.output_pop.activity

    def test(self, dataloader, store_history=False, status_bar=False):
        """

        :param dataloader: :class:'DataLoader';
            returns index (int), sample_data (tensor of float), and sample_target (tensor of float)
        :param store_history: bool
        :param status_bar: bool
        """
        num_samples = len(dataloader)

        if status_bar:
            from tqdm.autonotebook import tqdm

        epoch = 0
        epoch_sample_order = []
        if status_bar:
            dataloader_iter = tqdm(dataloader, desc='Samples')
        else:
            dataloader_iter = dataloader
        for sample_idx, sample_data, sample_target in dataloader_iter:
            sample_data = torch.squeeze(sample_data)
            sample_target = torch.squeeze(sample_target)
            epoch_sample_order.append(sample_idx)
            output = self.forward(sample_data, store_history)

            loss = self.criterion(output, sample_target)
            self.loss_history.append(loss.detach())

        epoch_sample_order = torch.concat(epoch_sample_order)
        self.sample_order.extend(epoch_sample_order)
        self.sorted_sample_indexes.extend(torch.add(epoch * num_samples, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.stack(self.loss_history)

        return loss.detach()

    def train(self, dataloader, epochs, store_history=False, store_weights=False, status_bar=False):
        """

        :param dataloader: :class:'DataLoader';
            returns index (int), sample_data (tensor of float), and sample_target (tensor of float)
        :param epochs: int
        :param store_history: bool
        :param store_weights: bool
        :param status_bar: bool
        """
        num_samples = len(dataloader)

        # Save weights & biases & activity
        if store_weights:
            self.param_history = [deepcopy(self.state_dict())]

        if status_bar:
            from tqdm.autonotebook import tqdm

        if status_bar:
            epoch_iter = tqdm(range(epochs), desc='Epochs')
        else:
            epoch_iter = range(epochs)

        self.target_history = []

        for epoch in epoch_iter:
            epoch_sample_order = []
            if status_bar and len(dataloader) > epochs:
                dataloader_iter = tqdm(dataloader, desc='Samples', leave=epoch == epochs - 1)
            else:
                dataloader_iter = dataloader

            for sample_idx, sample_data, sample_target in dataloader_iter:
                sample_data = torch.squeeze(sample_data).to(self.device)
                sample_target = torch.squeeze(sample_target).to(self.device)
                epoch_sample_order.append(sample_idx)
                output = self.forward(sample_data, store_history)

                loss = self.criterion(output, sample_target)
                self.loss_history.append(loss.detach())
                self.target_history.append(sample_target)

                # Update state variables required for weight and bias updates
                for backward in self.backward_methods:
                    backward(self, output, sample_target, store_history)

                # Step weights and biases
                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.step()
                            for projection in post_pop:
                                projection.learning_rule.step()

                self.constrain_weights_and_biases()

                # Store history of weights and biases
                if store_weights:
                    self.param_history.append(deepcopy(self.state_dict()))

            epoch_sample_order = torch.concat(epoch_sample_order)
            self.sample_order.extend(epoch_sample_order)
            self.sorted_sample_indexes.extend(torch.add(epoch * num_samples, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.stack(self.loss_history)

        return loss.detach()

    def train_and_validate(self, train_dataloader, val_dataloader, epochs, val_interval=(0, -1, 50),
                           store_history=False, store_weights=False, store_weights_interval=None,
                           save_to_file=None, status_bar=False):
        """
        Starting at validate_start, probe with the validate_data every validate_interval until >= validate_stop
        :param train_dataloader:
        :param val_dataloader:
        :param epochs:
        :param val_interval: tuple of int (start_index, stop_index, interval)
        :param store_history: bool
        :param store_weights: bool
        :param store_weights_interval: tuple of int (start_index, stop_index, interval)
        :param status_bar: bool
        :return:
        """
        num_samples = len(train_dataloader)

        train_step = 0
        # includes initial state before first train step
        train_step_range = torch.arange(epochs * num_samples + 1)

        # Load validation data & initialize intermediate variables
        assert len(val_dataloader) == 1, 'Validation Dataloader must have a single large batch'
        idx, val_data, val_target = next(iter(val_dataloader))
        val_data = val_data.to(self.device)
        val_target = val_target.to(self.device)
        val_output_history = []
        val_loss_history = []
        val_accuracy_history = []
        self.val_history_train_steps = []

        val_range = torch.arange(train_step_range[val_interval[0]], train_step_range[val_interval[1]] + 1,
                                 val_interval[2])

        # Compute validation loss
        if train_step in val_range:
            output = self.forward(val_data).detach()
            val_output_history.append(output)
            val_loss_history.append(self.criterion(output, val_target).detach())
            accuracy = 100 * torch.sum(torch.argmax(output, dim=1) == torch.argmax(val_target, dim=1)) / output.shape[0]
            val_accuracy_history.append(accuracy)
            self.val_history_train_steps.append(train_step)

        # Store history of weights and biases
        if store_weights:
            self.param_history = []
            self.param_history_steps = []
            if store_weights_interval is None:
                store_weights_range = val_range
            else:
                store_weights_range = torch.arange(train_step_range[store_weights_interval[0]],
                                                   train_step_range[store_weights_interval[1]] + 1,
                                                   store_weights_interval[2])
            if train_step in store_weights_range:
                self.param_history.append(deepcopy(self.state_dict()))
                self.param_history_steps.append(train_step)

        if status_bar:
            from tqdm.autonotebook import tqdm

        if status_bar:
            epoch_iter = tqdm(range(epochs), desc='Epochs')
        else:
            epoch_iter = range(epochs)

        self.target_history = []

        for epoch in epoch_iter:
            epoch_sample_order = []
            if status_bar and len(train_dataloader) > epochs:
                dataloader_iter = tqdm(train_dataloader, desc='Samples', leave=epoch == epochs - 1)
            else:
                dataloader_iter = train_dataloader

            for sample_idx, sample_data, sample_target in dataloader_iter:
                train_step += 1

                sample_data = torch.squeeze(sample_data).to(self.device)
                sample_target = torch.squeeze(sample_target).to(self.device)
                epoch_sample_order.append(sample_idx)
                output = self.forward(sample_data, store_history)

                loss = self.criterion(output, sample_target)
                self.loss_history.append(loss.detach())
                self.target_history.append(sample_target)

                # Update state variables required for weight and bias updates
                for backward in self.backward_methods:
                    backward(self, output, sample_target, store_history)

                # Step weights and biases
                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.step()
                            for projection in post_pop:
                                projection.learning_rule.step()

                self.constrain_weights_and_biases()

                # Store history of weights and biases
                if store_weights and train_step in store_weights_range:
                    self.param_history.append(deepcopy(self.state_dict()))
                    self.param_history_steps.append(train_step)

                # Compute validation loss
                if train_step in val_range:
                    output = self.forward(val_data).detach()
                    val_output_history.append(output)
                    val_loss_history.append(self.criterion(output, val_target).detach())
                    accuracy = 100 * torch.sum(torch.argmax(output, dim=1) == torch.argmax(val_target, dim=1)) / \
                               output.shape[0]
                    val_accuracy_history.append(accuracy)
                    self.val_history_train_steps.append(train_step)

            epoch_sample_order = torch.concat(epoch_sample_order)
            self.sample_order.extend(epoch_sample_order)
            self.sorted_sample_indexes.extend(torch.add(epoch * num_samples, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.stack(self.loss_history).cpu()
        self.val_output_history = torch.stack(val_output_history).cpu()
        self.val_loss_history = torch.stack(val_loss_history).cpu()
        self.val_accuracy_history = torch.stack(val_accuracy_history).cpu()
        self.val_target = val_target.cpu()

        if save_to_file is not None:
            self.save(path=save_to_file)

    def save(self, path=None, dir="saved_networks/", filename="datetime.datetime.now().strftime('%Y%m%d_%H%M%S')"):

        if path is None:
            path = dir + filename + ".pickle"
            if not os.path.exists(dir):
                os.makedirs(dir)
        else:
            path = path + ".pickle"

        if os.path.exists(path):
            overwrite = input('File already exists. Overwrite? (y/n)')
            if overwrite == 'y':
                # shutil.rmtree(path+filename)
                os.remove(path)
            else:
                print('Model not saved')
                return

        self.params_to_save.extend(['param_history', 'param_history_steps', 'sample_order', 'target_history',
                                    'sorted_sample_indexes', 'loss_history', 'val_output_history', 'val_loss_history',
                                    'val_accuracy_history', 'val_target', 'activity_history_list', '_activity_history',
                                    '_backward_activity_history', 'bias_history_list', '_bias_history',
                                    '_plateau_history', 'plateau_history_list'])

        data_dict = {'network': {param_name: value for param_name, value in self.__dict__.items()
                                 if param_name in self.params_to_save},
                     'layers': {},
                     'populations': {},
                     'final_state_dict': self.state_dict()}

        for layer in self:
            layer_data = {param_name: value for param_name, value in layer.__dict__.items()
                          if param_name in self.params_to_save}
            data_dict['layers'][layer.name] = layer_data

            for population in layer:
                population_data = {param_name: value for param_name, value in population.__dict__.items()
                                   if param_name in self.params_to_save}
                data_dict['populations'][population.fullname] = population_data

        with open(path, 'wb') as file:
            pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Model saved to {path}')

    def load(self, filepath):
        print(f"Loading model data from '{filepath}'...")
        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)

        print('Loading parameters into the network...')
        self.__dict__.update(data_dict['network'])

        for layer in self:
            layer_data = data_dict['layers'][layer.name]
            layer.__dict__.update(layer_data)

            for population in layer:
                population_data = data_dict['populations'][population.fullname]
                population.__dict__.update(population_data)

        self.load_state_dict(data_dict['final_state_dict'])

        print(f"Model successfully loaded from '{filepath}'")

    def __iter__(self):
        for layer in self.layers.values():
            yield layer


class AttrDict(dict):
    """
    Enables Layers, Populations, and Projections to be accessed as attributes of the network object.
    """
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
    def __init__(self, network, layer, name, size, activation, activation_kwargs=None, tau=None,
                 include_bias=False, bias_init=None, bias_init_args=None, bias_bounds=None,
                 bias_learning_rule=None, bias_learning_rule_kwargs=None, output_pop=False):
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
        :param output_pop: bool; a single population must be designated as the output population to compute network loss
        """
        # Constants
        self.network = network
        self.layer = layer
        self.name = name
        self.size = size
        self.fullname = layer.name+self.name
        if tau is None:
            self.tau = network.tau
        else:
            self.tau = tau

        if output_pop:
            self.network.output_pop = self

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
        self.bias = self.bias.to(network.device)
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

        self.bias_learning_rule = bias_learning_rule_class(self, **bias_learning_rule_kwargs)
        self.network.backward_methods.add(bias_learning_rule_class.backward)

        self.network.parameter_dict[self.fullname+'_bias'] = self.bias
        self.network.optimizer_params_list.append({'params': self.bias,
                                                   'lr':self.bias_learning_rule.learning_rate})

        # Initialize storage containers
        self.projections = {}
        self.backward_projections = []
        self.outgoing_projections = {}
        self.incoming_projections = {}
        self.reinit(network.device)
        self.reset_history()

    def reinit(self, device):
        """
        Method for resetting state variables of a population
        :param device:
        """
        self.activity = torch.zeros(self.size, device=device)
        self.state = torch.zeros(self.size, device=device)
        self.forward_steps_activity = []

    def reset_history(self):
        """

        """
        self.activity_history_list = []
        self._activity_history = None
        self.backward_activity_history_list = []
        self._backward_activity_history = None
        self.plateau_history_list = []
        self._plateau_history = None
        self.nudge_history_list = []
        self._nudge_history = None

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

        self.network.module_dict[projection.name] = projection
        self.network.optimizer_params_list.append({'params': projection.weight,
                                                   'lr':projection.learning_rule.learning_rate})

    def __iter__(self):
        for projections in self.projections.values():
            for projection in projections.values():
                yield projection

    @property
    def activity_history(self):
        if self._activity_history is None:
            if self.activity_history_list:
                self._activity_history = \
                    torch.stack([torch.stack(forward_steps_activity)
                                 for forward_steps_activity in self.activity_history_list])
                self.activity_history_list = []
        else:
            if self.activity_history_list:
                self._activity_history = \
                    torch.cat([self._activity_history,
                               torch.stack([torch.stack(forward_steps_activity)
                                            for forward_steps_activity in self.activity_history_list])])
                self.activity_history_list = []

        return self._activity_history

    @property
    def backward_activity_history(self):
        if not hasattr(self, '_backward_activity_history'):
            return None
        if self._backward_activity_history is None:
            if self.backward_activity_history_list:
                self._backward_activity_history = \
                    torch.stack([torch.stack(backward_steps_activity)
                                 for backward_steps_activity in self.backward_activity_history_list])
                self.backward_activity_history_list = []
        else:
            if self.backward_activity_history_list:
                self._backward_activity_history = \
                    torch.cat([self._backward_activity_history,
                               torch.stack([torch.stack(backward_steps_activity)
                                            for backward_steps_activity in self.backward_activity_history_list])])
                self.backward_activity_history_list = []

        return self._backward_activity_history

    @property
    def plateau_history(self):
        if not hasattr(self, '_plateau_history'):
            return None
        if self._plateau_history is None:
            if self.plateau_history_list:
                self._plateau_history = torch.stack(self.plateau_history_list)
                self.plateau_history_list = []
        else:
            if self.plateau_history_list:
                self._plateau_history = torch.cat([self._plateau_history, torch.stack(self.plateau_history_list)])
                self.plateau_history_list = []

        return self._plateau_history

    @property
    def nudge_history(self):
        if not hasattr(self, '_nudge_history'):
            return None
        if self._nudge_history is None:
            if self.nudge_history_list:
                self._nudge_history = torch.stack(self.nudge_history_list)
                self.nudge_history_list = []
        else:
            if self.nudge_history_list:
                self._nudge_history = torch.cat([self._nudge_history, torch.stack(self.nudge_history_list)])
                self.nudge_history_list = []

        return self._nudge_history

    @property
    def bias_history(self):
        """
        TODO: cache the bias_history so this doesn't have to rebuilt from scratch at each call
        :return:
        """
        param_history = self.network.param_history
        _bias_history = [param_history[t][f'parameter_dict.{self.fullname}_bias'] for t in range(len(param_history))]
        return torch.stack(_bias_history)


class Input(Population):
    def __init__(self, network, layer, name, size, *args, **kwargs):
        self.network = network
        self.layer = layer
        self.name = name
        self.fullname = layer.name+self.name
        self.size = size
        self.projections = {}
        self.backward_projections = []
        self.outgoing_projections = {}
        self.incoming_projections = {}
        self.reinit(network.device)
        self.reset_history()

    def reset_history(self):
        self.activity_history_list = []
        self._activity_history = None
        self.backward_activity_history_list = []
        self._backward_activity_history = None


class Projection(nn.Linear):
    def __init__(self, pre_pop, post_pop, weight_init=None, weight_init_args=None, weight_constraint=None,
                 weight_constraint_kwargs=None, weight_bounds=None, direction='forward', update_phase='forward',
                 compartment=None, learning_rule='Backprop', learning_rule_kwargs=None, device=None, dtype=None):
        """

        :param pre_pop: :class:'Population'
        :param post_pop: :class:'Population'
        :param weight_init: str
        :param weight_init_args: tuple
        :param weight_constraint: str
        :param weight_constraint_kwargs: dict
        :param weight_bounds: tuple of float
        :param direction: str in ['forward', 'recurrent', 'F', 'R']
        :param update_phase: str in ['forward', 'backward', 'F', B']
        :param compartment: None or str in ['soma', 'dend']
        :param learning_rule: str
        :param learning_rule_kwargs: dict
        :param device:
        :param dtype:
        """
        super().__init__(pre_pop.size, post_pop.size, bias=False, device=device, dtype=dtype)

        self.pre = pre_pop
        self.post = post_pop
        self.name = f'{post_pop.layer.name}{post_pop.name}_{pre_pop.layer.name}{pre_pop.name}'

        # Initialize weight parameters
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

        if direction not in ['forward', 'recurrent', 'F', 'R']:
            raise RuntimeError('Projection: direction (%s) must be forward or recurrent' % direction)
        self.direction = direction
        if update_phase not in ['forward', 'backward', 'all', 'F', 'B', 'A']:
            raise RuntimeError('Projection: update_phase (%s) must be forward, backward, or all' % update_phase)
        if update_phase in ['backward', 'B', 'all', 'A']:
            self.post.backward_projections.append(self)
        self.update_phase = update_phase

        if compartment is not None and compartment not in ['soma', 'dend']:
            raise RuntimeError('Projection: compartment (%s) must be None, soma, or dend' % compartment)
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

    @property
    def weight_history(self):
        param_history = self.post.network.param_history
        _weight_history = [param_history[t][f'module_dict.{self.name}.weight'] for t in range(len(param_history))]
        return torch.stack(_weight_history)