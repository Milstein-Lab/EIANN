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
from collections import defaultdict
from functools import partial

from EIANN.utils import half_kaiming_init, scaled_kaiming_init, linear, read_from_yaml
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
        self.device = torch.device(device)
        self.layer_config = layer_config
        self.projection_config = projection_config

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

        self.backward_methods = []
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
                            print(f'Network: appending a projection from {pre_pop.fullname} -> {post_pop.fullname}')

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
        self.reset_history()

    def init_weights_and_biases(self):
        for i, post_layer in enumerate(self):
            if i > 0:
                for post_pop in post_layer:
                    total_fan_in = 0
                    for projection in post_pop:
                        fan_in = projection.pre.size
                        total_fan_in += fan_in
                        if projection.weight_init is not None:
                            if projection.weight_init == 'half_kaiming':
                                 half_kaiming_init(projection.weight.data, fan_in, *projection.weight_init_args,
                                                   bounds=projection.weight_bounds)
                            elif projection.weight_init == 'scaled_kaiming':
                                scaled_kaiming_init(projection.weight.data, fan_in, *projection.weight_init_args)
                            else:
                                try:
                                    getattr(projection.weight.data,
                                            projection.weight_init)(*projection.weight_init_args)
                                except:
                                    raise RuntimeError('Network.init_weights_and_biases: callable for weight_init: %s '
                                                       'must be half_kaiming, scaled_kaiming, or a method of Tensor' %
                                                       projection.weight_init)
                    if post_pop.include_bias:
                        if post_pop.bias_init is None:
                            scaled_kaiming_init(post_pop.bias.data, total_fan_in)
                        elif post_pop.bias_init == 'scaled_kaiming':
                            scaled_kaiming_init(post_pop.bias.data, total_fan_in, *post_pop.bias_init_args)
                        else:
                            try:
                                getattr(post_pop.bias.data, post_pop.bias_init)(*post_pop.bias_init_args)
                            except:
                                raise RuntimeError('Network.init_weights_and_biases: callable for bias_init: %s '
                                                   'must be scaled_kaiming, or a method of Tensor' % post_pop.bias_init)

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
                        if (projection.constrain_weight is not None and
                                projection.weight_constraint_name != 'clone_weight'):
                            projection.constrain_weight()
        
        # After all constraints have been applied, clone weights
        for i, post_layer in enumerate(self):
            if i > 0:
                for post_pop in post_layer:
                    for projection in post_pop:
                        if (projection.constrain_weight is not None and
                                projection.weight_constraint_name == 'clone_weight'):
                            projection.constrain_weight()

    def reset_history(self):
        self.sample_order = []
        self.sorted_sample_indexes = []
        self.loss_history = []
        self.param_history = []
        self.param_history_steps = []
        self.prev_param_history = []
        self.target_history = []
        for layer in self:
            for population in layer:
                population.reinit(self.device)
                population.reset_history()

    def forward(self, sample, store_history=False, store_dynamics=False, no_grad=False):
        """

        :param sample: tensor
        :param store_history: bool
        :param store_dynamics: bool
        :param no_grad: bool
        :return: tensor
        """

        if len(sample.shape) > 1:
            batch_size = sample.shape[0]
        else:
            batch_size = 1
        for i, layer in enumerate(self):
            if i == 0:
                input_pop = next(iter(layer))
            for pop in layer:
                pop.reinit(self.device, batch_size=batch_size)
        input_pop.activity = torch.squeeze(sample)

        for t in range(self.forward_steps):
            if (t >= self.forward_steps - self.backward_steps) and not no_grad:
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
                        if store_dynamics:
                            post_pop.forward_steps_activity.append(post_pop.activity.detach().clone())

        if store_history:
            for layer in self:
                for pop in layer:
                    if store_dynamics:
                        pop.append_attribute_history('activity', pop.forward_steps_activity)
                    else:
                        pop.append_attribute_history('activity', pop.activity.detach().clone())

        return self.output_pop.activity

    def test(self, dataloader, store_history=False, store_dynamics=False, status_bar=False):
        """

        :param dataloader: :class:'DataLoader';
            returns index (int), sample_data (tensor of float), and sample_target (tensor of float)
        :param store_history: bool
        :param store_dynamics: bool
        :param status_bar: bool
        :return: float
        """
        if status_bar:
            from tqdm.autonotebook import tqdm

        if status_bar:
            dataloader_iter = tqdm(dataloader, desc='Samples')
        else:
            dataloader_iter = dataloader

        on_device = False

        for sample_idx, sample_data, sample_target in dataloader_iter:
            sample_data = torch.squeeze(sample_data)
            sample_target = torch.squeeze(sample_target)
            if not on_device:
                if sample_data.device == self.device:
                    on_device = True
                else:
                    sample_data = sample_data.to(self.device)
                    sample_target = sample_target.to(self.device)

            output = self.forward(sample_data, store_history=store_history, store_dynamics=store_dynamics, no_grad=True)
            loss = self.criterion(output, sample_target)

        return loss.item()

    def train(self, train_dataloader, val_dataloader=None, epochs=1, val_interval=(0, -1, 50), samples_per_epoch=None,
              store_history=False, store_dynamics=False, store_params=False, store_params_interval=None,
              save_to_file=None, status_bar=False):
        """
        Starting at validate_start, probe with the validate_data every validate_interval until >= validate_stop
        :param train_dataloader:
        :param val_dataloader:
        :param epochs:
        :param val_interval: tuple of int (start_index, stop_index, interval)
        :param samples_per_epoch: int
        :param store_history: bool
        :param store_dynamics: bool
        :param store_params: bool
        :param store_params_interval: tuple of int (start_index, stop_index, interval)
        :param save_to_file: None or file_path
        :param status_bar: bool
        """
        self.reset_history()
        if samples_per_epoch is None:
            samples_per_epoch = len(train_dataloader)

        train_step = 0
        # includes initial state before first train step
        train_step_range = torch.arange(epochs * samples_per_epoch + 1)

        val_data_on_device = False

        # Load validation data & initialize intermediate variables
        if val_dataloader is not None:
            assert len(val_dataloader) == 1, 'Validation Dataloader must have a single large batch'
            idx, val_data, val_target = next(iter(val_dataloader))
            if not val_data_on_device:
                if val_data.device == self.device:
                    val_data_on_device = True
                else:
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
                output = self.forward(val_data, store_dynamics=False, no_grad=True)
                val_output_history.append(output.detach().clone())
                val_loss_history.append(self.criterion(output, val_target).item())
                accuracy = 100 * torch.sum(torch.argmax(output, dim=1) == torch.argmax(val_target, dim=1)) / \
                           output.shape[0]
                val_accuracy_history.append(accuracy.item())
                self.val_history_train_steps.append(train_step-1)

        # Store history of weights and biases
        if store_params:
            self.param_history = []
            self.param_history_steps = []
            self.prev_param_history = []
            if store_params_interval is None:
                store_params_interval = val_interval
                store_params_range = val_range
            else:
                store_params_range = torch.arange(train_step_range[store_params_interval[0]],
                                                   train_step_range[store_params_interval[1]] + 1,
                                                   store_params_interval[2])
            if store_params_interval[2] == 1:
                self.param_history.append(deepcopy(self.state_dict()))
                self.param_history_steps.append(train_step-1)

        if status_bar:
            from tqdm.autonotebook import tqdm
        if status_bar:
            epoch_iter = tqdm(range(epochs), desc='Epochs')
        else:
            epoch_iter = range(epochs)

        self.target_history = []

        # initialize learning rule parameters
        for post_layer in self:
            for post_pop in post_layer:
                if post_pop.include_bias:
                    post_pop.bias_learning_rule.reinit()
                for projection in post_pop:
                    projection.learning_rule.reinit()

        train_data_on_device = False

        for epoch in epoch_iter:
            epoch_sample_order = []
            if status_bar and len(train_dataloader) > epochs:
                dataloader_iter = tqdm(train_dataloader, desc='Samples', total=samples_per_epoch,
                                       leave=epoch == epochs - 1)
            else:
                dataloader_iter = train_dataloader

            for sample_count, (sample_idx, sample_data, sample_target) in enumerate(dataloader_iter):
                if sample_count >= samples_per_epoch:
                    break
                sample_data = torch.squeeze(sample_data)
                sample_target = torch.squeeze(sample_target)
                if not train_data_on_device:
                    if sample_data.device == self.device:
                        train_data_on_device = True
                    else:
                        sample_data = sample_data.to(self.device)
                        sample_target = sample_target.to(self.device)
                epoch_sample_order.append(sample_idx)

                output = self.forward(sample_data, store_history=store_history, store_dynamics=store_dynamics)

                loss = self.criterion(output, sample_target)
                self.loss_history.append(loss.item())
                self.target_history.append(sample_target.clone())

                if store_params and (train_step in store_params_range) and store_params_interval[2] > 1: 
                    self.prev_param_history.append(deepcopy(self.state_dict())) # Store parameters for dW comparison
                        
                # Update state variables required for weight and bias updates
                for backward in self.backward_methods:
                    backward(self, output, sample_target, store_history=store_history, store_dynamics=store_dynamics)

                # Step weights and biases
                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.step()
                            for projection in post_pop:
                                projection.learning_rule.step()

                self.constrain_weights_and_biases()

                # update learning rule parameters
                for i, post_layer in enumerate(self):
                    if i > 0:
                        for post_pop in post_layer:
                            if post_pop.include_bias:
                                post_pop.bias_learning_rule.update()
                            for projection in post_pop:
                                projection.learning_rule.update()

                # Store history of weights and biases
                if store_params and train_step in store_params_range:
                    self.param_history.append(deepcopy(self.state_dict()))
                    self.param_history_steps.append(train_step)

                # Compute validation loss
                if val_dataloader is not None and train_step in val_range:
                    output = self.forward(val_data, store_dynamics=False, no_grad=True)
                    val_output_history.append(output.detach().clone())
                    val_loss_history.append(self.criterion(output, val_target).item())
                    accuracy = 100 * torch.sum(torch.argmax(output, dim=1) == torch.argmax(val_target, dim=1)) / \
                               output.shape[0]
                    val_accuracy_history.append(accuracy.item())
                    self.val_history_train_steps.append(train_step)

                train_step += 1

            epoch_sample_order = torch.concat(epoch_sample_order)
            self.sample_order.extend(epoch_sample_order)
            self.sorted_sample_indexes.extend(torch.add(epoch * samples_per_epoch, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.tensor(self.loss_history).cpu()
        self.target_history = torch.stack(self.target_history).cpu()
        if val_dataloader is not None:
            self.val_output_history = torch.stack(val_output_history).cpu()
            self.val_loss_history = torch.tensor(val_loss_history).cpu()
            self.val_accuracy_history = torch.tensor(val_accuracy_history).cpu()
            self.val_target = val_target.cpu()

        if save_to_file is not None:
            self.save(path=save_to_file)

    def save(self, path=None, dir='saved_networks', file_name_base=None, disp=True):
        """

        :param path: str (path to file)
        :param dir: str (path to dir)
        :param file_name_base: str
        :param disp: str
        """
        if path is None:
            if file_name_base is None:
                file_name_base = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = '%s/%s.pkl' % (dir, file_name_base)
            if not os.path.exists(dir):
                os.makedirs(dir)

        # if os.path.exists(path):
        #     overwrite = input('File already exists. Overwrite? (y/n)')
        #     if overwrite == 'y':
        #         os.remove(path)
        #     else:
        #         print('Model not saved')
        #         return

        self.params_to_save.extend(['param_history', 'param_history_steps', 'prev_param_history', 'sample_order',
                                    'target_history', 'sorted_sample_indexes', 'loss_history', 'val_output_history',
                                    'val_loss_history', 'val_history_train_steps', 'val_accuracy_history',
                                    'val_target', 'attribute_history_dict'])
        
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
        if disp:
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


class AttrDict:
    def __iter__(self):
        for key in self.__dict__:
            yield key


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
        self.bias = nn.Parameter(torch.zeros(self.size, device=network.device), requires_grad=False)
        self.bias_init = bias_init
        if bias_init_args is None:
            bias_init_args = ()
        self.bias_init_args = bias_init_args
        self.bias_bounds = bias_bounds
        if bias_learning_rule_kwargs is None:
            bias_learning_rule_kwargs = {}
        if bias_learning_rule is None:
            bias_learning_rule_class = rules.BiasLearningRule
            self.bias.is_learned = False
        else:
            self.bias.is_learned = True
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
        if bias_learning_rule_class.backward not in self.network.backward_methods:
            self.network.backward_methods.append(bias_learning_rule_class.backward)

        self.network.parameter_dict[self.fullname+'_bias'] = self.bias
        self.network.optimizer_params_list.append({'params': self.bias,
                                                   'lr':self.bias_learning_rule.learning_rate})

        # Initialize storage containers
        self.projections = {}
        self.backward_projections = []
        self.outgoing_projections = {}
        self.incoming_projections = {}
        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))
        self.reinit(network.device)
        self.reset_history()

    def append_attribute_history(self, attr_name, vals):
        self.attribute_history_dict[attr_name]['buffer'].append(vals)

    def get_attribute_history(self, attr_name):
        if self.attribute_history_dict[attr_name]['buffer']:
            if isinstance(self.attribute_history_dict[attr_name]['buffer'][0], torch.Tensor):
                temp_history = torch.stack(self.attribute_history_dict[attr_name]['buffer'])
            elif isinstance(self.attribute_history_dict[attr_name]['buffer'][0], list):
                temp_history = \
                    torch.stack(
                        [torch.stack(val_list) for val_list in self.attribute_history_dict[attr_name]['buffer']])
            self.attribute_history_dict[attr_name]['buffer'] = []
        else:
            return self.attribute_history_dict[attr_name]['history']
        if self.attribute_history_dict[attr_name]['history'] is None:
            self.attribute_history_dict[attr_name]['history'] = temp_history
        else:
            self.attribute_history_dict[attr_name]['history'] = torch.cat(
                [self.attribute_history_dict[attr_name]['history'], temp_history])

        return self.attribute_history_dict[attr_name]['history']
    
    def get_param_history(self, param_name):
        cached_param_history = self.get_attribute_history(param_name)
        if len(self.network.param_history) == 0:
            return cached_param_history
        if cached_param_history is None:
            start = 0
        elif len(cached_param_history) == len(self.network.param_history):
            return cached_param_history
        else:
            start = len(cached_param_history)
        for t in range(start, len(self.network.param_history)):
            this_param = self.network.param_history[t][f'parameter_dict.{self.fullname}_{param_name}']
            self.append_attribute_history(param_name, this_param)
        return self.get_attribute_history(param_name)
    
    def reinit(self, device, batch_size=1):
        """
        Method for resetting state variables of a population
        :param device:
        """
        if batch_size > 1:
            self.activity = torch.zeros((batch_size, self.size), device=device)
            self.state = torch.zeros((batch_size, self.size), device=device)
        else:
            self.activity = torch.zeros(self.size, device=device)
            self.state = torch.zeros(self.size, device=device)
        self.forward_steps_activity = []

    def reset_history(self):
        """

        """
        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))
    
    def append_projection(self, projection):
        """
        Register Projection parameters as Network module parameters. Enables convenient attribute access syntax.
        :param projection: :class:'Projection'
        """
        if projection.learning_rule.__class__.backward not in self.network.backward_methods:
            self.network.backward_methods.append(projection.learning_rule.__class__.backward)

        if projection.pre.layer.name not in self.projections:
            self.projections[projection.pre.layer.name] = {}
            self.__dict__[projection.pre.layer.name] = AttrDict()
        self.projections[projection.pre.layer.name][projection.pre.name] = projection
        self.__dict__[projection.pre.layer.name].__dict__[projection.pre.name] = projection

        self.network.module_dict[projection.name] = projection
        self.network.optimizer_params_list.append({'params': projection.weight,
                                                   'lr':projection.learning_rule.learning_rate})

    def __iter__(self):
        for projections in self.projections.values():
            for projection in projections.values():
                yield projection

    @property
    def activity_history(self):
        return self.get_attribute_history('activity')

    @property
    def bias_history(self):
        return self.get_param_history('bias')


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
        self.include_bias = False
        self.reinit(network.device)
        self.reset_history()

    def reset_history(self):
        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))


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
                self.weight_constraint_name = weight_constraint
                if hasattr(rules, weight_constraint):
                    weight_constraint = getattr(rules, weight_constraint)
                elif hasattr(external, weight_constraint):
                    weight_constraint = getattr(external, weight_constraint)
            if not callable(weight_constraint):
                raise RuntimeError \
                    ('Projection: weight_constraint: %s must be imported and callable' %
                     weight_constraint)
            else:
                self.weight_constraint_name = weight_constraint.__name__
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
            self.weight.is_learned = False
        else:
            self.weight.is_learned = True
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

        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))

    def append_attribute_history(self, attr_name, vals):
        self.attribute_history_dict[attr_name]['buffer'].append(vals)

    def get_attribute_history(self, attr_name):
        if self.attribute_history_dict[attr_name]['buffer']:
            if isinstance(self.attribute_history_dict[attr_name]['buffer'][0], torch.Tensor):
                temp_history = torch.stack(self.attribute_history_dict[attr_name]['buffer'])
            elif isinstance(self.attribute_history_dict[attr_name]['buffer'][0], list):
                temp_history = \
                    torch.stack(
                        [torch.stack(val_list) for val_list in self.attribute_history_dict[attr_name]['buffer']])
            self.attribute_history_dict[attr_name]['buffer'] = []
        else:
            return self.attribute_history_dict[attr_name]['history']
        if self.attribute_history_dict[attr_name]['history'] is None:
            self.attribute_history_dict[attr_name]['history'] = temp_history
        else:
            self.attribute_history_dict[attr_name]['history'] = torch.cat(
                [self.attribute_history_dict[attr_name]['history'], temp_history])

        return self.attribute_history_dict[attr_name]['history']

    def get_weight_history(self):
        cached_weight_history = self.get_attribute_history('weight')
        if len(self.post.network.param_history) == 0:
            return cached_weight_history
        if cached_weight_history is None:
            start = 0
        elif len(cached_weight_history) == len(self.post.network.param_history):
            return cached_weight_history
        else:
            start = len(cached_weight_history)
        for t in range(start, len(self.post.network.param_history)):
            this_weight = self.post.network.param_history[t][f'module_dict.{self.name}.weight']
            self.append_attribute_history('weight', this_weight)
        return self.get_attribute_history('weight')

    @property
    def weight_history(self):
        return self.get_weight_history()
    


def build_EIANN_from_config(config_path, network_seed=42):
    '''
    Build an EIANN network from a config file
    '''
    network_config = read_from_yaml(config_path)
    layer_config = network_config['layer_config']
    projection_config = network_config['projection_config']
    training_kwargs = network_config['training_kwargs']
    network = Network(layer_config, projection_config, seed=network_seed, **training_kwargs)
    return network
