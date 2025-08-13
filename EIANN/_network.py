import torch
import torch.nn as nn
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
import numpy as np
from copy import deepcopy
from collections import defaultdict
from functools import partial
from typing import Optional, Dict, Any, Union, Tuple, List

import EIANN.utils as ut
import EIANN.rules as rules
import EIANN.external as external


class Network(nn.Module):
    def __init__(self, layer_config, projection_config, learning_rate=None, optimizer=SGD, optimizer_kwargs=None,
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
        self.training_kwargs = {'learning_rate': learning_rate, 'optimizer': optimizer,
                                'optimizer_kwargs': optimizer_kwargs, 'criterion': criterion,
                                'criterion_kwargs': criterion_kwargs, 'device': device, 'tau': tau,
                                'forward_steps': forward_steps, 'backward_steps': backward_steps}

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

        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.backward_methods = []
        self.module_dict = nn.ModuleDict()
        self.parameter_dict = nn.ParameterDict()
        self.optimizer_params_list = []
        self.populations = {}
        self.projections = {}

        # Build network populations
        self.layers = {}
        for i, (layer_name, pop_config) in enumerate(layer_config.items()):
            layer = Layer(self, layer_name)
            self.layers[layer_name] = layer
            self.__dict__[layer_name] = layer
            for j, (pop_name, pop_kwargs) in enumerate(pop_config.items()):
                if i == 0 and j == 0:
                    pop = Input(self, layer, pop_name, **pop_kwargs)
                elif 'population_type' in pop_kwargs:
                    if pop_kwargs['population_type'] in ['Conv2D', 'conv2D', 'conv2d']:
                        pop = Conv2DPopulation(self, layer, pop_name, **pop_kwargs)
                    elif pop_kwargs['population_type'] in ['Flatten', 'flatten']:
                        pop = FlattenPopulation(self, layer, pop_name, **pop_kwargs)
                    elif pop_kwargs['population_type'] in ['MaxPool2D', 'maxpool2D', 'maxpool2d']:
                        pop = MaxPool2DPopulation(self, layer, pop_name, **pop_kwargs)
                    elif pop_kwargs['population_type'] == 'default':
                        pop = Population(self, layer, pop_name, **pop_kwargs)
                    else:
                        raise Exception('EIANN.Network: pop: %s; invalid population_type: %s' %
                                        (pop_name, pop_kwargs['population_type']))
                else:
                    pop = Population(self, layer, pop_name, **pop_kwargs)
                layer.append_population(pop)
                self.populations[pop.fullname] = pop
        
        # if no output_pop is designated, default to the first population specified in the final layer
        input_layer, *_, output_layer = list(self)
        self.output_pop = next(iter(output_layer))
        self.input_pop = next(iter(input_layer))

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
                        if 'projection_type' in projection_kwargs:
                            if projection_kwargs['projection_type'] in ['Conv2D', 'conv2D', 'conv2d']:
                                projection = Conv2DProjection(pre_pop, post_pop, device=device, **projection_kwargs)
                            elif projection_kwargs['projection_type'] in ['Linear', 'linear']:
                                projection = Projection(pre_pop, post_pop, device=device, **projection_kwargs)
                            else:
                                raise NotImplementedError('EIANN.Network: projection type: %s not implemented' %
                                                          projection_kwargs['projection_type'])
                        else:
                            projection = Projection(pre_pop, post_pop, device=device, **projection_kwargs)
                        post_pop.append_projection(projection)
                        post_pop.incoming_projections[projection.name] = projection
                        pre_pop.outgoing_projections[projection.name] = projection
                        self.projections[projection.name] = projection
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
            optimizer = optimizer(self.optimizer_params_list, **optimizer_kwargs)

        self.optimizer = optimizer
        self.init_weights_and_biases()
        self.reset_history()

    def init_weights_and_biases(self):
        for i, post_layer in enumerate(self):
            # if i > 0:
                for post_pop in post_layer:
                    total_fan_in = 0
                    for projection in post_pop:
                        fan_in = projection.pre.size
                        total_fan_in += fan_in
                        if projection.weight_init is not None:
                            if projection.weight_init == 'half_kaiming':
                                ut.half_kaiming_init(projection.weight.data, fan_in, *projection.weight_init_args,
                                                   bounds=projection.weight_bounds)
                            elif projection.weight_init == 'scaled_kaiming':
                                ut.scaled_kaiming_init(projection.weight.data, fan_in, *projection.weight_init_args)
                            elif projection.weight_init in ['clone', 'clone_weight']:
                                pass
                            else:
                                try:
                                    getattr(projection.weight.data,
                                            projection.weight_init)(*projection.weight_init_args)
                                except:
                                    raise RuntimeError('Network.init_weights_and_biases: callable for weight_init: %s '
                                                       'must be half_kaiming, scaled_kaiming, clone, or a method of '
                                                       'Tensor' % projection.weight_init)
                    if post_pop.include_bias:
                        if post_pop.bias_init is None:
                            ut.scaled_kaiming_init(post_pop.bias.data, total_fan_in)
                        elif post_pop.bias_init == 'scaled_kaiming':
                            ut.scaled_kaiming_init(post_pop.bias.data, total_fan_in, *post_pop.bias_init_args)
                        else:
                            try:
                                getattr(post_pop.bias.data, post_pop.bias_init)(*post_pop.bias_init_args)
                            except:
                                raise RuntimeError('Network.init_weights_and_biases: callable for bias_init: %s '
                                                   'must be scaled_kaiming, or a method of Tensor' % post_pop.bias_init)

        self.constrain_weights_and_biases()

        for projection in self.projections.values():
            if projection.weight_init in ['clone', 'clone_weight']:
                rules.weight_functions.clone_weight(projection, **projection.weight_init_args)
                if projection.weight_bounds is not None:
                    projection.weight.data = projection.weight.data.clamp(*projection.weight_bounds)
                if projection.constrain_weight is not None:
                    projection.constrain_weight()

    def constrain_weights_and_biases(self):
        for i, post_layer in enumerate(self):
            # if i > 0:
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
            # if i > 0:
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

    def forward(self, sample, store_history=False, store_dynamics=False, store_num_steps=None, no_grad=False):
        """
        :param sample: tensor
        :param store_history: bool
        :param store_dynamics: bool
        :param store_num_steps: int
        :param no_grad: bool
        :return: tensor
        """
        if store_num_steps is None:
            store_num_steps = self.forward_steps
        
        for population in self.populations.values():
            if sample.ndim == 1:
                population.reinit(self.device, batch_size=1)
            else:
                population.reinit(self.device, batch_size=sample.shape[0])
    
        if not hasattr(self, 'input_pop'):
            self.input_pop = next(iter(list(self)[0]))
        self.input_pop.activity = torch.squeeze(sample)

        for t in range(self.forward_steps):
            if (t >= self.forward_steps - self.backward_steps) and not no_grad:
                track_grad = True
            else:
                track_grad = False

            with torch.set_grad_enabled(track_grad):
                for population in self.populations.values():
                    population.prev_activity = population.activity
                for i, post_layer in enumerate(self):
                    for post_pop in post_layer:
                        if i > 0:
                            post_pop.forward()
                        if store_dynamics and t >= (self.forward_steps - store_num_steps):
                            post_pop.forward_steps_activity.append(post_pop.activity.detach().clone())

        if store_history:
            for pop in self.populations.values():
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

    def update_forward_state(self, store_history=False, store_dynamics=False, store_num_steps=None):
        """
        Clone forward neuronal activities and initialize forward dendritic states.
        """
        if store_num_steps is None:
            store_num_steps = self.forward_steps

        for post_pop in self.populations.values():
            post_pop.forward_activity = post_pop.activity.detach().clone()
            post_pop.forward_prev_activity = post_pop.prev_activity.detach().clone()
            
            init_dend_state = False
            for projection in post_pop:
                pre_pop = projection.pre
                if projection.compartment == 'dend':
                    if not init_dend_state:
                        post_pop.forward_dendritic_state = torch.zeros(post_pop.size, device=self.device)
                        init_dend_state = True
                        if store_dynamics:
                            post_pop.forward_dendritic_state_steps = torch.zeros(store_num_steps, post_pop.size, device=self.device)

                    if projection.direction in ['forward', 'F']:
                        post_pop.forward_dendritic_state = post_pop.forward_dendritic_state + projection(pre_pop.activity)
                        if store_dynamics:
                            pre_activity_dynamics = torch.stack(pre_pop.forward_steps_activity[self.forward_steps-store_num_steps:])
                            post_pop.forward_dendritic_state_steps = post_pop.forward_dendritic_state_steps + projection(pre_activity_dynamics)
                    elif projection.direction in ['recurrent', 'R']:
                        post_pop.forward_dendritic_state = post_pop.forward_dendritic_state + projection(pre_pop.prev_activity)
                        if store_dynamics:
                            pre_activity_dynamics = torch.stack(pre_pop.forward_steps_activity[self.forward_steps-store_num_steps:-1])
                            post_pop.forward_dendritic_state_steps[1:] = post_pop.forward_dendritic_state_steps[1:] + projection(pre_activity_dynamics)

            if store_history and hasattr(post_pop, 'forward_dendritic_state'):
                if store_dynamics:
                    post_pop.append_attribute_history('forward_dendritic_state', post_pop.forward_dendritic_state_steps.detach().clone())
                else:
                    post_pop.append_attribute_history('forward_dendritic_state', post_pop.forward_dendritic_state.detach().clone())
                       
    def train(self, train_dataloader, val_dataloader=None, epochs=1, val_interval=(0, -1, 50), samples_per_epoch=None,
              store_history=False, store_dynamics=False, store_params=False, store_history_interval=None, 
              store_params_interval=None, save_to_file=None, status_bar=False):
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
        :param store_history_interval: tuple of int (start_index, stop_index, interval)
        :param store_params_interval: tuple of int (start_index, stop_index, interval)
        :param save_to_file: None or file_path
        :param status_bar: bool
        """
        self.reset_history()
        if samples_per_epoch is None:
            samples_per_epoch = len(train_dataloader)
        
        # Define timepoints for validation
        train_step_range = torch.arange(epochs * samples_per_epoch)
        if val_interval[0] < 0 and abs(val_interval[0]) > len(train_step_range):
            val_start_index = 0
        else:
            val_start_index = train_step_range[val_interval[0]]
        val_end_index = train_step_range[val_interval[1]]
        val_step_size = val_interval[2]
        val_range = torch.arange(val_end_index, val_start_index - 1, -val_step_size).flip(0)
        if val_start_index == 0 and 0 not in val_range:
            val_range = torch.cat((torch.tensor([0]), val_range))
        
        # Load validation data and initialize intermediate variables
        if val_dataloader is not None:
            assert len(val_dataloader) == 1, 'Validation Dataloader must have a single large batch'
            idx, val_data, val_target = next(iter(val_dataloader))
            if not val_data.device == self.device:
                val_data = val_data.to(self.device)
                val_target = val_target.to(self.device)
            self.val_output_history = []
            self.val_loss_history = []
            self.val_accuracy_history = []
            self.val_history_train_steps = []
            
        # Store history of weights and biases
        if store_history:
            if store_history_interval is not None:
                if store_history_interval[0] < 0 and abs(store_history_interval[0]) > len(train_step_range):
                    store_history_start_index = 0
                else:
                    store_history_start_index = train_step_range[store_history_interval[0]]
                store_history_end_index = train_step_range[store_history_interval[1]]
                store_history_step_size = store_history_interval[2]
                store_history_range = (
                    torch.arange(store_history_end_index, store_history_start_index - 1,
                                 -store_history_step_size).flip(0))
                if store_history_start_index == 0 and 0 not in store_history_range:
                    store_history_range = torch.cat((torch.tensor([0]), store_history_range))
        
        # Store history of weights and biases
        if store_params:
            if store_params_interval is None:
                store_params_step_size = val_step_size
                store_params_range = val_range
            else:
                if store_params_interval[0] < 0 and abs(store_params_interval[0]) > len(train_step_range):
                    store_params_start_index = 0
                else:
                    store_params_start_index = train_step_range[store_params_interval[0]]
                store_params_end_index = train_step_range[store_params_interval[1]]
                store_params_step_size = store_params_interval[2]
                store_params_range = (
                    torch.arange(store_params_end_index, store_params_start_index - 1, -store_params_step_size).flip(0))
                if store_params_start_index == 0 and 0 not in store_params_range:
                    store_params_range = torch.cat((torch.tensor([0]), store_params_range))
            if (0 in store_params_range) and store_params_step_size == 1: # store initial state of the network
                self.param_history.append(deepcopy(self.state_dict()))
                self.param_history_steps.append(-1)
        
        if status_bar:
            from tqdm.autonotebook import tqdm
            epoch_iter = tqdm(range(epochs), desc='Epochs')
        else:
            epoch_iter = range(epochs)

        # Initialize learning rule parameters
        for post_layer in self:
            for post_pop in post_layer:
                if post_pop.include_bias:
                    post_pop.bias_learning_rule.reinit()
                for projection in post_pop:
                    projection.learning_rule.reinit()
        

        #######################################################
        #*************     Main training loop     *************
        #######################################################
        train_step = 0
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
                
                sample_target = torch.squeeze(sample_target)
                if not sample_data.device == self.device:
                    sample_data = sample_data.to(self.device)
                    sample_target = sample_target.to(self.device)
                epoch_sample_order.append(sample_idx)
                
                if store_history:
                    if store_history_interval is None:
                        this_train_step_store_history = True
                    else:
                        this_train_step_store_history = train_step in store_history_range
                else:
                    this_train_step_store_history = False

                output = self.forward(sample_data, store_history=this_train_step_store_history,
                                      store_dynamics=store_dynamics)
                
                loss = self.criterion(output, sample_target)
                self.loss_history.append(loss.item())
                self.target_history.append(sample_target.clone())

                if store_params and (train_step in store_params_range) and store_params_step_size > 1:
                    self.prev_param_history.append(deepcopy(self.state_dict()))  # Store parameters for dW comparison
                
                # Update state variables required for weight and bias updates
                self.update_forward_state(store_history=this_train_step_store_history, store_dynamics=store_dynamics)
                
                for backward in self.backward_methods:
                    backward(self, output, sample_target, store_history=this_train_step_store_history, store_dynamics=store_dynamics)
                
                # Step weights and biases
                for i, post_layer in enumerate(self):
                    # if i > 0:
                    for post_pop in post_layer:
                        if post_pop.include_bias:
                            post_pop.bias_learning_rule.step()
                        for projection in post_pop:
                            projection.learning_rule.step()
                
                self.constrain_weights_and_biases()

                # update learning rule parameters
                for i, post_layer in enumerate(self):
                    # if i > 0:
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
                    
                    self.val_output_history.append(output.detach().clone())
                    self.val_loss_history.append(self.criterion(output, val_target).item())
                    accuracy = 100 * torch.sum(torch.argmax(output, dim=1) == torch.argmax(val_target, dim=1)) / \
                               output.shape[0]
                    self.val_accuracy_history.append(accuracy.item())
                    self.val_history_train_steps.append(train_step)
                    if status_bar: # Display the current loss and accuracy on the progress bar
                        epoch_iter.set_description(f"Validation Loss: {self.val_loss_history[-1]:.4f}, Accuracy: {self.val_accuracy_history[-1]:.2f}% - Epoch")
                train_step += 1

            epoch_sample_order = torch.concat(epoch_sample_order)
            self.sample_order.extend(epoch_sample_order)
            self.sorted_sample_indexes.extend(torch.add(epoch * samples_per_epoch, torch.argsort(epoch_sample_order)))

        self.sample_order = torch.stack(self.sample_order)
        self.sorted_sample_indexes = torch.stack(self.sorted_sample_indexes)
        self.loss_history = torch.tensor(self.loss_history)
        self.target_history = torch.stack(self.target_history)
        if val_dataloader is not None:
            self.val_output_history = torch.stack(self.val_output_history)
            self.val_loss_history = torch.tensor(self.val_loss_history)
            self.val_accuracy_history = torch.tensor(self.val_accuracy_history)
            self.val_history_train_steps = torch.tensor(self.val_history_train_steps)
        if store_params:
            self.param_history_steps = torch.tensor(self.param_history_steps)
        
        if save_to_file is not None:
            ut.save_network(self, path=save_to_file)

    def __iter__(self):
        for layer in self.layers.values():
            yield layer


class AttrDict:
    def __iter__(self):
        for key in self.__dict__:
            yield key


class Layer(object):
    def __init__(self, network, name):
        self.name = name
        self.network = network
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
    def __init__(self, network, layer, name, size, activation='linear', activation_kwargs=None, tau=None,
                 include_bias=False, bias_init=None, bias_init_args=None, bias_bounds=None,
                 bias_learning_rule=None, bias_learning_rule_kwargs=None, custom_update=None, custom_update_kwargs=None,
                 output_pop=False):
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
        :param custom_update: str; name of imported callable
        :param custom_update_kwargs: dict
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
            if hasattr(ut, activation):
                activation = getattr(ut, activation)
            elif hasattr(torch.nn.functional, activation):
                activation = getattr(torch.nn.functional, activation) 
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
            if not any([bias_learning_rule_class.backward.__func__ is backward_method.__func__
                        for backward_method in self.network.backward_methods]):
                self.network.backward_methods.append(bias_learning_rule_class.backward)

        self.network.parameter_dict[self.fullname+'_bias'] = self.bias
        self.network.optimizer_params_list.append({'params': self.bias,
                                                   'lr': self.bias_learning_rule.learning_rate})
        
        # TODO: implement custom state updates per population
        # if custom_update is not None:
        #     if isinstance(custom_update, str):
        #         if hasattr(rules, custom_update):
        #             custom_update = getattr(rules, custom_update)
        #         elif hasattr(external, custom_update):
        #             weight_constraint = getattr(external, weight_constraint)
        #     if not callable(weight_constraint):
        #         raise RuntimeError \
        #             ('Projection: weight_constraint: %s must be imported and callable' %
        #              weight_constraint)
            
        
        # Initialize storage containers
        self.projections = {}
        self.backward_projections = []
        self.outgoing_projections = {}
        self.incoming_projections = {}
        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))
        self.reinit(network.device)
        self.reset_history()

    def register_attribute_history(self, attr_name):
        attr_history_name = attr_name + '_history'
        if not hasattr(self.__class__, attr_history_name):
            setattr(self.__class__, attr_history_name, property(lambda parent: parent.get_attribute_history(attr_name)))
    
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
    
    def forward(self):
        delta_state = -self.state + self.bias
        for projection in self:
            pre_pop = projection.pre
            if projection.compartment not in ['dend', 'dendrite']:
                if projection.update_phase in ['forward', 'all', 'F', 'A']:
                    if projection.direction in ['forward', 'F']:
                        delta_state = delta_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        delta_state = delta_state + projection(pre_pop.prev_activity)
        self.state = self.state + delta_state / self.tau
        self.activity = self.activation(self.state)
    
    def reinit(self, device, batch_size=1):
        """
        Method for resetting state variables of a population
        :param device:
        """
        if batch_size > 1:
            self.state = torch.zeros((batch_size, self.size), device=device)
        else:
            self.state = torch.zeros(self.size, device=device)
        self.state += self.bias
        self.activity = self.activation(self.state).to(device)
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
            if not any([projection.learning_rule.__class__.backward.__func__ is backward_method.__func__
                        for backward_method in self.network.backward_methods]):
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

    @property
    def activity_dynamics(self):
        return torch.stack(self.forward_steps_activity) if self.forward_steps_activity else None


class Conv2DPopulation(Population):
    def __init__(self, network, layer, name, size, population_type, activation='linear', activation_kwargs=None,
                 tau=None, include_bias=False, bias_init=None, bias_init_args=None, bias_bounds=None,
                 bias_learning_rule=None, bias_learning_rule_kwargs=None, custom_update=None, custom_update_kwargs=None,
                 output_pop=False, image_dim=None, kernel_size=5, image_source=None, **kwargs):
        """
        Class for population of neurons that receive projections of type Conv2DProjection from one or more populations
        of type Conv2DPopulation.
        Currently automatic calculation of output image_dim assumes default stride, padding, and dilation.
        :param network: :class:'Network'
        :param layer: :class:'Layer'
        :param name: str
        :param size: int
        :param population_type: str
        :param activation: str; name of imported callable
        :param activation_kwargs: dict
        :param include_bias: bool
        :param bias_init: str; name of imported callable
        :param bias_init_args: dict
        :param bias_bounds: tuple of float
        :param bias_learning_rule: str; name of imported callable
        :param bias_learning_rule_kwargs: dict
        :param custom_update: str; name of imported callable
        :param custom_update_kwargs: dict
        :param output_pop: bool; a single population must be designated as the output population to compute network loss
        :param image_dim: int; height and width of image for convolutional kernels
        :param kernel_size: int; height and width of kernels, incoming convolutional projections will inherit this value
        :param image_source: str; name of population with reference input image_dim
        """
        # Constants
        self.kernel_size = kernel_size
        if image_dim is None:
            if image_source is not None:
                try:
                    source_layer_name, source_pop_name = image_source.split('.')
                    source_pop = network.layers[source_layer_name].populations[source_pop_name]
                    source_image_dim = source_pop.image_dim
                except:
                    raise Exception('Population: invalid source image reference population: %s' % image_source)
            if source_image_dim is not None:
                image_dim = source_image_dim - (self.kernel_size - 1)
        self.image_dim = image_dim
        
        super().__init__(network, layer, name, size, activation=activation, activation_kwargs=activation_kwargs,
                         tau=tau, include_bias=include_bias, bias_init=bias_init, bias_init_args=bias_init_args,
                         bias_bounds=bias_bounds, bias_learning_rule=bias_learning_rule,
                         bias_learning_rule_kwargs=bias_learning_rule_kwargs, custom_update=custom_update,
                         custom_update_kwargs=custom_update_kwargs, output_pop=output_pop)
    
    def forward(self):
        delta_state = -self.state + self.bias.unsqueeze(-1).unsqueeze(-1)
        for projection in self:
            pre_pop = projection.pre
            if projection.compartment not in ['dend', 'dendrite']:
                if projection.update_phase in ['forward', 'all', 'F', 'A']:
                    if projection.direction in ['forward', 'F']:
                        delta_state = delta_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        delta_state = delta_state + projection(pre_pop.prev_activity)
        self.state = self.state + delta_state / self.tau
        self.activity = self.activation(self.state)
    
    def reinit(self, device, batch_size=1):
        """
        Method for resetting state variables of a population
        :param device:
        """
        if self.image_dim is None:
            if batch_size > 1:
                self.state = torch.zeros((batch_size, self.size), device=device)
            else:
                self.state = torch.zeros(self.size, device=device)
            self.state += self.bias
        else:
            if batch_size > 1:
                self.state = torch.zeros((batch_size, self.size, self.image_dim, self.image_dim), device=device)
            else:
                self.state = torch.zeros(self.size, self.image_dim, self.image_dim, device=device)
            self.state += self.bias.unsqueeze(-1).unsqueeze(-1)
        self.activity = self.activation(self.state).to(device)
        self.forward_steps_activity = []


class MaxPool2DPopulation(Population):
    def __init__(self, network, layer, name, population_type, custom_update=None, custom_update_kwargs=None,
                 output_pop=False, kernel_size=2, source=None, **kwargs):
        """
        Class for population of neurons that performs a MaxPool2D operation on the output of a Conv2DPopulation.
        Currently automatic calculation of output image_dim assumes default stride, padding, and dilation.
        :param network: :class:'Network'
        :param layer: :class:'Layer'
        :param name: str
        :param population_type: str
        :param custom_update: str; name of imported callable
        :param custom_update_kwargs: dict
        :param output_pop: bool; a single population must be designated as the output population to compute network loss
        :param kernel_size: int; height and width of pool kernel
        :param source: str; name of population with reference input image_dim
        """
        # Constants
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size, **kwargs)
        try:
            source_layer_name, source_pop_name = source.split('.')
            source_pop = network.layers[source_layer_name].populations[source_pop_name]
            source_image_dim = source_pop.image_dim
            assert(source_image_dim is not None)
        except:
            raise Exception('FlattenPopulation: invalid source population: %s' % source)
        self.source_pop = source_pop
        self.image_dim = source_image_dim // self.kernel_size
        size = self.source_pop.size
        
        super().__init__(network, layer, name, size, custom_update=custom_update,
                         custom_update_kwargs=custom_update_kwargs, output_pop=output_pop)
    
    def forward(self):
        self.activity = self.pool(self.source_pop.activity)
    
    def reinit(self, device, batch_size=1):
        """
        Method for resetting state variables of a population
        :param device:
        """
        if self.image_dim is None:
            if batch_size > 1:
                self.state = torch.zeros((batch_size, self.size), device=device)
            else:
                self.state = torch.zeros(self.size, device=device)
            self.state += self.bias
        else:
            if batch_size > 1:
                self.state = torch.zeros((batch_size, self.size, self.image_dim, self.image_dim), device=device)
            else:
                self.state = torch.zeros(self.size, self.image_dim, self.image_dim, device=device)
            self.state += self.bias.unsqueeze(-1).unsqueeze(-1)
        self.activity = self.activation(self.state).to(device)
        self.forward_steps_activity = []


class FlattenPopulation(Population):
    def __init__(self, network, layer, name, population_type, custom_update=None, custom_update_kwargs=None,
                 output_pop=False, source=None, **kwargs):
        """
        Class for population of neurons that performs a flatten operation on the output of a Conv2DPopulation or
        MaxPool2DPopulation.
        :param network: :class:'Network'
        :param layer: :class:'Layer'
        :param name: str
        :param population_type: str
        :param custom_update: str; name of imported callable
        :param custom_update_kwargs: dict
        :param output_pop: bool; a single population must be designated as the output population to compute network loss
        :param source: str; name of Conv2DPopulation to flatten
        """
        # Constants
        try:
            source_layer_name, source_pop_name = source.split('.')
            source_pop = network.layers[source_layer_name].populations[source_pop_name]
            source_image_dim = source_pop.image_dim
            assert(source_image_dim is not None)
        except:
            raise Exception('FlattenPopulation: invalid source population: %s' % source)
        self.source_pop = source_pop
        size = source_pop.size * source_pop.image_dim * source_pop.image_dim
        
        super().__init__(network, layer, name, size, custom_update=custom_update,
                         custom_update_kwargs=custom_update_kwargs, output_pop=output_pop)
    
    def forward(self):
        # last 3 dimensions are (size, source_image_dim, source_image_dim)
        self.activity = torch.flatten(self.source_pop.activity, -3)


class Input(Population):
    def __init__(self, network, layer, name, size, *args, image_dim=None, **kwargs):
        self.network = network
        self.layer = layer
        self.name = name
        self.fullname = layer.name + self.name
        self.size = size
        self.image_dim = image_dim
        self.projections = {}
        self.backward_projections = []
        self.outgoing_projections = {}
        self.incoming_projections = {}
        self.include_bias = False
        self.reinit(network.device)
        self.reset_history()
    
    def reinit(self, device, batch_size=1):
        """
        Method for resetting state variables of a population
        :param device:
        """
        self.forward_steps_activity = []


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
        :param compartment: None or str in ['soma', 'dend', 'dendrite']
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
        elif weight_init is None:
            raise RuntimeError('Projection: weight_init_args provided for unspecified method')
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

        if compartment is not None and compartment not in ['soma', 'dend', 'dendrite']:
            raise RuntimeError('Projection: compartment (%s) must be None, soma, dend, or dendrite' % compartment)
        self.compartment = compartment
        
        if self.compartment in ['dend', 'dendrite']:
            self.post.register_attribute_history('forward_dendritic_state')
        
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
        
        # learning rules with parameters that require gradient tracking must set requires_grad to True
        self.weight.requires_grad = False
        self.learning_rule = learning_rule_class(self, **learning_rule_kwargs)

        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))

    def register_attribute_history(self, attr_name):
        attr_history_name = attr_name + '_history'
        if not hasattr(self.__class__, attr_history_name):
            setattr(self.__class__, attr_history_name, property(lambda parent: parent.get_attribute_history(attr_name)))
    
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


class Conv2DProjection(nn.Conv2d):
    def __init__(self, pre_pop, post_pop, weight_init=None, weight_init_args=None, weight_constraint=None,
                 weight_constraint_kwargs=None, weight_bounds=None, direction='forward', update_phase='forward',
                 compartment=None, learning_rule='Backprop', learning_rule_kwargs=None, device=None, dtype=None,
                 kernel_size=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', **kwargs):
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
        :param compartment: None or str in ['soma', 'dend', 'dendrite']
        :param learning_rule: str
        :param learning_rule_kwargs: dict
        :param device:
        :param dtype:
        :param kernel_size: int
        :param stride: int
        :param padding: int
        :param dilation: int
        :param groups: int
        :param padding_mode: str
        """
        if kernel_size is None:
            kernel_size = post_pop.kernel_size
        self.kernel_size = kernel_size
        
        super().__init__(pre_pop.size, post_pop.size, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, padding_mode=padding_mode, bias=False, device=device,
                         dtype=dtype)
        
        self.pre = pre_pop
        self.post = post_pop
        self.name = f'{post_pop.layer.name}{post_pop.name}_{pre_pop.layer.name}{pre_pop.name}'
        
        # Initialize weight parameters
        self.weight_init = weight_init
        if weight_init_args is None:
            weight_init_args = ()
        elif weight_init is None:
            raise RuntimeError('Projection: weight_init_args provided for unspecified method')
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
        
        if compartment is not None and compartment not in ['soma', 'dend', 'dendrite']:
            raise RuntimeError('Projection: compartment (%s) must be None, soma, dend, or dendrite' % compartment)
        self.compartment = compartment
        
        if self.compartment in ['dend', 'dendrite']:
            self.post.register_attribute_history('forward_dendritic_state')
        
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
        
        # learning rules with parameters that require gradient tracking must set requires_grad to True
        self.weight.requires_grad = False
        self.learning_rule = learning_rule_class(self, **learning_rule_kwargs)
        
        self.attribute_history_dict = defaultdict(partial(deepcopy, {'buffer': [], 'history': None}))
    
    def register_attribute_history(self, attr_name):
        attr_history_name = attr_name + '_history'
        if not hasattr(self.__class__, attr_history_name):
            setattr(self.__class__, attr_history_name, property(lambda parent: parent.get_attribute_history(attr_name)))
    
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


class NetworkBuilder:
    """Builder for biologically inspired neural networks with layers, populations, and projections."""
    
    def __init__(self):
        self._layers = {}
        self._projections = {}
        self._training_kwargs = {}
        self._current_layer = None
        self._population_types = {}  # Track population types for automatic connection typing
        
    def layer(self, name: str) -> 'LayerBuilder':
        """Start building a layer with the given name."""
        if name not in self._layers:
            self._layers[name] = {}
        self._current_layer = name
        return LayerBuilder(self, name)
    
    def training(self, **kwargs) -> 'NetworkBuilder':
        """Set training parameters."""
        self._training_kwargs.update(kwargs)
        return self
    
    def connect(self, source: str, target: str) -> 'ProjectionBuilder':
        """Create a connection between populations using dot notation.
        
        Args:
            source: Source in format "layer_name.population_name" (e.g., "Input.E")
            target: Target in format "layer_name.population_name" (e.g., "H1.E")
            
        Returns:
            ProjectionBuilder for further configuration
            
        Example:
            network.connect(source='Input.E', target='H1.E')
            network.connect('H1.E', 'H2.E').learning_rule('Hebbian')
        """
        # Parse source
        if '.' not in source:
            raise ValueError(f"Source '{source}' must be in format 'layer.population'")
        source_layer, source_pop = source.split('.', 1)
        
        # Parse target  
        if '.' not in target:
            raise ValueError(f"Target '{target}' must be in format 'layer.population'")
        target_layer, target_pop = target.split('.', 1)
        
        projection_builder = ProjectionBuilder(
            self,
            source_layer,
            source_pop,
            target_layer, 
            target_pop
        )
        
        # Apply population type if it exists for the source population
        source_key = f"{source_layer}.{source_pop}"
        if source_key in self._population_types:
            pop_type, init_scale = self._population_types[source_key]
            projection_builder.type(pop_type, init_scale)
        
        return projection_builder
        
    def set_learning_rule_for_layer(self, 
                                    target_layer: str,
                                    rule: Optional[str], 
                                    learning_rate: Optional[float] = None,
                                    **kwargs) -> 'NetworkBuilder':
        """Set learning rule for all projections TO a specific layer.
        
        Args:
            target_layer: Name of the target layer
            rule: Learning rule name or None
            learning_rate: Learning rate to apply
            **kwargs: Additional learning rule parameters
        
        Returns:
            Self for method chaining
        """
        # Update projections TO this layer
        if target_layer in self._projections:
            for target_pop in self._projections[target_layer]:
                for source_layer in self._projections[target_layer][target_pop]:
                    for source_pop in self._projections[target_layer][target_pop][source_layer]:
                        projection = self._projections[target_layer][target_pop][source_layer][source_pop]
                        
                        projection['learning_rule'] = rule
                        
                        if learning_rate is not None or kwargs:
                            rule_kwargs = {}
                            if learning_rate is not None:
                                rule_kwargs['learning_rate'] = learning_rate
                            rule_kwargs.update(kwargs)
                            projection['learning_rule_kwargs'] = rule_kwargs
        
        # Update bias learning rules for populations IN this layer
        if target_layer in self._layers:
            for pop_name, pop_config in self._layers[target_layer].items():
                if pop_config.get('include_bias', False):
                    if rule is not None:
                        pop_config['bias_learning_rule'] = rule
                    if learning_rate is not None:
                        if 'bias_learning_rule_kwargs' not in pop_config:
                            pop_config['bias_learning_rule_kwargs'] = {}
                        pop_config['bias_learning_rule_kwargs']['learning_rate'] = learning_rate
        
        return self

    def set_learning_rule_for_population(self, 
                                        target_layer: str,
                                        target_population: str,
                                        rule: Optional[str], 
                                        learning_rate: Optional[float] = None,
                                        **kwargs) -> 'NetworkBuilder':
        """Set learning rule for all projections TO a specific population.
        
        Args:
            target_layer: Name of the target layer
            target_population: Name of the target population
            rule: Learning rule name or None
            learning_rate: Learning rate to apply
            **kwargs: Additional learning rule parameters
        
        Returns:
            Self for method chaining
        """
        # Update projections TO this population
        if (target_layer in self._projections and 
            target_population in self._projections[target_layer]):
            
            for source_layer in self._projections[target_layer][target_population]:
                for source_pop in self._projections[target_layer][target_population][source_layer]:
                    projection = self._projections[target_layer][target_population][source_layer][source_pop]
                    
                    projection['learning_rule'] = rule
                    
                    if learning_rate is not None or kwargs:
                        rule_kwargs = {}
                        if learning_rate is not None:
                            rule_kwargs['learning_rate'] = learning_rate
                        rule_kwargs.update(kwargs)
                        projection['learning_rule_kwargs'] = rule_kwargs
        
        # Update bias learning rule for this specific population
        if (target_layer in self._layers and 
            target_population in self._layers[target_layer]):
            
            pop_config = self._layers[target_layer][target_population]
            if pop_config.get('include_bias', False):
                if rule is not None:
                    pop_config['bias_learning_rule'] = rule
                if learning_rate is not None:
                    if 'bias_learning_rule_kwargs' not in pop_config:
                        pop_config['bias_learning_rule_kwargs'] = {}
                    pop_config['bias_learning_rule_kwargs']['learning_rate'] = learning_rate
        
        return self

    def set_learning_rule(self, 
                        rule: Optional[str], 
                        learning_rate: Optional[float] = None,
                        **kwargs) -> 'NetworkBuilder':
        """Set learning rule for ALL projections in the network.
        
        Args:
            rule: Learning rule name (e.g., 'Backprop', 'BTSP_19', etc.) or None
            learning_rate: Learning rate to apply to all projections
            **kwargs: Additional learning rule parameters to apply to all projections
        
        Returns:
            Self for method chaining
        """
        # Iterate through all projections and set the learning rule
        for target_layer in self._projections:
            for target_pop in self._projections[target_layer]:
                for source_layer in self._projections[target_layer][target_pop]:
                    for source_pop in self._projections[target_layer][target_pop][source_layer]:
                        projection = self._projections[target_layer][target_pop][source_layer][source_pop]
                        
                        # Set the learning rule
                        projection['learning_rule'] = rule
                        
                        # Set learning rule parameters if provided
                        if learning_rate is not None or kwargs:
                            rule_kwargs = {}
                            if learning_rate is not None:
                                rule_kwargs['learning_rate'] = learning_rate
                            rule_kwargs.update(kwargs)
                            projection['learning_rule_kwargs'] = rule_kwargs
        
        # Update bias learning rules for ALL populations
        for layer_name, layer_config in self._layers.items():
            for pop_name, pop_config in layer_config.items():
                if pop_config.get('include_bias', False):
                    if rule is not None:
                        pop_config['bias_learning_rule'] = rule
                    if learning_rate is not None:
                        if 'bias_learning_rule_kwargs' not in pop_config:
                            pop_config['bias_learning_rule_kwargs'] = {}
                        pop_config['bias_learning_rule_kwargs']['learning_rate'] = learning_rate
                        
        return self

    def get_layer_config(self) -> Dict[str, Any]:
        """Get the layer configuration dictionary."""
        return deepcopy(self._layers)
    
    def get_projection_config(self) -> Dict[str, Any]:
        """Get the projection configuration dictionary."""
        return deepcopy(self._projections)
    
    def get_training_kwargs(self) -> Dict[str, Any]:
        """Get the training parameters dictionary."""
        return deepcopy(self._training_kwargs)

    def print_architecture(self) -> None:
        """Print the network architecture in a readable format."""
        # First, collect all connections and sort them
        connections = []
        for target_layer in self._projections:
            for target_pop in self._projections[target_layer]:
                for source_layer in self._projections[target_layer][target_pop]:
                    for source_pop in self._projections[target_layer][target_pop][source_layer]:
                        projection = self._projections[target_layer][target_pop][source_layer][source_pop]
                        
                        # Get source population size
                        source_size = self._layers.get(source_layer, {}).get(source_pop, {}).get('size', '?')
                        
                        # Get target population size
                        target_size = self._layers.get(target_layer, {}).get(target_pop, {}).get('size', '?')
                        
                        # Build connection string
                        source_str = f"{source_layer}.{source_pop} ({source_size})"
                        target_str = f"{target_layer}.{target_pop} ({target_size})"
                        
                        # Determine connection type from weight bounds
                        weight_bounds = projection.get('weight_bounds', (None, None))
                        connection_type = "Unknown"
                        
                        if weight_bounds == (0, None):
                            connection_type = "Exc"
                        elif weight_bounds == (None, 0):
                            connection_type = "Inh"
                        # elif weight_bounds == (None, None):
                        #     connection_type = "Mixed"
                        else:
                            # Custom bounds
                            lower, upper = weight_bounds
                            if lower is not None and upper is not None:
                                connection_type = f"[{lower}, {upper}]"
                            elif lower is not None:
                                connection_type = f"[{lower}, ∞)"
                            elif upper is not None:
                                connection_type = f"(-∞, {upper}]"
                        
                        # Get learning rule info
                        learning_rule = projection.get('learning_rule', 'None')
                        rule_kwargs = projection.get('learning_rule_kwargs', {})
                        
                        if learning_rule and learning_rule != 'None':
                            if rule_kwargs:
                                # Format learning rate and other parameters
                                params = []
                                if 'learning_rate' in rule_kwargs:
                                    params.append(f"lr={rule_kwargs['learning_rate']}")
                                # Add other parameters
                                for key, value in rule_kwargs.items():
                                    if key != 'learning_rate':
                                        params.append(f"{key}={value}")
                                
                                rule_str = f"{learning_rule} ({', '.join(params)})"
                            else:
                                rule_str = learning_rule
                        else:
                            rule_str = "No learning rule"
                        
                        if connection_type != "Unknown":
                            connections.append(f"{source_str} -> {target_str} [{connection_type}]: {rule_str}")
                        else:
                            connections.append(f"{source_str} -> {target_str}: {rule_str}")
        
        # Sort connections by network flow order (Input -> Hidden -> Output)
        def connection_sort_key(connection_str):
            # Extract source layer name for sorting
            source_part = connection_str.split(' -> ')[0]
            layer_name = source_part.split('.')[0]
            
            # Define layer order priority
            if layer_name.lower() == 'input':
                return (0, layer_name)
            elif layer_name.lower() == 'output':
                return (999, layer_name)  # Output layers go last
            elif layer_name.lower().startswith('h'):
                # Hidden layers: try to extract number for proper ordering
                try:
                    num = int(''.join(filter(str.isdigit, layer_name)))
                    return (1, num, layer_name)
                except:
                    return (1, 0, layer_name)
            else:
                # Other layers go in the middle
                return (2, layer_name)
        
        connections.sort(key=connection_sort_key)
        
        # Find unconnected populations
        connected_populations = set()
        for target_layer in self._projections:
            for target_pop in self._projections[target_layer]:
                connected_populations.add(f"{target_layer}.{target_pop}")
                for source_layer in self._projections[target_layer][target_pop]:
                    for source_pop in self._projections[target_layer][target_pop][source_layer]:
                        connected_populations.add(f"{source_layer}.{source_pop}")
        
        unconnected_populations = []
        for layer_name, layer_config in self._layers.items():
            for pop_name, pop_config in layer_config.items():
                pop_key = f"{layer_name}.{pop_name}"
                if pop_key not in connected_populations:
                    size = pop_config.get('size', '?')
                    
                    # Determine population type if it exists
                    pop_type_str = ""
                    if pop_key in self._population_types:
                        pop_type, _ = self._population_types[pop_key]
                        if pop_type.lower() in ['excitatory', 'e', 'exc']:
                            pop_type_str = " [Exc]"
                        elif pop_type.lower() in ['inhibitory', 'i', 'inh']:
                            pop_type_str = " [Inh]"
                        else:
                            pop_type_str = f" [{pop_type}]"
                    
                    unconnected_populations.append(f"{pop_key} ({size}){pop_type_str}")
        
        # Print the architecture
        if connections or unconnected_populations:
            print('='*50)
            print("Network Architecture:")
            
            if connections:
                for connection in connections:
                    print(connection)
            else:
                print("No connections defined.")
            
            if unconnected_populations:
                print("\nUnconnected populations:")
                for pop in unconnected_populations:
                    print(pop)
            
            print('='*50)
            print()
        else:
            print("No layers or connections defined in network.")

    def build(self, seed=None) -> Tuple[Dict, Dict, Dict]:
        """Build and return all configuration dictionaries."""
        layer_config = self.get_layer_config()
        projection_config = self.get_projection_config()
        training_kwargs = self.get_training_kwargs()

        network = Network(layer_config, projection_config, **training_kwargs, seed=seed)
        print(network)
        return network


class LayerBuilder:
    """Builder for individual layers containing populations."""
    
    def __init__(self, network_builder: NetworkBuilder, layer_name: str):
        self._network_builder = network_builder
        self._layer_name = layer_name
        
    def population(self, 
                   pop_name: str, 
                   size: int, 
                   activation: Optional[str] = None,
                   bias: bool = False,
                   bias_learning_rule: Optional[str] = "Backprop",
                   bias_learning_rate: Optional[float] = None,
                   tau: Optional[float] = None) -> 'LayerBuilder':
        """Add a population to the current layer."""
        if self._layer_name not in self._network_builder._layers:
            self._network_builder._layers[self._layer_name] = {}
            
        pop_config = {'size': size}
        if activation is not None:
            pop_config['activation'] = activation
        
        if bias:
            pop_config['include_bias'] = bias
            pop_config['bias_learning_rule'] = bias_learning_rule
            pop_config['bias_learning_rule_kwargs'] = {'learning_rate': bias_learning_rate}
            
        if tau is not None:
            pop_config['tau'] = tau
            
        self._network_builder._layers[self._layer_name][pop_name] = pop_config
        return self
    
    def type(self, neuron_type: str, init_scale: float = 1.0) -> 'LayerBuilder':
        """Set the type of the last added population.
        
        Args:
            neuron_type: Type of neuron ('Exc', 'Inh', 'Excitatory', 'Inhibitory')
            init_scale: Initialization scale for connections from this population
            
        Returns:
            Self for method chaining
        """
        if not self._network_builder._layers[self._layer_name]:
            raise ValueError(f"No populations defined in layer {self._layer_name}")
        
        # Get the last added population
        last_pop = list(self._network_builder._layers[self._layer_name].keys())[-1]
        
        # Store the population type for future connections
        pop_key = f"{self._layer_name}.{last_pop}"
        self._network_builder._population_types[pop_key] = (neuron_type, init_scale)
        
        # Apply the type to any existing outgoing connections from this population
        self._apply_type_to_existing_connections(last_pop, neuron_type, init_scale)
        
        return self
    
    def _apply_type_to_existing_connections(self, source_pop: str, neuron_type: str, init_scale: float):
        """Apply the population type to existing outgoing connections."""
        # Find all projections where this population is the source
        for target_layer in self._network_builder._projections:
            for target_pop in self._network_builder._projections[target_layer]:
                if self._layer_name in self._network_builder._projections[target_layer][target_pop]:
                    if source_pop in self._network_builder._projections[target_layer][target_pop][self._layer_name]:
                        projection = self._network_builder._projections[target_layer][target_pop][self._layer_name][source_pop]
                        
                        # Apply the type settings
                        if neuron_type.lower() in ['excitatory', 'e', 'exc']:
                            projection['weight_bounds'] = (0, None)
                            projection['weight_init'] = 'half_kaiming'
                            projection['weight_init_args'] = (init_scale,)
                        elif neuron_type.lower() in ['inhibitory', 'i', 'inh']:
                            projection['weight_bounds'] = (None, 0)
                            projection['weight_init'] = 'half_kaiming'
                            projection['weight_init_args'] = (init_scale,)
    
    def connect_to(self, 
                   target_layer: str, 
                   target_population: str, 
                   source_population: Optional[str] = None) -> 'ProjectionBuilder':
        """Create a projection from this layer to a target layer/population."""
        # If no source population specified, assume we're connecting from the last added population
        if source_population is None:
            if not self._network_builder._layers[self._layer_name]:
                raise ValueError(f"No populations defined in layer {self._layer_name}")
            source_population = list(self._network_builder._layers[self._layer_name].keys())[-1]
        
        projection_builder = ProjectionBuilder(
            self._network_builder, 
            self._layer_name, 
            source_population,
            target_layer, 
            target_population
        )
        
        # Apply population type if it exists for the source population
        source_key = f"{self._layer_name}.{source_population}"
        if source_key in self._network_builder._population_types:
            pop_type, init_scale = self._network_builder._population_types[source_key]
            projection_builder.type(pop_type, init_scale)
        
        return projection_builder
    
    def connect_from(self, 
                     source_layer: str, 
                     source_population: str, 
                     target_population: Optional[str] = None) -> 'ProjectionBuilder':
        """Create a projection from a source layer/population to this layer."""
        # If no target population specified, assume we're connecting to the last added population
        if target_population is None:
            if not self._network_builder._layers[self._layer_name]:
                raise ValueError(f"No populations defined in layer {self._layer_name}")
            target_population = list(self._network_builder._layers[self._layer_name].keys())[-1]
        
        projection_builder = ProjectionBuilder(
            self._network_builder,
            source_layer,
            source_population, 
            self._layer_name,
            target_population
        )
        
        # Apply population type if it exists for the source population
        source_key = f"{source_layer}.{source_population}"
        if source_key in self._network_builder._population_types:
            pop_type, init_scale = self._network_builder._population_types[source_key]
            projection_builder.type(pop_type, init_scale)
        
        return projection_builder
    
    def layer(self, name: str) -> 'LayerBuilder':
        """Switch to building a different layer."""
        return self._network_builder.layer(name)
    
    def training(self, **kwargs) -> NetworkBuilder:
        """Set training parameters and return to network builder."""
        return self._network_builder.training(**kwargs)
    
    def build(self) -> Tuple[Dict, Dict, Dict]:
        """Build and return all configuration dictionaries."""
        return self._network_builder.build()


class ProjectionBuilder:
    """Builder for projections between populations."""
    
    def __init__(self, 
                 network_builder: NetworkBuilder,
                 source_layer: str,
                 source_pop: str, 
                 target_layer: str, 
                 target_pop: str):
        self._network_builder = network_builder
        self._source_layer = source_layer
        self._source_pop = source_pop
        self._target_layer = target_layer
        self._target_pop = target_pop
        
        # Initialize nested structure if needed
        projections = self._network_builder._projections
        if target_layer not in projections:
            projections[target_layer] = {}
        if target_pop not in projections[target_layer]:
            projections[target_layer][target_pop] = {}
        if source_layer not in projections[target_layer][target_pop]:
            projections[target_layer][target_pop][source_layer] = {}
        if source_pop not in projections[target_layer][target_pop][source_layer]:
            projections[target_layer][target_pop][source_layer][source_pop] = {}
            
        self._projection_config = projections[target_layer][target_pop][source_layer][source_pop]
    
    def weight_init(self, 
                    init_type: str, 
                    *args, 
                    **kwargs) -> 'ProjectionBuilder':
        """Set weight initialization."""
        self._projection_config['weight_init'] = init_type
        if args:
            self._projection_config['weight_init_args'] = args
        if kwargs:
            self._projection_config['weight_init_kwargs'] = kwargs
        return self
    
    def weight_bounds(self, 
                      lower: Optional[float] = None, 
                      upper: Optional[float] = None) -> 'ProjectionBuilder':
        """Set weight bounds (clipping)."""
        self._projection_config['weight_bounds'] = (lower, upper)
        return self
    
    def type(self, weight_type: str, init_scale: float = 1.0) -> 'ProjectionBuilder':
        """Set weight type (Excitatory or Inhibitory)."""
        if weight_type.lower() in ['excitatory', 'e', 'exc']:
            self.weight_bounds(0, None)
            self.weight_init('half_kaiming', init_scale)
        elif weight_type.lower() in ['inhibitory', 'i', 'inh']:
            self.weight_bounds(None, 0)
            self.weight_init('half_kaiming', init_scale)
        else:
            raise ValueError("Invalid weight type. Must be 'Exc' or 'Inh'")
        return self

    def weight_constraint(self, 
                          constraint_type: str, 
                          **kwargs) -> 'ProjectionBuilder':
        """Set weight constraints."""
        self._projection_config['weight_constraint'] = constraint_type
        if kwargs:
            self._projection_config['weight_constraint_kwargs'] = kwargs
        return self
    
    def direction(self, direction: str) -> 'ProjectionBuilder':
        """Set connection direction (F=forward, R=recurrent, B=backward)."""
        self._projection_config['direction'] = direction
        return self
    
    def learning_rule(self, 
                      rule: Optional[str], 
                      learning_rate: Optional[float] = None,
                      **kwargs) -> 'ProjectionBuilder':
        """Set learning rule and parameters."""
        self._projection_config['learning_rule'] = rule
        if learning_rate is not None or kwargs:
            rule_kwargs = {}
            if learning_rate is not None:
                rule_kwargs['learning_rate'] = learning_rate
            rule_kwargs.update(kwargs)
            self._projection_config['learning_rule_kwargs'] = rule_kwargs
        return self
    
    def update_phase(self, phase: str) -> 'ProjectionBuilder':
        """Set update phase (A or B)."""
        self._projection_config['update_phase'] = phase
        return self
    
    def compartment(self, compartment: str) -> 'ProjectionBuilder':
        """Set target compartment (soma, dend, etc.)."""
        self._projection_config['compartment'] = compartment
        return self
    
    def layer(self, name: str) -> LayerBuilder:
        """Switch to building a different layer."""
        return self._network_builder.layer(name)
    
    def training(self, **kwargs) -> NetworkBuilder:
        """Set training parameters and return to network builder."""
        return self._network_builder.training(**kwargs)
    
    def build(self) -> Tuple[Dict, Dict, Dict]:
        """Build and return all configuration dictionaries."""
        return self._network_builder.build()


