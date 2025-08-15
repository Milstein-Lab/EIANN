import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import gc

from EIANN import Network
from EIANN.utils import (read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, sort_by_class_averaged_val_output,
                         get_binned_mean_population_attribute_history_dict)
from EIANN.plot import (plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, plot_receptive_fields,
                        plot_representation_metrics)
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_config_updates import *
import EIANN.utils as utils


context = Context()


def config_controller():
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = str_to_bool(context.debug)


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.epochs = int(context.epochs)
    context.status_bar = str_to_bool(context.status_bar)
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = str_to_bool(context.debug)
    if 'verbose' not in context():
        context.verbose = False
    else:
        context.verbose = str_to_bool(context.verbose)
    if 'interactive' not in context():
        context.interactive = False
    else:
        context.interactive = str_to_bool(context.interactive)
    if 'eval_accuracy' not in context():
        context.eval_accuracy = 'final'
    else:
        context.eval_accuracy = str(context.eval_accuracy)
    if 'store_history' not in context():
        context.store_history = False
    else:
        context.store_history = str_to_bool(context.store_history)
    if 'store_dynamics' not in context():
        context.store_dynamics = False
    else:
        context.store_dynamics = str_to_bool(context.store_dynamics)
        
    if 'store_params' not in context():
        context.store_params = False
    else:
        context.store_params = str_to_bool(context.store_params)
    if context.debug:
        context.store_num_steps = None
    elif 'store_num_steps' not in context():
        if context.store_dynamics:
            context.store_num_steps = None
        else:
            context.store_num_steps = 2
    else:
        context.store_num_steps = int(context.store_num_steps)
    if 'full_analysis' not in context():
        context.full_analysis = False
    else:
        context.full_analysis = str_to_bool(context.full_analysis)
    if 'equilibration_activity_tolerance' not in context():
        context.equilibration_activity_tolerance = 0.2
    else:
        context.equilibration_activity_tolerance = float(context.equilibration_activity_tolerance)
    if 'compute_receptive_fields' not in context():
        context.compute_receptive_fields = False
    else:
        context.compute_receptive_fields = str_to_bool(context.compute_receptive_fields)
    if 'constrain_equilibration_dynamics' not in context():
        context.constrain_equilibration_dynamics = True
    else:
        context.constrain_equilibration_dynamics = str_to_bool(context.constrain_equilibration_dynamics)
    if 'export_network_config_file_path' not in context():
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        if context.label is None:
            context.export_network_config_file_path = f"{context.output_dir}/{network_name}_optimized.yaml"
        else:
            context.export_network_config_file_path = \
                f"{context.output_dir}/{network_name}_{context.label}_optimized.yaml"
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)
    if not context.retrain:
        if 'data_file_path' not in context() or not os.path.exists(context.data_file_path):
            raise Exception('nested_optimize_EIANN_spiral_2_hidden: missing valid data_file_path to load network from '
                            'file')
    if 'plot_initial' not in context():
        context.plot_initial = False
    else:
        context.plot_initial - str_to_bool(context.plot_initial)
    if 'include_dend_loss_objective' not in context():
        context.include_dend_loss_objective = False
    else:
        context.include_dend_loss_objective = str_to_bool(context.include_dend_loss_objective)
    if 'include_equilibration_dynamics_objective' not in context():
        context.include_equilibration_dynamics_objective = False
    else:
        context.include_equilibration_dynamics_objective = str_to_bool(context.include_equilibration_dynamics_objective)
    
    if 'store_history_interval' not in context():
        context.store_history_interval = None
    
    context.train_steps = int(context.train_steps)
    
    history_interval = max(min(250, int(context.train_steps / 200)), 100)
    if 'store_params_interval' not in context():
        context.store_params_interval = (0, -1, history_interval)
    
    if context.full_analysis:
        context.val_interval = (0, -1, history_interval)
        context.store_params_interval = context.val_interval
        context.store_params = True
        context.store_num_steps = None
        if context.store_history_interval is not None:
            context.store_history_interval = context.val_interval
    
    if context.include_dend_loss_objective:
        if not context.store_history:
            context.store_history = True
            if context.store_history_interval is None:
                context.store_history_interval = context.val_interval
    
    if 'data_file_path' in context():
        context.base_data_file_path = context.data_file_path
    else:
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        context.base_data_file_path = f"{context.output_dir}/{network_name}.pkl"
    
    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']
    
    if 'criterion' in context():
        context.training_kwargs['criterion'] = context.criterion
    
    # Load dataset
    if context.interactive:
        download = True
    else:
        download = False
    
    if 'flatten_data' not in context():
        context.flatten_data = True
    else:
        context.flatten_data = str_to_bool(context.flatten_data)
    
    if context.flatten_data:
        tensor_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(torch.flatten)])
    else:
        tensor_transform = T.ToTensor()
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(root=context.output_dir + '/datasets/CIFAR10_data/',
                                                             train=True, download=download, transform=tensor_transform)
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(root=context.output_dir + '/datasets/CIFAR10_data/',
                                                            train=False, download=download, transform=tensor_transform)

    # Add index to train & test data
    CIFAR10_train = []
    for idx, (data, target) in enumerate(CIFAR10_train_dataset):
        target = torch.eye(len(CIFAR10_train_dataset.classes))[target]
        CIFAR10_train.append((idx, data, target))

    CIFAR10_test = []
    for idx, (data, target) in enumerate(CIFAR10_test_dataset):
        target = torch.eye(len(CIFAR10_test_dataset.classes))[target]
        CIFAR10_test.append((idx, data, target))

    # Put data in dataloader
    context.data_generator = torch.Generator()
    context.train_sub_dataloader = \
        torch.utils.data.DataLoader(CIFAR10_train[0:-10000], shuffle=True, generator=context.data_generator)
    context.val_dataloader = torch.utils.data.DataLoader(CIFAR10_train[-10000:], batch_size=10000, shuffle=False)
    context.test_dataloader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=10000, shuffle=False)


def get_mean_forward_dend_loss(network, num_steps, abs=True):
    """
    
    :param network:
    :param num_steps: int
    :param: abs: bool
    :return: tensor
    """
    attr_name = 'forward_dendritic_state'
    all_pop_attr_history_list = []
    
    for pop_name, pop in network.populations.items():
        attr_history = pop.get_attribute_history(attr_name)
        if attr_history is None:
            continue
        attr_history = attr_history.detach().clone()
        if abs:
            attr_history = torch.abs(attr_history)
        all_pop_attr_history_list.append(attr_history)
    
    all_pop_attr_history_tensor = torch.concatenate(all_pop_attr_history_list, dim=1)
    mean_attr_history = torch.mean(all_pop_attr_history_tensor, dim=1)
    
    return torch.mean(mean_attr_history[-num_steps:]).item()


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print('network_seeds:', network_seeds, 'data_seeds:', data_seeds)
        sys.stdout.flush()
    return [network_seeds, data_seeds]


def compute_features(x, seed, data_seed, model_id=None, export=False, plot=False):
    """

    :param x: array of float
    :param seed: int
    :param data_seed: int
    :param model_id: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    update_source_contexts(x, context)
    
    data_generator = context.data_generator
    train_sub_dataloader = context.train_sub_dataloader
    val_dataloader = context.val_dataloader
    test_dataloader = context.test_dataloader
    
    epochs = context.epochs
    
    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
    
    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_cifar10: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))
    
    if plot:
        try:
            network.Output.E.H1.E.initial_weight = network.Output.E.H1.E.weight.data.detach().clone()
            network.H1.E.Output.E.initial_weight = network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            title = 'Initial (%i, %i)' % (seed, data_seed)
            plot_batch_accuracy(network, test_dataloader, title=title)  # population='all',
    
    if not context.retrain:
        network = utils.load_network(context.data_file_path)
        if context.disp:
            print('nested_optimize_EIANN_cifar10: pid: %i loaded network history from %s' %
                  (os.getpid(), context.data_file_path))
    else:
        data_generator.manual_seed(data_seed)
        if context.debug:
            import time
            current_time = time.time()
        network.train(train_sub_dataloader, val_dataloader, epochs=epochs,
                      val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                      samples_per_epoch=context.train_steps, store_history=context.store_history,
                      store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar)
    
    if plot:
        try:
            from EIANN.plot import plot_FB_weight_alignment
            plot_FB_weight_alignment(network.Output.E.H1.E, network.H1.E.Output.E)
        except:
            pass

    # reorder output units if using unsupervised learning rule
    if not context.supervised:
        if context.eval_accuracy == 'final':
            min_loss_idx = len(network.val_loss_history) - 1
            sorted_output_idx = sort_by_class_averaged_val_output(network, val_dataloader)
        elif context.eval_accuracy == 'best':
            min_loss_idx, sorted_output_idx = sort_by_val_history(network, val_dataloader, plot=plot)
        else:
            raise Exception('nested_optimize_EIANN_cifar10: eval_accuracy must be final or best, not %s' %
                            context.eval_accuracy)
        sorted_val_loss_history, sorted_val_accuracy_history = \
            recompute_validation_loss_and_accuracy(network, val_dataloader, sorted_output_idx=sorted_output_idx,
                                                   store=True)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = None
        sorted_val_loss_history = network.val_loss_history
        sorted_val_accuracy_history = network.val_accuracy_history
    
    if context.store_history and (context.store_history_interval is None):
        binned_train_loss_steps, sorted_train_loss_history, sorted_train_accuracy_history = \
            recompute_train_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=plot)
    
    # Select for stability by computing mean accuracy in a window after the best validation step
    val_stepsize = int(context.val_interval[2])
    num_val_steps_accuracy_window = int(context.num_training_steps_accuracy_window) // val_stepsize
    
    if context.eval_accuracy == 'final':
        final_loss = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
        final_argmax_accuracy = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])
        
        results = {'loss': final_loss,
                   'accuracy': final_argmax_accuracy}
    elif context.eval_accuracy == 'best':
        if min_loss_idx + num_val_steps_accuracy_window > len(
                sorted_val_loss_history):  # if best loss too close to the end
            best_accuracy_window = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])
            best_loss_window = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
        else:
            best_accuracy_window = \
                torch.mean(sorted_val_accuracy_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])
            best_loss_window = torch.mean(
                sorted_val_loss_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])
        
        results = {'loss': best_loss_window,
                   'accuracy': best_accuracy_window}
    else:
        raise Exception('nested_optimize_EIANN_cifar10: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)

    if np.isnan(results['loss']) or np.isinf(results['loss']):
        if context.debug and context.interactive:
            context.update(locals())
        return dict()
    
    if context.include_dend_loss_objective:
        if context.store_history_interval is None:
            dend_loss_window = int(context.num_training_steps_accuracy_window)
        else:
            dend_loss_window = num_val_steps_accuracy_window
        mean_forward_dend_loss = get_mean_forward_dend_loss(network, dend_loss_window)
        results['mean_forward_dend_loss'] = mean_forward_dend_loss
    
    if plot:
        title = 'Final (%i, %i)' % (seed, data_seed)
        plot_batch_accuracy(network, test_dataloader, sorted_output_idx=sorted_output_idx, title=title)  # population='all',
        plot_train_loss_history(network)
        plot_validate_loss_history(network)
    
    # if 'H1' in network.layers:
    #     if context.compute_receptive_fields:
    #         # Compute receptive fields
    #         population = network.H1.E
    #         receptive_fields = utils.compute_maxact_receptive_fields(population)
    #     else:
    #         receptive_fields = network.H1.E.Input.E.weight.detach()
    #
    #     if plot:
    #         plot_receptive_fields(receptive_fields, sort=True, num_cols=10, num_rows=10)
    
    if context.full_analysis:
        # if 'H1' in network.layers:
        #     metrics_dict = utils.compute_representation_metrics(network.H1.E, test_dataloader, receptive_fields)
        #     plot_representation_metrics(metrics_dict)
        
        test_loss_history, test_accuracy_history = \
            compute_test_loss_and_accuracy_history(network, test_dataloader, sorted_output_idx=sorted_output_idx,
                                                   plot=plot, status_bar=context.status_bar)
    
    if context.constrain_equilibration_dynamics or context.debug:
        residuals = check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance,
                                                 store_num_steps=context.store_num_steps, disp=context.disp, plot=plot)
        if context.include_equilibration_dynamics_objective:
            results['dynamics_residuals'] = residuals
        elif residuals > 0. and not context.debug:
            if context.interactive:
                context.update(locals())
            return dict()
    
    if export:
        base_data_file_path_prefix = context.base_data_file_path.split('.')[0]
        if context.label is None:
            this_data_file_path = f"{base_data_file_path_prefix}_{seed}_{data_seed}.pkl"
        else:
            this_data_file_path = f"{base_data_file_path_prefix}_{seed}_{data_seed}_{context.label}.pkl"
        
        utils.save_network(network, path=this_data_file_path, disp=False)
        if context.disp:
            print('nested_optimize_EIANN_cifar10: pid: %i exported network history to %s' %
                  (os.getpid(), this_data_file_path))
    
    if not context.interactive:
        del network
        gc.collect()
    else:
        context.update(locals())

    return results


def filter_features(primitives, current_features, model_id=None, export=False, plot=False):

    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
            if np.isnan(val) or np.isinf(val):
                return dict()
            if key not in features:
                features[key] = []
            features[key].append(val)
    for key, val in features.items():
        features[key] = np.mean(val)

    return features


def get_objectives(features, model_id=None, export=False, plot=False):
    objectives = {}
    for key, val in features.items():
        if 'accuracy' in key:
            objectives[key] = 100. - val
        elif key == 'mean_forward_dend_loss':
            objectives[key] = np.abs(val)
        else:
            objectives[key] = val
    return features, objectives
