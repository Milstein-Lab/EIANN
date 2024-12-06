# Using: EIANN\optimize\optimize_config\spiral\nested_optimize_EIANN_2_hidden_dend_EI_contrast_fixed_bias.yaml

# Based on: EIANN\optimize\nested_optimize_EIANN_1_hidden_mnist.py

import torch
import os
import sys
import numpy as np
import time
import gc

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, \
            sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
            recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, generate_spiral_data, \
            sort_by_class_averaged_val_output
from EIANN.plot import plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, plot_receptive_fields, \
            plot_representation_metrics
from EIANN.optimize.network_config_updates import *
import EIANN.utils as utils

from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts

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
    if 'store_params_interval' not in context():
        context.store_params_interval = (0, -1, 100)
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
            context.export_network_config_file_path = f"{context.output_dir}/{network_name}_{context.label}_optimized.yaml"
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)
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
    
    context.store_history_interval = None
    if context.include_dend_loss_objective:
        if not context.store_history:
            context.store_history = True
            context.store_history_interval = context.val_interval

    context.train_steps = int(context.train_steps)
    
    if context.full_analysis:
        context.val_interval = (0, -1, 100)
        context.store_params_interval = (0, -1, 100)
        context.store_params = True
        context.store_num_steps = None
    
    network_config = read_from_yaml(context.network_config_file_path)
    # network_config = utils.convert_config_dict(network_config) # TODO fix this for simplified configs
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    # Load data
    spiral_train = generate_spiral_data(arm_size=1400)
    spiral_val = generate_spiral_data(arm_size=300)
    spiral_test = generate_spiral_data(arm_size=300)

    # Put in dataloaders
    context.data_generator = torch.Generator()
    context.train_dataloader = torch.utils.data.DataLoader(spiral_train, shuffle=True, generator=context.data_generator)
    context.val_dataloader = torch.utils.data.DataLoader(spiral_val, shuffle=False, batch_size=len(spiral_val))
    context.test_dataloader = torch.utils.data.DataLoader(spiral_test, shuffle=False, batch_size=len(spiral_test))

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
    :return: dict
    """
    update_source_contexts(x, context)
    
    data_generator = context.data_generator
    train_dataloader = context.train_dataloader
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
            print(f'nested_optimize_EIANN_spiral_2_hidden: pid: {os.getpid()} exported network config to {context.export_network_config_file_path}')
    
    if plot:
        try:
            network.H2.E.H1.E.initial_weight = network.H2.E.H1.E.weight.data.detach().clone()
            network.H1.E.H2.E.initial_weight = network.H1.E.H2.E.weight.data.detach().clone()
            network.Output.E.H2.E.initial_weight = network.Output.E.H2.E.weight.data.detach().clone()
            network.H2.E.Output.E.initial_weight = network.H2.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            plot_batch_accuracy(network, test_dataloader, population='all', title='Initial')
    
    if 'data_file_path' not in context():
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        if context.label is None:
            context.data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}.pkl"
        else:
            context.data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}_{context.label}.pkl"
    
    if os.path.exists(context.data_file_path) and not context.retrain:
        network = utils.load_network(context.data_file_path)
        if context.disp:
            print(f'nested_optimize_EIANN_spiral_1_hidden: pid: {os.getpid()} loaded network history from {context.data_file_path}')
    else:
        data_generator.manual_seed(data_seed)
        if context.debug:
            current_time = time.time()
        network.train(train_dataloader, val_dataloader, epochs=epochs,
                      val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                      samples_per_epoch=context.train_steps, store_history=context.store_history,
                      store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar, debug=context.debug)
    
    if plot:
        try:
            from EIANN.plot import plot_FB_weight_alignment
            plot_FB_weight_alignment(network.H2.E.H1.E, network.H1.E.H2.E)
            plot_FB_weight_alignment(network.Output.E.H2.E, network.H2.E.Output.E)
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
            raise Exception(f'nested_optimize_EIANN_spiral_1_hidden: eval_accuracy must be final or best, not {context.eval_accuracy}')
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
        if min_loss_idx + num_val_steps_accuracy_window > len(sorted_val_loss_history):  # if best loss too close to the end
            best_accuracy_window = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])
            best_loss_window = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
        else:
            best_accuracy_window = torch.mean(sorted_val_accuracy_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])
            best_loss_window = torch.mean(sorted_val_loss_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])
        
        results = {'loss': best_loss_window,
                   'accuracy': best_accuracy_window}
    else:
        raise Exception(f'nested_optimize_EIANN_spiral_1_hidden: eval_accuracy must be final or best, not {context.eval_accuracy}')

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
        # print('Weights match: %s' % torch.all(final_weights == network.Output.E.H1.E.weight.data))
        plot_batch_accuracy(network, test_dataloader, population='all', sorted_output_idx=sorted_output_idx,
                            title='Final')
        plot_train_loss_history(network)
        plot_validate_loss_history(network)
    
    # if context.compute_receptive_fields:
    #     # Compute receptive fields
    #     population = network.H1.E
    #     receptive_fields = utils.compute_maxact_receptive_fields(population)
    # else:
    #     receptive_fields = network.H1.E.Input.E.weight.detach()
    #
    # if plot:
    #     plot_receptive_fields(receptive_fields, sort=True, num_cols=10, num_rows=10)

    if context.full_analysis:
        # metrics_dict = utils.compute_representation_metrics(network.H1.E, test_dataloader, receptive_fields)
        # plot_representation_metrics(metrics_dict)
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
        utils.save_network(network, path=context.data_file_path, disp=False)
        if context.disp:
            print(f'nested_optimize_EIANN_spiral_1_hidden: pid: {os.getpid()} exported network history to {context.data_file_path}')
    
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