import torch
import os
import sys
import numpy as np

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, generate_spiral_data
from EIANN.plot import plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history
from EIANN.optimize.network_config_updates import *

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
    if 'export_network_config_file_path' not in context():
        context.export_network_config_file_path = None
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
    if 'full_analysis' not in context():
        context.full_analysis = False
    else:
        context.full_analysis = str_to_bool(context.full_analysis)
        if context.full_analysis:
            context.val_interval = (0, -1, 100)
            context.store_params_interval = (0, -1, 100)
            context.store_params = True
    if 'equilibration_activity_tolerance' not in context():
        context.equilibration_activity_tolerance = 0.2
    else:
        context.equilibration_activity_tolerance = float(context.equilibration_activity_tolerance)
    if 'constrain_equilibration_dynamics' not in context():
        context.constrain_equilibration_dynamics = True
    else:
        context.constrain_equilibration_dynamics = str_to_bool(context.constrain_equilibration_dynamics)
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)

    network_config = read_from_yaml(context.network_config_file_path)
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

def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    return [network_seeds, data_seeds]

def compute_features(x, seed, data_seed, export=False, plot=False):
    '''
    Parameters:
    - x: array of float
    - seed: int
    - data_seed: int
    - export: bool
    - plot: bool

    return: dict
    '''
    update_source_contexts(x, context)

    data_generator = context.data_generator
    train_dataloader = context.train_dataloader
    test_dataloader = context.test_dataloader
    val_dataloader = context.val_dataloader

    epochs = context.epochs

    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))
            
    if plot:
        try:
            network.Output.E.H1.E.initial_weight = network.Output.E.H1.E.weight.data.detach().clone()
            network.H1.E.Output.E.initial_weight = network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            plot_batch_accuracy(network, test_dataloader, population='all', title='Initial')
    
    if 'data_file_path1' not in context():
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        if context.label is None:
            context.data_file_path1 = f"{context.output_dir}/{network_name}_phase1_{seed}_{data_seed}.pkl"
        else:
            context.data_file_path1 = f"{context.output_dir}/{network_name}_phase1_{seed}_{data_seed}_{context.label}.pkl"
    
    if os.path.exists(context.data_file_path1) and not context.retrain:
        network = utils.load_network(context.data_file_path1)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_CL_mnist: pid: %i loaded phase1 network history from %s' %
                  (os.getpid(), context.data_file_path1))
    else:
        data_generator.manual_seed(data_seed)
        if context.debug:
            import time
            current_time = time.time()
        network.train(phase1_train_dataloader, phase1_val_dataloader, epochs=epochs,
                      val_interval=context.val_interval,  # e.g. (-201, -1, 10)
                      samples_per_epoch=context.train_steps1,
                      store_history=context.store_history, store_dynamics=context.store_dynamics,
                      store_history_interval=context.store_history_interval,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar, debug=context.debug)