import torch
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import gc

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history
from EIANN.plot import plot_EIANN_1_hidden_autoenc_config_summary, plot_batch_accuracy, plot_train_loss_history, \
    plot_validate_loss_history
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_config_updates import *


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

    input_size = 21
    dataset = torch.eye(input_size)
    target = torch.eye(dataset.shape[0])

    sample_indexes = torch.arange(len(dataset))
    context.data_generator = torch.Generator()
    autoenc_train_data = list(zip(sample_indexes, dataset, target))
    context.train_dataloader = DataLoader(autoenc_train_data, shuffle=True, generator=context.data_generator)
    context.test_dataloader = DataLoader(autoenc_train_data, batch_size=len(dataset), shuffle=False)


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print(network_seeds, data_seeds)
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
    val_dataloader = context.test_dataloader
    test_dataloader = context.test_dataloader
    epochs = context.epochs

    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if plot:
        plot_EIANN_1_hidden_autoenc_config_summary(network, test_dataloader, title='Initial')

    network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
    saved_network_path = f"{context.output_dir}/{network_name}_{seed}.pkl"
    if os.path.exists(saved_network_path) and not context.retrain:
        network.load(saved_network_path)
    else:
        data_generator.manual_seed(data_seed)
        network.train(train_dataloader, val_dataloader, epochs=epochs,
                      val_interval=context.val_interval,  # e.g. (-201, -1, 10)
                      store_history=context.store_history, store_dynamics=context.store_dynamics,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar)
        if export:
            network.save(path=saved_network_path)

    # reorder output units if using unsupervised/Hebbian rule
    if not context.supervised:
        min_loss_idx, sorted_output_idx = sort_by_val_history(network, plot=plot)
        sorted_val_loss_history, sorted_val_accuracy_history = \
            recompute_validation_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, store=True)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = None
        sorted_val_loss_history = network.val_loss_history
        sorted_val_accuracy_history = network.val_accuracy_history

    if context.store_history:
        binned_train_loss_steps, sorted_train_loss_history, sorted_train_accuracy_history = \
            recompute_train_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=plot)

    # Select for stability by computing mean accuracy in a window after the best validation step
    val_stepsize = int(context.val_interval[2])
    val_start_idx = int(context.val_interval[0])
    num_val_steps_accuracy_window = int(context.num_training_steps_accuracy_window) // val_stepsize
    if min_loss_idx + num_val_steps_accuracy_window > len(sorted_val_loss_history):  # if best loss too close to the end
        best_accuracy_window = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])
        best_loss_window = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
    else:
        best_accuracy_window = \
            torch.mean(sorted_val_accuracy_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])
        best_loss_window = torch.mean(
            sorted_val_loss_history[min_loss_idx:min_loss_idx + num_val_steps_accuracy_window])

    final_loss = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
    final_argmax_accuracy = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])

    if context.eval_accuracy == 'final':
        results = {'loss': final_loss,
                   'accuracy': final_argmax_accuracy}
    elif context.eval_accuracy == 'best':
        results = {'loss': best_loss_window,
                   'accuracy': best_accuracy_window}
    else:
        raise Exception('nested_optimize_EIANN_1_hidden: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)

    if torch.isnan(results['loss']):
        return dict()

    if context.constrain_equilibration_dynamics or context.debug:
        if not check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance,
                                            context.debug, context.disp):
            if not context.debug:
                return dict()

    if plot:
        plot_EIANN_1_hidden_autoenc_config_summary(network, test_dataloader, sorted_output_idx=sorted_output_idx,
                            title='Final')
        plot_train_loss_history(network)
        plot_validate_loss_history(network)

    if context.full_analysis:
        test_loss_history, test_accuracy_history = \
            compute_test_loss_and_accuracy_history(network, test_dataloader, sorted_output_idx=sorted_output_idx, plot=plot,
                                           status_bar=context.status_bar)

    if context.export_network_config_file_path is not None:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))

    if export:
        if context.temp_output_path is not None:

            end_index = start_index + len(train_dataloader)
            output_pop = network.output_pop

            with h5py.File(context.temp_output_path, 'a') as f:
                if context.label is None:
                    label = str(len(f))
                else:
                    label = context.label
                group = f.create_group(label)
                model_group = group.create_group(str(seed))
                activity_group = model_group.create_group('activity')
                metrics_group = model_group.create_group('metrics')
                for post_layer in network:
                    post_layer_activity = activity_group.create_group(post_layer.name)
                    for post_pop in post_layer:
                        activity_data = \
                            post_pop.activity_history[network.sorted_sample_indexes, -1, :][start_index:end_index, :].T
                        if post_pop == output_pop:
                            activity_data = activity_data[sorted_idx,:]
                        post_layer_activity.create_dataset(post_pop.name, data=activity_data)
                metrics_group.create_dataset('binned_train_loss_steps', data=binned_train_loss_steps)
                metrics_group.create_dataset('loss', data=sorted_train_loss_history)
                metrics_group.create_dataset('accuracy', data=sorted_train_accuracy_history)

    if not context.interactive:
        del network
        gc.collect()
    elif context.debug:
        context.update(locals())

    return results


def filter_features(primitives, current_features, model_id=None, export=False, plot=False):

    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
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
        else:
            objectives[key] = val
    return features, objectives
