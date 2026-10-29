import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset, Subset
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
from EIANN.optimize.nested_optimize_EIANN_1_hidden_CL_mnist import (get_task_incremental_projection_config,
                                                                    get_mean_forward_dend_loss)
import EIANN.utils as utils


context = Context()

# run a single seed (must be run from the root directory of EIANN):
# python -m nested.analyze --framework=serial \
#   --config-file-path=optimize/optimize_config/cifar10_CL/5_tasks/20261026_nested_optimize_EIANN_convnet_A_CL_cifar10_5_tasks_van_bp.yaml \
#   --output-dir=data --label=van_bp --export --num_instances=1 --store_history=True --retrain=False \
#   --full_analysis=False --status_bar=True

# CL params (same as nested_optimize_EIANN_1_hidden_CL_mnist):
# num_splits (int) determines how many different tasks to split cifar10 into
# train_steps_per_class (int) number of train steps per class in each task (a task with n classes trains for
#   n * train_steps_per_class steps); override from the command line with --train_steps_per_class=<n>
# cumulative_val_set decides whether or not to have val set be cumulative to all seen tasks (default to true)
# task_incremental (bool) trains each task only on its own classes' output units and evaluates each sample only within
#   its own task's output units (default to false). Requires continual-learning rules (subclasses of
#   ContinualLearningMixin)
# multihead (bool) with task_incremental, makes the output layer one independent head per task (default to true)
# CIFAR10 params (same as nested_optimize_EIANN_cifar10):
# flatten_data (bool) flattens each image into a vector; set to False for convnets (default to true)
# val_batch_size (int) evaluates validation and test sets in batches of this size to limit memory (default: full set)


def config_controller():
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = str_to_bool(context.debug)


def get_dataloader(dataset, batch_size=None):
    """
    Returns a non-shuffled DataLoader that evaluates dataset in batches of batch_size (default: a single batch).
    :param dataset: Dataset
    :param batch_size: int
    :return: :class:'DataLoader'
    """
    if batch_size is None:
        batch_size = len(dataset)
    # a trailing batch of size 1 would be squeezed to an unbatched sample
    elif len(dataset) % batch_size == 1:
        raise Exception('nested_optimize_EIANN_CL_cifar10: val_batch_size: %i leaves a final batch of size 1 for a '
                        'dataset of size %i' % (batch_size, len(dataset)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_dataloader_output(network, dataloader):
    """
    Forward a non-shuffled dataloader batch by batch.
    :param network: :class:'Network'
    :param dataloader: :class:'DataLoader'
    :return: tuple of tensor (output, target)
    """
    output = torch.cat([network.forward(data.to(network.device), no_grad=True) for _, data, _ in dataloader])
    target = utils.get_dataloader_targets(dataloader).to(network.device)
    return output, target


def save_network_without_batch_state(network, path):
    """
    Save network without the population states and activities left over from its last forward pass. After validation
    these hold the last val batch (~1 MB per image for convnets, so ~10 GB for 10000 val images). They are restored
    after saving, so training continues unchanged.
    :param network: :class:'Network'
    :param path: str
    """
    cached = {}
    for name, population in network.populations.items():
        cached[name] = {attr: getattr(population, attr) for attr in ('state', 'activity', 'prev_activity',
                                                                     'forward_steps_activity')
                        if hasattr(population, attr)}
        population.reinit(device=network.device, batch_size=1)
        population.prev_activity = population.activity
    try:
        utils.save_network(network, path=path, disp=False)
    finally:
        for name, population in network.populations.items():
            for attr, val in cached[name].items():
                setattr(population, attr, val)


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.epochs = int(context.epochs)
    context.status_bar = str_to_bool(context.status_bar)

    # CL specific #
    context.train_steps_per_class = int(context.train_steps_per_class)
    context.num_splits = int(context.num_splits)

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
    # train steps between stored summaries of the learning rules' lr_mod, stored only when exporting the phase
    # pickles; 0: not stored
    if 'store_lr_mod_interval' not in context():
        context.store_lr_mod_interval = 20
    else:
        context.store_lr_mod_interval = int(context.store_lr_mod_interval)
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
        context.plot_initial = str_to_bool(context.plot_initial)
    if 'include_dend_loss_objective' not in context():
        context.include_dend_loss_objective = False
    else:
        context.include_dend_loss_objective = str_to_bool(context.include_dend_loss_objective)
    if 'include_equilibration_dynamics_objective' not in context():
        context.include_equilibration_dynamics_objective = False
    else:
        context.include_equilibration_dynamics_objective = str_to_bool(context.include_equilibration_dynamics_objective)
    if 'cumulative_val_set' not in context():
        context.cumulative_val_set = True
    else:
        context.cumulative_val_set = str_to_bool(context.cumulative_val_set)
    if 'task_incremental' not in context():
        context.task_incremental = False
    else:
        context.task_incremental = str_to_bool(context.task_incremental)
    if 'multihead' not in context():
        context.multihead = True
    else:
        context.multihead = str_to_bool(context.multihead)
    if context.task_incremental and not context.supervised:
        raise Exception('nested_optimize_EIANN_CL_cifar10: task_incremental is not supported with supervised=False')
    if 'flatten_data' not in context():
        context.flatten_data = True
    else:
        context.flatten_data = str_to_bool(context.flatten_data)
    if 'val_batch_size' not in context() or context.val_batch_size is None:
        context.val_batch_size = None
    else:
        context.val_batch_size = int(context.val_batch_size)
    if 'store_history_interval' not in context():
        context.store_history_interval = None

    history_interval = max(int(context.train_steps_per_class * 10 // context.num_splits * context.epochs / 200), 100)
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

    if 'data_file_paths' not in context():
        context.data_file_paths = []

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
    if context.flatten_data:
        tensor_transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    else:
        tensor_transform = T.ToTensor()
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(root=context.output_dir + '/datasets/CIFAR10_data/',
                                                         train=True, download=download, transform=tensor_transform)
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(root=context.output_dir + '/datasets/CIFAR10_data/',
                                                        train=False, download=download, transform=tensor_transform)

    # split data
    num_classes = len(CIFAR10_train_dataset.classes)
    classes_per_task = num_classes // context.num_splits

    labels_in_tasks = [list(range(t, min(num_classes, t+classes_per_task))) for t in range(0, num_classes, classes_per_task)]
    context.labels_in_tasks = labels_in_tasks
    context.train_steps_per_task = [len(x) * context.train_steps_per_class for x in labels_in_tasks]

    train_datasets = [[] for _ in range(len(labels_in_tasks))]
    val_datasets = [[] for _ in range(len(labels_in_tasks))]
    test_datasets = [[] for _ in range(len(labels_in_tasks))]
    full_test_dataset = []

    # the last 10000 train images are the validation set, as in nested_optimize_EIANN_cifar10
    num_train = len(CIFAR10_train_dataset) - 10000
    for idx, (data, label) in enumerate(CIFAR10_train_dataset):
        target = torch.eye(num_classes)[label]
        task_membership = [i for i, labels in enumerate(labels_in_tasks) if label in labels][0]

        if idx < num_train:
            train_datasets[task_membership].append((idx, data, target))
        else:
            val_datasets[task_membership].append((idx, data, target))

    for idx, (data, label) in enumerate(CIFAR10_test_dataset):
        target = torch.eye(num_classes)[label]
        task_membership = [i for i, labels in enumerate(labels_in_tasks) if label in labels][0]

        full_test_dataset.append((idx, data, target))
        test_datasets[task_membership].append((idx, data, target))


    context.train_dataloaders = []
    context.val_dataloaders = []
    context.test_dataloaders = []

    # put data into dataloaders
    context.data_generator = torch.Generator()

    if not context.cumulative_val_set:
        full_val_dataset = ConcatDataset(val_datasets)
        context.full_val_dataloader = get_dataloader(full_val_dataset, context.val_batch_size)
    else:
        original_datasets = list(val_datasets)
        val_datasets = [ConcatDataset(original_datasets[:i + 1]) for i in range(len(original_datasets))]

    for task_train, task_val, task_test in zip(train_datasets, val_datasets, test_datasets):
        context.train_dataloaders.append(DataLoader(task_train, shuffle=True, generator=context.data_generator))
        context.val_dataloaders.append(get_dataloader(task_val, context.val_batch_size))
        context.test_dataloaders.append(get_dataloader(task_test, context.val_batch_size))

    context.full_test_dataloader = get_dataloader(full_test_dataset, context.val_batch_size)


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
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
    if context.task_incremental:
        projection_config = get_task_incremental_projection_config(context.projection_config, context.labels_in_tasks,
                                                                   multihead=context.multihead)
    else:
        projection_config = context.projection_config

    data_generator = context.data_generator
    full_test_dataloader = context.full_test_dataloader
    if context.flatten_data:
        plot_population = 'all'
    else:
        plot_population = 'OutputE'

    epochs = context.epochs

    network = Network(context.layer_config, projection_config, seed=seed, **context.training_kwargs)

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': projection_config,
                       'training_kwargs': context.training_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        if context.disp:
            print('nested_optimize_EIANN_CL_cifar10: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))

    if plot:
        try:
            network.Output.E.H1.E.initial_weight = network.Output.E.H1.E.weight.data.detach().clone()
            network.H1.E.Output.E.initial_weight = network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        if context.plot_initial:
            title = 'Initial (%i, %i)' % (seed, data_seed)
            plot_batch_accuracy(network, full_test_dataloader, population=plot_population, title=title)

    results = {}
    for i, (train_steps, train_loader, val_loader, test_loader) in enumerate(zip(context.train_steps_per_task, context.train_dataloaders, context.val_dataloaders, context.test_dataloaders)):
        if f'data_file_path{i}' not in context():
            network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
            if not os.path.exists(f"{context.output_dir}/{network_name}/{seed}"):
                os.makedirs(f"{context.output_dir}/{network_name}/{seed}", exist_ok=True)
            if context.label is None:
                context.data_file_paths.append(f"{context.output_dir}/{network_name}/{seed}/{network_name}_phase{i}_{seed}_{data_seed}.pkl")
            else:
                context.data_file_paths.append(f"{context.output_dir}/{network_name}/{seed}/{network_name}_phase{i}_{seed}_{data_seed}_{context.label}.pkl")

        if context.debug:
            output_weights_before = {projection.name: projection.weight.detach().cpu().clone()
                                     for projection in network.output_pop}

        if os.path.exists(context.data_file_paths[-1]) and not context.retrain:
            network = utils.load_network(context.data_file_paths[-1])
            if context.disp:
                print('nested_optimize_EIANN_CL_cifar10: pid: %i loaded phase%i network history from %s' %
                    (os.getpid(), i, context.data_file_paths[-1]))
        else:
            data_generator.manual_seed(data_seed)
            if context.debug:
                import time
                current_time = time.time()
            network.train(train_loader, val_loader, epochs=epochs,
                        val_interval=context.val_interval,  # e.g. (-201, -1, 10)
                        samples_per_epoch=train_steps,
                        store_history=context.store_history, store_dynamics=context.store_dynamics,
                        store_history_interval=context.store_history_interval,
                        store_params=context.store_params, store_params_interval=context.store_params_interval,
                        store_lr_mod_interval=(context.store_lr_mod_interval or None) if export else None,
                        store_val_output_history=context.task_incremental, status_bar=context.status_bar)

        # reorder output units if using unsupervised/Hebbian rule
        if not context.supervised:
            if context.eval_accuracy == 'final':
                min_loss_idx = len(network.val_loss_history) - 1
                sorted_output_idx = sort_by_class_averaged_val_output(network, val_loader)
            elif context.eval_accuracy == 'best':
                min_loss_idx, sorted_output_idx = sort_by_val_history(network, val_loader, plot=plot)
            else:
                raise Exception('nested_optimize_EIANN_CL_cifar10: eval_accuracy must be final or best, not %s' %
                                context.eval_accuracy)
            sorted_val_loss_history, sorted_val_accuracy_history = \
                recompute_validation_loss_and_accuracy(network, val_loader, sorted_output_idx=sorted_output_idx,
                                                    store=True)
        elif context.task_incremental:
            # score each val sample only within its own task's output units
            sorted_output_idx = None
            val_target = utils.get_dataloader_targets(val_loader).to(network.device)
            task_incremental_val_history = \
                [utils.compute_task_incremental_loss_and_accuracy(val_output, val_target, context.labels_in_tasks,
                                                                  network.criterion)
                 for val_output in network.val_output_history]
            sorted_val_loss_history = torch.tensor([loss for loss, _ in task_incremental_val_history])
            sorted_val_accuracy_history = torch.tensor([accuracy for _, accuracy in task_incremental_val_history])
            min_loss_idx = torch.argmin(sorted_val_loss_history)
        else:
            min_loss_idx = torch.argmin(network.val_loss_history)
            sorted_output_idx = None
            sorted_val_loss_history = network.val_loss_history
            sorted_val_accuracy_history = network.val_accuracy_history

        if context.debug:
            print(f'\n===== debug: end of phase {i} (task_incremental={context.task_incremental}, '
                  f'cumulative_val_set={context.cumulative_val_set}) =====')
            label_str = '' if context.label is None else f'_{context.label}'
            # report_phase_debug expects a single-batch val_loader
            utils.report_phase_debug(network, i, get_dataloader(val_loader.dataset), context.labels_in_tasks,
                                     output_weights_before=output_weights_before,
                                     fig_path=f'{context.output_dir}/debug_plots/phase{i}_{seed}{label_str}.png')

        if context.store_history and (context.store_history_interval is None):
            binned_train_loss_steps, sorted_train_loss_history, sorted_train_accuracy_history = \
                recompute_train_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=plot, title=f'Phase {i}')

        # Select for stability by computing mean accuracy in a window after the best validation step
        val_stepsize = int(context.val_interval[2])
        num_val_steps_accuracy_window = int(context.num_training_steps_accuracy_window) // val_stepsize

        if context.eval_accuracy == 'final':
            final_loss = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
            final_argmax_accuracy = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])

            results[f'phase{i}_loss'] = final_loss
            results[f'phase{i}_accuracy'] = final_argmax_accuracy

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

            results[f'phase{i}_loss'] = best_loss_window
            results[f'phase{i}_accuracy'] = best_accuracy_window

        else:
            raise Exception('nested_optimize_EIANN_CL_cifar10: eval_accuracy must be final or best, not %s' %
                            context.eval_accuracy)

        if torch.isnan(results[f'phase{i}_loss']) or np.isinf(results[f'phase{i}_loss']):
            if context.debug and context.interactive:
                context.update(locals())
            return dict()

        if context.include_dend_loss_objective:
            if context.store_history_interval is None:
                dend_loss_window = int(context.num_training_steps_accuracy_window)
            else:
                dend_loss_window = num_val_steps_accuracy_window
            mean_forward_dend_loss = get_mean_forward_dend_loss(network, dend_loss_window)
            results[f'phase{i}_mean_forward_dend_loss'] = mean_forward_dend_loss

        if plot:
            plot_batch_accuracy(network, full_test_dataloader, population=plot_population,
                                sorted_output_idx=sorted_output_idx, title=f'After Phase {i}')
            plot_train_loss_history(network, title=f'Phase {i}')
            plot_validate_loss_history(network, title=f'Phase {i}')

        if context.full_analysis:
            test_loss_history, test_accuracy_history = \
                compute_test_loss_and_accuracy_history(network, test_loader,
                                                    sorted_output_idx=sorted_output_idx, plot=plot,
                                                    status_bar=context.status_bar)

        if context.constrain_equilibration_dynamics or context.debug:
            residuals = check_equilibration_dynamics(network, test_loader, context.equilibration_activity_tolerance,
                                                    store_num_steps=context.store_num_steps, disp=context.disp, plot=plot)
            if context.include_equilibration_dynamics_objective:
                results[f'phase{i}_dynamics_residuals'] = residuals
            elif residuals > 0. and not context.debug:
                if context.interactive:
                    context.update(locals())
                return dict()

        if export:
            save_network_without_batch_state(network, context.data_file_paths[-1])
            if context.disp:
                print('nested_optimize_EIANN_CL_cifar10: pid: %i exported phase%i network history to %s' %
                    (os.getpid(), i, context.data_file_paths[-1]))

        network.reset_history()

        network.update_CL_states()
        if context.task_incremental and context.multihead:
            # re-apply weight constraints so the task masks switch to the next task's head before its first train step
            network.constrain_weights_and_biases()

    if context.constrain_equilibration_dynamics or context.debug:
        residuals = check_equilibration_dynamics(network, full_test_dataloader,
                                                 context.equilibration_activity_tolerance,
                                                 store_num_steps=context.store_num_steps, disp=context.disp,
                                                 plot=plot)
        if context.include_equilibration_dynamics_objective:
            results['final_dynamics_residuals'] = residuals
        elif residuals > 0. and not context.debug:
            if context.interactive:
                context.update(locals())
            return dict()

    if not context.cumulative_val_set:
        full_val_output, full_val_target = get_dataloader_output(network, context.full_val_dataloader)
        if context.task_incremental:
            final_total_loss, final_total_accuracy = (
                utils.compute_task_incremental_loss_and_accuracy(full_val_output, full_val_target,
                                                                 context.labels_in_tasks, network.criterion))
        else:
            if sorted_output_idx is not None:
                full_val_output = full_val_output[:, sorted_output_idx]
            final_total_loss = network.criterion(full_val_output, full_val_target).item()
            final_total_accuracy = (100 * torch.sum(torch.argmax(full_val_output, dim=1) ==
                                                    torch.argmax(full_val_target, dim=1)) /
                                    full_val_output.shape[0]).item()
        results['final_loss'] = final_total_loss
        results['final_accuracy'] = final_total_accuracy

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
        elif 'mean_forward_dend_loss' in key:
            objectives[key] = np.abs(val)
        else:
            objectives[key] = val
    return features, objectives
