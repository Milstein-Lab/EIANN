import torch
# torch.use_deterministic_algorithms(mode=True)
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import gc

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history
from EIANN.plot import plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, plot_receptive_fields
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_config_updates import *
import EIANN.utils as utils


context = Context()

# run 5 random seeds in parallel:
# mpirun -n 6 python -m mpi4py.futures -m nested.analyze --framework=mpi \
#   --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_BTSP_config_D1.yaml \
#   --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=BTSP_D1 --output-dir=optimize/data --label=btsp \
#   --export --store_history=True --retrain=False --full_analysis=True --status_bar=True

# mpirun -n 6 python -m mpi4py.futures -m nested.analyze --framework=mpi \
#   --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_bpDale_softplus_SGD_1_inh_config_A.yaml \
#   --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=bpDale_softplus_1_inh_A --output-dir=optimize/data --label=bpDale \
#   --export --export-file-path=multiseed_mnist_metrics.hdf5 --store_history=True --retrain=False --full_analysis=True --status_bar=True

# run a single seed (must be run from the root directory of EIANN):
# python -m nested.analyze --framework=serial \
#   --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_BTSP_config_D1.yaml \
#   --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=BTSP_D1 --output-dir=optimize/data --label=btsp \
#   --export --compute_receptive_fields=False --num_instances=1 --store_history=True --retrain=False --full_analysis=True --status_bar=True

# python -m nested.analyze --framework=serial --config-file-path=optimize/config/mnist/nested_optimize_EIANN_1_hidden_mnist_bpDale_softplus_SGD_1_inh_config_A.yaml --param-file-path=optimize/config/mnist/20230301_nested_optimize_mnist_1_hidden_1_inh_params.yaml --model-key=bpDale_softplus_1_inh_A --output-dir=optimize/data --label=btsp --compute_receptive_fields=False --num_instances=1 --store_history=True --retrain=False --status_bar=True --plot --full_analysis=True


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
        context.export_network_config_file_path = f"{context.output_dir}/{network_name}_optimized.yaml"
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)

    context.train_steps = int(context.train_steps)
    
    if context.full_analysis:
        context.val_interval = (0, -1, 100)
        context.store_params_interval = (0, -1, 100)
        context.store_params = True
        context.store_num_steps = None
    
    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=context.output_dir + '/datasets/MNIST_data/', train=True,
                                                     download=True, transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=context.output_dir + '/datasets/MNIST_data/', train=False,
                                                    download=False, transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    for idx, (data, target) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[target]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx, (data, target) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[target]
        MNIST_test.append((idx, data, target))

    # Put data in dataloader
    context.data_generator = torch.Generator()
    context.train_sub_dataloader = \
        torch.utils.data.DataLoader(MNIST_train[0:-10000], shuffle=True, generator=context.data_generator)
    context.val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    context.test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)


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
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network config to %s' %
                  (os.getpid(), context.export_network_config_file_path))
    
    if plot:
        try:
            network.Output.E.H1.E.initial_weight = network.Output.E.H1.E.weight.data.detach().clone()
            network.H1.E.Output.E.initial_weight = network.H1.E.Output.E.weight.data.detach().clone()
        except:
            pass
        plot_batch_accuracy(network, test_dataloader, population='all', title='Initial')
    
    if 'data_file_path' not in context():
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        if context.label is None:
            context.data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}.pkl"
        else:
            context.data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}_{context.label}.pkl"
    
    if os.path.exists(context.data_file_path) and not context.retrain:
        network.load(context.data_file_path)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i loaded network history from %s' %
                  (os.getpid(), context.data_file_path))
    else:
        data_generator.manual_seed(data_seed)
        network.train(train_sub_dataloader, val_dataloader, epochs=epochs,
                      val_interval=context.val_interval, # e.g. (-201, -1, 10),
                      samples_per_epoch=context.train_steps,
                      store_history=context.store_history, store_dynamics=context.store_dynamics,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar)
        # final_weights = deepcopy(network.Output.E.H1.E.weight.data)
        if export:
            network.save(path=context.data_file_path, disp=False)
            if context.disp:
                print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network history to %s' %
                      (os.getpid(), context.data_file_path))

    if plot:
        try:
            from EIANN.plot import plot_FB_weight_alignment
            plot_FB_weight_alignment(network.Output.E.H1.E, network.H1.E.Output.E)
        except:
            pass

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
    if min_loss_idx + num_val_steps_accuracy_window > len(sorted_val_loss_history): # if best loss too close to the end
        best_accuracy_window = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])
        best_loss_window = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
    else:
        best_accuracy_window = \
            torch.mean(sorted_val_accuracy_history[min_loss_idx:min_loss_idx+num_val_steps_accuracy_window])
        best_loss_window = torch.mean(sorted_val_loss_history[min_loss_idx:min_loss_idx+num_val_steps_accuracy_window])

    final_loss = torch.mean(sorted_val_loss_history[-num_val_steps_accuracy_window:])
    final_argmax_accuracy = torch.mean(sorted_val_accuracy_history[-num_val_steps_accuracy_window:])

    if context.eval_accuracy == 'final':
        results = {'loss': final_loss,
                   'accuracy': final_argmax_accuracy}
    elif context.eval_accuracy == 'best':
        results = {'loss': best_loss_window,
                   'accuracy': best_accuracy_window}
    else:
        raise Exception('nested_optimize_EIANN_1_hidden_mnist: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)

    if torch.isnan(results['loss']):
        if context.debug and context.interactive:
            context.update(locals())
        return dict()
    
    if plot:
        # print('Weights match: %s' % torch.all(final_weights == network.Output.E.H1.E.weight.data))
        plot_batch_accuracy(network, test_dataloader, population='all', sorted_output_idx=sorted_output_idx,
                            title='Final')
        plot_train_loss_history(network)
        plot_validate_loss_history(network)

    if context.compute_receptive_fields:
        # Compute receptive fields
        population = network.H1.E
        receptive_fields, _ = utils.compute_maxact_receptive_fields(population, test_dataloader, sigmoid=False)
        _, activity_preferred_inputs = utils.compute_act_weighted_avg(network.H1.E, test_dataloader)
    else:
        receptive_fields = network.H1.E.Input.E.weight.detach()
        activity_preferred_inputs = None
    if plot:
        plot_receptive_fields(receptive_fields, activity_preferred_inputs, sort=True, num_cols=10, num_rows=10)

    if context.full_analysis:
        metrics_dict = utils.compute_representation_metrics(network.H1.E, test_dataloader, receptive_fields,
                                                            plot=plot)
        test_loss_history, test_accuracy_history = \
            compute_test_loss_and_accuracy_history(network, test_dataloader, sorted_output_idx=sorted_output_idx,
                                                   plot=plot, status_bar=context.status_bar)
    
    if context.constrain_equilibration_dynamics or context.debug:
        if not check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance,
                                            store_num_steps=context.store_num_steps, debug=context.debug,
                                            disp=context.disp, plot=plot):
            if not context.debug:
                if context.interactive:
                    context.update(locals())
                return dict()
    
    # if export:
    #     if context.temp_output_path is not None:
    #         # Compute test activity and metrics
    #         idx, data, target = next(iter(test_dataloader))
    #         network.forward(data)
    #         network.output_pop.activity = network.output_pop.activity[:, sorted_output_idx]
    #
    #         with h5py.File(context.temp_output_path, 'a') as f:
    #             if context.label is None:
    #                 label = str(len(f))
    #             else:
    #                 label = context.label
    #             group = f.create_group(label)
    #             model_group = group.create_group(str(seed))
    #
    #             activity_group = model_group.create_group('activity')
    #             metrics_group = model_group.create_group('metrics')
    #
    #             for layer in network:
    #                 layer_activity = activity_group.create_group(layer.name)
    #                 for pop in layer:
    #                     # can't call .T with dim > 2, check for store_dynamics
    #                     activity_data = pop.activity.T.detach()
    #                     layer_activity.create_dataset(pop.name, data=activity_data)
    #
    #             metrics_group.create_dataset('val_loss', data=sorted_val_loss_history)
    #             metrics_group.create_dataset('val_loss_steps', data=network.val_history_train_steps)
    #             metrics_group.create_dataset('val_accuracy', data=sorted_val_accuracy_history)
    #             if context.store_history:
    #                 metrics_group.create_dataset('train_loss', data=sorted_train_loss_history)
    #                 metrics_group.create_dataset('binned_train_loss_steps', data=binned_train_loss_steps)
    #                 metrics_group.create_dataset('train_accuracy', data=sorted_train_accuracy_history)
    #
    #             if context.full_analysis:
    #                 metrics_group.create_dataset('test_loss', data=test_loss_history)
    #                 metrics_group.create_dataset('test_loss_steps', data=network.param_history_steps)
    #                 metrics_group.create_dataset('test_accuracy', data=test_accuracy_history)
    #                 for metric in metrics_dict:
    #                     metrics_group.create_dataset(metric, data=metrics_dict[metric])

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
