import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys
from copy import deepcopy
import numpy as np
import h5py

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy
from EIANN.plot import plot_simple_EIANN_config_summary, plot_simple_EIANN_weight_history_diagnostic, \
    plot_batch_accuracy
from nested.utils import Context, param_array_to_dict
from nested.optimize_utils import update_source_contexts


context = Context()


def config_controller():
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = bool(context.debug)


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.epochs = int(context.epochs)
    context.status_bar = bool(context.status_bar)
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = bool(context.debug)
    if 'verbose' not in context():
        context.verbose = False
    else:
        context.verbose = bool(context.verbose)
    if 'export_network_config_file_path' not in context():
        context.export_network_config_file_path = None
    if 'eval_accuracy' not in context():
        context.eval_accuracy = 'final'
    else:
        context.eval_accuracy = str(context.eval_accuracy)
    if 'store_weights' not in context():
        context.store_weights = False
    else:
        context.store_weights = bool(context.store_weights)

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print(network_seeds, data_seeds)
        sys.stdout.flush()
    return [network_seeds, data_seeds]


def update_EIANN_config_1_hidden_mnist_backprop_Dale_softplus_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    learning_rate = param_dict['learning_rate']
    softplus_beta = param_dict['softplus_beta']
    H1_FBI_size = int(param_dict['H1_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.training_kwargs['optimizer'] = 'SGD'
    context.training_kwargs['learning_rate'] = learning_rate


def update_EIANN_config_1_hidden_mnist_backprop_Dale_relu_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_size = int(param_dict['H1_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_BTSP_D(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_max_weight = param_dict['H1_E_Input_E_max_weight']
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                   context.layer_config['Input']['E']['size']
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight = param_dict['H1_E_Output_E_max_weight']
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']

    H1_E_H1_FBI_weight = param_dict['H1_E_H1_FBI_weight']
    H1_I_dend_H1_E_weight_scale = param_dict['H1_I_dend_H1_E_weight_scale']
    H1_I_dend_H1_E_learning_rate = param_dict['H1_I_dend_H1_E_learning_rate']
    H1_E_H1_I_dend_init_weight_scale = param_dict['H1_E_H1_I_dend_init_weight_scale']
    H1_E_H1_I_dend_learning_rate = param_dict['H1_E_H1_I_dend_learning_rate']
    H1_I_dend_H1_I_dend_weight_scale = param_dict['H1_I_dend_H1_I_dend_weight_scale']
    H1_I_dend_H1_I_dend_learning_rate = param_dict['H1_I_dend_H1_I_dend_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']
    H1_I_dend_size = int(param_dict['H1_I_dend_size'])

    context.layer_config['H1']['Dend_I']['size'] = H1_I_dend_size

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_I_dend_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_I_dend_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (FBI_E_weight,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_I_dend_H1_E_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_I_dend_H1_E_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['weight_constraint_kwargs']['scale'] = \
        H1_I_dend_H1_I_dend_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_I_dend_H1_I_dend_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (0, Output_E_H1_E_max_init_weight)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (Output_E_Output_FBI_weight,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (FBI_E_weight,)


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

    input_size = 784

    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/', train=True, download=True,
                                                     transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root='../datasets/MNIST_data/',
                                                    train=False, download=False,
                                                    transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    for idx, (data, target) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[target]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx, (data, target) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[target]
        MNIST_test.append((idx, data, target))

    context.train_steps = int(context.train_steps)

    # Put data in dataloader
    data_generator = torch.Generator()
    train_dataloader = torch.utils.data.DataLoader(MNIST_train, shuffle=True, generator=data_generator)
    train_sub_dataloader = \
        torch.utils.data.DataLoader(MNIST_train[0:context.train_steps], shuffle=True, generator=data_generator)
    val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)

    epochs = context.epochs

    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if plot:
        plot_batch_accuracy(network, test_dataloader, title='Initial')

    data_generator.manual_seed(data_seed)
    network.train_and_validate(train_sub_dataloader,
                               val_dataloader,
                               epochs=epochs,
                               val_interval=context.val_interval, # e.g. (-200, -1, 10)
                               store_history=False,
                               store_weights=context.store_weights,
                               status_bar=context.status_bar)

    if not context.supervised: #reorder output units if using unsupervised/Hebbian rule
        min_loss_idx, sorted_output_idx = sort_by_val_history(network, plot=plot)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = torch.arange(0, network.val_output_history.shape[-1])

    sorted_val_loss_history, sorted_val_accuracy_history = \
        recompute_validation_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=plot)

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
        raise Exception('nested_optimize_EIANN_1_hidden: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)
    if plot:
        plot_batch_accuracy(network, test_dataloader, title='Final')

    if context.debug:
        print('pid: %i, seed: %i, sample_order: %s, final_output: %s' % (os.getpid(), seed, network.sample_order,
                                                                         network.Output.E.activity))
        context.update(locals())

    if export:
        # TODO: refactor
        if context.export_network_config_file_path is not None:
            config_dict = {'layer_config': context.layer_config,
                           'projection_config': context.projection_config,
                           'training_kwargs': context.training_kwargs}
            write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
            if context.disp:
                print('nested_optimize_EIANN_1_hidden: pid: %i exported network config to %s' %
                      (os.getpid(), context.export_network_config_file_path))
        if context.temp_output_path is not None:

            end_index = start_index + context.num_training_steps_argmax_accuracy_window
            output_layer = list(network)[-1]
            output_pop = next(iter(output_layer))

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
                            activity_data = activity_data[sorted_output_idx,:]
                        post_layer_activity.create_dataset(post_pop.name, data=activity_data)
                metrics_group.create_dataset('loss', data=sorted_val_loss_history)
                metrics_group.create_dataset('accuracy', data=sorted_val_accuracy_history)

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
