import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy
from EIANN.plot import plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history
from nested.utils import Context, param_array_to_dict
from nested.optimize_utils import update_source_contexts
from .nested_optimize_EIANN_1_hidden import update_EIANN_config_1_hidden_Gjorgjieva_Hebb_C


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
    if 'store_history' not in context():
        context.store_history = False
    else:
        context.store_history = bool(context.store_history)
    if 'store_weights' not in context():
        context.store_weights = False
    else:
        context.store_weights = bool(context.store_weights)
    if 'store_weights_interval' not in context():
        context.store_weights_interval = (0, -1, 100)

    context.train_steps = int(context.train_steps)

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
        torch.utils.data.DataLoader(MNIST_train[0:context.train_steps], shuffle=True, generator=context.data_generator)
    context.val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    context.test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)


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

    softplus_beta = param_dict['softplus_beta']
    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_backprop_Dale_softplus_SGD_B(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['FFI']['size'] = H1_I_size
    context.layer_config['H1']['FBI']['size'] = H1_I_size
    context.layer_config['Output']['FFI']['size'] = Output_I_size
    context.layer_config['Output']['FBI']['size'] = Output_I_size

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['FFI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['E']['H1']['FFI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['FFI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['Output']['E']['Output']['FFI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_backprop_Dale_softplus_SGD_C(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['I']['size'] = H1_I_size
    context.layer_config['Output']['I']['size'] = Output_I_size

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_backprop_Dale_softplus_SGD_D(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['FFI']['size'] = H1_I_size
    context.layer_config['Output']['FFI']['size'] = Output_I_size

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['FFI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FFI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FFI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FFI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_backprop_Dale_relu_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_A(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale']
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale']
    I_I_learning_rate = param_dict['I_I_learning_rate']

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale']
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_B(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['FBI']['size']) / 2
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale'] * \
                                         math.sqrt(context.layer_config['Output']['FBI']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_1_inh_A(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FBI_weight = -param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight = param_dict['H1_FBI_H1_E_weight_scale'] / \
                         math.sqrt(context.layer_config['H1']['E']['size']) / 2

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FBI_weight = -param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight = param_dict['Output_FBI_Output_E_weight_scale'] / \
                                 math.sqrt(context.layer_config['Output']['E']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight,)

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (Output_E_Output_FBI_weight,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight,)


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_1_inh_B(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FFI_weight = -param_dict['H1_E_H1_FFI_weight_scale']
    H1_FFI_Input_E_weight = param_dict['H1_FFI_Input_E_weight_scale'] / \
                         math.sqrt(context.layer_config['Input']['E']['size']) / 2

    H1_E_H1_FBI_weight = -param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight = param_dict['H1_FBI_H1_E_weight_scale'] / \
                         math.sqrt(context.layer_config['H1']['E']['size']) / 2

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FFI_weight = -param_dict['Output_E_Output_FFI_weight_scale']
    Output_FFI_H1_E_weight = param_dict['Output_FFI_H1_E_weight_scale'] / \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2

    Output_E_Output_FBI_weight = -param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight = param_dict['Output_FBI_Output_E_weight_scale'] / \
                                 math.sqrt(context.layer_config['Output']['E']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['FFI']['weight_init_args'] = (H1_E_H1_FFI_weight,)
    context.projection_config['H1']['FFI']['Input']['E']['weight_init_args'] = (H1_FFI_Input_E_weight,)

    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight,)

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['FFI']['weight_init_args'] = (Output_E_Output_FFI_weight,)
    context.projection_config['Output']['FFI']['H1']['E']['weight_init_args'] = (Output_FFI_H1_E_weight,)

    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (Output_E_Output_FBI_weight,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight,)


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_1_inh_C(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_I_weight = -param_dict['H1_E_H1_I_weight_scale']
    H1_I_Input_E_weight = param_dict['H1_I_Input_E_weight_scale'] / \
                          math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_I_H1_E_weight = param_dict['H1_I_H1_E_weight_scale'] / \
                       math.sqrt(context.layer_config['H1']['E']['size']) / 2

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_I_weight = -param_dict['Output_E_Output_I_weight_scale']
    Output_I_H1_E_weight = param_dict['Output_I_H1_E_weight_scale'] / \
                           math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_E_weight = param_dict['Output_I_Output_E_weight_scale'] / \
                               math.sqrt(context.layer_config['Output']['E']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_weight,)
    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_weight,)
    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_weight,)

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = (Output_E_Output_I_weight,)
    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = (Output_I_H1_E_weight,)
    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = (Output_I_Output_E_weight,)


def update_EIANN_config_1_hidden_mnist_Gjorgjieva_Hebb_1_inh_D(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FFI_weight = -param_dict['H1_E_H1_FFI_weight_scale']
    H1_FFI_Input_E_weight = param_dict['H1_FFI_Input_E_weight_scale'] / \
                            math.sqrt(context.layer_config['Input']['E']['size']) / 2

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FFI_weight = -param_dict['Output_E_Output_FFI_weight_scale']
    Output_FFI_H1_E_weight = param_dict['Output_FFI_H1_E_weight_scale'] / \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['FFI']['weight_init_args'] = (H1_E_H1_FFI_weight,)
    context.projection_config['H1']['FFI']['Input']['E']['weight_init_args'] = (H1_FFI_Input_E_weight,)

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['FFI']['weight_init_args'] = (Output_E_Output_FFI_weight,)
    context.projection_config['Output']['FFI']['H1']['E']['weight_init_args'] = (Output_FFI_H1_E_weight,)


def update_EIANN_config_1_hidden_mnist_BTSP_C2(x, context):
    """
    This config has 1 static soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in H1.
    Only E_Dend_I is learned. Inits are half-kaining with parameterized scale or _fill.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale']
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']

    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight_scale,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_Dend_I_H1_E_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)


def update_EIANN_config_1_hidden_mnist_BTSP_Clone_Dend_I_1_bad(x, context):
    """
    This config has 1 static soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in H1.
    All incoming and outgoing Dend_I projections are cloned to duplicate activity of Output.E neurons.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_E_H1_FBI_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)


def update_EIANN_config_1_hidden_mnist_BTSP_Clone_Dend_I_1(x, context):
    """
    This config has 1 static soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in H1.
    All incoming and outgoing Dend_I projections are cloned to duplicate activity of Output.E neurons.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)


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
    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale']
    H1_Dend_I_H1_E_learning_rate = param_dict['H1_Dend_I_H1_E_learning_rate']
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']
    H1_Dend_I_H1_Dend_I_weight_scale = param_dict['H1_Dend_I_H1_Dend_I_weight_scale']
    H1_Dend_I_H1_Dend_I_learning_rate = param_dict['H1_Dend_I_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']
    H1_Dend_I_size = int(param_dict['H1_Dend_I_size'])

    context.layer_config['H1']['Dend_I']['size'] = H1_Dend_I_size

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (FBI_E_weight,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_E_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_Dend_I_H1_E_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_Dend_I_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_Dend_I_H1_Dend_I_learning_rate

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


def update_EIANN_config_1_hidden_mnist_BTSP_E(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    H1_Dend_I_size = int(param_dict['H1_Dend_I_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['H1']['Dend_I']['size'] = H1_Dend_I_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    E_FBI_learning_rate = param_dict['E_FBI_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale']
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale']
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']
    H1_Dend_I_H1_Dend_I_weight_scale = param_dict['H1_Dend_I_H1_Dend_I_weight_scale']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_FBI_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_E_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_Dend_I_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (0, Output_E_H1_E_max_init_weight)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_FBI_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_mnist_BTSP_F(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    H1_Dend_I_size = int(param_dict['H1_Dend_I_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['H1']['Dend_I']['size'] = H1_Dend_I_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    E_FBI_learning_rate = param_dict['E_FBI_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']
    H1_Dend_I_H1_Dend_I_weight_scale = param_dict['H1_Dend_I_H1_Dend_I_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['Dend_I']['size']) / 2

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['FBI']['size']) / 2
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['FBI']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_FBI_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_E_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_Dend_I_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (0, Output_E_H1_E_max_init_weight)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_FBI_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_mnist_BTSP_G(x, context):
    """
    Uses BTSP_2 rule.
    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_FBI_size = int(param_dict['H1_FBI_size'])
    H1_Dend_I_size = int(param_dict['H1_Dend_I_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['H1']['Dend_I']['size'] = H1_Dend_I_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    E_FBI_learning_rate = param_dict['E_FBI_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']
    H1_Dend_I_H1_Dend_I_weight_scale = param_dict['H1_Dend_I_H1_Dend_I_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['Dend_I']['size']) / 2

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['FBI']['size']) / 2
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale'] * \
                                math.sqrt(context.layer_config['Output']['FBI']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_FBI_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_E_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['weight_constraint_kwargs']['scale'] = \
        H1_Dend_I_H1_Dend_I_weight_scale
    context.projection_config['H1']['Dend_I']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (0, Output_E_H1_E_max_init_weight)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate

    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_FBI_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


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

    if plot:
        plot_batch_accuracy(network, test_dataloader, title='Initial')

    data_generator.manual_seed(data_seed)
    network.train_and_validate(train_sub_dataloader,
                               val_dataloader,
                               epochs=epochs,
                               val_interval=context.val_interval, # e.g. (-201, -1, 10)
                               store_history=context.store_history,
                               store_weights=context.store_weights,
                               store_weights_interval=context.store_weights_interval,
                               status_bar=context.status_bar)

    # reorder output units if using unsupervised/Hebbian rule
    if not context.supervised:
        min_loss_idx, sorted_output_idx = sort_by_val_history(network, plot=plot)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = torch.arange(0, network.val_output_history.shape[-1])

    sorted_val_loss_history, sorted_val_accuracy_history = \
        recompute_validation_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, store=True, plot=plot)

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

    if torch.isnan(results['loss']):
        return dict()

    if plot:
        plot_batch_accuracy(network, test_dataloader, sorted_output_idx=sorted_output_idx, title='Final')
        plot_train_loss_history(network)
        plot_validate_loss_history(network)

    if context.debug:
        print('pid: %i, seed: %i, network.val_loss_history: %s' % (os.getpid(), seed, network.val_loss_history))
        sys.stdout.flush()
        context.update(locals())

    if export:
        if context.export_network_config_file_path is not None:
            config_dict = {'layer_config': context.layer_config,
                           'projection_config': context.projection_config,
                           'training_kwargs': context.training_kwargs}
            write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
            if context.disp:
                print('nested_optimize_EIANN_1_hidden_mnist: pid: %i exported network config to %s' %
                      (os.getpid(), context.export_network_config_file_path))

        # TODO: refactor
        """
        if context.temp_output_path is not None:

            end_index = start_index + context.num_training_steps_argmax_accuracy_window
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
                            activity_data = activity_data[sorted_output_idx,:]
                        post_layer_activity.create_dataset(post_pop.name, data=activity_data)
                metrics_group.create_dataset('loss', data=sorted_val_loss_history)
                metrics_group.create_dataset('accuracy', data=sorted_val_accuracy_history)
        """

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
