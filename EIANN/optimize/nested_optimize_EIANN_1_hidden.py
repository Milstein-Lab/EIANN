import torch
from torch.utils.data import DataLoader
import os, sys
from copy import deepcopy
import numpy as np
import h5py
import math

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_unsupervised_by_best_epoch, check_equilibration_dynamics
from EIANN.plot import plot_simple_EIANN_config_summary, plot_simple_EIANN_weight_history_diagnostic
from nested.utils import Context, param_array_to_dict, str_to_bool
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
    if 'export_network_config_file_path' not in context():
        context.export_network_config_file_path = None
    if 'eval_accuracy' not in context():
        context.eval_accuracy = 'final'
    else:
        context.eval_accuracy = str(context.eval_accuracy)
    if 'store_weights' not in context():
        context.store_weights = False
    else:
        context.store_weights = str_to_bool(context.store_weights)
    if 'constrain_equilibration_dynamics' not in context():
        context.constrain_equilibration_dynamics = True
    else:
        context.constrain_equilibration_dynamics = str_to_bool(context.constrain_equilibration_dynamics)
    if 'equilibration_activity_tolerance' not in context():
        context.equilibration_activity_tolerance = 0.2
    else:
        context.equilibration_activity_tolerance = float(context.equilibration_activity_tolerance)

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    context.input_size = 21
    context.dataset = torch.eye(context.input_size)
    context.target = torch.eye(context.dataset.shape[0])

    context.sample_indexes = torch.arange(len(context.dataset))
    context.data_generator = torch.Generator()
    autoenc_train_data = list(zip(context.sample_indexes, context.dataset, context.target))
    context.dataloader = DataLoader(autoenc_train_data, shuffle=True, generator=context.data_generator)
    context.test_dataloader = DataLoader(autoenc_train_data, batch_size=len(context.dataset), shuffle=False)


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print(network_seeds, data_seeds)
        sys.stdout.flush()
    return [network_seeds, data_seeds]


def update_EIANN_config_1_hidden_backprop_relu_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    learning_rate = param_dict['learning_rate']

    context.training_kwargs['optimizer'] = 'SGD'
    context.training_kwargs['learning_rate'] = learning_rate


def update_EIANN_config_1_hidden_backprop_softplus_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    learning_rate = param_dict['learning_rate']
    softplus_beta = param_dict['softplus_beta']

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.training_kwargs['optimizer'] = 'SGD'
    context.training_kwargs['learning_rate'] = learning_rate


def update_EIANN_config_1_hidden_bpDale_softplus_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

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


def update_EIANN_config_1_hidden_bpDale_1_inh_static_softplus_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']

    E_E_learning_rate = param_dict['E_E_learning_rate']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                                       math.sqrt(context.layer_config['Output']['E']['size'])

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)
    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_1_hidden_Gjorgjieva_Hebb_A(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']
    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale']
    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale']
    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale']
    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale


def update_EIANN_config_1_hidden_Gjorgjieva_Hebb_C(x, context):
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


def update_EIANN_config_2_hidden_Gjorgjieva_Hebb_C(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H_FBI_size = int(param_dict['H_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    context.layer_config['H1']['FBI']['size'] = H_FBI_size
    context.layer_config['H2']['FBI']['size'] = H_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H2_E_H1_E_weight_scale = param_dict['H2_E_H1_E_weight_scale'] * \
                                math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    H2_E_H2_FBI_weight_scale = param_dict['H2_E_H2_FBI_weight_scale'] * \
                               math.sqrt(context.layer_config['H2']['FBI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H2_FBI_H2_E_weight_scale = param_dict['H2_FBI_H2_E_weight_scale'] * \
                               math.sqrt(context.layer_config['H2']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['FBI']['size']) / 2
    H2_FBI_H2_FBI_weight_scale = param_dict['H2_FBI_H2_FBI_weight_scale'] * \
                                 math.sqrt(context.layer_config['H2']['FBI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    Output_E_H2_E_weight_scale = param_dict['Output_E_H2_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H2']['E']['size']) / 2
    Output_E_E_learning_rate = param_dict['Output_E_E_learning_rate']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['FBI']['size']) / 2
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale'] * \
                                         math.sqrt(context.layer_config['Output']['FBI']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate

    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H2']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = H2_E_H1_E_weight_scale
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate

    context.projection_config['H2']['E']['H2']['FBI']['weight_constraint_kwargs']['scale'] = H2_E_H2_FBI_weight_scale
    context.projection_config['H2']['E']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H2']['FBI']['H2']['E']['weight_constraint_kwargs']['scale'] = H2_FBI_H2_E_weight_scale
    context.projection_config['H2']['FBI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H2']['FBI']['H2']['FBI']['weight_constraint_kwargs']['scale'] = \
        H2_FBI_H2_FBI_weight_scale
    context.projection_config['H2']['FBI']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H2']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H2_E_weight_scale
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_E_learning_rate

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


def update_EIANN_config_1_hidden_1_inh_Gjorgjieva_Hebb(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    E_E_learning_rate = param_dict['E_E_learning_rate']
    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale']
    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale']
    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)


def update_EIANN_config_1_hidden_Gjorgjieva_Hebb_1_inh_static_C(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                                       math.sqrt(context.layer_config['Output']['E']['size'])

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)
    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)


def update_EIANN_config_1_hidden_Gjorgjieva_anti_Hebb_A(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']
    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale']
    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale']
    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale']
    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['weight_bounds'] = (None, context.I_floor_weight)
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_bounds'] = (None, context.I_floor_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['weight_bounds'] = (None, context.I_floor_weight)
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_bounds'] = (None, context.I_floor_weight)


def update_EIANN_config_1_hidden_Gjorgjieva_anti_Hebb_B(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']
    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale']
    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale']
    Output_FBI_Output_FBI_weight_scale = param_dict['Output_FBI_Output_FBI_weight_scale']
    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale']
    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale']
    H1_FBI_H1_FBI_weight_scale = param_dict['H1_FBI_H1_FBI_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_constraint_kwargs']['scale'] = H1_E_H1_FBI_weight_scale
    context.projection_config['H1']['E']['H1']['FBI']['weight_bounds'] = (None, context.I_floor_weight)
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_FBI_H1_E_weight_scale
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate
    context.projection_config['H1']['FBI']['H1']['FBI']['weight_constraint_kwargs']['scale'] = \
        H1_FBI_H1_FBI_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_FBI_weight_scale
    context.projection_config['Output']['E']['Output']['FBI']['weight_bounds'] = (None, context.I_floor_weight)
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_E_weight_scale
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['weight_constraint_kwargs']['scale'] = \
        Output_FBI_Output_FBI_weight_scale


def update_EIANN_config_1_hidden_BTSP_A(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_max_weight = param_dict['H1_E_Input_E_max_weight']
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                   context.layer_config['Input']['E']['size']
    H1_E_BTSP_learning_rate = param_dict['H1_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight = param_dict['H1_E_Output_E_max_weight']
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']

    H1_E_H1_FBI_weight = param_dict['H1_E_H1_FBI_weight']
    H1_Dend_I_H1_E_weight = param_dict['H1_Dend_I_H1_E_weight']
    H1_E_H1_Dend_I_init_weight = param_dict['H1_E_H1_Dend_I_init_weight']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_BTSP_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (FBI_E_weight,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_Dend_I_H1_E_weight,)

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


def update_EIANN_config_1_hidden_BTSP_B(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H1_E_Input_E_max_weight = param_dict['H1_E_Input_E_max_weight']
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                   context.layer_config['Input']['E']['size']
    H1_E_BTSP_learning_rate = param_dict['H1_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight = param_dict['H1_E_Output_E_max_weight']
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['Output']['E']['size']

    H1_E_H1_FBI_weight = param_dict['H1_E_H1_FBI_weight']
    H1_Dend_I_H1_E_weight = param_dict['H1_Dend_I_H1_E_weight']
    H1_E_H1_Dend_I_init_weight = param_dict['H1_E_H1_Dend_I_init_weight']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_BTSP_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_Dend_I_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_BTSP_learning_rate

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (FBI_E_weight,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_Dend_I_H1_E_weight,)

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


def update_EIANN_config_1_hidden_BTSP_C(x, context):
    """
    1 static somatic interneuron; 1 dendritic interneuron with learned I -> E_Dend
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
    H1_Dend_I_H1_E_weight = param_dict['H1_Dend_I_H1_E_weight']
    H1_E_H1_Dend_I_init_weight = param_dict['H1_E_H1_Dend_I_init_weight']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    context.layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (0, H1_E_Input_E_max_init_weight)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight,)
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
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_Dend_I_H1_E_weight,)

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


def update_EIANN_config_1_hidden_BTSP_C2(x, context):
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


def update_EIANN_config_1_hidden_BTSP_C4(x, context):
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
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                               math.sqrt(context.layer_config['Output']['E']['size'])

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


def update_EIANN_config_1_hidden_BTSP_E1(x, context):
    """
    This config has 1 static soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in H1.
    Only E_Dend_I is learned. Inits are half-kaining with parameterized scale or _fill. E cells use the
    DendriticBiasLearning rule.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_bias_learning_rate = param_dict['H1_E_bias_learning_rate']
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
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']
    H1_E_H1_Dend_I_learning_rate = param_dict['H1_E_H1_Dend_I_learning_rate']

    Output_E_bias_learning_rate = param_dict['Output_E_bias_learning_rate']
    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                               math.sqrt(context.layer_config['Output']['E']['size'])

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.layer_config['H1']['E']['bias_learning_rule_kwargs']['learning_rate'] = H1_E_bias_learning_rate
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

    context.layer_config['Output']['E']['bias_learning_rule_kwargs']['learning_rate'] = Output_E_bias_learning_rate
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


def update_EIANN_config_1_hidden_BTSP_D2(x, context):
    """
    This config has 1 soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in H1.
    E_Dend_I is learned with the DendriticLoss rule. E_FBI and FBI_E are learned with backprop.
    Inits are half-kaining with parameterized scale or _fill.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

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

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])
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
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_weight_scale,)
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
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
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
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
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate


def update_EIANN_config_1_hidden_BTSP_Clone_Dend_I_1(x, context):
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
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                               math.sqrt(context.layer_config['Output']['E']['size'])

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


def update_EIANN_config_1_hidden_BTSP_D(x, context):
    """
    1 static somatic interneuron; 7 hidden dendritic interneurons with learned E_Dend <- I, I <- E and I <- I
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

    target = context.target
    data_generator = context.data_generator
    dataloader = context.dataloader
    test_dataloader = context.test_dataloader
    epochs = context.epochs

    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if plot:
        network.test(dataloader, store_history=True, status_bar=context.status_bar)
        plot_simple_EIANN_config_summary(network, num_samples=len(dataloader), label='Initial')
        network.reset_history()

    # if context.debug:
    #     context.update(locals())
    #     return dict()

    data_generator.manual_seed(data_seed)
    network.train(dataloader, epochs, store_history=True, store_weights=context.store_weights,
                  status_bar=context.status_bar)

    if context.constrain_equilibration_dynamics or context.debug:
        if not check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance,
                                            context.debug, context.disp, context.debug and plot):
            if not context.debug:
                return dict()

    if not context.supervised:
        #TODO: this should depend on value of eval_accuracy
        sorted_idx = sort_unsupervised_by_best_epoch(network, target, plot=plot)
    else:
        sorted_idx = torch.arange(0, len(target))

    best_epoch_index, epoch_loss, epoch_argmax_accuracy = \
        analyze_simple_EIANN_epoch_loss_and_accuracy(network, target, sorted_output_idx=sorted_idx, plot=plot)

    if best_epoch_index + context.num_epochs_argmax_accuracy > epochs:
        best_argmax_accuracy = torch.mean(epoch_argmax_accuracy[-context.num_epochs_argmax_accuracy:])
        best_epoch_loss = torch.mean(epoch_loss[-context.num_epochs_argmax_accuracy:])
    else:
        best_argmax_accuracy = \
            torch.mean(epoch_argmax_accuracy[best_epoch_index:best_epoch_index + context.num_epochs_argmax_accuracy])
        best_epoch_loss = torch.mean(epoch_loss[best_epoch_index:best_epoch_index + context.num_epochs_argmax_accuracy])
    final_epoch_loss = torch.mean(epoch_loss[-context.num_epochs_argmax_accuracy:])
    final_argmax_accuracy = torch.mean(epoch_argmax_accuracy[-context.num_epochs_argmax_accuracy:])

    if context.eval_accuracy == 'final':
        start_index = (epochs - 1) * len(dataloader)
        results = {'loss': final_epoch_loss,
                   'accuracy': final_argmax_accuracy}
    elif context.eval_accuracy == 'best':
        start_index = best_epoch_index * len(dataloader)
        results = {'loss': best_epoch_loss,
                   'accuracy': best_argmax_accuracy}
    else:
        raise Exception('nested_optimize_EIANN_1_hidden: eval_accuracy must be final or best, not %s' %
                        context.eval_accuracy)
    if plot:
        plot_simple_EIANN_config_summary(network, num_samples=len(dataloader), start_index=start_index,
                                         sorted_output_idx=sorted_idx, label='Final')

    if context.debug:
        print('pid: %i, seed: %i, sample_order: %s, final_output: %s' % (os.getpid(), seed, network.sample_order,
                                                                         network.Output.E.activity))
        context.update(locals())
        #if plot:
        #    plot_simple_EIANN_weight_history_diagnostic(network)

    if export:
        if context.export_network_config_file_path is not None:
            config_dict = {'layer_config': context.layer_config,
                           'projection_config': context.projection_config,
                           'training_kwargs': context.training_kwargs}
            write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
            if context.disp:
                print('nested_optimize_EIANN_1_hidden: pid: %i exported network config to %s' %
                      (os.getpid(), context.export_network_config_file_path))
        if context.temp_output_path is not None:

            end_index = start_index + len(dataloader)
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
                metrics_group.create_dataset('loss', data=epoch_loss)
                metrics_group.create_dataset('accuracy', data=epoch_argmax_accuracy)

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
