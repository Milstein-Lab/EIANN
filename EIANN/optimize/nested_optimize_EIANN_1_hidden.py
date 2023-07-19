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


def update_EIANN_config_1_hidden_backprop_Dale_softplus_SGD_F(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']

    context.layer_config['H1']['I']['size'] = H_I_size
    context.layer_config['Output']['I']['size'] = Output_I_size

    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    Output_E_H1_E_init_weight_scale = param_dict['Output_E_H1_E_init_weight_scale']
    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['I']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate
    context.projection_config['H1']['I']['H1']['I']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = \
        (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate
    context.projection_config['Output']['I']['Output']['I']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)

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


def update_EIANN_config_1_hidden_Gjorgjieva_Hebb_F(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['I']['size'] = H1_I_size
    context.layer_config['Output']['I']['size'] = Output_I_size

    H1_E_Input_E_weight_scale = param_dict['H1_E_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']

    H1_E_H1_I_weight_scale = param_dict['H1_E_H1_I_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['I']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_I_H1_E_weight_scale = param_dict['H1_I_H1_E_weight_scale'] * \
                               math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_I_Input_E_weight_scale = param_dict['H1_I_Input_E_weight_scale'] * \
                             math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_I_H1_I_weight_scale = param_dict['H1_I_H1_I_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['I']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['I']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                       math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                         math.sqrt(context.layer_config['Output']['I']['size']) / 2

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['I']['weight_constraint_kwargs']['scale'] = H1_E_H1_I_weight_scale
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['I']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_I_Input_E_weight_scale
    context.projection_config['H1']['I']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_I_H1_E_weight_scale
    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['H1']['I']['weight_constraint_kwargs']['scale'] = \
        H1_I_H1_I_weight_scale
    context.projection_config['H1']['I']['H1']['I']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['I']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['I']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['I']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['Output']['I']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['I']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
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


def update_EIANN_config_1_hidden_BTSP_F1(x, context):
    """
    H1.SomaI and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule. H1.DendI clones weights from H1.SomaI.
    H1.E.H1.DendI weights are learned with the DendriticLoss_3 rule.
    Inits are half-kaining with parameterized scale.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H1_I_size
    context.layer_config['H1']['DendI']['size'] = H1_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

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

    H1_E_H1_SomaI_weight_scale = param_dict['H1_E_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_I_H1_E_weight_scale = param_dict['H1_I_H1_E_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_I_Input_E_weight_scale = param_dict['H1_I_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_I_H1_I_weight_scale = param_dict['H1_I_H1_I_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H1_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['E']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_E_H1_SomaI_weight_scale
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_constraint_kwargs']['scale'] = \
        H1_I_Input_E_weight_scale
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_I_H1_E_weight_scale
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_I_H1_I_weight_scale
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

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

    context.projection_config['Output']['E']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_BTSP_F2(x, context):
    """
    H1.SomaI and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule. H1.DendI clones weights from H1.SomaI.
    H1.E.H1.DendI weights are learned with the DendriticLoss_3 rule.
    Inits are half-kaining with parameterized scale.
    E<-E weights are learned with the BTSP_5 rule.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H1_I_size
    context.layer_config['H1']['DendI']['size'] = H1_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_weight_scale = param_dict['H1_E_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_I_H1_E_weight_scale = param_dict['H1_I_H1_E_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_I_Input_E_weight_scale = param_dict['H1_I_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_I_H1_I_weight_scale = param_dict['H1_I_H1_I_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2

    BTSP_temporal_discount = param_dict['BTSP_temporal_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['temporal_discount'] = \
        BTSP_temporal_discount

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['temporal_discount'] = \
        BTSP_temporal_discount

    context.projection_config['H1']['E']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_E_H1_SomaI_weight_scale
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_constraint_kwargs']['scale'] = \
        H1_I_Input_E_weight_scale
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_I_H1_E_weight_scale
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_I_H1_I_weight_scale
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['temporal_discount'] = \
        BTSP_temporal_discount

    context.projection_config['Output']['E']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_BTSP_F3(x, context):
    """
    H1.SomaI and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule. H1.DendI clones weights from H1.SomaI.
    H1.E.H1.DendI weights are learned with the DendriticLoss_4 rule.
    E<-E weights are learned with the BTSP_6 rule.
    Inits are half-kaining with parameterized scale.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H1_I_size
    context.layer_config['H1']['DendI']['size'] = H1_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_weight_scale = param_dict['H1_E_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_I_H1_E_weight_scale = param_dict['H1_I_H1_E_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_I_Input_E_weight_scale = param_dict['H1_I_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_I_H1_I_weight_scale = param_dict['H1_I_H1_I_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2

    BTSP_decay_tau = param_dict['BTSP_decay_tau']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['H1']['E']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_E_H1_SomaI_weight_scale
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_constraint_kwargs']['scale'] = \
        H1_I_Input_E_weight_scale
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = H1_I_H1_E_weight_scale
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_I_H1_I_weight_scale
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['Output']['E']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_BTSP_F5(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    E<-E weights are learned with the BTSP_7 rule.
    Inits are half-kaining with parameterized scale.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H1_I_size
    context.layer_config['H1']['DendI']['size'] = H1_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_weight_scale = param_dict['H1_E_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_SomaI_H1_E_weight_scale = param_dict['H1_SomaI_H1_E_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_SomaI_Input_E_weight_scale = param_dict['H1_SomaI_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_SomaI_H1_SomaI_weight_scale = param_dict['H1_SomaI_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_DendI_H1_E_weight_scale = param_dict['H1_DendI_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_DendI_H1_DendI_weight_scale = param_dict['H1_DendI_H1_DendI_weight_scale'] * \
                                     math.sqrt(context.layer_config['H1']['DendI']['size']) / 2
    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2

    BTSP_decay_tau = param_dict['BTSP_decay_tau']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['H1']['E']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_E_H1_SomaI_weight_scale
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_Input_E_weight_scale
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_H1_E_weight_scale
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_H1_SomaI_weight_scale
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H1']['DendI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_DendI_H1_E_weight_scale
    context.projection_config['H1']['DendI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['DendI']['H1']['DendI']['weight_constraint_kwargs']['scale'] = \
        H1_DendI_H1_DendI_weight_scale
    context.projection_config['H1']['DendI']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau

    context.projection_config['Output']['E']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate


def update_EIANN_config_1_hidden_BTSP_F6(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    E<-E weights are learned with the BTSP_8 rule.
    Inits are half-kaining with parameterized scale.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)
    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H1_I_size
    context.layer_config['H1']['DendI']['size'] = H1_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']
    BTSP_anti_hebb_th = param_dict['BTSP_anti_hebb_th']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_weight_scale = param_dict['H1_E_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    E_I_learning_rate = param_dict['E_I_learning_rate']
    H1_SomaI_H1_E_weight_scale = param_dict['H1_SomaI_H1_E_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_SomaI_Input_E_weight_scale = param_dict['H1_SomaI_Input_E_weight_scale'] * \
                                math.sqrt(context.layer_config['Input']['E']['size']) / 2
    I_E_learning_rate = param_dict['I_E_learning_rate']
    H1_SomaI_H1_SomaI_weight_scale = param_dict['H1_SomaI_H1_SomaI_weight_scale'] * \
                             math.sqrt(context.layer_config['H1']['SomaI']['size']) / 2
    I_I_learning_rate = param_dict['I_I_learning_rate']

    H1_DendI_H1_E_weight_scale = param_dict['H1_DendI_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    H1_DendI_H1_DendI_weight_scale = param_dict['H1_DendI_H1_DendI_weight_scale'] * \
                                     math.sqrt(context.layer_config['H1']['DendI']['size']) / 2
    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_weight_scale = param_dict['Output_E_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2
    Output_I_Output_E_weight_scale = param_dict['Output_I_Output_E_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['E']['size']) / 2
    Output_I_H1_E_weight_scale = param_dict['Output_I_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_I_Output_I_weight_scale = param_dict['Output_I_Output_I_weight_scale'] * \
                                     math.sqrt(context.layer_config['Output']['SomaI']['size']) / 2

    BTSP_decay_tau = param_dict['BTSP_decay_tau']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_th'] = BTSP_anti_hebb_th

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_th'] = BTSP_anti_hebb_th

    context.projection_config['H1']['E']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_E_H1_SomaI_weight_scale
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_Input_E_weight_scale
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_H1_E_weight_scale
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_constraint_kwargs']['scale'] = \
        H1_SomaI_H1_SomaI_weight_scale
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H1']['DendI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        H1_DendI_H1_E_weight_scale
    context.projection_config['H1']['DendI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['DendI']['H1']['DendI']['weight_constraint_kwargs']['scale'] = \
        H1_DendI_H1_DendI_weight_scale
    context.projection_config['H1']['DendI']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_th'] = BTSP_anti_hebb_th

    context.projection_config['Output']['E']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_E_Output_I_weight_scale
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate

    context.projection_config['Output']['SomaI']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_H1_E_weight_scale
    context.projection_config['Output']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_E_weight_scale
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_constraint_kwargs']['scale'] = \
        Output_I_Output_I_weight_scale
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = \
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
