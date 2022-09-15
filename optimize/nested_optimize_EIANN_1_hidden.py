from EIANN import *
from EIANN_utils import *
from nested.utils import Context, param_array_to_dict, read_from_yaml, write_to_yaml
from nested.optimize_utils import update_source_contexts
import os
from copy import deepcopy


context = Context()


def config_worker():
    context.start_instance = int(context.start_instance)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.epochs = int(context.epochs)
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

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']


def get_random_seeds():
    return [[int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
            for instance_id in
             range(context.start_instance, context.start_instance + context.num_instances)]]


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


def update_EIANN_config_1_hidden_Gjorgieva_Hebb_A(x, context):
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


def update_EIANN_config_1_hidden_Gjorgieva_anti_Hebb_A(x, context):
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


def update_EIANN_config_1_hidden_Gjorgieva_anti_Hebb_B(x, context):
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
    H1_I_dend_H1_E_weight = param_dict['H1_I_dend_H1_E_weight']
    H1_E_H1_I_dend_init_weight = param_dict['H1_E_H1_I_dend_init_weight']
    H1_E_H1_I_dend_learning_rate = param_dict['H1_E_H1_I_dend_learning_rate']

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
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_I_dend_init_weight,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_I_dend_learning_rate
    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = \
        (H1_E_Output_E_min_init_weight, H1_E_Output_E_max_init_weight)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (FBI_E_weight,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_I_dend_H1_E_weight,)

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


def compute_features(x, seed, model_id=None, export=False, plot=False):
    """

    :param x: array of float
    :param model_id: str
    :param export: bool
    :return: dict
    """
    update_source_contexts(x, context)

    input_size = 21
    dataset = torch.eye(input_size)
    target = torch.eye(dataset.shape[0])

    epochs = context.epochs

    network = EIANN(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if plot:
        for sample in dataset:
            network.forward(sample, store_history=True)
        plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=context.supervised, label='Initial')
        network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=context.status_bar)

    loss_history, epoch_argmax_accuracy = \
        analyze_EIANN_loss(network, target, supervised=context.supervised, plot=plot)

    final_epoch_loss = torch.mean(loss_history[-target.shape[0]:])
    final_argmax_accuracy = torch.mean(epoch_argmax_accuracy[-context.num_epochs_argmax_accuracy:])

    if plot:
        plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=context.supervised, label='Final')

    if context.debug:
        print('pid: %i, seed: %i, sample_order: %s, final_output: %s' % (os.getpid(), seed, network.sample_order,
                                                                         network.Output.E.activity))
        context.update(locals())

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs}
        if context.export_network_config_file_path is None:
            raise Exception('nested_optimize_EIANN_1_hidden: missing required export_network_config_file_path')
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)

    return {'loss': final_epoch_loss,
            'accuracy': final_argmax_accuracy}


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
        if key in ['accuracy']:
            objectives[key] = 100. - val
        else:
            objectives[key] = val
    return features, objectives
