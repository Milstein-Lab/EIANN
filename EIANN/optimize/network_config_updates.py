from nested.utils import param_array_to_dict
import math


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


def update_EIANN_config_1_hidden_backprop_Dale_softplus_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']

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
    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['I']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['I']['weight_init_args'] = (Output_I_Output_I_init_weight_scale,)

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


def update_EIANN_config_1_hidden_Gjorgjieva_Hebb_G(x, context):
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

    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']

    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']

    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    Output_E_H1_E_weight_scale = param_dict['Output_E_H1_E_weight_scale'] * \
                                 math.sqrt(context.layer_config['H1']['E']['size']) / 2
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']

    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['weight_constraint_kwargs']['scale'] = H1_E_Input_E_weight_scale
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate

    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['I']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_constraint_kwargs']['scale'] = \
        Output_E_H1_E_weight_scale
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate

    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['I']['weight_init_args'] = (Output_I_Output_I_init_weight_scale,)


def update_EIANN_config_1_hidden_BCM_G(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['I']['size'] = H1_I_size
    context.layer_config['Output']['I']['size'] = Output_I_size

    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']
    H1_E_theta_tau = param_dict['H1_E_theta_tau']
    H1_E_BCM_k = param_dict['H1_E_BCM_k']

    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    Output_E_H1_E_init_weight_scale = param_dict['Output_E_H1_E_init_weight_scale']
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']
    Output_E_theta_tau = param_dict['Output_E_theta_tau']
    Output_E_BCM_k = param_dict['Output_E_BCM_k']

    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['theta_tau'] = H1_E_theta_tau
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['k'] = H1_E_BCM_k

    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['I']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['theta_tau'] = \
        Output_E_theta_tau
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['k'] = \
        Output_E_BCM_k

    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['I']['weight_init_args'] = (Output_I_Output_I_init_weight_scale,)


def update_EIANN_config_1_hidden_BCM_cotuned_I_H(x, context):
    """

    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    H1_I_size = int(param_dict['H1_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['I']['size'] = H1_I_size
    context.layer_config['Output']['I']['size'] = Output_I_size

    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_Input_E_learning_rate = param_dict['H1_E_Input_E_learning_rate']
    H1_E_theta_tau = param_dict['H1_E_theta_tau']
    H1_E_BCM_k = param_dict['H1_E_BCM_k']

    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['E_I_learning_rate']
    I_I_learning_rate = param_dict['E_I_learning_rate']

    H1_I_theta_tau = param_dict['H1_I_theta_tau']
    H1_I_BCM_k = param_dict['H1_I_BCM_k']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    Output_E_H1_E_init_weight_scale = param_dict['Output_E_H1_E_init_weight_scale']
    Output_E_H1_E_learning_rate = param_dict['Output_E_H1_E_learning_rate']
    Output_E_theta_tau = param_dict['Output_E_theta_tau']
    Output_E_BCM_k = param_dict['Output_E_BCM_k']

    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']

    Output_I_theta_tau = param_dict['Output_I_theta_tau']
    Output_I_BCM_k = param_dict['Output_I_BCM_k']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['theta_tau'] = H1_E_theta_tau
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['k'] = H1_E_BCM_k

    context.projection_config['H1']['E']['H1']['I']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['theta_tau'] = H1_E_theta_tau
    context.projection_config['H1']['E']['H1']['I']['learning_rule_kwargs']['k'] = H1_E_BCM_k

    context.projection_config['H1']['I']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['theta_tau'] = H1_I_theta_tau
    context.projection_config['H1']['I']['Input']['E']['learning_rule_kwargs']['k'] = H1_I_BCM_k

    context.projection_config['H1']['I']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['theta_tau'] = H1_I_theta_tau
    context.projection_config['H1']['I']['H1']['E']['learning_rule_kwargs']['k'] = H1_I_BCM_k

    context.projection_config['H1']['I']['H1']['I']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)
    context.projection_config['H1']['I']['H1']['I']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate
    context.projection_config['H1']['I']['H1']['I']['learning_rule_kwargs']['theta_tau'] = H1_I_theta_tau
    context.projection_config['H1']['I']['H1']['I']['learning_rule_kwargs']['k'] = H1_I_BCM_k

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H1_E_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['theta_tau'] = Output_E_theta_tau
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['k'] = Output_E_BCM_k

    context.projection_config['Output']['E']['Output']['I']['weight_init_args'] = (Output_E_Output_I_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['theta_tau'] = Output_E_theta_tau
    context.projection_config['Output']['E']['Output']['I']['learning_rule_kwargs']['k'] = Output_E_BCM_k

    context.projection_config['Output']['I']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['theta_tau'] = Output_I_theta_tau
    context.projection_config['Output']['I']['H1']['E']['learning_rule_kwargs']['k'] = Output_I_BCM_k

    context.projection_config['Output']['I']['Output']['E']['weight_init_args'] = (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['theta_tau'] = Output_I_theta_tau
    context.projection_config['Output']['I']['Output']['E']['learning_rule_kwargs']['k'] = Output_I_BCM_k

    context.projection_config['Output']['I']['Output']['I']['weight_init_args'] = (Output_I_Output_I_init_weight_scale,)
    context.projection_config['Output']['I']['Output']['I']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate
    context.projection_config['Output']['I']['Output']['I']['learning_rule_kwargs']['theta_tau'] = Output_I_theta_tau
    context.projection_config['Output']['I']['Output']['I']['learning_rule_kwargs']['k'] = Output_I_BCM_k


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
    Uses a single BTSP_anti_hebb_th for all layers.
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


def update_EIANN_config_1_hidden_BTSP_F7(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    E<-E weights are learned with the BTSP_8 rule.
    Inits are half-kaining with parameterized scale.
    Uses a separate BTSP_anti_hebb_th for H1 vs. Output layer cells.
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
    H1_E_BTSP_anti_hebb_th = param_dict['H1_E_BTSP_anti_hebb_th']
    Output_E_BTSP_anti_hebb_th = param_dict['Output_E_BTSP_anti_hebb_th']

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
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_th'] = H1_E_BTSP_anti_hebb_th

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_th'] = H1_E_BTSP_anti_hebb_th

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
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        Output_E_BTSP_anti_hebb_th

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


def update_EIANN_config_1_hidden_BTSP_F8(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are learned with the Gjorgjieva_Hebb_2 rule.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    E<-E weights are learned with the BTSP_10 rule.
    Inits are half-kaining with parameterized scale.
    BTSP_anti_hebb_th is specified separately for H1 vs. Output layer cells. It is parameterized as a tuple of float
    bounds.
    anti_hebb_th_learning_rate is specified separately for H1 vs. Output layer cells, and is specified separately from
    BTSP_learning_rate.
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
    H1_E_BTSP_anti_hebb_th_min = param_dict['H1_E_BTSP_anti_hebb_th_min']
    H1_E_BTSP_anti_hebb_th_max = param_dict['H1_E_BTSP_anti_hebb_th_max']
    Output_E_BTSP_anti_hebb_th = param_dict['Output_E_BTSP_anti_hebb_th']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Input_E_anti_hebb_learning_rate = param_dict['H1_E_Input_E_anti_hebb_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_Output_E_anti_hebb_learning_rate = param_dict['H1_E_Output_E_anti_hebb_learning_rate']
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
    Output_E_anti_hebb_learning_rate = param_dict['Output_E_anti_hebb_learning_rate']
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
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Input_E_anti_hebb_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Output_E_anti_hebb_learning_rate

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
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (0., Output_E_BTSP_anti_hebb_th)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        Output_E_anti_hebb_learning_rate

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


def update_EIANN_config_1_hidden_BTSP_G(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are not learned.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    E<-E weights are learned with the BTSP_10 rule.
    Inits are half-kaining with parameterized scale.
    BTSP_anti_hebb_th is specified separately for H1 vs. Output layer cells. It is parameterized as a tuple of float
    bounds.
    anti_hebb_th_learning_rate is specified separately for H1 vs. Output layer cells, and is specified separately from
    BTSP_learning_rate.
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
    H1_E_BTSP_anti_hebb_th_min = param_dict['H1_E_BTSP_anti_hebb_th_min']
    H1_E_BTSP_anti_hebb_th_max = param_dict['H1_E_BTSP_anti_hebb_th_max']
    Output_E_BTSP_anti_hebb_th = param_dict['Output_E_BTSP_anti_hebb_th']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Input_E_anti_hebb_learning_rate = param_dict['H1_E_Input_E_anti_hebb_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_Output_E_anti_hebb_learning_rate = param_dict['H1_E_Output_E_anti_hebb_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_init_weight_scale = param_dict['H1_E_H1_SomaI_init_weight_scale']
    H1_SomaI_H1_E_init_weight_scale = param_dict['H1_SomaI_H1_E_init_weight_scale']
    H1_SomaI_Input_E_init_weight_scale = param_dict['H1_SomaI_Input_E_init_weight_scale']
    H1_SomaI_H1_SomaI_init_weight_scale = param_dict['H1_SomaI_H1_SomaI_init_weight_scale']

    H1_DendI_H1_E_init_weight_scale = param_dict['H1_DendI_H1_E_init_weight_scale']
    H1_DendI_H1_DendI_init_weight_scale = param_dict['H1_DendI_H1_DendI_init_weight_scale']
    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_anti_hebb_learning_rate = param_dict['Output_E_anti_hebb_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    BTSP_decay_tau = param_dict['BTSP_decay_tau']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Input_E_anti_hebb_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Output_E_anti_hebb_learning_rate

    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_SomaI_init_weight_scale,)

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_SomaI_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_SomaI_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_SomaI_H1_SomaI_init_weight_scale,)

    context.projection_config['H1']['DendI']['H1']['E']['weight_init_args'] = (H1_DendI_H1_E_init_weight_scale,)
    context.projection_config['H1']['DendI']['H1']['DendI']['weight_init_args'] = (H1_DendI_H1_DendI_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (0., Output_E_BTSP_anti_hebb_th)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        Output_E_anti_hebb_learning_rate

    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['SomaI']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)


def update_EIANN_config_1_hidden_BTSP_G3(x, context):
    """
    H1.SomaI, H1.DendI, and Output.SomaI are not learned.
    H1.E.H1.DendI weights are learned with the DendriticLoss_5 rule.
    BTSP_G used the BTSP_10 rule, which had a bug, E<-E weights are learned with the BTSP_11 rule.
    Inits are half-kaining with parameterized scale.
    BTSP_anti_hebb_th is specified separately for H1 vs. Output layer cells. It is parameterized as a tuple of float
    bounds.
    anti_hebb_th_learning_rate is specified separately for H1 vs. Output layer cells, and is specified separately from
    BTSP_learning_rate.

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
    H1_E_BTSP_anti_hebb_th_min = param_dict['H1_E_BTSP_anti_hebb_th_min']
    H1_E_BTSP_anti_hebb_th_max = param_dict['H1_E_BTSP_anti_hebb_th_max']
    Output_E_BTSP_anti_hebb_th = param_dict['Output_E_BTSP_anti_hebb_th']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_Input_E_BTSP_learning_rate = param_dict['H1_E_Input_E_BTSP_learning_rate']
    H1_E_Input_E_anti_hebb_learning_rate = param_dict['H1_E_Input_E_anti_hebb_learning_rate']
    H1_E_Output_E_BTSP_learning_rate = param_dict['H1_E_Output_E_BTSP_learning_rate']
    H1_E_Output_E_anti_hebb_learning_rate = param_dict['H1_E_Output_E_anti_hebb_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_Output_E_max_weight_scale = param_dict['H1_E_Output_E_max_weight_scale']
    H1_E_Output_E_max_weight = H1_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H1_E_Output_E_init_weight_scale = H1_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_SomaI_init_weight_scale = param_dict['H1_E_H1_SomaI_init_weight_scale']
    H1_SomaI_H1_E_init_weight_scale = param_dict['H1_SomaI_H1_E_init_weight_scale']
    H1_SomaI_Input_E_init_weight_scale = param_dict['H1_SomaI_Input_E_init_weight_scale']
    H1_SomaI_H1_SomaI_init_weight_scale = param_dict['H1_SomaI_H1_SomaI_init_weight_scale']

    H1_DendI_H1_E_init_weight_scale = param_dict['H1_DendI_H1_E_init_weight_scale']
    H1_DendI_H1_DendI_init_weight_scale = param_dict['H1_DendI_H1_DendI_init_weight_scale']
    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale'] / \
                                      math.sqrt(context.layer_config['H1']['DendI']['size'])
    H1_E_H1_DendI_learning_rate = param_dict['H1_E_H1_DendI_learning_rate']

    Output_E_H1_E_max_weight_scale = param_dict['Output_E_H1_E_max_weight_scale']
    Output_E_H1_E_max_weight = Output_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    Output_E_H1_E_init_weight_scale = Output_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_anti_hebb_learning_rate = param_dict['Output_E_anti_hebb_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']

    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H1_E_init_weight_scale = param_dict['Output_I_H1_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    BTSP_decay_tau = param_dict['BTSP_decay_tau']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Input_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Input_E_anti_hebb_learning_rate

    context.projection_config['H1']['E']['Output']['E']['weight_init_args'] = (H1_E_Output_E_init_weight_scale,)
    context.projection_config['H1']['E']['Output']['E']['weight_bounds'] = (0, H1_E_Output_E_max_weight)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H1_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_Output_E_BTSP_learning_rate
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (H1_E_BTSP_anti_hebb_th_min, H1_E_BTSP_anti_hebb_th_max)
    context.projection_config['H1']['E']['Output']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        H1_E_Output_E_anti_hebb_learning_rate

    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_SomaI_init_weight_scale,)

    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H1_E_H1_DendI_learning_rate

    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_SomaI_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_SomaI_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_SomaI_H1_SomaI_init_weight_scale,)

    context.projection_config['H1']['DendI']['H1']['E']['weight_init_args'] = (H1_DendI_H1_E_init_weight_scale,)
    context.projection_config['H1']['DendI']['H1']['DendI']['weight_init_args'] = (H1_DendI_H1_DendI_init_weight_scale,)

    context.projection_config['Output']['E']['H1']['E']['weight_init_args'] = (Output_E_H1_E_init_weight_scale,)
    context.projection_config['Output']['E']['H1']['E']['weight_bounds'] = (0, Output_E_H1_E_max_weight)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['decay_tau'] = \
        BTSP_decay_tau
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_th'] = \
        (0., Output_E_BTSP_anti_hebb_th)
    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['anti_hebb_learning_rate'] = \
        Output_E_anti_hebb_learning_rate

    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['SomaI']['H1']['E']['weight_init_args'] = (Output_I_H1_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)


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


def update_EIANN_config_2_hidden_mnist_backprop_Dale_softplus_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H_FBI_size = int(param_dict['H_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['FBI']['size'] = H_FBI_size
    context.layer_config['H2']['FBI']['size'] = H_FBI_size
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

    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H2']['E']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H2']['FBI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_2_hidden_mnist_backprop_Dale_softplus_SGD_B(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H_FBI_size = int(param_dict['H_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']

    context.layer_config['H1']['FBI']['size'] = H_FBI_size
    context.layer_config['H2']['FBI']['size'] = H_FBI_size
    context.layer_config['Output']['FBI']['size'] = Output_FBI_size

    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_H1_FBI_init_weight_scale = param_dict['H1_E_H1_FBI_init_weight_scale']
    H1_FBI_H1_E_init_weight_scale = param_dict['H1_FBI_H1_E_init_weight_scale']
    H2_E_H1_E_init_weight_scale = param_dict['H2_E_H1_E_init_weight_scale']
    H2_E_H2_FBI_init_weight_scale = param_dict['H2_E_H2_FBI_init_weight_scale']
    H2_FBI_H2_E_init_weight_scale = param_dict['H2_FBI_H2_E_init_weight_scale']
    Output_E_H2_E_init_weight_scale = param_dict['Output_E_H2_E_init_weight_scale']
    Output_E_Output_FBI_init_weight_scale = param_dict['Output_E_Output_FBI_init_weight_scale']
    Output_FBI_Output_E_init_weight_scale = param_dict['Output_FBI_Output_E_init_weight_scale']

    for i, layer in enumerate(context.layer_config.values()):
        if i > 0:
            for pop in layer.values():
                if 'activation' in pop and pop['activation'] == 'softplus':
                    if 'activation_kwargs' not in pop:
                        pop['activation_kwargs'] = {}
                    pop['activation_kwargs']['beta'] = softplus_beta

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (H1_E_H1_FBI_init_weight_scale,)
    context.projection_config['H1']['FBI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_init_weight_scale,)

    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H2']['E']['H2']['FBI']['weight_init_args'] = (H2_E_H2_FBI_init_weight_scale,)
    context.projection_config['H2']['FBI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H2']['FBI']['H2']['E']['weight_init_args'] = (H2_FBI_H2_E_init_weight_scale,)

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = \
        (Output_E_Output_FBI_init_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = \
        (Output_FBI_Output_E_init_weight_scale,)

    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_2_hidden_mnist_backprop_Dale_softplus_SGD_E(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']

    context.layer_config['H1']['FBI']['size'] = H1_FBI_size
    context.layer_config['H2']['FBI']['size'] = H1_FBI_size
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
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['H2']['E']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = E_I_learning_rate
    context.projection_config['H2']['FBI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = I_E_learning_rate
    context.projection_config['H2']['FBI']['H2']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate

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


def update_EIANN_config_1_hidden_mnist_backprop_Dale_softplus_SGD_E(x, context):
    """
    Includes FBI <- FBI connections.
    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)

    softplus_beta = param_dict['softplus_beta']
    H1_FBI_size = int(param_dict['H1_FBI_size'])
    Output_FBI_size = int(param_dict['Output_FBI_size'])

    E_E_learning_rate = param_dict['E_E_learning_rate']
    E_I_learning_rate = param_dict['E_I_learning_rate']
    I_E_learning_rate = param_dict['I_E_learning_rate']
    I_I_learning_rate = param_dict['I_I_learning_rate']

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
    context.projection_config['H1']['FBI']['H1']['FBI']['learning_rule_kwargs']['learning_rate'] = I_I_learning_rate

    context.projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = E_E_learning_rate
    context.projection_config['Output']['E']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        E_I_learning_rate
    context.projection_config['Output']['FBI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        I_E_learning_rate
    context.projection_config['Output']['FBI']['Output']['FBI']['learning_rule_kwargs']['learning_rate'] = \
        I_I_learning_rate

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


def update_EIANN_config_2_hidden_mnist_BTSP_D1(x, context):
    """
    This config has 1 static soma-targeting FBI cell per layer, and 1 static dend-targeting Dend_I cell in each H layer.
    Only E_Dend_I is learned. Inits are half-kaining with parameterized scale or _fill.
    :param x:
    :param context:

    """
    param_dict = param_array_to_dict(x, context.param_names)

    FF_BTSP_init_weight_factor = param_dict['FF_BTSP_init_weight_factor']
    FB_BTSP_init_weight_factor = param_dict['FB_BTSP_init_weight_factor']

    H_E_FF_BTSP_learning_rate = param_dict['H_E_FF_BTSP_learning_rate']
    H_E_FB_BTSP_learning_rate = param_dict['H_E_FB_BTSP_learning_rate']
    H_E_BTSP_pos_loss_th = param_dict['H_E_BTSP_pos_loss_th']
    H_E_BTSP_neg_loss_th = param_dict['H_E_BTSP_neg_loss_th']
    H_E_H_Dend_I_learning_rate = param_dict['H_E_H_Dend_I_learning_rate']

    H1_E_Input_E_max_weight_scale = param_dict['H1_E_Input_E_max_weight_scale']
    H1_E_Input_E_max_weight = H1_E_Input_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H1_E_Input_E_max_weight_scale * FF_BTSP_init_weight_factor
    H1_E_H2_E_max_weight_scale = param_dict['H1_E_H2_E_max_weight_scale']
    H1_E_H2_E_max_weight = H1_E_H2_E_max_weight_scale / math.sqrt(context.layer_config['H2']['E']['size'])
    H1_E_H2_E_init_weight_scale = H1_E_H2_E_max_weight_scale * FB_BTSP_init_weight_factor

    H1_E_H1_FBI_weight_scale = param_dict['H1_E_H1_FBI_weight_scale']
    H1_FBI_H1_E_weight_scale = param_dict['H1_FBI_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])

    H1_Dend_I_H1_E_weight_scale = param_dict['H1_Dend_I_H1_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H1']['E']['size'])
    H1_E_H1_Dend_I_init_weight_scale = param_dict['H1_E_H1_Dend_I_init_weight_scale']

    H2_E_H1_E_max_weight_scale = param_dict['H2_E_H1_E_max_weight_scale']
    H2_E_H1_E_max_weight = H2_E_H1_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    H2_E_H1_E_init_weight_scale = H2_E_H1_E_max_weight_scale * FF_BTSP_init_weight_factor
    H2_E_Output_E_max_weight_scale = param_dict['H2_E_Output_E_max_weight_scale']
    H2_E_Output_E_max_weight = H2_E_Output_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H2_E_Output_E_init_weight_scale = H2_E_Output_E_max_weight_scale * FB_BTSP_init_weight_factor

    H2_E_H2_FBI_weight_scale = param_dict['H2_E_H2_FBI_weight_scale']
    H2_FBI_H2_E_weight_scale = param_dict['H2_FBI_H2_E_weight_scale'] / \
                               math.sqrt(context.layer_config['H2']['E']['size'])

    H2_Dend_I_H2_E_weight_scale = param_dict['H2_Dend_I_H2_E_weight_scale'] / \
                                  math.sqrt(context.layer_config['H2']['E']['size'])
    H2_E_H2_Dend_I_init_weight_scale = param_dict['H2_E_H2_Dend_I_init_weight_scale']

    Output_E_H2_E_max_weight_scale = param_dict['Output_E_H2_E_max_weight_scale']
    Output_E_H2_E_max_weight = Output_E_H2_E_max_weight_scale / math.sqrt(context.layer_config['H2']['E']['size'])
    Output_E_H2_E_init_weight_scale = Output_E_H2_E_max_weight_scale * FF_BTSP_init_weight_factor
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']

    Output_E_Output_FBI_weight_scale = param_dict['Output_E_Output_FBI_weight_scale']
    Output_FBI_Output_E_weight_scale = param_dict['Output_FBI_Output_E_weight_scale'] / \
                               math.sqrt(context.layer_config['Output']['E']['size'])

    BTSP_neg_loss_ET_discount = param_dict['BTSP_neg_loss_ET_discount']

    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['pos_loss_th'] = H_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_th'] = H_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_FF_BTSP_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H1']['E']['H1']['FBI']['weight_init_args'] = (-H1_E_H1_FBI_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['weight_init_args'] = (H1_E_H1_Dend_I_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H_E_H_Dend_I_learning_rate
    context.projection_config['H1']['E']['H2']['E']['weight_init_args'] = (H1_E_H2_E_init_weight_scale,)
    context.projection_config['H1']['E']['H2']['E']['weight_bounds'] = (0, H1_E_H2_E_max_weight)
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['pos_loss_th'] = H_E_BTSP_pos_loss_th
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['neg_loss_th'] = H_E_BTSP_neg_loss_th
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_FB_BTSP_learning_rate
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H1']['FBI']['H1']['E']['weight_init_args'] = (H1_FBI_H1_E_weight_scale,)
    context.projection_config['H1']['Dend_I']['H1']['E']['weight_init_args'] = (H1_Dend_I_H1_E_weight_scale,)

    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H1']['E']['weight_bounds'] = (0, H2_E_H1_E_max_weight)
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['pos_loss_th'] = H_E_BTSP_pos_loss_th
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_th'] = H_E_BTSP_neg_loss_th
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_FF_BTSP_learning_rate
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['H2']['E']['H2']['FBI']['weight_init_args'] = (-H2_E_H2_FBI_weight_scale,)
    context.projection_config['H2']['E']['H2']['Dend_I']['weight_init_args'] = (H2_E_H2_Dend_I_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['Dend_I']['learning_rule_kwargs']['learning_rate'] = \
        H_E_H_Dend_I_learning_rate
    context.projection_config['H2']['E']['Output']['E']['weight_init_args'] = (H2_E_Output_E_init_weight_scale,)
    context.projection_config['H2']['E']['Output']['E']['weight_bounds'] = (0, H2_E_Output_E_max_weight)
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['pos_loss_th'] = H_E_BTSP_pos_loss_th
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_th'] = H_E_BTSP_neg_loss_th
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_FB_BTSP_learning_rate
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount

    context.projection_config['H2']['FBI']['H2']['E']['weight_init_args'] = (H2_FBI_H2_E_weight_scale,)
    context.projection_config['H2']['Dend_I']['H2']['E']['weight_init_args'] = (H2_Dend_I_H2_E_weight_scale,)

    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['H2']['E']['weight_bounds'] = (0, Output_E_H2_E_max_weight)
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['pos_loss_th'] = \
        Output_E_BTSP_pos_loss_th
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['neg_loss_th'] = \
        Output_E_BTSP_neg_loss_th
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_BTSP_learning_rate
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['neg_loss_ET_discount'] = \
        BTSP_neg_loss_ET_discount
    context.projection_config['Output']['E']['Output']['FBI']['weight_init_args'] = (-Output_E_Output_FBI_weight_scale,)
    context.projection_config['Output']['FBI']['Output']['E']['weight_init_args'] = (Output_FBI_Output_E_weight_scale,)


