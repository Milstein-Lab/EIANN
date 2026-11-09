from nested.utils import param_array_to_dict


def update_EIANN_config_2_hidden_2ACDC_van_bp_relu_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H_learning_rate = param_dict['H_learning_rate']
    H1_init_weight_scale = param_dict['H1_init_weight_scale']
    H1_H1_init_weight_scale = param_dict['H1_H1_init_weight_scale']
    H2_init_weight_scale = param_dict['H2_init_weight_scale']

    Output_learning_rate = param_dict['Output_learning_rate']
    Output_init_weight_scale = param_dict['Output_init_weight_scale']

    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_H1_init_weight_scale,)

    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_init_weight_scale,)

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = Output_learning_rate
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_init_weight_scale,)


def update_tau(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    context.training_kwargs['tau'] = param_dict['tau']


def update_q_learning_kwargs(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    context.algorithm_kwargs['gamma'] = param_dict['gamma']
    context.algorithm_kwargs['epsilon'] = param_dict['epsilon']
    context.algorithm_kwargs['epsilon_decay'] = param_dict['epsilon_decay']



def set_learning_rule_kwarg(projection_config, learning_rule, key, value, post_layers=None):
    """
    Set learning_rule_kwargs[key] on every projection of a nested projection_config that uses learning_rule.
    :param post_layers: list of str; only projections onto these layers (default: all)
    """
    for post_layer, post_pops in projection_config.items():
        if post_layers is not None and post_layer not in post_layers:
            continue
        for pre_layers in post_pops.values():
            for pre_pops in pre_layers.values():
                for projection_kwargs in pre_pops.values():
                    if projection_kwargs.get('learning_rule') == learning_rule:
                        projection_kwargs['learning_rule_kwargs'][key] = value


def set_hidden_and_output_kwarg(context, learning_rule, key, H_value, Output_value):
    """
    Set learning_rule_kwargs[key] to Output_value on projections onto the Output layer and to H_value on all others.
    """
    hidden_layers = [layer for layer in context.projection_config if layer != 'Output']
    set_learning_rule_kwarg(context.projection_config, learning_rule, key, H_value, post_layers=hidden_layers)
    set_learning_rule_kwarg(context.projection_config, learning_rule, key, Output_value, post_layers=['Output'])


def update_SI_lambda(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    set_hidden_and_output_kwarg(context, 'Backprop_SI', 'si_lambda', param_dict['H_si_lambda'],
                                param_dict['Output_si_lambda'])


def update_EWC_lambda(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    set_hidden_and_output_kwarg(context, 'Backprop_EWC', 'ewc_lambda', param_dict['H_ewc_lambda'],
                                param_dict['Output_ewc_lambda'])
