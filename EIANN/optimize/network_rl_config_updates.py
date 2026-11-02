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
