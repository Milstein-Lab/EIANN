from nested.utils import param_array_to_dict
import math

def update_EIANN_config_2_hidden_van_bp_relu_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)
    
    H_learning_rate = param_dict['H_learning_rate']
    H1_init_weight_scale = param_dict['H1_init_weight_scale']
    H1_H1_init_weight_scale = param_dict['H1_H1_init_weight_scale']
    H2_init_weight_scale = param_dict['H2_init_weight_scale']
    H2_H2_init_weight_scale = param_dict['H2_H2_init_weight_scale']
    
    Output_learning_rate = param_dict['Output_learning_rate']
    Output_init_weight_scale = param_dict['Output_init_weight_scale']
    
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_H1_init_weight_scale,)
    
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H2']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['E']['weight_init_args'] = (H2_H2_init_weight_scale,)
    
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = (
        Output_learning_rate)
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_init_weight_scale,)
    
    context.training_kwargs['optimizer'] = 'SGD'


def update_EIANN_config_2_hidden_eprop_relu_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)
    
    H_learning_rate = param_dict['H_learning_rate']
    H1_init_weight_scale = param_dict['H1_init_weight_scale']
    H1_H1_init_weight_scale = param_dict['H1_H1_init_weight_scale']
    H2_init_weight_scale = param_dict['H2_init_weight_scale']
    H2_H2_init_weight_scale = param_dict['H2_H2_init_weight_scale']
    
    Output_learning_rate = param_dict['Output_learning_rate']
    Output_init_weight_scale = param_dict['Output_init_weight_scale']
    
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_H1_init_weight_scale,)
    
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H2']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_learning_rate
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['E']['weight_init_args'] = (H2_H2_init_weight_scale,)
    
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = (
        Output_learning_rate)
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_init_weight_scale,)
    
    context.training_kwargs['optimizer'] = 'SGD'
    context.training_kwargs['tau'] = param_dict['tau']

def update_EIANN_config_2_hidden_backprop_Dale_relu_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)
    
    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])
    
    context.layer_config['H1']['SomaI']['size'] = H_I_size
    context.layer_config['H2']['SomaI']['size'] = H_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size
    
    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']
    H_E_I_learning_rate = param_dict['H_E_I_learning_rate']
    H_I_E_learning_rate = param_dict['H_I_E_learning_rate']
    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_H1_E_init_weight_scale = param_dict['H1_E_H1_E_init_weight_scale']
    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']
    
    H2_E_H1_E_init_weight_scale = param_dict['H2_E_H1_E_init_weight_scale']
    H2_E_H2_E_init_weight_scale = param_dict['H2_E_H2_E_init_weight_scale']
    H2_E_H2_I_init_weight_scale = param_dict['H2_E_H2_I_init_weight_scale']
    H2_I_H1_E_init_weight_scale = param_dict['H2_I_H1_E_init_weight_scale']
    H2_I_H2_E_init_weight_scale = param_dict['H2_I_H2_E_init_weight_scale']
    H2_I_H2_I_init_weight_scale = param_dict['H2_I_H2_I_init_weight_scale']
    
    Output_E_E_learning_rate = param_dict['Output_E_E_learning_rate']
    Output_E_H2_E_init_weight_scale = param_dict['Output_E_H2_E_init_weight_scale']
    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_H2_E_init_weight_scale = param_dict['Output_I_H2_E_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']
    
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_E_H1_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_E_I_learning_rate
    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)
    
    context.projection_config['H1']['SomaI']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H1']['SomaI']['H1']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)
    
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H2']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H2']['E']['H2']['E']['weight_init_args'] = (H2_E_H2_E_init_weight_scale,)
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_E_I_learning_rate
    context.projection_config['H2']['E']['H2']['SomaI']['weight_init_args'] = (H2_E_H2_I_init_weight_scale,)
    
    context.projection_config['H2']['SomaI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H2']['SomaI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H2']['SomaI']['H2']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['H2']['SomaI']['H1']['E']['weight_init_args'] = (H2_I_H1_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['E']['weight_init_args'] = (H2_I_H2_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['SomaI']['weight_init_args'] = (H2_I_H2_I_init_weight_scale,)
    
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = (
        Output_E_E_learning_rate)
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_E_I_learning_rate
    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)
    
    context.projection_config['Output']['SomaI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['E']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['Output']['SomaI']['Output']['SomaI']['learning_rule_kwargs']['learning_rate'] = H_I_E_learning_rate
    context.projection_config['Output']['SomaI']['H2']['E']['weight_init_args'] = (Output_I_H2_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)

    context.training_kwargs['optimizer'] = 'SGD'

def update_EIANN_config_2_hidden_backprop_Dale_fixedI_relu_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H_I_size
    context.layer_config['H2']['SomaI']['size'] = H_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']
    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_H1_E_init_weight_scale = param_dict['H1_E_H1_E_init_weight_scale']
    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    H2_E_H1_E_init_weight_scale = param_dict['H2_E_H1_E_init_weight_scale']
    H2_E_H2_E_init_weight_scale = param_dict['H2_E_H2_E_init_weight_scale']
    H2_E_H2_I_init_weight_scale = param_dict['H2_E_H2_I_init_weight_scale']
    H2_I_H1_E_init_weight_scale = param_dict['H2_I_H1_E_init_weight_scale']
    H2_I_H2_E_init_weight_scale = param_dict['H2_I_H2_E_init_weight_scale']
    H2_I_H2_I_init_weight_scale = param_dict['H2_I_H2_I_init_weight_scale']

    Output_E_E_learning_rate = param_dict['Output_E_E_learning_rate']
    Output_E_H2_E_init_weight_scale = param_dict['Output_E_H2_E_init_weight_scale']
    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_H2_E_init_weight_scale = param_dict['Output_I_H2_E_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']

    # I cells are fixed: only the E->E projections are learned, so learning rates are set only on those.
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_E_H1_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H2']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H2']['E']['H2']['E']['weight_init_args'] = (H2_E_H2_E_init_weight_scale,)
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['SomaI']['weight_init_args'] = (H2_E_H2_I_init_weight_scale,)

    context.projection_config['H2']['SomaI']['H1']['E']['weight_init_args'] = (H2_I_H1_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['E']['weight_init_args'] = (H2_I_H2_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['SomaI']['weight_init_args'] = (H2_I_H2_I_init_weight_scale,)

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = (
        Output_E_E_learning_rate)
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)

    context.projection_config['Output']['SomaI']['H2']['E']['weight_init_args'] = (Output_I_H2_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)

    context.training_kwargs['optimizer'] = 'SGD'

def update_EIANN_config_2_hidden_backprop_Dale_CA1_relu_SGD_G(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])

    context.layer_config['H1']['SomaI']['size'] = H_I_size
    context.layer_config['H2']['SomaI']['size'] = H_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size

    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']
    H1_E_Input_E_init_weight_scale = param_dict['H1_E_Input_E_init_weight_scale']
    H1_E_H1_E_init_weight_scale = param_dict['H1_E_H1_E_init_weight_scale']
    H1_E_H1_I_init_weight_scale = param_dict['H1_E_H1_I_init_weight_scale']
    H1_I_Input_E_init_weight_scale = param_dict['H1_I_Input_E_init_weight_scale']
    H1_I_H1_E_init_weight_scale = param_dict['H1_I_H1_E_init_weight_scale']
    H1_I_H1_I_init_weight_scale = param_dict['H1_I_H1_I_init_weight_scale']

    H2_E_H1_E_init_weight_scale = param_dict['H2_E_H1_E_init_weight_scale']
    H2_E_H2_I_init_weight_scale = param_dict['H2_E_H2_I_init_weight_scale']
    H2_I_H1_E_init_weight_scale = param_dict['H2_I_H1_E_init_weight_scale']
    H2_I_H2_E_init_weight_scale = param_dict['H2_I_H2_E_init_weight_scale']
    H2_I_H2_I_init_weight_scale = param_dict['H2_I_H2_I_init_weight_scale']

    Output_E_E_learning_rate = param_dict['Output_E_E_learning_rate']
    Output_E_H2_E_init_weight_scale = param_dict['Output_E_H2_E_init_weight_scale']
    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_H2_E_init_weight_scale = param_dict['Output_I_H2_E_init_weight_scale']

    # I cells are fixed: only the E->E projections are learned, so learning rates are set only on those.
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H1']['E']['H1']['E']['weight_init_args'] = (H1_E_H1_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_I_init_weight_scale,)

    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_I_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_I_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_I_H1_I_init_weight_scale,)

    # H2 has no E->E recurrence in the CA1 variant, so only the feedforward H1.E -> H2.E projection is learned here.
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = H_E_E_learning_rate
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['SomaI']['weight_init_args'] = (H2_E_H2_I_init_weight_scale,)

    context.projection_config['H2']['SomaI']['H1']['E']['weight_init_args'] = (H2_I_H1_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['E']['weight_init_args'] = (H2_I_H2_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['SomaI']['weight_init_args'] = (H2_I_H2_I_init_weight_scale,)

    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = (
        Output_E_E_learning_rate)
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)
    context.projection_config['Output']['SomaI']['H2']['E']['weight_init_args'] = (Output_I_H2_E_init_weight_scale,)

    context.training_kwargs['optimizer'] = 'SGD'
