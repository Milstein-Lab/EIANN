from nested.utils import param_array_to_dict
import math

def update_EIANN_config_2_hidden_van_bp_relu_SGD(x, context):
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


def update_EIANN_config_2_hidden_eprop_relu_SGD(x, context):
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

def update_EIANN_config_2_hidden_backprop_Dale_relu_SGD(x, context):
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

def update_EIANN_config_2_hidden_backprop_Dale_fixedI_relu_SGD(x, context):
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

def update_EIANN_config_2_hidden_backprop_Dale_CA1_relu_SGD(x, context):
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

def update_EIANN_config_2_hidden_BTSP(x, context):
    """
    H.SomaI, and Output.SomaI are not learned.
    H.DendI.H.E and H.DendI.H.DendI are learned with the Hebb_WeightNorm rule.
    H.E.H.DendI weights are learned with the DendriticLoss_6 rule.
    E<-E weights are learned with the BTSP_19 rule.
    H2.E.Output.E and H1.E.H2.E weights are learned with BTSP_19 rule.
    Inits are half-kaiming with parameterized scale.
    :param x:
    :param context:
    """
    param_dict = param_array_to_dict(x, context.param_names)
    max_pop_fraction = param_dict['max_pop_fraction']
    H_I_size = int(param_dict['H_I_size'])
    Output_I_size = int(param_dict['Output_I_size'])
    
    context.layer_config['H1']['SomaI']['size'] = H_I_size
    context.layer_config['H2']['SomaI']['size'] = H_I_size
    context.layer_config['H1']['DendI']['size'] = H_I_size
    context.layer_config['H2']['DendI']['size'] = H_I_size
    context.layer_config['Output']['SomaI']['size'] = Output_I_size
    
    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']
    H_E_DendI_learning_rate = param_dict['H_E_DendI_learning_rate']
    DendI_E_learning_rate = param_dict['DendI_E_learning_rate']
    DendI_DendI_learning_rate = param_dict['DendI_DendI_learning_rate']
    
    H1_E_Input_E_init_weight_factor = param_dict['H1_E_Input_E_init_weight_factor']
    H_E_FF_E_max_weight_scale = param_dict['H_E_FF_E_max_weight_scale']
    H1_E_Input_E_max_weight = H_E_FF_E_max_weight_scale / math.sqrt(context.layer_config['Input']['E']['size'])
    H1_E_Input_E_init_weight_scale = H_E_FF_E_max_weight_scale * H1_E_Input_E_init_weight_factor
    
    H1_E_H2_E_init_weight_factor = param_dict['H1_E_H2_E_init_weight_factor']
    H_E_TD_E_max_weight_scale = param_dict['H_E_TD_E_max_weight_scale']
    H1_E_H2_E_max_weight = H_E_TD_E_max_weight_scale / math.sqrt(context.layer_config['H2']['E']['size'])
    H1_E_H2_E_init_weight_scale = H_E_TD_E_max_weight_scale * H1_E_H2_E_init_weight_factor
    
    neg_rate_th = param_dict['neg_rate_th']
    temporal_discount = param_dict['temporal_discount']
    
    H1_E_H1_SomaI_init_weight_scale = param_dict['H1_E_H1_SomaI_init_weight_scale']
    H1_SomaI_H1_E_init_weight_scale = param_dict['H1_SomaI_H1_E_init_weight_scale']
    H1_SomaI_Input_E_init_weight_scale = param_dict['H1_SomaI_Input_E_init_weight_scale']
    H1_SomaI_H1_SomaI_init_weight_scale = param_dict['H1_SomaI_H1_SomaI_init_weight_scale']
    
    H1_DendI_H1_E_weight_scale = (param_dict['H1_DendI_H1_E_weight_scale'] *
                                  math.sqrt(context.layer_config['H1']['E']['size']) / 2)
    H1_DendI_H1_DendI_weight_scale = (param_dict['H1_DendI_H1_DendI_weight_scale'] *
                                      math.sqrt(context.layer_config['H1']['DendI']['size']) / 2)
    H1_E_H1_DendI_init_weight_scale = param_dict['H1_E_H1_DendI_init_weight_scale']
    
    H2_E_H1_E_init_weight_factor = param_dict['H2_E_H1_E_init_weight_factor']
    H2_E_H1_E_max_weight = H_E_FF_E_max_weight_scale / math.sqrt(context.layer_config['H1']['E']['size'])
    H2_E_H1_E_init_weight_scale = H_E_FF_E_max_weight_scale * H2_E_H1_E_init_weight_factor
    
    H2_E_Output_E_init_weight_factor = param_dict['H2_E_Output_E_init_weight_factor']
    H2_E_Output_E_max_weight = H_E_TD_E_max_weight_scale / math.sqrt(context.layer_config['Output']['E']['size'])
    H2_E_Output_E_init_weight_scale = H_E_TD_E_max_weight_scale * H2_E_Output_E_init_weight_factor
    
    H2_E_H2_SomaI_init_weight_scale = param_dict['H2_E_H2_SomaI_init_weight_scale']
    H2_SomaI_H2_E_init_weight_scale = param_dict['H2_SomaI_H2_E_init_weight_scale']
    H2_SomaI_H1_E_init_weight_scale = param_dict['H2_SomaI_H1_E_init_weight_scale']
    H2_SomaI_H2_SomaI_init_weight_scale = param_dict['H2_SomaI_H2_SomaI_init_weight_scale']
    
    H2_DendI_H2_E_weight_scale = (param_dict['H2_DendI_H2_E_weight_scale'] *
                                  math.sqrt(context.layer_config['H2']['E']['size']) / 2)
    H2_DendI_H2_DendI_weight_scale = (param_dict['H2_DendI_H2_DendI_weight_scale'] *
                                      math.sqrt(context.layer_config['H2']['DendI']['size']) / 2)
    H2_E_H2_DendI_init_weight_scale = param_dict['H2_E_H2_DendI_init_weight_scale']
    
    Output_E_H2_E_init_weight_factor = param_dict['Output_E_H2_E_init_weight_factor']
    Output_E_H2_E_max_weight_scale = param_dict['Output_E_H2_E_max_weight_scale']
    Output_E_H2_E_max_weight = Output_E_H2_E_max_weight_scale / math.sqrt(context.layer_config['H2']['E']['size'])
    Output_E_H2_E_init_weight_scale = Output_E_H2_E_max_weight_scale * Output_E_H2_E_init_weight_factor
    Output_E_H2_E_learning_rate = param_dict['Output_E_H2_E_learning_rate']
    
    Output_E_Output_I_init_weight_scale = param_dict['Output_E_Output_I_init_weight_scale']
    Output_I_Output_E_init_weight_scale = param_dict['Output_I_Output_E_init_weight_scale']
    Output_I_H2_E_init_weight_scale = param_dict['Output_I_H2_E_init_weight_scale']
    Output_I_Output_I_init_weight_scale = param_dict['Output_I_Output_I_init_weight_scale']
    
    context.projection_config['H1']['E']['Input']['E']['weight_init_args'] = (H1_E_Input_E_init_weight_scale,)
    context.projection_config['H1']['E']['Input']['E']['weight_bounds'] = (0, H1_E_Input_E_max_weight)
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['neg_rate_th'] = neg_rate_th
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['max_pop_fraction'] = max_pop_fraction
    context.projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['temporal_discount'] = temporal_discount
    
    context.projection_config['H1']['E']['H1']['SomaI']['weight_init_args'] = (H1_E_H1_SomaI_init_weight_scale,)
    
    context.projection_config['H1']['E']['H1']['DendI']['weight_init_args'] = (H1_E_H1_DendI_init_weight_scale,)
    context.projection_config['H1']['E']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H_E_DendI_learning_rate
    
    context.projection_config['H1']['E']['H2']['E']['weight_init_args'] = (H1_E_H2_E_init_weight_scale,)
    context.projection_config['H1']['E']['H2']['E']['weight_bounds'] = (0., H1_E_H2_E_max_weight)
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['neg_rate_th'] = neg_rate_th
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['max_pop_fraction'] = max_pop_fraction
    context.projection_config['H1']['E']['H2']['E']['learning_rule_kwargs']['temporal_discount'] = temporal_discount
    
    context.projection_config['H1']['SomaI']['Input']['E']['weight_init_args'] = (H1_SomaI_Input_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['E']['weight_init_args'] = (H1_SomaI_H1_E_init_weight_scale,)
    context.projection_config['H1']['SomaI']['H1']['SomaI']['weight_init_args'] = (H1_SomaI_H1_SomaI_init_weight_scale,)
    
    context.projection_config['H1']['DendI']['H1']['E']['weight_constraint_kwargs']['scale'] = (
        H1_DendI_H1_E_weight_scale)
    context.projection_config['H1']['DendI']['H1']['E']['learning_rule_kwargs']['learning_rate'] = DendI_E_learning_rate
    context.projection_config['H1']['DendI']['H1']['DendI']['weight_constraint_kwargs']['scale'] = (
        H1_DendI_H1_DendI_weight_scale)
    context.projection_config['H1']['DendI']['H1']['DendI']['learning_rule_kwargs']['learning_rate'] = (
        DendI_DendI_learning_rate)
    
    context.projection_config['H2']['E']['H1']['E']['weight_init_args'] = (H2_E_H1_E_init_weight_scale,)
    context.projection_config['H2']['E']['H1']['E']['weight_bounds'] = (0, H2_E_H1_E_max_weight)
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['neg_rate_th'] = neg_rate_th
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['max_pop_fraction'] = max_pop_fraction
    context.projection_config['H2']['E']['H1']['E']['learning_rule_kwargs']['temporal_discount'] = temporal_discount
    
    context.projection_config['H2']['E']['H2']['SomaI']['weight_init_args'] = (H2_E_H2_SomaI_init_weight_scale,)
    
    context.projection_config['H2']['E']['H2']['DendI']['weight_init_args'] = (H2_E_H2_DendI_init_weight_scale,)
    context.projection_config['H2']['E']['H2']['DendI']['learning_rule_kwargs']['learning_rate'] = \
        H_E_DendI_learning_rate
    
    context.projection_config['H2']['E']['Output']['E']['weight_init_args'] = (H2_E_Output_E_init_weight_scale,)
    context.projection_config['H2']['E']['Output']['E']['weight_bounds'] = (0., H2_E_Output_E_max_weight)
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['neg_rate_th'] = neg_rate_th
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['learning_rate'] = \
        H_E_E_learning_rate
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['max_pop_fraction'] = max_pop_fraction
    context.projection_config['H2']['E']['Output']['E']['learning_rule_kwargs']['temporal_discount'] = temporal_discount
    
    context.projection_config['H2']['SomaI']['H1']['E']['weight_init_args'] = (H2_SomaI_H1_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['E']['weight_init_args'] = (H2_SomaI_H2_E_init_weight_scale,)
    context.projection_config['H2']['SomaI']['H2']['SomaI']['weight_init_args'] = (H2_SomaI_H2_SomaI_init_weight_scale,)
    
    context.projection_config['H2']['DendI']['H2']['E']['weight_constraint_kwargs']['scale'] = (
        H2_DendI_H2_E_weight_scale)
    context.projection_config['H2']['DendI']['H2']['E']['learning_rule_kwargs']['learning_rate'] = DendI_E_learning_rate
    context.projection_config['H2']['DendI']['H2']['DendI']['weight_constraint_kwargs']['scale'] = (
        H2_DendI_H2_DendI_weight_scale)
    context.projection_config['H2']['DendI']['H2']['DendI']['learning_rule_kwargs']['learning_rate'] = (
        DendI_DendI_learning_rate)
    
    context.projection_config['Output']['E']['H2']['E']['weight_init_args'] = (Output_E_H2_E_init_weight_scale,)
    context.projection_config['Output']['E']['H2']['E']['weight_bounds'] = (0, Output_E_H2_E_max_weight)
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = \
        Output_E_H2_E_learning_rate
    context.projection_config['Output']['E']['H2']['E']['learning_rule_kwargs']['temporal_discount'] = temporal_discount
    
    context.projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (Output_E_Output_I_init_weight_scale,)
    
    context.projection_config['Output']['SomaI']['H2']['E']['weight_init_args'] = (Output_I_H2_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (Output_I_Output_E_init_weight_scale,)
    context.projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (Output_I_Output_I_init_weight_scale,)
