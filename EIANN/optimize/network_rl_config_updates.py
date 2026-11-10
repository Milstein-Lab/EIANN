import math

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


def update_qnext_van_bp(x, context):
    """
    DPPQL's QNext network (Input -> H1 -> Output, no recurrence). Its params are prefixed QNext_; the Q network's are
    set by the Q network's own update function.
    """
    param_dict = param_array_to_dict(x, context.param_names)

    projection_config = context.qnext_projection_config
    projection_config['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = \
        param_dict['QNext_H_learning_rate']
    projection_config['H1']['E']['Input']['E']['weight_init_args'] = (param_dict['QNext_H1_init_weight_scale'],)

    projection_config['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = \
        param_dict['QNext_Output_learning_rate']
    projection_config['Output']['E']['H1']['E']['weight_init_args'] = (param_dict['QNext_Output_init_weight_scale'],)


def set_bpDale_config(param_dict, layer_config, projection_config, prefix=''):
    """
    Map E/I (Dale's law) params onto a nested layer_config and projection_config, with the old branch's names
    (update_EIANN_config_2_hidden_backprop_Dale_relu_SGD), for any layers:
    - sizes: {prefix}H_I_size for the SomaI of every hidden layer, {prefix}Output_I_size for Output's SomaI;
    - learning rates: {prefix}H_E_E_learning_rate onto hidden E from E, {prefix}Output_E_E_learning_rate onto Output E
      from E, {prefix}H_E_I_learning_rate onto any E from SomaI, {prefix}H_I_E_learning_rate onto any SomaI;
    - weight scales: {prefix}<post layer>_<E|I>_<pre layer>_<E|I>_init_weight_scale for every projection.
    Projections without a learning rule (learning_rule: null, e.g. the fixed SomaI projections of eprop_bpDale) get no
    learning rate, so a learning rate param is only read when some projection of its group is learned.
    :param prefix: str; e.g. 'QNext_' for DPPQL's QNext network
    """
    def tag(pop):
        return 'I' if pop == 'SomaI' else pop

    for layer, pops in layer_config.items():
        if 'SomaI' in pops:
            size_name = 'Output_I_size' if layer == 'Output' else 'H_I_size'
            pops['SomaI']['size'] = int(param_dict[f'{prefix}{size_name}'])

    for post_layer, post_pops in projection_config.items():
        for post_pop, pre_layers in post_pops.items():
            for pre_layer, pre_pops in pre_layers.items():
                for pre_pop, projection_kwargs in pre_pops.items():
                    if post_pop == 'E' and pre_pop == 'E':
                        learning_rate_name = 'Output_E_E' if post_layer == 'Output' else 'H_E_E'
                    elif post_pop == 'E':
                        learning_rate_name = 'H_E_I'
                    else:
                        learning_rate_name = 'H_I_E'
                    if projection_kwargs.get('learning_rule') is not None:
                        projection_kwargs['learning_rule_kwargs']['learning_rate'] = \
                            param_dict[f'{prefix}{learning_rate_name}_learning_rate']
                    projection_kwargs['weight_init_args'] = (
                        param_dict[f'{prefix}{post_layer}_{tag(post_pop)}_{pre_layer}_{tag(pre_pop)}_init_weight_scale'],)


def update_EIANN_config_2_hidden_2ACDC_bpDale_relu_SGD(x, context):
    param_dict = param_array_to_dict(x, context.param_names)

    set_bpDale_config(param_dict, context.layer_config, context.projection_config)


def update_qnext_bpDale(x, context):
    """
    DPPQL's QNext network with E and SomaI populations; params as in set_bpDale_config, prefixed QNext_.
    """
    param_dict = param_array_to_dict(x, context.param_names)

    set_bpDale_config(param_dict, context.qnext_layer_config, context.qnext_projection_config, prefix='QNext_')


def update_EIANN_config_2_hidden_2ACDC_BTSP_Continuous(x, context):
    """
    2-hidden E/SomaI/DendI network with BTSP_Continuous (rules/btsp_continuous.py) on every learned E <- E projection,
    ported from the old branch's update_EIANN_config_2_hidden_BTSP (treadmill_RL), with a learned recurrent
    H1.E <- H1.E projection added to hold the cue.
    - BTSP_Continuous: Input.E -> H1.E, H1.E -> H1.E (recurrent), H1.E -> H2.E, H2.E -> Output.E. Max weights are
      <max_weight_scale> / sqrt(fan-in); init scales are max_weight_scale * init_weight_factor.
    - Top-down H2.E -> H1.E and Output.E -> H2.E (dendritic) are transposed copies of the feedforward weights
      (clone_weight), scaled by H1_E_H2_E_weight_scale and H2_E_Output_E_weight_scale.
    - H.E <- H.DendI learned with DendriticLoss_6; H.DendI <- H.E and H.DendI <- H.DendI with Hebb_WeightNorm
      (normalize_weight scale = param * sqrt(pre size) / 2). SomaI projections are not learned.
    """
    param_dict = param_array_to_dict(x, context.param_names)
    layer_config = context.layer_config
    projection_config = context.projection_config

    H_I_size = int(param_dict['H_I_size'])
    for layer in ('H1', 'H2'):
        layer_config[layer]['SomaI']['size'] = H_I_size
        layer_config[layer]['DendI']['size'] = H_I_size
    layer_config['Output']['SomaI']['size'] = int(param_dict['Output_I_size'])

    btsp_kwargs = {'neg_rate_th': param_dict['neg_rate_th'], 'max_pop_fraction': param_dict['max_pop_fraction'],
                   'tau_ET': param_dict['tau_ET'], 'tau_IS': param_dict['tau_IS'], 'dep_th': param_dict['dep_th'],
                   'dep_width': param_dict['dep_width']}

    def set_btsp(post_layer, pre_layer, max_weight_scale, init_weight_factor, learning_rate):
        projection_kwargs = projection_config[post_layer]['E'][pre_layer]['E']
        fan_in = layer_config[pre_layer]['E']['size']
        projection_kwargs['weight_bounds'] = (0., max_weight_scale / math.sqrt(fan_in))
        projection_kwargs['weight_init_args'] = (max_weight_scale * init_weight_factor,)
        projection_kwargs['learning_rule_kwargs'].update(btsp_kwargs)
        projection_kwargs['learning_rule_kwargs']['learning_rate'] = learning_rate

    H_E_FF_E_max_weight_scale = param_dict['H_E_FF_E_max_weight_scale']
    H_E_E_learning_rate = param_dict['H_E_E_learning_rate']
    set_btsp('H1', 'Input', H_E_FF_E_max_weight_scale, param_dict['H1_E_Input_E_init_weight_factor'],
             H_E_E_learning_rate)
    set_btsp('H1', 'H1', param_dict['H1_E_H1_E_max_weight_scale'], param_dict['H1_E_H1_E_init_weight_factor'],
             H_E_E_learning_rate)
    set_btsp('H2', 'H1', H_E_FF_E_max_weight_scale, param_dict['H2_E_H1_E_init_weight_factor'], H_E_E_learning_rate)
    set_btsp('Output', 'H2', param_dict['Output_E_H2_E_max_weight_scale'],
             param_dict['Output_E_H2_E_init_weight_factor'], param_dict['Output_E_H2_E_learning_rate'])

    projection_config['H1']['E']['H2']['E']['weight_constraint_kwargs']['scale'] = param_dict['H1_E_H2_E_weight_scale']
    projection_config['H2']['E']['Output']['E']['weight_constraint_kwargs']['scale'] = \
        param_dict['H2_E_Output_E_weight_scale']

    for layer, pre_layer in (('H1', 'Input'), ('H2', 'H1')):
        projection_config[layer]['E'][layer]['SomaI']['weight_init_args'] = \
            (param_dict[f'{layer}_E_{layer}_SomaI_init_weight_scale'],)
        projection_config[layer]['SomaI'][pre_layer]['E']['weight_init_args'] = \
            (param_dict[f'{layer}_SomaI_{pre_layer}_E_init_weight_scale'],)
        projection_config[layer]['SomaI'][layer]['E']['weight_init_args'] = \
            (param_dict[f'{layer}_SomaI_{layer}_E_init_weight_scale'],)
        projection_config[layer]['SomaI'][layer]['SomaI']['weight_init_args'] = \
            (param_dict[f'{layer}_SomaI_{layer}_SomaI_init_weight_scale'],)

        projection_config[layer]['E'][layer]['DendI']['weight_init_args'] = \
            (param_dict[f'{layer}_E_{layer}_DendI_init_weight_scale'],)
        projection_config[layer]['E'][layer]['DendI']['learning_rule_kwargs']['learning_rate'] = \
            param_dict['H_E_DendI_learning_rate']

        DendI_E = projection_config[layer]['DendI'][layer]['E']
        DendI_E['weight_constraint_kwargs']['scale'] = \
            param_dict[f'{layer}_DendI_{layer}_E_weight_scale'] * math.sqrt(layer_config[layer]['E']['size']) / 2
        DendI_E['learning_rule_kwargs']['learning_rate'] = param_dict['DendI_E_learning_rate']
        DendI_DendI = projection_config[layer]['DendI'][layer]['DendI']
        DendI_DendI['weight_constraint_kwargs']['scale'] = \
            param_dict[f'{layer}_DendI_{layer}_DendI_weight_scale'] * math.sqrt(H_I_size) / 2
        DendI_DendI['learning_rule_kwargs']['learning_rate'] = param_dict['DendI_DendI_learning_rate']

    projection_config['Output']['E']['Output']['SomaI']['weight_init_args'] = \
        (param_dict['Output_E_Output_I_init_weight_scale'],)
    projection_config['Output']['SomaI']['H2']['E']['weight_init_args'] = (param_dict['Output_I_H2_E_init_weight_scale'],)
    projection_config['Output']['SomaI']['Output']['E']['weight_init_args'] = \
        (param_dict['Output_I_Output_E_init_weight_scale'],)
    projection_config['Output']['SomaI']['Output']['SomaI']['weight_init_args'] = \
        (param_dict['Output_I_Output_I_init_weight_scale'],)
