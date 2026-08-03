from nested.utils import param_array_to_dict
import math


def update_EIANN_van_bp_relu_SGD(x, context):
    """
    Update the config for BOTH networks in the dual-network Q learning (DNQL) setup
    from a single flat parameter vector.

    Q-network params are prefixed 'Q_', QNext-network params 'QNext_'. Each network
    owns a namespaced copy of its config on the context (context.q_* / context.qnext_*),
    populated in init_context, so the two networks are tuned independently.

    Q network:     Input -> H1 (recurrent) -> H2 -> Output
    QNext network: Input (= Q hidden activity) -> H1 -> Output  (no recurrence)
    """
    param_dict = param_array_to_dict(x, context.param_names)

    # ------------------------------------------------------------------ #
    # Q network
    # ------------------------------------------------------------------ #
    q_H_learning_rate = param_dict['Q_H_learning_rate']
    q_Output_learning_rate = param_dict['Q_Output_learning_rate']
    q_H1_init_weight_scale = param_dict['Q_H1_init_weight_scale']
    q_H1_H1_init_weight_scale = param_dict['Q_H1_H1_init_weight_scale']
    q_H2_init_weight_scale = param_dict['Q_H2_init_weight_scale']
    q_Output_init_weight_scale = param_dict['Q_Output_init_weight_scale']

    q_proj = context.q_projection_config
    q_proj['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = q_H_learning_rate
    q_proj['H1']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = q_H_learning_rate
    q_proj['H1']['E']['Input']['E']['weight_init_args'] = (q_H1_init_weight_scale,)
    q_proj['H1']['E']['H1']['E']['weight_init_args'] = (q_H1_H1_init_weight_scale,)

    q_proj['H2']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = q_H_learning_rate
    q_proj['H2']['E']['H1']['E']['weight_init_args'] = (q_H2_init_weight_scale,)

    q_proj['Output']['E']['H2']['E']['learning_rule_kwargs']['learning_rate'] = q_Output_learning_rate
    q_proj['Output']['E']['H2']['E']['weight_init_args'] = (q_Output_init_weight_scale,)

    context.q_training_kwargs['optimizer'] = 'SGD'

    # ------------------------------------------------------------------ #
    # QNext network
    # ------------------------------------------------------------------ #
    qnext_H_learning_rate = param_dict['QNext_H_learning_rate']
    qnext_Output_learning_rate = param_dict['QNext_Output_learning_rate']
    qnext_H1_init_weight_scale = param_dict['QNext_H1_init_weight_scale']
    qnext_Output_init_weight_scale = param_dict['QNext_Output_init_weight_scale']

    qnext_proj = context.qnext_projection_config
    qnext_proj['H1']['E']['Input']['E']['learning_rule_kwargs']['learning_rate'] = qnext_H_learning_rate
    qnext_proj['H1']['E']['Input']['E']['weight_init_args'] = (qnext_H1_init_weight_scale,)

    qnext_proj['Output']['E']['H1']['E']['learning_rule_kwargs']['learning_rate'] = qnext_Output_learning_rate
    qnext_proj['Output']['E']['H1']['E']['weight_init_args'] = (qnext_Output_init_weight_scale,)

    context.qnext_training_kwargs['optimizer'] = 'SGD'
