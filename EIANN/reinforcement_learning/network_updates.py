import torch


def reinit_learning_rules(network):
    """
    Initialize learning rule parameters, as at the start of Network.train.
    """
    for post_layer in network:
        for post_pop in post_layer:
            if post_pop.include_bias:
                post_pop.bias_learning_rule.reinit()
            for projection in post_pop:
                projection.learning_rule.reinit()


def apply_learning_rules(network, output, target, store_history=False, store_dynamics=False):
    """
    Update weights and biases from one forward pass, as in one train step of Network.train: run the backward methods
    (compute_learning_rule_updates), then step, constrain and update every learning rule (step_learning_rules).

    Parameters
    ----------
    network : Network
    output : torch.Tensor
        Output of the forward pass to learn from (with grad, for rules that use autograd).
    target : torch.Tensor
        Same shape as output.
    store_history : bool, default False
    store_dynamics : bool, default False
    """
    compute_learning_rule_updates(network, output, target, store_history=store_history, store_dynamics=store_dynamics)
    step_learning_rules(network)


def compute_learning_rule_updates(network, output, target, store_history=False, store_dynamics=False):
    """
    Update the forward state and run every registered backward method. Rules that accumulate updates in backward and
    apply them in step() (e.g. Eprop) can run this for several forward passes before step_learning_rules; rules that
    apply their update inside backward (e.g. Backprop) are updated here.
    """
    network.update_forward_state(store_history=store_history, store_dynamics=store_dynamics)

    for backward in network.backward_methods:
        backward(network, output, target, store_history=store_history, store_dynamics=store_dynamics)


def step_learning_rules(network):
    """
    Step, constrain and update every learning rule.
    """
    with torch.no_grad():
        for post_layer in network:
            for post_pop in post_layer:
                if post_pop.include_bias:
                    post_pop.bias_learning_rule.step()
                for projection in post_pop:
                    projection.learning_rule.step()

        network.constrain_weights_and_biases()

        for post_layer in network:
            for post_pop in post_layer:
                if post_pop.include_bias:
                    post_pop.bias_learning_rule.update()
                for projection in post_pop:
                    projection.learning_rule.update()
