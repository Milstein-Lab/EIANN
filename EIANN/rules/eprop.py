import torch
from .base_classes import LearningRule


class Eprop(LearningRule):
    """
    Symmetric e-prop (Bellec et al. 2020, https://doi.org/10.1038/s41467-020-17236-y) for networks run one sample at a
    time over a sequence, e.g. the steps of an episode with EIANN.recurrent_network.RecurrentNetwork.

    Every integration substep updates the post state as s_t = (1 - 1/tau) s_{t-1} + (bias + sum_i W_i x_i,t) / tau,
    where x_t is the presynaptic activity (of the previous substep for recurrent projections). e-prop keeps only the
    path from a weight to the post state through the post unit's own leak, giving the eligibility trace
        e_t = (1 - 1/tau) e_{t-1} + x_t / tau,
    ticked once per substep, which depends only on the presynaptic unit and is stored as a vector. The learning signal
    L_t = dL_t/ds_t is backpropagated within the current substep only, and is nonzero only at the last substep of a step,
    where the loss is. The gradient is the sum over steps of the outer product L_t e_t. With tau == 1 the trace has no
    memory and the gradient equals backprop truncated to the current substep.

    With forward_steps == 1, a step is one substep. With forward_steps > 1, the traces must tick every substep: run the
    network with RecurrentNetwork(trace_clock='substep') (RL algorithms: trace_clock='substep'). Otherwise the trace
    would see only the last substep's activity once per step, and backward raises.

    The rule's methods split the work:
      forward()  update the eligibility trace after every forward step (called by RecurrentNetwork.step, which also
                 resets the trace at the start of every sequence via reset_eligibility_trace),
      backward() add L_t e_t for the current step to the accumulated gradient,
      step()     apply the accumulated gradient with the network's optimizer and clear it.
    Calling step() after every backward() updates online; calling it once per episode accumulates over the episode.

    Requirements: forward_steps == 1 or trace_clock 'substep'; somatic projections; backward runs on the output of a
    single step; all trainable projections of the network use Eprop. Autapses of recurrent projections are learned like any other weight; remove
    them with the no_autapses weight constraint.
    """
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)
        assert projection.compartment not in ['dend', 'dendrite'], 'Eprop: only somatic projections are supported'
        # gradients are only taken with respect to population states, but the states must be part of the graph
        projection.weight.requires_grad = True
        self.eligibility_trace = None
        self.accumulated_grad = torch.zeros_like(projection.weight)

    def reinit(self):
        self.eligibility_trace = None
        self.accumulated_grad.zero_()

    def reset_eligibility_trace(self):
        self.eligibility_trace = None

    def forward(self):
        """
        Update the eligibility trace from the presynaptic activity of the integration substep that just ran.
        """
        pre = self.projection.pre
        tau = self.projection.post.tau
        # recurrent projections drive the post population with the presynaptic activity of the previous step
        if self.projection.direction in ['recurrent', 'R']:
            pre_activity = pre.prev_activity.detach()
        else:
            pre_activity = pre.activity.detach()
        if self.eligibility_trace is None:
            self.eligibility_trace = pre_activity / tau
        else:
            self.eligibility_trace = (1. - 1. / tau) * self.eligibility_trace + pre_activity / tau

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        assert output.shape == network.output_pop.activity.shape, \
            'Eprop: backward takes the output of a single step (e.g. OnlineQLearning)'
        rules = [projection.learning_rule for projection in network.projections.values()
                 if isinstance(projection.learning_rule, cls)]
        if network.forward_steps > 1 and getattr(network, 'trace_clock', 'step') != 'substep':
            raise RuntimeError(f'Eprop: forward_steps is {network.forward_steps}, so eligibility traces must tick every '
                               f'integration substep: run the network with RecurrentNetwork(trace_clock=\'substep\') '
                               f'(RL algorithms: trace_clock=\'substep\')')
        if any(rule.eligibility_trace is None for rule in rules):
            raise RuntimeError('Eprop: eligibility traces are updated by forward(); run the network with '
                               'EIANN.recurrent_network.RecurrentNetwork.step')

        post_pops = {rule.projection.post.fullname: rule.projection.post for rule in rules}
        post_pops = {name: pop for name, pop in post_pops.items() if pop.state.requires_grad}
        loss = network.criterion(output, target)
        learning_signals = dict(zip(post_pops, torch.autograd.grad(
            loss, [pop.state for pop in post_pops.values()], allow_unused=True)))

        for rule in rules:
            learning_signal = learning_signals.get(rule.projection.post.fullname)
            if learning_signal is not None:
                rule.accumulated_grad += torch.outer(learning_signal, rule.eligibility_trace)

    def step(self):
        # backward never sets .grad, so the optimizer only updates this projection's weight (with the learning rate of
        # its param group); its state (e.g. momentum, Adam moments) is kept by the network's optimizer as usual
        weight = self.projection.weight
        weight.grad = self.accumulated_grad.clone()
        self.projection.post.network.optimizer.step()
        weight.grad = None
        self.accumulated_grad.zero_()
