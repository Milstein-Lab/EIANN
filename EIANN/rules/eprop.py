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

    With forward_steps == 1, a step is one substep. With forward_steps > 1, the clock must be chosen explicitly:
    - RecurrentNetwork(trace_clock='substep') (RL algorithms: trace_clock='substep') ticks the traces every substep,
      which gives the gradient above exactly;
    - trace_clock='step' ticks them once per step with the last substep's activity: an approximation in which tau is in
      steps, and the earlier substeps of a step reach the gradient only through the learning signal of the last one.
    With no clock set (trace_clock=None, the default), backward raises.

    The rule's methods split the work:
      forward()  update the eligibility trace after every forward step (called by RecurrentNetwork.step, which also
                 resets the trace at the start of every sequence via reset_eligibility_trace),
      backward() add L_t e_t for the current step to the accumulated gradient,
      step()     apply the accumulated gradient with the network's optimizer and clear it.
    Calling step() after every backward() updates online; calling it once per episode accumulates over the episode.

    Requirements: forward_steps == 1 or an explicit trace_clock ('substep', or 'step' as an approximation); somatic projections; backward runs on the output of a
    single step; all trainable projections of the network use Eprop (or its subclass Eprop_SomaI, for projections onto
    populations that reach the loss only through recurrent projections). Autapses of recurrent projections are learned like any other weight; remove
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

    def signal_target(self):
        """
        The tensor whose gradient with respect to the loss is this rule's learning signal: the post state of the last
        substep. Subclasses (Eprop_SomaI) take it elsewhere.
        """
        return self.projection.post.state

    def signal_trace(self):
        """
        The eligibility trace paired with the learning signal of signal_target.
        """
        return self.eligibility_trace

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        assert output.shape == network.output_pop.activity.shape, \
            'Eprop: backward takes the output of a single step (e.g. OnlineQLearning)'
        # Eprop and its subclasses share this backward, which is registered once per network under whichever class
        # registered first, so gather by Eprop rather than cls
        rules = [projection.learning_rule for projection in network.projections.values()
                 if isinstance(projection.learning_rule, Eprop)]
        if network.forward_steps > 1 and getattr(network, 'trace_clock', None) is None:
            raise RuntimeError(f'Eprop: forward_steps is {network.forward_steps}, so choose when eligibility traces '
                               f'tick: RecurrentNetwork(trace_clock=\'substep\') for every integration substep (exact), '
                               f'or trace_clock=\'step\' for once per step (approximate); RL algorithms take the same '
                               f'trace_clock kwarg')
        if any(rule.eligibility_trace is None for rule in rules):
            raise RuntimeError('Eprop: eligibility traces are updated by forward(); run the network with '
                               'EIANN.recurrent_network.RecurrentNetwork.step')

        if any(isinstance(rule, Eprop_SomaI) for rule in rules):
            Eprop_SomaI.check_network(network, [rule for rule in rules if isinstance(rule, Eprop_SomaI)])

        # one gradient target per tensor: rules onto the same population with the same kind of signal share it. All
        # signals come from one autograd.grad call, since a second call would traverse the freed graph
        targets = {}
        for rule in rules:
            signal_target = rule.signal_target()
            if signal_target is not None and signal_target.requires_grad:
                targets.setdefault(id(signal_target), signal_target)
        loss = network.criterion(output, target)
        learning_signals = dict(zip(targets, torch.autograd.grad(loss, list(targets.values()), allow_unused=True)))

        for rule in rules:
            signal_target, signal_trace = rule.signal_target(), rule.signal_trace()
            learning_signal = None if signal_target is None else learning_signals.get(id(signal_target))
            if learning_signal is not None and signal_trace is not None:
                rule.accumulated_grad += torch.outer(learning_signal, signal_trace)

    def step(self):
        # backward never sets .grad, so the optimizer only updates this projection's weight (with the learning rate of
        # its param group); its state (e.g. momentum, Adam moments) is kept by the network's optimizer as usual
        weight = self.projection.weight
        weight.grad = self.accumulated_grad.clone()
        self.projection.post.network.optimizer.step()
        weight.grad = None
        self.accumulated_grad.zero_()


class Eprop_SomaI(Eprop):
    """
    e-prop for projections onto a population whose activity reaches the loss only through recurrent projections, like
    the SomaI populations of the Dale's law networks (weights from Input, E and SomaI onto SomaI). Used alongside Eprop,
    which handles every other projection, including E <- SomaI.

    Why Eprop gives these projections no learning signal
    ------------------------------------------------------
    Eprop's learning signal is L_T = dLoss/ds_post(T), backpropagated within the last substep T of a step, the only
    substep with a loss. SomaI drives E only through recurrent projections (E <- SomaI, direction R), which read SomaI's
    activity of the previous substep. A change to SomaI's state at T therefore reaches E only at T + 1, after the loss:
    dLoss/ds_SomaI(T) = 0, and Eprop would leave every projection onto SomaI frozen without a warning. (E <- SomaI
    itself learns fine with Eprop: its post population is E, whose state at T is on the path to the loss.)

    What this rule does instead
    ----------------------------
    SomaI's state one substep earlier, s_I(T-1), does reach the loss, through
        s_I(T-1) -> a_I(T-1) -> [E <- SomaI at substep T] -> s_E(T) -> ... -> output(T) -> loss.
    So the learning signal is taken one substep back, L_I = dLoss/ds_I(T-1), and paired with the eligibility trace as it
    stood after substep T-1, e(T-1):
        gradient += outer(L_I, e(T-1)).
    This is e-prop's usual truncation with the signal delayed by one substep: the trace keeps only the path from the
    weight to s_I through SomaI's own leak, and L_I keeps only paths that reach the loss within substep T. The other
    paths from s_I(T-1) (to s_I(T) through the leak, to SomaI(T) through SomaI <- SomaI) do not reach the loss at T, so
    L_I is the exact partial derivative.

    How L_I is computed
    -------------------
    With backward_steps 1 the earlier substeps run without autograd, so s_I(T-1) is not in the graph. Instead, after
    every substep (when RecurrentNetwork ticks the traces, i.e. learning_rule.forward()), the first Eprop_SomaI rule
    onto a population makes its state a gradient leaf and recomputes the activity from it:
        leaf = post.state.detach().requires_grad_()
        post.activity = post.activation(leaf)        # same values, now in a graph rooted at leaf
    The next substep's network.forward copies activity into prev_activity, which the E <- SomaI projection reads, so the
    graph of output(T) runs back to the leaf made after substep T-1. backward then takes the learning signal as the
    gradient with respect to that leaf (autograd also applies SomaI's activation derivative), in the same
    torch.autograd.grad call as Eprop's signals. The population keeps the latest leaf and the one before it, which is
    the one used. Rules onto the same population share both, so the activity is replaced once per substep.

    Only the identity of post.activity changes, not its values, so forward dynamics, histories and the learning
    signals of the Eprop rules are the same as with these projections fixed.

    Requirements: trace_clock='substep' and forward_steps > 1 (with forward_steps 1, T-1 is the previous step, which
    RL algorithms cut from the graph with RecurrentNetwork.detach); the post population must not drive any projection
    in the forward direction (its activity must reach the loss only one substep later); somatic projections, as Eprop.
    """
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)
        assert projection.post is not projection.post.network.output_pop, \
            'Eprop_SomaI: the output population reaches the loss directly; use Eprop'
        # trace after substep T-1, paired with the learning signal at T-1
        self.prev_eligibility_trace = None

    def reinit(self):
        super().reinit()
        self.reset_eligibility_trace()

    def reset_eligibility_trace(self):
        super().reset_eligibility_trace()
        self.prev_eligibility_trace = None
        post = self.projection.post
        post.eprop_state_leaf = post.eprop_prev_state_leaf = post.eprop_leaf_activity = None

    def forward(self):
        """
        After every substep: make the post state a gradient leaf (once per population), then shift and tick the trace.
        """
        post = self.projection.post
        # a fresh activity tensor means this substep's leaf has not been made yet by another rule onto post
        if getattr(post, 'eprop_leaf_activity', None) is not post.activity:
            post.eprop_prev_state_leaf = getattr(post, 'eprop_state_leaf', None)
            leaf = post.state.detach().requires_grad_()
            post.activity = post.activation(leaf)
            post.eprop_state_leaf, post.eprop_leaf_activity = leaf, post.activity
        self.prev_eligibility_trace = self.eligibility_trace
        super().forward()

    def signal_target(self):
        return getattr(self.projection.post, 'eprop_prev_state_leaf', None)

    def signal_trace(self):
        return self.prev_eligibility_trace

    @staticmethod
    def check_network(network, rules):
        if getattr(network, 'trace_clock', None) != 'substep' or network.forward_steps < 2:
            raise RuntimeError(f'Eprop_SomaI: needs trace_clock=\'substep\' and forward_steps > 1 (got trace_clock='
                               f'{getattr(network, "trace_clock", None)!r}, forward_steps={network.forward_steps}): '
                               f'its learning signal is taken at the substep before the last')
        post_pops = {rule.projection.post for rule in rules}
        for projection in network.projections.values():
            if projection.pre in post_pops and projection.direction in ['forward', 'F']:
                raise RuntimeError(f'Eprop_SomaI: {projection.pre.fullname} drives {projection.name} in the forward '
                                   f'direction, so its activity reaches the loss within the last substep and the '
                                   f'delayed learning signal would miss that path; use Eprop')
