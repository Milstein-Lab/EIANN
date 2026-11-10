"""
BTSP rules with continuously decaying eligibility traces (ET) and instructive signals (IS).

The rules in btsp.py pair presynaptic activity and plateaus within one sample (BTSP_20, BTSP_CL and their variants), or
add a single hard-coded cross-sample term (BTSP_19, temporal_discount). Here ET and IS are leaky state variables that
decay once per network step, so a plateau can potentiate synapses that were active several steps before it (ET still
elevated) or several steps after it (IS still elevated): the seconds-long, bidirectional window of BTSP. They are meant
for sequential tasks (RL, memory) run one step at a time with EIANN.recurrent_network.RecurrentNetwork.

BTSP_Eprop and BTSP_Eprop_BP merge e-prop and BTSP: an e-prop-style eligibility trace carries presynaptic activity
across steps, and the weight update has BTSP's weight-dependent form. By default (tau_IS=0) the current step's plateau is
an instantaneous learning signal and the window is causal; tau_IS > 0 adds decaying instructive signals and BTSP's
plateau-before-pre side.
"""
import math

import torch

from .btsp import BTSP_CL, BTSP_BP_CL


def trace_decay(tau, dt):
    """
    Fraction of a trace left after one step of duration dt: exp(-dt / tau), or 0 (no memory) when tau == 0.
    """
    if tau < 0. or dt <= 0.:
        raise ValueError(f'trace_decay: tau must be >= 0 and dt > 0, not tau={tau}, dt={dt}')
    return 0. if tau == 0. else math.exp(-dt / tau)


class BTSPTraceMixin(object):
    """
    Decaying ET and IS shared by the BTSP_Continuous rules. Traces are leaky integrators clamped to [0, 1]:
        ET_j <- clamp(ET_j * exp(-dt / tau_ET) + clamp(x_j, 0, 1) / n, 0, 1)   (presynaptic activity x)
        IS_i <- clamp(IS_i * exp(-dt / tau_IS) + max(plateau_i, 0), 0, 1)
    With the overlap S_ij = IS_i * ET_j, every step with a learning update adds
        dW_ij = (w_max - W_ij) * S_ij - W_ij * dep_ratio * q_dep(S_ij) + min(plateau_i, 0) * clamp(x_j, 0, 1),
    the BTSP_19 update with the plateau amplitude folded into IS. Negative plateaus have no trace: as in BTSP_20 and
    BTSP_CL, they give a linear update with the presynaptic activity of the same step. A positive plateau at step t
    pairs with ET_t, which holds decayed presynaptic activity of earlier steps (pre before plateau); its IS then decays
    over later steps and pairs with later presynaptic activity (plateau before pre). The weight change integrates IS * ET over time, and the
    asymmetry of the window is set by tau_ET vs. tau_IS. With tau_ET == tau_IS == 0, only the current step contributes.

    The clock is the network step (one call to RecurrentNetwork.step, e.g. one environment step), with tau in units of
    dt (dt=1 by default: tau in steps). With RecurrentNetwork(trace_clock='substep'), the traces tick n = forward_steps
    times per step (set_ticks_per_step), each tick lasting dt / n, so tau keeps its meaning; ET adds 1/n of the
    presynaptic activity per tick, so a constant input adds about as much per step as with one tick. IS still receives
    plateaus once per step, at backward. With n = 1 (the default) the rule is unchanged.

    The methods split the work as in Eprop:
      forward()       decay all traces and add the presynaptic activity of the step that just ran to ET (called by
                      RecurrentNetwork.step after every step, so ET sees forward activity, before any nudging),
      reset_traces()  clear the traces (called by RecurrentNetwork.step at the start of every sequence, whether or not
                      population states persist across steps),
      backward()      (defined by each concrete class) compute plateaus with the parent rule's backward, add them to
                      IS, and add this step's dW to the accumulated update,
      step()          apply the accumulated update and clear it.
    Calling step() after every backward() updates online (OnlineQLearning); with accumulate_updates, the summed updates
    of an episode are applied at its end, at the weights the episode ran with.

    Requirements: the network is run with RecurrentNetwork.step; backward runs on the output of a single step (e.g.
    OnlineQLearning, OnlineDPPQL), not on a stacked episode (QLearning). Network.train is not supported.
    """
    def __init__(self, projection, tau_ET=1., tau_IS=1., dt=1., dep_th=0.2, dep_width=0.1, **kwargs):
        """
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0; decay time constant of the eligibility trace, in units of dt
        :param tau_IS: float >= 0; decay time constant of the instructive signals, in units of dt
        :param dt: float > 0; duration of one network step
        :param dep_th: float in [0, 1]; threshold of q_dep on the overlap S. Higher than the 0.01 default of the btsp.py
            rules: there q_dep acts on presynaptic activity (usually near 1), here on a product of decaying traces, and
            with a threshold of 0.01 nearly every step of the trace tails adds full depression, so pairings net
            depression. With dep_th ~0.2-0.3 and dep_width ~0.1, a single pairing potentiates most at coincidence and
            less on both sides
        :param dep_width: float > 0; width of q_dep
        :param kwargs: passed to the parent BTSP rule (e.g. dep_ratio, learning_rate, relu_gate, task_incremental,
            task_classes)
        """
        super().__init__(projection, dep_th=dep_th, dep_width=dep_width, **kwargs)
        self.tau_ET = float(tau_ET)
        self.tau_IS = float(tau_IS)
        self.dt = float(dt)
        self.set_ticks_per_step(1)
        self.accumulated_delta = torch.zeros_like(projection.weight.data)
        self.reset_traces()

    def set_ticks_per_step(self, ticks_per_step):
        """
        Number of trace ticks (calls to forward) per step: 1 by default, forward_steps with
        RecurrentNetwork(trace_clock='substep'). Each tick lasts dt / ticks_per_step.
        :param ticks_per_step: int >= 1
        """
        ticks_per_step = int(ticks_per_step)
        if ticks_per_step < 1:
            raise ValueError(f'{type(self).__name__}: ticks_per_step must be >= 1, not {ticks_per_step}')
        self.ticks_per_step = ticks_per_step
        self.ET_decay = trace_decay(self.tau_ET, self.dt / ticks_per_step)
        self.IS_decay = trace_decay(self.tau_IS, self.dt / ticks_per_step)

    def reinit(self):
        self.reset_traces()
        self.accumulated_delta.zero_()

    def reset_traces(self):
        self.ET = None
        self.IS = None
        self.pre_activity = None

    def forward(self):
        """
        Decay the traces by one tick and add the presynaptic activity of the forward step (or substep) that just ran
        to ET.
        """
        pre = self.projection.pre
        post = self.projection.post
        # recurrent projections drive the post population with the presynaptic activity of the previous step
        if self.projection.direction in ['recurrent', 'R']:
            pre_activity = pre.prev_activity
        else:
            pre_activity = pre.activity
        pre_activity = torch.clamp(pre_activity.detach(), 0., 1.)
        self.pre_activity = pre_activity
        if self.ET is None:
            self.ET = pre_activity / self.ticks_per_step
            self.IS = torch.zeros(post.size, device=post.network.device)
        else:
            self.ET = torch.clamp(self.ET * self.ET_decay + pre_activity / self.ticks_per_step, 0., 1.)
            self.IS = self.IS * self.IS_decay

    def compute_delta_weight(self, plateau):
        """
        Unscaled weight update of the current step: BTSP on the trace overlap, plus the linear update for negative
        plateaus of this step.
        :param plateau: tensor; plateaus of the post population at this step
        :return: tensor
        """
        weight = self.projection.weight.detach()
        overlap = self.IS.unsqueeze(1) * self.ET.unsqueeze(0)
        neg_plateau = torch.clamp(plateau, max=0.)
        return ((self.w_max - weight) * overlap - weight * self.dep_ratio * self.q_dep(overlap) +
                neg_plateau.unsqueeze(1) * self.pre_activity.unsqueeze(0))

    def accumulate_plateau(self):
        """
        Add the positive plateaus of the current step (set by backward) to IS, and this step's weight update to the
        accumulated update.
        """
        if self.ET is None:
            raise RuntimeError(f'{type(self).__name__}: traces are updated by forward(); run the network with '
                               f'EIANN.recurrent_network.RecurrentNetwork.step')
        plateau = self.projection.post.plateau.detach()
        self.IS = torch.clamp(self.IS + torch.clamp(plateau, min=0.), 0., 1.)
        self.accumulated_delta += self.compute_delta_weight(plateau)

    @classmethod
    def check_single_step(cls, network, output):
        assert output.shape == network.output_pop.activity.shape, \
            f'{cls.__name__}: backward takes the output of a single step (e.g. OnlineQLearning)'

    @classmethod
    def accumulate_trace_updates(cls, network):
        for projection in network.projections.values():
            if isinstance(projection.learning_rule, cls):
                projection.learning_rule.accumulate_plateau()

    def step(self):
        self.projection.weight.data += self.learning_rate * self.accumulated_delta
        self.accumulated_delta.zero_()


class BTSP_Continuous(BTSPTraceMixin, BTSP_CL):
    def __init__(self, projection, tau_ET=1., tau_IS=1., dt=1., **kwargs):
        """
        BTSP with decaying ET and IS (see BTSPTraceMixin), with plateaus from dendritic top-down inputs and somatic
        nudging as in BTSP_CL (including task_incremental output errors). Note that with persistent population states,
        the nudged somatic state carries into the next step.
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0
        :param tau_IS: float >= 0
        :param dt: float > 0
        :param kwargs: passed to BTSP_CL (e.g. neg_rate_th, dep_ratio, max_pop_fraction,
            stochastic, learning_rate, relu_gate, task_incremental, task_classes)
        """
        super().__init__(projection, tau_ET=tau_ET, tau_IS=tau_IS, dt=dt, **kwargs)

    # defined here rather than in the mixin: Network registers backward methods by function identity, so every
    # concrete class needs its own
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        cls.check_single_step(network, output)
        super(BTSP_Continuous, cls).backward(network, output, target, store_history=store_history,
                                             store_dynamics=store_dynamics)
        cls.accumulate_trace_updates(network)


class BTSP_Continuous_BP(BTSPTraceMixin, BTSP_BP_CL):
    def __init__(self, projection, tau_ET=1., tau_IS=1., dt=1., **kwargs):
        """
        BTSP with decaying ET and IS (see BTSPTraceMixin), with plateaus computed by backprop as in BTSP_BP_CL
        (-dL/da of the step, scaled by error_scale and clamped to [-1, 1]; no somatic nudging). Requires
        backward_steps >= 1. max_pop_fraction, stochastic and neg_rate_th are ignored.
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0
        :param tau_IS: float >= 0
        :param dt: float > 0
        :param kwargs: passed to BTSP_BP_CL (e.g. error_scale, dep_ratio, learning_rate, relu_gate,
            task_incremental, task_classes)
        """
        super().__init__(projection, tau_ET=tau_ET, tau_IS=tau_IS, dt=dt, **kwargs)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        cls.check_single_step(network, output)
        super(BTSP_Continuous_BP, cls).backward(network, output, target, store_history=store_history,
                                                store_dynamics=store_dynamics)
        cls.accumulate_trace_updates(network)


class BTSPEpropMixin(BTSPTraceMixin):
    """
    e-prop-style credit with BTSP updates, shared by BTSP_Eprop and BTSP_Eprop_BP. Each projection keeps an e-prop
    eligibility trace of its presynaptic activity, a normalized leaky integrator (unit DC gain, as in Eprop but with its
    own time constant tau_ET instead of the post population's tau), ticked once per network step, or once per
    integration substep with RecurrentNetwork(trace_clock='substep') (see BTSPTraceMixin; the normalized trace needs no
    rescaling of its input):
        e_j <- d_ET * e_j + (1 - d_ET) * clamp(x_j, 0, 1),    d_ET = exp(-dt / tau_ET)
    Positive plateaus drive an instructive signal, a leaky integrator:
        IS_i <- clamp(IS_i * d_IS + max(plateau_i, 0), 0, 1),    d_IS = exp(-dt / tau_IS)
    With the overlap S_ij = IS_i * e_j, every step with a learning update adds
        dW_ij = (w_max - W_ij) * S_ij - W_ij * dep_ratio * q_dep(S_ij) + min(plateau_i, 0) * e_j,
    BTSP's update for positive plateaus and a direct linear update for negative ones, on the step of the negative
    plateau only (no negative trace).

    With tau_IS == 0 (the default), IS is the plateau of the current step, an instantaneous learning signal as e-prop's
    L_t: the window is causal, crediting presynaptic activity of the same and earlier steps only, and both signs pair
    with the same trace. With tau_IS > 0, IS persists after a positive plateau and pairs with later presynaptic activity,
    adding BTSP's plateau-before-pre side for potentiation. With tau_ET == tau_IS == 0, only the current step
    contributes.

    Hooks, requirements and the accumulated update are those of BTSPTraceMixin.
    """
    def __init__(self, projection, tau_ET=1., tau_IS=0., dt=1., **kwargs):
        """
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0; decay time constant of the eligibility trace, in units of dt
        :param tau_IS: float >= 0; decay time constant of the instructive signal, in units of dt (0: no IS memory)
        :param dt: float > 0; duration of one network step
        :param kwargs: passed to BTSPTraceMixin and the parent BTSP rule (e.g. dep_th, dep_width, dep_ratio,
            learning_rate, relu_gate, task_incremental, task_classes)
        """
        super().__init__(projection, tau_ET=tau_ET, tau_IS=tau_IS, dt=dt, **kwargs)

    def forward(self):
        """
        Decay the instructive signal and update the eligibility trace with the presynaptic activity of the forward step
        that just ran.
        """
        pre = self.projection.pre
        post = self.projection.post
        # recurrent projections drive the post population with the presynaptic activity of the previous step
        if self.projection.direction in ['recurrent', 'R']:
            pre_activity = pre.prev_activity
        else:
            pre_activity = pre.activity
        pre_activity = torch.clamp(pre_activity.detach(), 0., 1.)
        self.pre_activity = pre_activity
        if self.ET is None:
            self.ET = (1. - self.ET_decay) * pre_activity
            self.IS = torch.zeros(post.size, device=post.network.device)
        else:
            self.ET = self.ET_decay * self.ET + (1. - self.ET_decay) * pre_activity
            self.IS = self.IS * self.IS_decay

    def compute_delta_weight(self, plateau):
        """
        Unscaled weight update of the current step.
        :param plateau: tensor; plateaus of the post population at this step
        :return: tensor
        """
        weight = self.projection.weight.detach()
        overlap = self.IS.unsqueeze(1) * self.ET.unsqueeze(0)
        return ((self.w_max - weight) * overlap - weight * self.dep_ratio * self.q_dep(overlap) +
                torch.clamp(plateau, max=0.).unsqueeze(1) * self.ET.unsqueeze(0))

    def accumulate_plateau(self):
        """
        Add the positive plateaus of the current step (set by backward) to the instructive signal, and this step's weight
        update to the accumulated update.
        """
        if self.ET is None:
            raise RuntimeError(f'{type(self).__name__}: traces are updated by forward(); run the network with '
                               f'EIANN.recurrent_network.RecurrentNetwork.step')
        plateau = self.projection.post.plateau.detach()
        self.IS = torch.clamp(self.IS + torch.clamp(plateau, min=0.), 0., 1.)
        self.accumulated_delta += self.compute_delta_weight(plateau)


class BTSP_Eprop(BTSPEpropMixin, BTSP_CL):
    def __init__(self, projection, tau_ET=1., tau_IS=0., dt=1., **kwargs):
        """
        e-prop-style eligibility trace with BTSP updates (see BTSPEpropMixin), with plateaus from dendritic top-down
        inputs and somatic nudging as in BTSP_CL (including task_incremental output errors).
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0
        :param tau_IS: float >= 0
        :param dt: float > 0
        :param kwargs: passed to BTSP_CL (e.g. neg_rate_th, dep_th, dep_width, dep_ratio, max_pop_fraction, stochastic,
            learning_rate, relu_gate, task_incremental, task_classes)
        """
        super().__init__(projection, tau_ET=tau_ET, tau_IS=tau_IS, dt=dt, **kwargs)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        cls.check_single_step(network, output)
        super(BTSP_Eprop, cls).backward(network, output, target, store_history=store_history,
                                        store_dynamics=store_dynamics)
        cls.accumulate_trace_updates(network)


class BTSP_Eprop_BP(BTSPEpropMixin, BTSP_BP_CL):
    def __init__(self, projection, tau_ET=1., tau_IS=0., dt=1., **kwargs):
        """
        e-prop-style eligibility trace with BTSP updates (see BTSPEpropMixin), with plateaus computed by backprop as in
        BTSP_BP_CL: -dL/da of the step, scaled by error_scale and clamped to [-1, 1]; for relu units with relu_gate this
        is e-prop's learning signal -dL/ds. No somatic nudging. Requires backward_steps >= 1.
        :param projection: :class:'nn.Linear'
        :param tau_ET: float >= 0
        :param tau_IS: float >= 0
        :param dt: float > 0
        :param kwargs: passed to BTSP_BP_CL (e.g. error_scale, dep_th, dep_width, dep_ratio, learning_rate, relu_gate,
            task_incremental, task_classes)
        """
        super().__init__(projection, tau_ET=tau_ET, tau_IS=tau_IS, dt=dt, **kwargs)

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        cls.check_single_step(network, output)
        super(BTSP_Eprop_BP, cls).backward(network, output, target, store_history=store_history,
                                           store_dynamics=store_dynamics)
        cls.accumulate_trace_updates(network)
