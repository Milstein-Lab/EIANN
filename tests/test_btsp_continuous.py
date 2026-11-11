import math

import pytest
import torch

import EIANN.network as nt
from EIANN.recurrent_network import RecurrentNetwork
from EIANN.reinforcement_learning import QLearning, OnlineQLearning
from EIANN.reinforcement_learning.environments import CueTreadmillEnv, RandomSchedule, ScheduledEnv
from EIANN.reinforcement_learning.network_updates import compute_learning_rule_updates, step_learning_rules


def build_single_synapse_network(tau_ET=2., tau_IS=1., dep_ratio=1., rule='BTSP_Continuous_BP', **rule_kwargs):
    """
    Two input units -> one linear output unit through a projection with a backprop-plateau rule (BTSP_Continuous_BP or
    BTSP_Eprop_BP; learning rate 1, no gating). Only input unit 0 is ever active; the synapse under test is
    weight[0, 0].
    """
    kwargs = {'tau_ET': tau_ET, 'tau_IS': tau_IS, 'dep_ratio': dep_ratio, 'relu_gate': False, **rule_kwargs}
    network = nt.Network({'Input': {'E': {'size': 2}}, 'Output': {'E': {'size': 1, 'activation': 'linear'}}},
                         {'Output': {'E': {'Input': {'E': {'direction': 'F',
                                                           'learning_rule': rule,
                                                           'learning_rule_kwargs': kwargs}}}}},
                         learning_rate=1., forward_steps=1, backward_steps=1, tau=1, seed=0)
    projection = network.projections['OutputE_InputE']
    projection.weight.data.fill_(0.5)
    return network, projection


def run_pairing(network, pre_step, plateau_step, num_steps, error=0.25):
    """
    Presynaptic pulse at pre_step, output error (a plateau) only at plateau_step. Updates are accumulated at fixed weights
    and applied at the end. Returns the plateau amplitude.
    """
    model = RecurrentNetwork(network, persistent_state=False)
    model.reset()
    plateau = None
    for t in range(num_steps):
        output = model.step(torch.tensor([1. if t == pre_step else 0., 0.]))
        target = output.detach() + (error if t == plateau_step else 0.)
        compute_learning_rule_updates(network, output, target)
        if t == plateau_step:
            plateau = network.output_pop.plateau.item()
        model.detach()
    step_learning_rules(network)
    return plateau


def expected_delta(rule, w, plateau, pre_step, plateau_step, num_steps):
    delta = 0.
    for t in range(max(pre_step, plateau_step), num_steps):
        S = torch.tensor(plateau * rule.IS_decay ** (t - plateau_step) * rule.ET_decay ** (t - pre_step))
        delta += ((rule.w_max - w) * S - w * rule.dep_ratio * rule.q_dep(S)).item()
    return delta


@pytest.mark.parametrize('order', ['pre_first', 'plateau_first'])
@pytest.mark.parametrize('dep_ratio', [0., 1.])
def test_bidirectional_window_matches_closed_form(order, dep_ratio):
    deltas = []
    for k in (1, 2, 3):
        network, projection = build_single_synapse_network(dep_ratio=dep_ratio)
        rule = projection.learning_rule
        pre_step, plateau_step = (0, k) if order == 'pre_first' else (k, 0)
        num_steps = k + 6
        plateau = run_pairing(network, pre_step, plateau_step, num_steps)
        assert plateau > 0.
        delta = projection.weight[0, 0].item() - 0.5
        assert math.isclose(delta, expected_delta(rule, 0.5, plateau, pre_step, plateau_step, num_steps),
                            rel_tol=1e-5)
        deltas.append(delta)
    if dep_ratio == 0.:
        # potentiation alone: the window decays on both sides of the plateau
        assert deltas[0] > deltas[1] > deltas[2] > 0.


def test_no_memory_when_tau_is_0():
    network, projection = build_single_synapse_network(tau_ET=0., tau_IS=0.)
    run_pairing(network, pre_step=0, plateau_step=1, num_steps=4)
    assert projection.weight[0, 0].item() == 0.5

    network, projection = build_single_synapse_network(tau_ET=0., tau_IS=0.)
    rule = projection.learning_rule
    plateau = run_pairing(network, pre_step=1, plateau_step=1, num_steps=4)
    S = torch.tensor(plateau)
    expected = ((rule.w_max - 0.5) * S - 0.5 * rule.dep_ratio * rule.q_dep(S)).item()
    assert math.isclose(projection.weight[0, 0].item() - 0.5, expected, rel_tol=1e-6)


def test_negative_plateau_is_linear_and_instantaneous():
    # as in BTSP_20: dW = negative plateau * presynaptic activity of the same step, with no trace
    network, projection = build_single_synapse_network(tau_ET=2., tau_IS=1.)
    plateau = run_pairing(network, pre_step=2, plateau_step=2, num_steps=6, error=-0.25)
    assert plateau < 0.
    assert math.isclose(projection.weight[0, 0].item() - 0.5, plateau, rel_tol=1e-6)

    network, projection = build_single_synapse_network(tau_ET=2., tau_IS=1.)
    run_pairing(network, pre_step=0, plateau_step=2, num_steps=6, error=-0.25)
    assert projection.weight[0, 0].item() == 0.5


def test_traces_last_the_sequence_without_persistent_state():
    network, projection = build_single_synapse_network(tau_ET=2.)
    rule = projection.learning_rule
    model = RecurrentNetwork(network, persistent_state=False)
    model.reset()
    model.step(torch.tensor([1., 0.]))
    model.step(torch.tensor([0., 0.]))
    assert math.isclose(rule.ET[0].item(), math.exp(-1. / 2.), rel_tol=1e-6)
    model.reset()
    model.step(torch.tensor([0., 0.]))
    assert torch.all(rule.ET == 0.) and rule.IS.item() == 0.


def test_updates_apply_only_at_step():
    network, projection = build_single_synapse_network()
    model = RecurrentNetwork(network)
    model.reset()
    for t in range(3):
        output = model.step(torch.tensor([1., 0.]))
        compute_learning_rule_updates(network, output, output.detach() + 0.25)
        assert projection.weight[0, 0].item() == 0.5
        model.detach()
    step_learning_rules(network)
    assert projection.weight[0, 0].item() > 0.5
    assert torch.all(projection.learning_rule.accumulated_delta == 0.)


def test_requires_recurrent_network_step():
    network, projection = build_single_synapse_network()
    output = network.forward(torch.tensor([1., 0.]))
    with pytest.raises(RuntimeError):
        compute_learning_rule_updates(network, output, output.detach() + 0.25)


def build_q_network(learning_rule, **learning_rule_kwargs):
    """
    Small feedforward Q network for the treadmill (the traces carry history across steps). Without inhibition,
    excitatory recurrence would make activity diverge. The dendritic variant gets top-down Output.E -> H1.E dendritic
    inputs.
    """
    proj = lambda direction: {'weight_init': 'scaled_kaiming', 'weight_init_args': (1.,), 'direction': direction,
                              'weight_bounds': (0., 2.), 'learning_rule': learning_rule,
                              'learning_rule_kwargs': dict(learning_rule_kwargs)}
    h1_projections = {'Input': {'E': proj('F')}}
    if learning_rule in ('BTSP_Continuous', 'BTSP_Eprop'):
        h1_projections['Output'] = {'E': {'weight_init': 'half_kaiming', 'weight_bounds': (0., None),
                                          'compartment': 'dend', 'learning_rule': None}}
    return nt.Network({'Input': {'E': {'size': 5}}, 'H1': {'E': {'size': 16, 'activation': 'relu'}},
                       'Output': {'E': {'size': 2, 'activation': 'relu'}}},
                      {'H1': {'E': h1_projections}, 'Output': {'E': {'H1': {'E': proj('F')}}}},
                      learning_rate=0.05, forward_steps=1, backward_steps=1, tau=1, seed=0)


@pytest.mark.parametrize('learning_rule, trace_kwargs', [('BTSP_Continuous', {'tau_ET': 3., 'tau_IS': 1.}),
                                                         ('BTSP_Continuous_BP', {'tau_ET': 3., 'tau_IS': 1.}),
                                                         ('BTSP_Eprop', {'tau_ET': 3.}),
                                                         ('BTSP_Eprop_BP', {'tau_ET': 3.})])
@pytest.mark.parametrize('accumulate_updates', [False, True])
def test_online_q_learning_treadmill(learning_rule, trace_kwargs, accumulate_updates):
    network = build_q_network(learning_rule, **trace_kwargs)
    weights_before = {name: projection.weight.detach().clone() for name, projection in network.projections.items()}
    tracks = [CueTreadmillEnv(cue=0, rewarded_zone=0), CueTreadmillEnv(cue=1, rewarded_zone=1)]
    OnlineQLearning(network, accumulate_updates=accumulate_updates, seed=0).learn(
        ScheduledEnv(RandomSchedule(tracks, seed=0)), episodes=5)
    changed = False
    for name, projection in network.projections.items():
        assert not torch.isnan(projection.weight).any()
        if projection.learning_rule.__class__.__name__ == learning_rule:
            assert projection.weight.min() >= 0. and projection.weight.max() <= 2.
            changed |= not torch.equal(projection.weight, weights_before[name])
    assert changed


@pytest.mark.parametrize('learning_rule', ['BTSP_Continuous_BP', 'BTSP_Eprop_BP'])
def test_episode_batched_q_learning_raises(learning_rule):
    with pytest.raises(AssertionError):
        QLearning(build_q_network(learning_rule), persistent_state=True).learn(CueTreadmillEnv(), episodes=1)


def expected_eprop_delta(rule, w, plateau, trace):
    S = torch.tensor(max(plateau, 0.) * trace)
    return ((rule.w_max - w) * S - w * rule.dep_ratio * rule.q_dep(S)).item() + min(plateau, 0.) * trace


def test_eprop_trace_window_is_causal():
    deltas = []
    for k in (0, 1, 2, 3):
        network, projection = build_single_synapse_network(tau_ET=3., tau_IS=0., dep_ratio=0., rule='BTSP_Eprop_BP')
        rule = projection.learning_rule
        plateau = run_pairing(network, pre_step=0, plateau_step=k, num_steps=k + 2)
        trace = (1. - rule.ET_decay) * rule.ET_decay ** k
        delta = projection.weight[0, 0].item() - 0.5
        assert math.isclose(delta, expected_eprop_delta(rule, 0.5, plateau, trace), rel_tol=1e-5)
        deltas.append(delta)
    assert deltas[0] > deltas[1] > deltas[2] > deltas[3] > 0.

    # a plateau does not credit presynaptic activity that comes after it
    network, projection = build_single_synapse_network(tau_ET=3., tau_IS=0., rule='BTSP_Eprop_BP')
    run_pairing(network, pre_step=2, plateau_step=0, num_steps=5)
    assert projection.weight[0, 0].item() == 0.5


def test_eprop_negative_plateau_pairs_with_trace():
    network, projection = build_single_synapse_network(tau_ET=3., tau_IS=0., rule='BTSP_Eprop_BP')
    rule = projection.learning_rule
    plateau = run_pairing(network, pre_step=0, plateau_step=2, num_steps=4, error=-0.25)
    assert plateau < 0.
    trace = (1. - rule.ET_decay) * rule.ET_decay ** 2
    assert math.isclose(projection.weight[0, 0].item() - 0.5, plateau * trace, rel_tol=1e-5)


def test_eprop_same_step_when_tau_is_0():
    network, projection = build_single_synapse_network(tau_ET=0., tau_IS=0., rule='BTSP_Eprop_BP')
    rule = projection.learning_rule
    plateau = run_pairing(network, pre_step=1, plateau_step=1, num_steps=3)
    assert math.isclose(projection.weight[0, 0].item() - 0.5, expected_eprop_delta(rule, 0.5, plateau, 1.),
                        rel_tol=1e-6)

    network, projection = build_single_synapse_network(tau_ET=0., tau_IS=0., rule='BTSP_Eprop_BP')
    run_pairing(network, pre_step=0, plateau_step=1, num_steps=3)
    assert projection.weight[0, 0].item() == 0.5


def expected_eprop_IS_delta(rule, w, plateau, pre_step, plateau_step, num_steps):
    """
    Closed-form dW (weights fixed during the sequence) for one presynaptic pulse and one plateau, with decaying IS.
    """
    delta = 0.
    for t in range(num_steps):
        e = (1. - rule.ET_decay) * rule.ET_decay ** (t - pre_step) if t >= pre_step else 0.
        IS = max(plateau, 0.) * rule.IS_decay ** (t - plateau_step) if t >= plateau_step else 0.
        S = torch.tensor(IS * e)
        delta += ((rule.w_max - w) * S - w * rule.dep_ratio * rule.q_dep(S)).item()
        if t == plateau_step:
            delta += min(plateau, 0.) * e
    return delta


@pytest.mark.parametrize('order', ['pre_first', 'plateau_first'])
def test_eprop_decaying_IS_window_is_bidirectional(order):
    deltas = []
    for k in (1, 2, 3):
        network, projection = build_single_synapse_network(tau_ET=3., tau_IS=1., dep_ratio=0., rule='BTSP_Eprop_BP')
        rule = projection.learning_rule
        pre_step, plateau_step = (0, k) if order == 'pre_first' else (k, 0)
        plateau = run_pairing(network, pre_step, plateau_step, num_steps=k + 6)
        delta = projection.weight[0, 0].item() - 0.5
        assert math.isclose(delta, expected_eprop_IS_delta(rule, 0.5, plateau, pre_step, plateau_step, k + 6),
                            rel_tol=1e-5)
        deltas.append(delta)
    assert deltas[0] > deltas[1] > deltas[2] > 0.


def test_eprop_negative_plateau_has_no_trace():
    # negative plateaus give a direct linear update on their own step only, even with tau_IS > 0
    network, projection = build_single_synapse_network(tau_ET=3., tau_IS=1., rule='BTSP_Eprop_BP')
    plateau = run_pairing(network, pre_step=2, plateau_step=0, num_steps=6, error=-0.25)
    assert plateau < 0.
    assert projection.weight[0, 0].item() == 0.5

    network, projection = build_single_synapse_network(tau_ET=3., tau_IS=1., rule='BTSP_Eprop_BP')
    rule = projection.learning_rule
    plateau = run_pairing(network, pre_step=0, plateau_step=2, num_steps=6, error=-0.25)
    trace = (1. - rule.ET_decay) * rule.ET_decay ** 2
    assert math.isclose(projection.weight[0, 0].item() - 0.5, plateau * trace, rel_tol=1e-5)


@pytest.mark.parametrize('rule', ['BTSP_Continuous_BP', 'BTSP_Eprop_BP'])
def test_substep_clock_keeps_time_constants(rule):
    # a presynaptic pulse lasting one step, then silence. With the substep clock (5 ticks of dt/5 per step), ET still
    # decays by exp(-1/tau_ET) per step, and IS_decay ** 5 == exp(-1/tau_IS)
    tau_ET, tau_IS, n = 3., 2., 5
    ET = {}
    for trace_clock in ('step', 'substep'):
        network, projection = build_single_synapse_network(tau_ET=tau_ET, tau_IS=tau_IS, rule=rule)
        network.forward_steps = network.backward_steps = n
        model = RecurrentNetwork(network, persistent_state=True, trace_clock=trace_clock)
        rule_instance = projection.learning_rule
        assert rule_instance.ticks_per_step == (n if trace_clock == 'substep' else 1)
        assert rule_instance.IS_decay ** rule_instance.ticks_per_step == pytest.approx(math.exp(-1. / tau_IS))
        model.reset()
        ET[trace_clock] = []
        for t in range(4):
            model.step(torch.tensor([1. if t == 0 else 0., 0.]), no_grad=True)
            ET[trace_clock].append(rule_instance.ET[0].item())
    for values in ET.values():
        for t in range(1, 4):
            assert values[t] / values[t - 1] == pytest.approx(math.exp(-1. / tau_ET))
    d = math.exp(-1. / (n * tau_ET))
    if rule == 'BTSP_Eprop_BP':
        # the normalized trace reaches 1 - exp(-1/tau_ET) after a one-step pulse with either clock
        assert ET['substep'][0] == pytest.approx(ET['step'][0]) == pytest.approx(1. - math.exp(-1. / tau_ET))
    else:
        # ET adds 1/n of the activity per tick: sum of d^k / n over the 5 ticks of the pulse, vs 1 with one tick
        assert ET['step'][0] == pytest.approx(1.)
        assert ET['substep'][0] == pytest.approx(sum(d ** k for k in range(n)) / n)


def test_ticks_per_step_must_be_positive():
    network, projection = build_single_synapse_network()
    with pytest.raises(ValueError):
        projection.learning_rule.set_ticks_per_step(0)
